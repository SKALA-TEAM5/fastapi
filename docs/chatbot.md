# 산안비 챗봇 에이전트 (Chatbot Agent)

산업안전보건관리비(산안비) 관련 질문에 대해 Qdrant 벡터 DB 기반 RAG 검색 및 LangGraph 멀티 에이전트 파이프라인으로 정확한 답변을 스트리밍 방식으로 제공하는 챗봇 시스템입니다.

---

## 목차

1. [Overview](#1-overview)
2. [Tech Stack](#2-tech-stack)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [Agent 그래프 상세](#4-agent-그래프-상세)
5. [API 명세](#5-api-명세)
6. [MemorySaver — 대화 연속성](#6-memorysaver--대화-연속성)
7. [Qdrant 접속 구조](#7-qdrant-접속-구조)
8. [파일 구조](#8-파일-구조)
9. [환경변수](#9-환경변수)
10. [의존성](#10-의존성)
11. [k8s 배포](#11-k8s-배포)
12. [로컬 개발 환경 세팅](#12-로컬-개발-환경-세팅)
13. [테스트](#13-테스트)
14. [평가 결과](#14-평가-결과)

---

## 1. Overview

### 무엇을 하는가

사용자가 산안비 관련 질문을 입력하면, 시스템이 다음 단계를 거쳐 답변을 생성합니다.

1. 질문 의도(intent)를 LLM으로 분류
2. Qdrant에서 관련 법령·가이드 문서 검색 (BM25 + 벡터 앙상블)
3. Cross-encoder로 검색 결과 재순위 및 관련성 필터링
4. 관련 문서를 context로 LLM 답변 생성
5. 토큰 단위 SSE 스트리밍으로 프론트에 전달

### 주요 특징

- **멀티 에이전트 (LangGraph)** — intent 분류 → retrieval → grading → rewrite → generation 노드를 그래프로 연결
- **Agentic RAG** — 관련 문서가 없으면 질문을 자동 재작성 후 재검색 (최대 3회)
- **SSE Streaming** — 토큰 단위 실시간 응답, 사용자가 LLM 응답을 즉시 확인 가능
- **인메모리 대화 기록** — session_id 기반 멀티턴 대화 지원, 탭 닫으면 자동 소멸
- **범위 외 질문 처리** — 산안비와 무관한 질문은 RAG 없이 fallback 응답 반환

---

## 2. Tech Stack

| 분류 | 기술 |
|---|---|
| Framework | FastAPI |
| Agent Orchestration | LangGraph |
| LLM (분류·평가) | OpenAI GPT-4.1-mini (`OPENAI_MODEL`) |
| LLM (답변 생성) | OpenAI GPT-4.1 (`OPENAI_REPORT_MODEL`) |
| Vector DB | Qdrant (`legal_documents` 컬렉션) |
| Embedding | `jhgan/ko-sroberta-multitask` (HuggingFace) |
| Retrieval | BM25 + Vector Ensemble + Cross-encoder Reranking |
| 대화 기록 | LangGraph `MemorySaver` (인메모리) |
| 응답 방식 | SSE (`text/event-stream`) |

---

## 3. 시스템 아키텍처

```
[Frontend]
    │  POST /api/v1/chat
    │  { "question": "...", "session_id": "abc-123" }
    ▼
[FastAPI — chatbot.py router]
    │
    ▼
[chatbot_service.py]
    │  session_id 없으면 UUID 생성
    │  MemorySaver checkpointer (서버 싱글톤)
    ▼
[ChatbotAgent — LangGraph 그래프]
    │  thread_id = session_id
    │  MemorySaver에서 이전 대화 자동 로드
    ▼
  [노드 실행 — 아래 §4 참조]
    │
    ▼
[SSE StreamingResponse]
    │  text/event-stream
    │  data: {"type": "session_id", "value": "..."}
    │  data: {"type": "intent", "value": "..."}
    │  data: {"type": "token", "value": "..."}
    │  data: {"type": "sources", "value": [...]}
    │  data: [DONE]
    ▼
[Frontend]
```

---

## 4. Agent 그래프 상세

### 그래프 흐름

```
질문 입력
    │
    ▼
[Node 1] intent_classifier
    │
    ├── "기타" ──────────────────────────────────────────────────┐
    │                                                            │
    └── "카테고리판단" | "법령한도" | "적법성판단"               │
            │                                                    │
            ▼                                                    ▼
    [Node 2] retriever                               [Node 3] fallback_handler
            │                                                    │
            ▼                                                    │
    [Node 4] doc_grader                                          │
            │                                                    │
            ├── relevance_ok == False AND retry < 3              │
            │       ▼                                            │
            │   [Node 5] rewrite_query ──► Node 2 (루프백)       │
            │                                                    │
            └── relevance_ok == True OR retry >= 3               │
                    │                                            │
                    ▼                                            │
            [Node 6] answer_generator ◄──────────────────────────┘
                    │
                    ▼
              SSE Streaming 응답
```

### 각 노드 상세

#### Node 1 — `intent_classifier`

| 항목 | 내용 |
|---|---|
| 입력 | `question: str` |
| 출력 | `intent: str` |
| 방법 | `llm.with_structured_output(_IntentOutput)` |
| 분류값 | `카테고리판단` / `법령한도` / `적법성판단` / `기타` |

분류 기준:

- **카테고리판단** — "안전모는 몇 번 카테고리?", "방진마스크가 CAT_03인가요?"
- **법령한도** — "CAT_02 한도율이 얼마?", "보호구 구입 상한이 있나요?"
- **적법성판단** — "냉방기 구입이 산안비로 인정되나요?", "강사비 써도 되나요?"
- **기타** — 산안비와 무관하거나 시스템이 답변 불가한 질문

> **구현 노트:** `StrOutputParser` + JSON 파싱 방식이 아닌 `with_structured_output()`을 사용하여 function calling으로 타입 안전하게 반환받는다. JSON 파싱 실패 원천 차단.

#### Node 2 — `retriever`

| 항목 | 내용 |
|---|---|
| 입력 | `question`, `intent` |
| 출력 | `retrieved_docs: list[Document]` |
| 방법 | `core/rag.py`의 `build_retriever()` 재사용 |
| 검색 수 | top-k = 8 |

- BM25 (형태소 분석: kiwipiepy) + Vector 앙상블 (가중치 각 0.5)
- `core/storage.py`의 `load_vectorstore()` 싱글톤 캐시 활용

#### Node 3 — `fallback_handler`

| 항목 | 내용 |
|---|---|
| 입력 | `question` |
| 출력 | 안내 메시지 (고정 문자열) |
| 방법 | RAG 없이 `FALLBACK_MESSAGE` 상수 반환 |

예시 응답:
```
산업안전보건관리비(산안비) 관련 질문에만 답변드릴 수 있습니다.
• 카테고리 판단: '안전모는 몇 번 카테고리인가요?'
• 법령 한도: 'CAT_02 안전시설비 한도율이 얼마인가요?'
• 적법성 판단: '냉방기 구입비를 산안비로 사용할 수 있나요?'
```

#### Node 4 — `doc_grader`

| 항목 | 내용 |
|---|---|
| 입력 | `question`, `retrieved_docs` |
| 출력 | `graded_docs: list[Document]` |
| 방법 | `llm.with_structured_output(_GradeOutput)` |

hallucination의 주요 원인인 관련 없는 문서를 context에서 제거한다. "조금이라도 관련 있으면 relevant" 원칙을 적용하여 recall을 높인다.

#### Node 5 — `rewrite_query`

| 항목 | 내용 |
|---|---|
| 입력 | `question`, `retry_count` |
| 출력 | `question: str` (재작성), `retry_count + 1` |
| 방법 | `CHATBOT_REWRITE_PROMPT` — 구어체 → 법령 용어 변환 |
| 최대 횟수 | 3회. 초과 시 graded_docs가 비어있어도 answer_generator로 진행 |

#### Node 6 — `answer_generator`

| 항목 | 내용 |
|---|---|
| 입력 | `question`, `graded_docs`, `messages` (대화 기록) |
| 출력 | streaming tokens, `sources: list[str]` |
| 방법 | `ChatOpenAI(streaming=True).astream()` |
| 모델 | `OPENAI_REPORT_MODEL` (기본: `gpt-4.1`) |

답변 형식 규칙:
- **카테고리 판단** → 첫 문장에서 `CAT_XX(카테고리명)에 해당합니다` 결론 먼저
- **법령 한도** → 한도율·수치를 첫 문장에서 명시
- **적법성 판단** → `인정됩니다` / `인정되지 않습니다` 결론 먼저
- 법령 조항은 본문에 자연스럽게 인용
- 서론 도입부 (`본 답변은`, `검토 결과`) 금지

---

## 5. API 명세

### POST `/api/v1/chat`

#### 요청

```json
{
  "question": "추락방지망 설치비가 산안비로 인정되나요?",
  "session_id": "abc-1234"
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `question` | string | ✅ | 사용자 질문 |
| `session_id` | string | ❌ | 대화 세션 ID. 미전달 시 서버에서 UUID 생성 |

#### 응답 — SSE (`Content-Type: text/event-stream`)

```
data: {"type": "session_id", "value": "abc-1234"}

data: {"type": "intent", "value": "적법성판단"}

data: {"type": "token", "value": "CAT_02"}
data: {"type": "token", "value": "(안전시설비)에"}
data: {"type": "token", "value": " 해당합니다."}
...

data: {"type": "sources", "value": ["건설업 산업안전보건관리비 계상 및 사용기준"]}

data: [DONE]
```

| 이벤트 타입 | 시점 | 설명 |
|---|---|---|
| `session_id` | 최초 | 세션 ID 확정 (신규 생성 또는 그대로 반환) |
| `intent` | 분류 완료 후 | 질문 의도 분류 결과 |
| `token` | 답변 생성 중 | LLM 토큰 단위 스트리밍 (`answer_generator` 노드에서만 발생) |
| `sources` | 답변 완료 후 | 참조한 법령/문서 목록 |
| `[DONE]` | 종료 | 스트리밍 완료 신호 |

#### 오류 응답

```
data: {"type": "error", "value": "일시적인 오류가 발생했습니다. 다시 시도해 주세요."}
data: [DONE]
```

> **구현 노트:** `on_chat_model_stream` 이벤트 필터링 시 `event['metadata']['langgraph_node'] == 'answer_generator'` 조건으로 intent_classifier·doc_grader의 structured_output 내부 토큰을 차단한다.

---

## 6. MemorySaver — 대화 연속성

### 구조

LangGraph 내장 `MemorySaver`를 사용한다. **추가 패키지, DB, 인프라 없음.**

서버 프로세스 메모리에 세션별 대화 상태를 보관하며, 탭을 닫거나 서버가 재시작되면 자동으로 사라진다.

```python
from langgraph.checkpoint.memory import MemorySaver

# 앱 시작 시 1회 초기화 — 싱글톤으로 재사용
_checkpointer = MemorySaver()
_graph = build_chatbot_graph().compile(checkpointer=_checkpointer)

# thread_id = session_id → 세션별 대화 상태 자동 분리
await _graph.astream_events(
    {"question": "..."},
    config={"configurable": {"thread_id": session_id}},
    version="v2",
)
```

### session_id 관리

| 상황 | 처리 |
|---|---|
| 프론트에서 `session_id` 전달 | 해당 ID로 대화 기록 로드 |
| `session_id` 없음 | 서버에서 `uuid4()` 생성 |
| 새로 생성된 경우 | `session_id` 이벤트로 프론트에 전달 |

### 생명주기

| 상황 | 대화 기록 |
|---|---|
| 같은 탭에서 계속 질문 | 유지 (session_id 동일) |
| 탭 닫기 / 새로고침 | 소멸 (session_id 분실) |
| 서버 재시작 | 전체 소멸 |

### 멀티턴 동작 예시

```
[1번 대화]
Q: "안전모가 CAT_03이야?"
A: "CAT_03(보호구)에 해당합니다. ..."

[2번 대화 — 같은 session_id]
Q: "그러면 방진마스크는?"
→ MemorySaver에서 이전 대화 로드 → LLM이 이어지는 질문으로 인식
A: "방진마스크도 CAT_03(보호구)에 해당합니다. ..."
```

---

## 7. Qdrant 접속 구조

기존 `core/storage.py`의 `load_vectorstore()`를 그대로 재사용한다. **새로 추가할 코드 없음.**

```python
# nodes.py 내부
from src.core.storage import load_vectorstore, DEFAULT_COLLECTION
from src.core.rag import build_retriever

vectorstore = load_vectorstore(DEFAULT_COLLECTION)       # 싱글톤 캐시
retriever   = build_retriever(vectorstore, DEFAULT_COLLECTION, k=8)
```

| 항목 | 값 |
|---|---|
| 컬렉션 | `legal_documents` |
| 임베딩 모델 | `jhgan/ko-sroberta-multitask` |
| 검색 방식 | BM25 (kiwipiepy 형태소) + Vector 앙상블 (0.5 : 0.5) |
| Reranking | `BAAI/bge-reranker-v2-m3` (Cross-encoder) |
| URL | `.env`의 `QDRANT_URL` |

---

## 8. 파일 구조

### 새로 생성한 파일

```
src/
├── agents/
│   └── chatbot_agent/
│       ├── __init__.py       # 패키지 진입점
│       ├── state.py          # ChatbotState (MessagesState 상속)
│       ├── prompts.py        # 5개 프롬프트 상수
│       ├── nodes.py          # 6개 노드 함수 + 2개 라우팅 조건
│       └── agent.py          # LangGraph 그래프 조립 + MemorySaver 싱글톤
│
├── api/routers/
│   └── chatbot.py            # POST /api/v1/chat
│
├── schemas/
│   └── chatbot.py            # ChatRequest, ChatEvent
│
└── services/
    └── chatbot_service.py    # astream_events → SSE 변환

tests/
└── agents/
    └── chatbot_agent/
        ├── __init__.py
        ├── test_chatbot_agent.py   # 평가 스크립트
        └── cases/
            └── inputs.json         # 10개 테스트 케이스
```

### 기존 파일 수정 (최소)

```
src/main.py   ← chatbot router import + include_router 2줄 추가
```

**DB 스키마 변경: 없음.**

### 각 파일 역할

| 파일 | 역할 |
|---|---|
| `state.py` | `ChatbotState` TypedDict. `messages`는 `MessagesState` 상속으로 자동 누적 |
| `prompts.py` | INTENT / DOC_GRADE / ANSWER / REWRITE / FALLBACK 프롬프트 |
| `nodes.py` | 6개 노드 함수. intent·grade는 `with_structured_output()` 사용 |
| `agent.py` | `StateGraph` 조립. `get_compiled_graph()` 싱글톤 반환 |
| `chatbot_service.py` | `astream_events` v2 소비 → SSE 포맷 변환. `langgraph_node` 메타데이터로 토큰 필터링 |
| `chatbot.py` | `StreamingResponse(media_type="text/event-stream")` 반환 |

---

## 9. 환경변수

챗봇 기능을 위한 **추가 환경변수 없음.** 기존 `.env`를 그대로 사용한다.

| 변수 | 용도 | 기본값 |
|---|---|---|
| `OPENAI_API_KEY` | LLM 인증 | 필수 |
| `OPENAI_MODEL` | intent 분류 · doc grading 모델 | `gpt-4.1-mini` |
| `OPENAI_REPORT_MODEL` | 답변 생성 모델 | `gpt-4.1` |
| `QDRANT_URL` | Qdrant 서버 주소 | `http://localhost:6333` |

---

## 10. 의존성

`MemorySaver`는 `langgraph` 패키지에 내장되어 있어 **추가 의존성 없음.**

기존 `pyproject.toml` 그대로 사용한다.

---

## 11. k8s 배포

**추가 yaml 없음.** MemorySaver는 서버 프로세스 내에서만 동작하므로 별도 인프라가 필요하지 않다.

기존 `fastapi-deployment.yaml`, `fastapi-service.yaml`, `fastapi-configmap.yaml` 그대로 사용한다.

> **주의:** k8s에서 FastAPI `replicas`를 2 이상으로 늘리면 인스턴스 간 대화 기록이 공유되지 않는다. MemorySaver는 `replicas=1` 환경을 전제한다. 향후 수평 확장이 필요하면 Redis checkpointer 전환을 검토할 것.

---

## 12. 로컬 개발 환경 세팅

### Qdrant port-forward

```bash
kubectl port-forward svc/team5-qdrant 6333:6333 -n skala3-finalproj-class2-team5
```

### 서버 실행

```bash
cd fastapi
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

### 동작 확인

```bash
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "안전모는 몇 번 카테고리인가요?"}' \
  --no-buffer
```

예상 출력:

```
data: {"type": "session_id", "value": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}
data: {"type": "intent", "value": "카테고리판단"}
data: {"type": "token", "value": "CAT_03"}
data: {"type": "token", "value": "(보호구)에"}
data: {"type": "token", "value": " 해당합니다."}
...
data: {"type": "sources", "value": ["건설업 산업안전 보건관리비 해설 및 질의회시집(최종)"]}
data: [DONE]
```

---

## 13. 테스트

### 실행 방법

```bash
cd fastapi

# 전체 10개 케이스 실행
python -m tests.agents.chatbot_agent.test_chatbot_agent

# 특정 케이스만
python -m tests.agents.chatbot_agent.test_chatbot_agent --id chat_01

# 키워드 검증 없이 답변만 확인
python -m tests.agents.chatbot_agent.test_chatbot_agent --no-check
```

### 테스트 케이스 목록

| ID | 유형 | 질문 요약 | 검증 키워드 |
|---|---|---|---|
| chat_01 | 카테고리판단 | 안전모 카테고리 | CAT_03, 보호구 |
| chat_02 | 카테고리판단 | 핫팩 CAT_03 vs CAT_06 | CAT_06, 건강장해예방 |
| chat_03 | 카테고리판단 | 추락방지망 설치 인건비 | CAT_02, 안전시설비 |
| chat_04 | 법령한도 | CAT_02 한도율 | 한도, 20% |
| chat_05 | 적법성판단 | 사무실 냉방기 구입 | 인정되지 않습니다, 경비 |
| chat_06 | 적법성판단 | 외부 강사비 | CAT_05, 안전보건교육 |
| chat_07 | 카테고리판단 | 방진마스크 카테고리 | CAT_03, 보호구 |
| chat_08 | 기타 | 날씨 질문 (fallback) | 산안비, 카테고리 |
| chat_09 | 카테고리판단 | 안전관리자 급여 | CAT_01, 인건비 |
| chat_10 | 적법성판단 | 위험성평가 외부용역 | CAT_09, 위험성평가 |

### 출력 구성

케이스마다 intent 분류 결과, 전체 답변(토큰 조합), 참조 출처, 누락 키워드를 출력한다. 종료 시 intent 정확도·키워드 통과율 요약 및 `tests/agents/chatbot_agent/output/results_YYYYMMDD_HHMM.json` 저장.

---

## 14. 평가 결과

2026-06-04 기준 10개 케이스 평가 결과:

| 항목 | 결과 |
|---|---|
| Intent 정확도 | **10/10 (100%)** |
| 키워드 통과율 | **10/10 (100%)** |
| 결론 우선 원칙 준수 | ✅ 모든 케이스에서 `CAT_XX(카테고리명)에 해당합니다` 형식 |
| 법령 조항 인용 | ✅ 제7조제1항제X호 수준까지 인용 |
| 혼동 케이스 처리 | ✅ 핫팩 CAT_03↔CAT_06 정확 판단 |
| Fallback 동작 | ✅ 산안비 무관 질문 정상 차단 |

### 주요 답변 예시

**카테고리 판단 (chat_01)**
> CAT_03(보호구)에 해당합니다. 안전모는 산업안전보건법 시행령 제74조제1항제3호, 제77조제1항제3호에 따라 보호구로 명확히 규정되어 있습니다.

**혼동 케이스 (chat_02 — 핫팩)**
> CAT_06(근로자 건강장해예방비)에 해당합니다. 근로자에게 일률적으로 지급하는 보냉·보온장구(핫팩, 장갑 등)는 보호구(CAT_03)로 인정하지 않습니다.

**적법성 판단 (chat_05 — 사무실 냉방기)**
> 인정되지 않습니다. 현장 사무실 냉방기는 공사원가 계산 시 경비로 계상하도록 규정되어 있어 산안비 집행 대상이 아닙니다.

---

## 참고

- 기존 `core/rag.py` — BM25 + Vector Ensemble, rerank 구현체
- 기존 `core/storage.py` — load_vectorstore, load_collection_documents
- 기존 `core/llm_config.py` — LLM 싱글톤
- LangGraph 공식 문서: https://langchain-ai.github.io/langgraph/
- LangGraph MemorySaver: https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.memory.MemorySaver
