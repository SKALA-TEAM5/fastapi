# 산안비 챗봇 에이전트 (Chatbot Agent)

산업안전보건관리비(산안비) 관련 질문에 대해 Qdrant 벡터 DB 기반 RAG 검색 및 LangGraph 멀티 에이전트 파이프라인으로 정확한 답변을 스트리밍 방식으로 제공하는 챗봇 시스템입니다.

---

## 목차

1. [Overview](#1-overview)
2. [Tech Stack](#2-tech-stack)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [Agent 그래프 상세](#4-agent-그래프-상세)
5. [API 명세](#5-api-명세)
6. [SSE 스트리밍 구조](#6-sse-스트리밍-구조)
7. [안전 장치](#7-안전-장치)
8. [Spring 연동 구조](#8-spring-연동-구조)
9. [LangSmith 모니터링](#9-langsmith-모니터링)
10. [MemorySaver — 대화 연속성](#10-memorysaver--대화-연속성)
11. [Qdrant 접속 구조](#11-qdrant-접속-구조)
12. [파일 구조](#12-파일-구조)
13. [환경변수](#13-환경변수)
14. [의존성](#14-의존성)
15. [k8s 배포](#15-k8s-배포)
16. [로컬 개발 환경 세팅](#16-로컬-개발-환경-세팅)
17. [테스트](#17-테스트)
18. [평가 결과](#18-평가-결과)

---

## 1. Overview

### 무엇을 하는가

사용자가 산안비 관련 질문을 입력하면, 시스템이 다음 단계를 거쳐 답변을 생성합니다.

1. 질문 의도(intent)를 LLM으로 분류 (멀티턴 맥락 포함)
2. Qdrant에서 관련 법령·가이드 문서 검색 (BM25 + 벡터 앙상블)
3. Cross-encoder로 검색 결과 재순위 및 관련성 필터링
4. 관련 문서를 context로 LLM 답변 생성
5. 토큰 단위 SSE 스트리밍으로 프론트에 전달

### 주요 특징

- **멀티 에이전트 (LangGraph)** — intent 분류 → retrieval → grading → rewrite → generation 노드를 그래프로 연결
- **Agentic RAG** — 관련 문서가 없으면 질문을 자동 재작성 후 재검색 (최대 3회)
- **근거 없는 답변 차단** — 재시도 소진 후에도 관련 문서가 없으면 답변 생성을 차단하고 fallback 반환
- **SSE Streaming** — 토큰 단위 실시간 응답, 노드별 진행 상태(status) 이벤트 포함
- **멀티턴 대화** — session_id 기반, 이전 대화 맥락을 intent 분류에도 반영
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
| Reranker | `BAAI/bge-reranker-v2-m3` (HuggingFace) |
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
    │  세션 만료 감지 → session_reset 이벤트
    │  동시 요청 중복 방지 (session lock)
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
    │  data: {"type": "session_id",    "value": "..."}
    │  data: {"type": "session_reset", "value": "..."}  ← 세션 만료 시
    │  data: {"type": "status",        "value": "..."}  ← 노드별 진행 상태
    │  data: {"type": "intent",        "value": "..."}
    │  data: {"type": "token",         "value": "..."}
    │  data: {"type": "sources",       "value": [...]}
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
[Node 1] intent_classifier  ← 이전 대화(chat history) 포함
    │
    ├── "기타" ──────────────────────────────────────────────────────┐
    │                                                                │
    └── "카테고리판단" | "법령한도" | "적법성판단"                   │
        "계상기준"    | "도급귀속"                                    │
            │                                                        │
            ▼                                                        ▼
    [Node 2] retriever                                   [Node 3] fallback_handler
            │                                                        │
            ▼                                                        │
    [Node 4] doc_grader                                              │
            │                                                        │
            ├── graded_docs 없음 AND retry < 3                       │
            │       ▼                                                │
            │   [Node 5] rewrite_query ──► Node 2 (루프백)           │
            │                                                        │
            ├── graded_docs 없음 AND retry >= 3 ────────────────────►│
            │   (근거 없는 답변 생성 차단 → fallback)                 │
            │                                                        │
            └── graded_docs 있음                                     │
                    │                                                │
                    ▼                                                │
            [Node 6] answer_generator ◄──────────────────────────────┘
                    │
                    ▼
              SSE Streaming 응답
```

### 각 노드 상세

#### Node 1 — `intent_classifier`

| 항목 | 내용 |
|---|---|
| 입력 | `question: str`, `messages` (이전 대화 기록) |
| 출력 | `intent: str` |
| 방법 | `llm.with_structured_output(_IntentOutput)` |
| 분류값 | `카테고리판단` / `법령한도` / `적법성판단` / `계상기준` / `도급귀속` / `기타` |

분류 기준:

- **카테고리판단** — "안전모는 몇 번 카테고리?", "방진마스크가 CAT_03인가요?"
- **법령한도** — "CAT_02 한도율이 얼마?", "보호구 구입 상한이 있나요?"
- **적법성판단** — "냉방기 구입이 산안비로 인정되나요?", "강사비 써도 되나요?"
- **계상기준** — "산안비 총액 어떻게 계산해?", "대상액 5억 미만 요율이 얼마?", "설계변경 시 재계상 해야 해?"
- **도급귀속** — "하청업체도 산안비 직접 집행 가능해?", "공동도급 현장에서 귀속은?", "누가 산안비 책임져?"
- **기타** — 산안비와 무관하거나 시스템이 답변 불가한 질문

> **구현 노트 1:** `StrOutputParser` + JSON 파싱 방식이 아닌 `with_structured_output()`을 사용하여 function calling으로 타입 안전하게 반환받는다. JSON 파싱 실패 원천 차단.

> **구현 노트 2:** `INTENT_PROMPT`에 최근 대화 기록(`chat_history`)을 포함하여 "그러면 방진마스크는?", "방금 질문 다시 해줘" 같은 후속 질문도 올바른 intent로 분류한다.

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
| 입력 | `question`, `intent`, `graded_docs`, `retry_count` |
| 출력 | 안내 메시지 (고정 문자열) |
| 방법 | RAG 없이 상황별 고정 메시지 반환 |

두 가지 케이스를 구분하여 다른 메시지를 반환한다.

| 진입 경로 | 메시지 |
|---|---|
| `intent == "기타"` | 산안비 무관 질문 안내 + 예시 질문 목록 |
| 재시도 소진 후 근거 문서 없음 | "질문과 관련된 법령 근거를 찾지 못했습니다. 질문을 좀 더 구체적으로 바꿔서 다시 시도해 주세요." |

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
| 최대 횟수 | 3회. 초과 시 graded_docs가 비어있으면 **answer_generator가 아닌 fallback_handler로** 분기 |

> **구현 노트:** 재시도 3회 소진 후에도 근거 문서가 없으면 답변 생성을 차단한다. 이전에는 빈 context로 answer_generator를 실행했으나, 법령 도메인 특성상 근거 없는 답변 생성이 위험하므로 fallback으로 변경.

#### Node 6 — `answer_generator`

| 항목 | 내용 |
|---|---|
| 입력 | `question`, `graded_docs`, `messages` (대화 기록, 최근 20개) |
| 출력 | streaming tokens, `sources: list[str]` |
| 방법 | `ChatOpenAI(streaming=True).astream()` |
| 모델 | `OPENAI_REPORT_MODEL` (기본: `gpt-4.1`) |

답변 형식 규칙:
- **카테고리 판단** → 첫 문장에서 `CAT_XX(카테고리명)에 해당합니다` 결론 먼저
- **법령 한도** → 한도율·수치를 첫 문장에서 명시
- **적법성 판단** → `인정됩니다` / `인정되지 않습니다` 결론 먼저
- **계상기준** → 요율·금액 수치 우선, 공사 규모 구간(5억 미만 / 5억~50억 / 50억 이상) 구분 설명
- **도급귀속** → 계상·집행 주체 결론 먼저, 도급인·수급인 역할 명확히 구분
- 법령 조항은 본문에 자연스럽게 인용
- 서론 도입부 (`본 답변은`, `검토 결과`) 금지

> **구현 노트:** 대화 길이 누적 방지를 위해 `messages`가 `MAX_MESSAGES(20)`를 초과하면 오래된 메시지부터 제거한다.

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

| 필드 | 타입 | 필수 | 제약 | 설명 |
|---|---|---|---|---|
| `question` | string | ✅ | 1~500자 | 사용자 질문. 연속 공백·줄바꿈 자동 정리 |
| `session_id` | string | ❌ | - | 대화 세션 ID. 미전달 시 서버에서 UUID 생성 |

#### 응답 — SSE (`Content-Type: text/event-stream`)

```
data: {"type": "session_id",    "value": "abc-1234"}
data: {"type": "session_reset", "value": "이전 대화 기록이 만료되었습니다. 새 대화를 시작합니다."}

data: {"type": "status", "value": "질문 유형 분석 중..."}
data: {"type": "intent", "value": "적법성판단"}
data: {"type": "status", "value": "관련 법령 검색 중..."}
data: {"type": "status", "value": "답변 생성 중..."}

data: {"type": "token", "value": "CAT_02"}
data: {"type": "token", "value": "(안전시설비)에"}
...

data: {"type": "sources", "value": ["건설업 산업안전보건관리비 계상 및 사용기준"]}

data: [DONE]
```

| 이벤트 타입 | 시점 | 설명 |
|---|---|---|
| `session_id` | 최초 | 세션 ID 확정 (신규 생성 또는 그대로 반환) |
| `session_reset` | 세션 만료 감지 시 | 서버 재시작 등으로 이전 대화가 소멸된 경우 알림 |
| `status` | 각 노드 진입 직전 | 현재 처리 단계 안내 문자열 (스피너 표시용) |
| `intent` | 분류 완료 후 | 질문 의도 분류 결과 |
| `token` | 답변 생성 중 | LLM 토큰 단위 스트리밍 (`answer_generator` 노드에서만 발생) |
| `sources` | 답변 완료 후 | 참조한 법령/문서 목록 |
| `error` | 오류 발생 시 | 오류 안내 메시지 |
| `[DONE]` | 종료 | 스트리밍 완료 신호 |

#### 오류 응답

```
data: {"type": "error", "value": "이전 답변을 생성 중입니다. 잠시 후 다시 시도해 주세요."}
data: {"type": "error", "value": "응답 시간이 초과되었습니다. 다시 시도해 주세요."}
data: {"type": "error", "value": "일시적인 오류가 발생했습니다. 다시 시도해 주세요."}
data: [DONE]
```

> **구현 노트:** `on_chat_model_stream` 이벤트 필터링 시 `event['metadata']['langgraph_node'] == 'answer_generator'` 조건으로 intent_classifier·doc_grader의 structured_output 내부 토큰을 차단한다.

---

## 6. SSE 스트리밍 구조

### HTTP 통신 방식 비교

일반 REST API는 요청과 응답이 한 쌍으로 끝나요. 서버가 처리를 완료한 뒤 응답을 한 번에 보내고 연결이 끊겨요.

```
클라이언트 → 요청
서버       → (10초 처리) → 응답 전체 한 번에 반환 → 연결 종료
```

LLM 답변처럼 생성에 시간이 걸리는 경우, 사용자는 10초 동안 아무것도 볼 수 없어요.

SSE(Server-Sent Events)는 연결을 열어둔 채로 데이터를 조금씩 흘려보내는 방식이에요.

```
클라이언트 → 요청 (연결 유지)
서버       → "안" → "전" → "모" → "는" → ... → [DONE] → 연결 종료
```

생성하는 즉시 보내기 때문에 체감 속도가 훨씬 빨라요.

### SSE 포맷

SSE는 HTTP 위에서 동작하는 표준 포맷이에요. 규칙은 두 가지예요.

- `data:` 로 시작
- 이벤트 끝에 빈 줄 두 개 (`\n\n`) 로 구분

```
data: {"type": "token", "value": "안"}\n\n
data: {"type": "token", "value": "전"}\n\n
data: [DONE]\n\n
```

브라우저는 `EventSource` API로 이 포맷을 natively 지원해요.

### 이벤트 흐름과 타이밍

각 이벤트가 언제 발생하는지 노드 흐름과 함께 보면 이래요.

```
사용자: "안전모는 몇 번 카테고리?"

→ data: {"type": "session_id", "value": "uuid"}

[intent_classifier 노드 진입]
→ data: {"type": "status",  "value": "질문 유형 분석 중..."}   ← on_chain_start

[intent_classifier 노드 완료]
→ data: {"type": "intent",  "value": "카테고리판단"}            ← on_chain_end

[retriever 노드 진입]
→ data: {"type": "status",  "value": "관련 법령 검색 중..."}    ← on_chain_start

[doc_grader 노드 진입]
→ data: {"type": "status",  "value": "문서 관련성 검토 중..."}  ← on_chain_start

[answer_generator 노드 진입]
→ data: {"type": "status",  "value": "답변 생성 중..."}         ← on_chain_start

[LLM 토큰 생성]
→ data: {"type": "token",   "value": "CAT_03"}
→ data: {"type": "token",   "value": "(보호구)에"}
→ data: {"type": "token",   "value": " 해당합니다."}
...

[answer_generator 노드 완료]
→ data: {"type": "sources", "value": ["산업안전보건법 제72조"]}

→ data: [DONE]
```

`status`는 `on_chain_start` (노드 진입 직전) 에 발생해요. 실제 작업보다 항상 먼저 도착하기 때문에 프론트에서 스피너를 표시하기 적합해요.

### 프론트엔드 처리 전략

```
session_reset 수신 → "이전 대화가 만료되었습니다" 토스트 표시
status  수신       → 스피너 ON + 텍스트 변경 ("관련 법령 검색 중...")
token   수신 (첫 번째) → 스피너 OFF, 타이핑 시작
token   수신 (이후) → 화면에 계속 붙이기
sources 수신       → 출처 목록 표시
[DONE]  수신       → 연결 종료
```

### `chatbot_service.py` 이벤트 변환 규칙

| LangGraph 이벤트 | 조건 | SSE 이벤트 |
|---|---|---|
| `on_chain_start` | `ev_name in _NODE_STATUS` | `{"type": "status", "value": "...중..."}` |
| `on_chain_end` | `ev_name == "intent_classifier"` | `{"type": "intent", "value": intent}` |
| `on_chat_model_stream` | `langgraph_node == "answer_generator"` | `{"type": "token", "value": chunk}` |
| `on_chain_end` | `ev_name == "answer_generator"` | `{"type": "sources", "value": [...]}` |
| `on_chain_end` | `ev_name == "fallback_handler"` | `{"type": "token", "value": message}` |

> **구현 노트:** `on_chat_model_stream`은 intent_classifier, doc_grader의 structured_output 내부에서도 발생해요. `langgraph_node == "answer_generator"` 조건으로 필터링하지 않으면 의도치 않은 토큰이 프론트로 새어 나와요.

---

## 7. 안전 장치

운영 안정성을 위해 `chatbot_service.py`에 다섯 가지 안전 장치가 구현되어 있다.

### 1. 동시 요청 중복 방지

같은 `session_id`로 동시에 두 요청이 오면 이전 스트리밍 상태가 꼬일 수 있다. `_session_locks` 딕셔너리로 세션별 `asyncio.Lock`을 관리하여 중복 요청을 즉시 차단한다.

```python
_session_locks: dict[str, asyncio.Lock] = {}
```

Lock이 점유 중이면 `{"type": "error", "value": "이전 답변을 생성 중입니다."}` 이벤트를 반환하고 즉시 종료한다.

### 2. 스트리밍 타임아웃

LLM API가 중간에 끊기면 클라이언트가 무한 대기에 빠진다. 두 가지 타임아웃을 적용한다.

| 타임아웃 | 값 | 설명 |
|---|---|---|
| 전체 타임아웃 | 120초 | 전체 스트리밍 최대 허용 시간 |
| 토큰 간 타임아웃 | 30초 | 마지막 이벤트 이후 대기 최대 시간 |

초과 시 `{"type": "error", "value": "응답 시간이 초과되었습니다."}` 이벤트를 반환한다.

### 3. 세션 만료 감지

서버 재시작 시 MemorySaver가 소멸하여 이전 대화 기록이 날아간다. 프론트가 기존 `session_id`로 요청할 때 MemorySaver에 해당 thread 상태가 없으면 `session_reset` 이벤트를 먼저 전송한다.

```
data: {"type": "session_reset", "value": "이전 대화 기록이 만료되었습니다. 새 대화를 시작합니다."}
```

### 4. 대화 길이 트리밍

MemorySaver는 전체 대화를 메모리에 계속 누적한다. `answer_generator` 노드 진입 시 `messages`가 `MAX_MESSAGES(20)`를 초과하면 오래된 메시지부터 제거한다. 프롬프트에는 최근 6개 메시지만 포함(`_format_chat_history`)되므로 LLM 비용은 이미 제어되고 있고, 이 트리밍은 메모리 누적 방지 목적이다.

### 5. 질문 전처리

`ChatRequest` 스키마에서 자동으로 처리된다.

| 처리 | 내용 |
|---|---|
| 길이 제한 | `max_length=500` — 초과 시 422 에러 자동 반환 |
| 공백 정리 | 연속 공백 → 단일 공백, 연속 줄바꿈 압축 |
| 앞뒤 공백 | 자동 strip |

---

## 8. Spring 연동 구조

### 배포 환경에서의 접근 제한

k8s Ingress 설정상 FastAPI는 외부에 노출되지 않아요.

```yaml
# 외부 접근 가능
team5-iveri.skala25a.project.skala-ai.com      → frontend :3000
api-team5-iveri.skala25a.project.skala-ai.com  → backend  :8000

# 외부 접근 불가 (클러스터 내부 전용)
team5-fastapi:8001                              → FastAPI
```

프론트에서 FastAPI에 직접 요청할 수 없어요. Spring이 반드시 중간에서 프록시해야 해요.

### Spring 연동 흐름

```
[Frontend]
  POST /chat { question, session_id }
  쿠키: access_token=xxx
      ↓
[Spring MVC — JwtAuthenticationFilter]
  쿠키에서 JWT 파싱 → AuthenticatedUser 주입
  (기존 필터가 /chat도 자동으로 인증 처리)
      ↓
[ChatbotController.java]
  SseEmitter 생성 (timeout: 5분)
  ChatbotClient.stream() 호출
      ↓
[ChatbotClient.java — WebClient]
  POST http://team5-fastapi:8001/api/v1/chat
  응답을 Flux<String>으로 비동기 수신
      ↓
[FastAPI]
  LangGraph 실행 → SSE 이벤트 생성
      ↓
[ChatbotClient.java]
  이벤트 한 줄 올 때마다 → SseEmitter.send()
      ↓
[Frontend]
  EventSource로 수신 → 화면에 렌더링
```

### 왜 WebClient + SseEmitter 조합이냐

Spring MVC는 기본적으로 동기/블로킹 방식이에요. 기존 `FastApiAgentClient`가 쓰는 `RestClient`로 SSE를 받으면 전체 응답이 끝날 때까지 스레드가 블로킹돼요. 30~60초짜리 SSE 응답 동안 스레드가 묶이면 동시 요청 처리가 거의 불가능해요.

`WebClient`는 비동기/논블로킹이에요. FastAPI에서 토큰 하나가 올 때마다 이벤트 루프가 깨어나서 처리하고 다시 대기 상태로 돌아가요. 스레드가 묶이지 않아요.

```
RestClient  → 스레드 블로킹 → SSE에 부적합
WebClient   → 논블로킹     → SSE에 적합
SseEmitter  → Spring MVC에서 프론트로 SSE 중계하는 도구
```

`spring-boot-starter-webflux` 의존성을 추가하는데, 이건 WebFlux를 web framework로 쓰는 게 아니라 WebClient만 가져오는 거예요. 기존 MVC 동작에 영향 없어요.

### Spring 담당자 작업 범위

FastAPI 코드는 변경 없어요. Spring 담당자가 추가할 파일은 아래 세 가지예요.

| 파일 | 역할 |
|---|---|
| `build.gradle` | `spring-boot-starter-webflux` 의존성 추가 |
| `ChatbotClient.java` | WebClient로 FastAPI SSE 구독 |
| `ChatbotController.java` | SseEmitter로 프론트에 중계 |

SecurityConfig 변경 불필요. `/chat` 엔드포인트는 기존 `anyRequest().authenticated()` 룰이 자동 적용돼요.

### FastAPI 측 제공 스펙

Spring 담당자에게 전달할 인터페이스 명세예요.

```
엔드포인트: POST http://team5-fastapi:8001/api/v1/chat
Content-Type: application/json

요청 바디:
{
  "question": "string (필수, 1~500자)",
  "session_id": "string | null"
}

응답: text/event-stream
data: {"type": "session_id",    "value": "uuid"}
data: {"type": "session_reset", "value": "이전 대화 기록이 만료되었습니다."}
data: {"type": "status",        "value": "질문 유형 분석 중..."}
data: {"type": "intent",        "value": "카테고리판단"}
data: {"type": "status",        "value": "관련 법령 검색 중..."}
data: {"type": "token",         "value": "토큰"}
data: {"type": "sources",       "value": ["법령1", "법령2"]}
data: {"type": "error",         "value": "오류 메시지"}
data: [DONE]
```

---

## 9. LangSmith 모니터링

### 개요

LangSmith는 LangGraph 실행을 추적하는 모니터링 도구예요. 코드 변경 없이 환경변수만 추가하면 자동으로 트레이싱이 활성화돼요.

### 연동 방법

`.env` (로컬) 또는 k8s Secret (운영) 에 추가:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__xxxxxxxx      # LangSmith API 키
LANGCHAIN_PROJECT=sananbi-chatbot   # 프로젝트 이름
```

`LANGCHAIN_API_KEY`는 민감한 값이므로 k8s ConfigMap이 아닌 **Secret으로 분리**할 것.

### API 키 발급

[smith.langchain.com](https://smith.langchain.com) → Settings → API Keys → Create API Key

무료 플랜 기준 월 5,000 트레이스까지 지원해요.

### 트레이싱에서 볼 수 있는 것

```
Run: stream_chat
├── intent_classifier     → 입력/출력, 토큰 수, 응답 시간
├── retriever             → 검색된 문서 목록
├── doc_grader            → 문서별 관련성 점수
│   └── (재시도 시)
│       ├── rewrite_query → 재작성된 질문
│       └── retriever     → 재검색 결과
└── answer_generator      → LLM 입출력, 토큰 수, 비용
```

노드별 토큰 사용량, 지연 시간, LLM 프롬프트/응답 전문이 자동으로 기록돼요.

---

## 10. MemorySaver — 대화 연속성

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
| 기존 ID인데 서버 재시작 후 | `session_reset` 이벤트로 만료 알림 |

### 생명주기

| 상황 | 대화 기록 |
|---|---|
| 같은 탭에서 계속 질문 | 유지 (session_id 동일) |
| 탭 닫기 / X 버튼 | 소멸 (session_id 분실) — 휘발성 설계 의도 |
| 서버 재시작 | 전체 소멸, `session_reset` 이벤트로 프론트 알림 |

### 멀티턴 동작 예시

```
[1번 대화]
Q: "안전모가 CAT_03이야?"
A: "CAT_03(보호구)에 해당합니다. ..."

[2번 대화 — 같은 session_id]
Q: "그러면 방진마스크는?"
→ intent_classifier가 chat_history 보고 → 카테고리판단으로 분류
→ MemorySaver에서 이전 대화 로드 → LLM이 이어지는 질문으로 인식
A: "방진마스크도 CAT_03(보호구)에 해당합니다. ..."
```

---

## 11. Qdrant 접속 구조

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

## 12. 파일 구조

### 챗봇 관련 파일

```
src/
├── main.py                        ← lifespan 워밍업 추가
├── agents/
│   └── chatbot_agent/
│       ├── __init__.py
│       ├── state.py               # ChatbotState (MessagesState 상속)
│       ├── prompts.py             # 6개 intent 프롬프트 + NO_DOCS_FALLBACK_MESSAGE
│       ├── nodes.py               # 6개 노드 함수 + 2개 라우팅 조건
│       └── agent.py               # LangGraph 그래프 조립 + MemorySaver 싱글톤
│
├── api/routers/
│   └── chatbot.py                 # POST /api/v1/chat
│
├── schemas/
│   └── chatbot.py                 # ChatRequest (max_length, sanitize), ChatEvent
│
└── services/
    └── chatbot_service.py         # SSE 변환 + 5가지 안전 장치

scripts/
├── chat.py                        # CLI 테스트 스크립트
└── export_graph.py                # 그래프 PNG 저장 스크립트

tests/
└── agents/
    └── chatbot_agent/
        ├── test_chatbot_agent.py  # 단일턴 + 멀티턴 평가
        └── cases/
            └── inputs.json        # 16개 테스트 케이스
```

### 기존 파일 수정 내역

| 파일 | 변경 내용 |
|---|---|
| `src/main.py` | lifespan 워밍업 (HuggingFace 모델 선로딩) |
| `src/agents/chatbot_agent/prompts.py` | intent 4종 → 6종, 답변 형식 규칙 추가, NO_DOCS_FALLBACK_MESSAGE |
| `src/agents/chatbot_agent/nodes.py` | valid set 확장, chat_history 주입, MAX_MESSAGES, fallback 분기 |
| `src/agents/chatbot_agent/agent.py` | doc_grader → fallback 라우팅 추가 |
| `src/schemas/chatbot.py` | max_length=500, sanitize_question validator |
| `src/services/chatbot_service.py` | 5가지 안전 장치, status 이벤트, session_reset |

**DB 스키마 변경: 없음.**

### 각 파일 역할

| 파일 | 역할 |
|---|---|
| `state.py` | `ChatbotState` TypedDict. `messages`는 `MessagesState` 상속으로 자동 누적 |
| `prompts.py` | INTENT / DOC_GRADE / ANSWER / REWRITE / FALLBACK / NO_DOCS_FALLBACK 프롬프트 |
| `nodes.py` | 6개 노드 함수. intent·grade는 `with_structured_output()` 사용 |
| `agent.py` | `StateGraph` 조립. `get_compiled_graph()` 싱글톤 반환 |
| `chatbot_service.py` | `astream_events` v2 소비 → SSE 포맷 변환 + 안전 장치 |
| `chatbot.py` | `StreamingResponse(media_type="text/event-stream")` 반환 |

---

## 13. 환경변수

| 변수 | 용도 | 기본값 |
|---|---|---|
| `OPENAI_API_KEY` | LLM 인증 | 필수 |
| `OPENAI_MODEL` | intent 분류 · doc grading 모델 | `gpt-4.1-mini` |
| `OPENAI_REPORT_MODEL` | 답변 생성 모델 | `gpt-4.1` |
| `QDRANT_URL` | Qdrant 서버 주소 | `http://localhost:6333` |
| `LANGCHAIN_TRACING_V2` | LangSmith 트레이싱 활성화 | `false` |
| `LANGCHAIN_API_KEY` | LangSmith API 키 | 없음 (k8s Secret 권장) |
| `LANGCHAIN_PROJECT` | LangSmith 프로젝트 이름 | `sananbi-chatbot` |

---

## 14. 의존성

`MemorySaver`는 `langgraph` 패키지에 내장되어 있어 **추가 의존성 없음.**

기존 `pyproject.toml` 그대로 사용한다.

---

## 15. k8s 배포

**추가 yaml 없음.** MemorySaver는 서버 프로세스 내에서만 동작하므로 별도 인프라가 필요하지 않다.

기존 `fastapi-deployment.yaml`, `fastapi-service.yaml`, `fastapi-configmap.yaml` 그대로 사용한다.

### 서버 시작 시 모델 워밍업

`main.py`의 `lifespan`이 서버 시작 시 자동으로 HuggingFace 모델을 선로딩한다. 첫 요청 지연(콜드 스타트)을 없애기 위한 조치다.

```
서버 시작
  → jhgan/ko-sroberta-multitask 로딩 (임베딩)
  → BAAI/bge-reranker-v2-m3 로딩 (reranker)
  → LangGraph 그래프 컴파일
  → "Application startup complete." — 이후부터 요청 수신
```

> **주의 1:** k8s에서 FastAPI `replicas`를 2 이상으로 늘리면 인스턴스 간 대화 기록이 공유되지 않는다. MemorySaver는 `replicas=1` 환경을 전제한다. 향후 수평 확장이 필요하면 Redis checkpointer 전환을 검토할 것.

> **주의 2:** HuggingFace 모델은 컨테이너 기동 시 다운로드한다. 네트워크 환경에 따라 초기 기동 시간이 길어질 수 있다. (`readinessProbe`의 `initialDelaySeconds` 여유 확보 권장)

---

## 16. 로컬 개발 환경 세팅

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

### CLI로 챗봇 직접 테스트

```bash
python scripts/chat.py
```

질문을 입력하면 SSE 스트리밍 응답이 터미널에 출력된다. `q` 입력 시 종료.

### curl로 동작 확인

```bash
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "안전모는 몇 번 카테고리인가요?"}' \
  --no-buffer
```

예상 출력:

```
data: {"type": "session_id",    "value": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}
data: {"type": "status",        "value": "질문 유형 분석 중..."}
data: {"type": "intent",        "value": "카테고리판단"}
data: {"type": "status",        "value": "관련 법령 검색 중..."}
data: {"type": "status",        "value": "문서 관련성 검토 중..."}
data: {"type": "status",        "value": "답변 생성 중..."}
data: {"type": "token",         "value": "CAT_03"}
data: {"type": "token",         "value": "(보호구)에"}
data: {"type": "token",         "value": " 해당합니다."}
...
data: {"type": "sources",       "value": ["건설업 산업안전 보건관리비 해설 및 질의회시집(최종)"]}
data: [DONE]
```

---

## 17. 테스트

### 실행 방법

```bash
cd fastapi

# 전체 16개 단일턴 케이스 실행
python -m tests.agents.chatbot_agent.test_chatbot_agent

# 특정 케이스만
python -m tests.agents.chatbot_agent.test_chatbot_agent --id chat_11

# 키워드 검증 없이 답변만 확인
python -m tests.agents.chatbot_agent.test_chatbot_agent --no-check

# 멀티턴 케이스 실행
python -m tests.agents.chatbot_agent.test_chatbot_agent --multiturn
```

### 단일턴 테스트 케이스 목록

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
| chat_11 | 계상기준 | 산안비 총액 계산 방법 | 대상액, 비율 |
| chat_12 | 계상기준 | 대상액 5억 미만 요율 | 별표, 비율 |
| chat_13 | 계상기준 | 설계변경 시 재계상 의무 | 조정 계상, 대상액 |
| chat_14 | 도급귀속 | 하청업체 집행 가능 여부 | 도급인, 수급인 |
| chat_15 | 도급귀속 | 공동도급 현장 귀속 | 도급인 |
| chat_16 | 계상기준 | 공사 규모별 총액 산정 | 대상액, 비율 |

### 멀티턴 테스트 케이스 목록

| ID | 설명 | 검증 포인트 |
|---|---|---|
| multi_01 | 후속 질문 맥락 추적 | "그러면 방진마스크는?" → 카테고리판단 분류 |
| multi_02 | 방금 질문 복원 | "방금 질문 다시 해줘" → 기타로 빠지지 않고 법령한도 분류 |

### 출력 구성

케이스마다 intent 분류 결과, 전체 답변(토큰 조합), 참조 출처, 누락 키워드를 출력한다. 종료 시 intent 정확도·키워드 통과율 요약 및 `tests/agents/chatbot_agent/output/results_YYYYMMDD_HHMM.json` 저장.

---

## 18. 평가 결과

2026-06-05 기준 평가 결과:

### 단일턴 (16개 케이스)

| 항목 | 결과 |
|---|---|
| Intent 정확도 | **16/16 (100%)** |
| 키워드 통과율 | **16/16 (100%)** |
| 결론 우선 원칙 준수 | ✅ 모든 케이스에서 `CAT_XX(카테고리명)에 해당합니다` 형식 |
| 법령 조항 인용 | ✅ 제7조제1항제X호 수준까지 인용 |
| 혼동 케이스 처리 | ✅ 핫팩 CAT_03↔CAT_06 정확 판단 |
| 계상기준 신규 intent | ✅ 3개 케이스 모두 정확 분류 및 답변 |
| 도급귀속 신규 intent | ✅ 2개 케이스 모두 정확 분류 및 답변 |
| Fallback 동작 | ✅ 산안비 무관 질문 정상 차단 |

### 멀티턴 (2개 케이스)

| 항목 | 결과 |
|---|---|
| Intent 정확도 (전 턴 통과) | **2/2 (100%)** |
| 키워드 통과율 (전 턴 통과) | **2/2 (100%)** |
| 후속 질문 맥락 추적 | ✅ "그러면 방진마스크는?" → 카테고리판단 정확 분류 |
| 방금 질문 복원 | ✅ "방금 질문 다시 해줘" → 기타로 빠지지 않고 정확 분류 |

### 주요 답변 예시

**카테고리 판단 (chat_01)**
> CAT_03(보호구)에 해당합니다. 안전모는 산업안전보건법 시행령 제74조제1항제3호, 제77조제1항제3호에 따라 보호구로 명확히 규정되어 있습니다.

**혼동 케이스 (chat_02 — 핫팩)**
> CAT_06(근로자 건강장해예방비)에 해당합니다. 근로자에게 일률적으로 지급하는 보냉·보온장구(핫팩, 장갑 등)는 보호구(CAT_03)로 인정하지 않습니다.

**적법성 판단 (chat_05 — 사무실 냉방기)**
> 인정되지 않습니다. 현장 사무실 냉방기는 공사원가 계산 시 경비로 계상하도록 규정되어 있어 산안비 집행 대상이 아닙니다.

**계상기준 (chat_11 — 총액 계산)**
> 산안비 총액은 공사별 대상액에 별표 1의 요율을 곱하여 산정합니다. 대상액 구간에 따라 5억 미만, 5억~50억, 50억 이상으로 구분하여 요율이 달리 적용됩니다.

**도급귀속 (chat_14 — 하청 집행)**
> 하청업체(관계수급인)도 산안비를 직접 사용할 수 있습니다. 도급인은 산안비 범위 내에서 관계수급인에게 적정하게 지급하고, 하청업체가 이를 적법하게 사용한 경우 인정됩니다.

---

## 참고

- 기존 `core/rag.py` — BM25 + Vector Ensemble, rerank 구현체
- 기존 `core/storage.py` — load_vectorstore, load_collection_documents
- 기존 `core/llm_config.py` — LLM 싱글톤
- LangGraph 공식 문서: https://langchain-ai.github.io/langgraph/
- LangGraph MemorySaver: https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.memory.MemorySaver
