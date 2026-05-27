# 산업안전보건관리비 AI 감사 시스템

건설업 산업안전보건관리비 사용내역서의 항목 분류 및 집행 적정성을 자동 검증하는 AI 에이전트 시스템입니다.

---

## 목차

1. [Project Overview](#1-project-overview)
2. [Quick Start — 전체 실행 순서](#2-quick-start--전체-실행-순서)
3. [Getting Started](#3-getting-started)
4. [Database Architecture & Setup](#4-database-architecture--setup)
5. [Ingestion Pipeline](#5-ingestion-pipeline)
6. [Project Structure](#6-project-structure)
7. [Development & Contribution](#7-development--contribution)
8. [Usage](#8-usage)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Project Overview

### 소개

이 시스템은 두 개의 AI 에이전트로 구성됩니다.

- **Classifier Agent** — 사용내역서 항목을 9개 카테고리(CAT_01 ~ CAT_09)로 분류합니다.
- **Validator Agent** — 각 카테고리의 집행 내역이 법령 기준에 적합한지 검증하고 감사 의견을 생성합니다.

### 주요 기능

- 사용내역서 항목을 법령 기반으로 카테고리 자동 분류 및 이상 항목 탐지
- 카테고리별 집행 적정성 판정 (`적정`, `부적정`, `검토필요`)
- 법령 조항 인용 및 출처 기반 감사 사유 자동 생성
- BM25 + 벡터 앙상블 RAG 검색 + Cross-encoder Reranking
- PostgreSQL 법령 지식베이스 + Qdrant 벡터 DB 이중 구조
- **hash 기반 청크 단위 변경 감지** — 새벽 배치에서 변경된 조문만 선택적 재적재

### Tech Stack

| 분류 | 기술 |
|------|------|
| Language | Python 3.11.9 |
| Package Manager | uv |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Orchestration | LangGraph |
| Vector DB | Qdrant |
| Relational DB | PostgreSQL (`legal_rag` 스키마) |
| Embedding | sentence-transformers (HuggingFace) |
| Reranking | HuggingFace Cross-encoder |
| PDF 변환 | Docling |
| Law Scraping | requests + BeautifulSoup |

---

## 2. Quick Start — 전체 실행 순서

처음 세팅하는 사람을 위한 순서입니다. **이 순서대로 실행하면 시스템 전체가 동작합니다.**

```
Step 1. 저장소 클론 & 의존성 설치
Step 2. 환경변수 설정 (.env)
Step 3. PDF 법령 파일 배치
Step 4. Qdrant 컨테이너 실행
Step 5. PostgreSQL 스키마 생성 (Flyway)
Step 6. 전체 인덱싱 파이프라인 실행
Step 7. 동작 확인
Step 8. 에이전트 실행
```

---

### Step 1 — 저장소 클론 & 의존성 설치

```bash
git clone <repository-url>
cd rag
uv python install 3.11.9
uv sync
```

---

### Step 2 — 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```dotenv
OPENAI_API_KEY=sk-...
LAW_API_KEY=your_law_api_key        # open.law.go.kr 발급
DATABASE_URL=postgresql://safety_user:safety_password@localhost:5432/safety
QDRANT_URL=http://localhost:6333
```

---

### Step 3 — PDF 법령 파일 배치

아래 3개 파일을 `data/` 디렉토리에 복사합니다.

```
data/
  건설업 산업안전보건관리비 계상 및 사용기준(고용노동부고시)(제2025-11호)(20250212).pdf
  건설업 산업안전 보건관리비 해설 및 질의회시집(최종).pdf
  부록4산업안전보건관리비항목별사용불가내역.hwp.pdf
```

---

### Step 4 — Qdrant 컨테이너 실행

```bash
make vector-db
```

실행 확인: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

### Step 5 — PostgreSQL 스키마 생성

Flyway V2 마이그레이션으로 `legal_rag` 스키마와 3개 테이블(`legal_master`, `legal_rule_profiles`, `law_log`)을 생성합니다.

```bash
# db 프로젝트 디렉토리에서
make db-migrate
```

실행 확인:

```bash
psql $DATABASE_URL -c "\dt legal_rag.*"
# legal_master, legal_rule_profiles, law_log 가 보이면 성공
```

---

### Step 6 — 전체 인덱싱 파이프라인 실행

PDF, 법제처 Open API, 산안비 사용기준 고시 세 소스를 **Qdrant + PostgreSQL에 동시 적재**합니다.

```bash
uv run python -c "
from src.services.ingestion_service import run_pipeline
run_pipeline(force=True)
"
```

완료 시 출력 예시:

```
[PDF] 3개 파일 처리 시작
  [1/3] 건설업 산업안전보건관리비 계상 및 사용기준...
  [2/3] 건설업 산업안전 보건관리비 해설...
  [3/3] 부록4...

[PDF → RDB] legal_master 적재 시작...
[PDF → RDB] 완료: Master 842개 (Corpus 410 / Rules 432), Profiles 97개

[법제처 Open API] 조문 수집 및 Qdrant+RDB 적재 시작...
[법제처 Open API] 완료: 87개 조문

[산안비 사용기준] 고시 수집 및 Qdrant+RDB 적재 시작...
[산안비 사용기준] 완료: Qdrant 63개, Master 63개 ...
```

> 첫 실행은 PDF 변환(Docling)과 임베딩 시간이 포함되어 20~40분 소요될 수 있습니다.

---

### Step 7 — 동작 확인

```bash
# Qdrant 포인트 수 확인
uv run python -c "
from src.core.storage import _get_qdrant_client
client = _get_qdrant_client()
info = client.get_collection('legal_documents')
print(f'Qdrant points: {info.points_count}')
"

# PostgreSQL 적재 수 확인
psql $DATABASE_URL -c "
SELECT record_type, count(*) FROM legal_rag.legal_master GROUP BY record_type;
"
```

---

### Step 8 — 에이전트 실행

```bash
# Classifier Agent
uv run python -m src.agents.classifier_agent.cli \
  --input examples/classifier_agent/sample_input.json

# Validator Agent
uv run python -m src.agents.validator_agent.cli \
  --input examples/validator_agent/sample_input.json

# API 서버
uvicorn main:app --reload
# → http://localhost:8000/docs
```

---

## 3. Getting Started

> 아래 내용은 Quick Start의 각 단계에 대한 상세 설명입니다.

### Prerequisites

- Python 3.11.9
- [uv](https://github.com/astral-sh/uv) 패키지 매니저
- Docker & Docker Compose (Qdrant 실행용)
- PostgreSQL + Flyway (법령 DB 스키마 생성)
- `psql` CLI

### Installation

```bash
# 1. 저장소 클론
git clone <repository-url>
cd rag

# 2. Python 설치 (uv 사용)
uv python install 3.11.9

# 3. 의존성 설치
uv sync
```

### Environment Variables

프로젝트 루트에 `.env` 파일을 생성합니다.

```dotenv
# LLM
OPENAI_API_KEY=sk-...

# 법제처 Open API 키 (open.law.go.kr 발급)
LAW_API_KEY=your_law_api_key

# PostgreSQL (법령 지식베이스)
DATABASE_URL=postgresql://safety_user:safety_password@localhost:5432/safety

# Qdrant (벡터 DB) — 기본값: http://localhost:6333
QDRANT_URL=http://localhost:6333

# Qdrant 포트 (docker-compose용)
VECTOR_DB_PORT=6333
VECTOR_DB_GRPC_PORT=6334
```

---

## 4. Database Architecture & Setup

이 시스템은 두 종류의 DB를 사용합니다.

| DB | 역할 |
|----|------|
| **PostgreSQL** (`legal_rag` 스키마) | 법령 원문, 구조화 규칙, 운영형 보강 룰, 변경 이력 |
| **Qdrant** | 법령 텍스트 벡터 임베딩 (시맨틱 검색) |

> **참고:** 사용내역서 업무 DB는 백엔드가 별도로 관리합니다. 여기서 구축하는 것은 Classifier/Validator가 참조하는 **법령 지식베이스**입니다.

---

### 3-1. PostgreSQL: 법령 지식베이스

#### 스키마 구조

`legal_rag` 스키마 아래 **3개 테이블**로 구성됩니다.

| 테이블 | 역할 |
|--------|------|
| `legal_master` | 법령 원문 + 규칙 통합 마스터 (corpus / rule 단일 테이블) |
| `legal_rule_profiles` | 운영형 보강 룰 (synonym, allow_terms, disallow_terms 등) |
| `law_log` | 새벽 배치 변경 이력 (added / updated / deleted) |

#### `legal_master` 핵심 컬럼

| 컬럼 | 설명 |
|------|------|
| `id` | 레코드 고유 ID (예: `law_api:12345:article:72`) |
| `record_type` | `corpus` (법령 원문) 또는 `rule` (판정 규칙) |
| `source_type` | `law` / `guideline` / `qa` |
| `body` | 법령 본문 또는 규칙 원문 |
| `hash` | `sha256(body)` — 새벽 배치 변경 감지 기준 |
| `chunk_id` | Qdrant point ID와 1:1 대응 (`uuid5(id)`) |
| `article_no` | 조 (예: `제7조`) |
| `paragraph_no` | 항 (예: `제1항`) |
| `item_no` | 호 (예: `제2호`) |
| `category_code` | 카테고리 코드 (`CAT_01` ~ `CAT_09`) |
| `allowed` | 허용 여부 (`true` / `false` / `null`=조건부) |
| `limit_pct` | 법령상 한도 비율 (예: `0.2` = 20%) |

#### `chunk_id` — Qdrant 연결 키

```
legal_master.id  →  uuid5(NAMESPACE, id)  →  Qdrant point_id
"law_api:99999:article:72"  →  "bfec898d-e796-52af-83eb-185bce3478a6"
```

같은 `id`는 항상 같은 `chunk_id`를 반환하므로, DB 조회 없이 양쪽 저장소를 동기화할 수 있습니다.

#### `law_log` — 변경 이력

새벽 배치가 실행될 때마다 변경된 청크를 `run_id` 단위로 기록합니다.

| 컬럼 | 설명 |
|------|------|
| `run_id` | 배치 실행 묶음 ID |
| `change_type` | `added` / `updated` / `deleted` |
| `prev_hash` | 변경 이전 hash |
| `new_hash` | 변경 이후 hash |

#### Step 1 — PostgreSQL 스키마 생성

백엔드 팀이 Flyway V2 마이그레이션으로 `legal_rag` 스키마와 3개 테이블을 생성합니다.

```bash
# db 프로젝트에서 Flyway 마이그레이션 실행
make db-migrate
```

마이그레이션 파일: `db/migrations/V2__schema_legal_rag.sql`

#### Step 2 — Qdrant 컨테이너 실행

```bash
make vector-db
# 또는
docker compose up -d vector_db
```

Qdrant 대시보드: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

#### Step 3 — 전체 인덱싱 파이프라인 실행

`run_pipeline()` 한 번으로 **PDF, 법제처 Open API, 산안비 사용기준 고시** 세 소스가 Qdrant와 `legal_master`(RDB)에 동시 적재됩니다.

```bash
uv run python -c "
from src.services.ingestion_service import run_pipeline
run_pipeline(force=True)
"
```

> `DATABASE_URL` 환경변수가 설정되어 있으면 RDB에도 자동 적재됩니다.

---

### 3-2. 데이터 소스별 PDF 파일 배치

아래 3개 PDF 파일을 `data/` 디렉토리에 넣습니다.

```
data/
  건설업 산업안전보건관리비 계상 및 사용기준(고용노동부고시)(제2025-11호)(20250212).pdf
  건설업 산업안전 보건관리비 해설 및 질의회시집(최종).pdf
  부록4산업안전보건관리비항목별사용불가내역.hwp.pdf
```

---

## 5. Ingestion Pipeline

### 4-1. 전체 흐름

```
3개 데이터 소스
  ├── PDF 법령          (data/*.pdf)
  ├── 법제처 Open API   (open.law.go.kr)
  └── 산안비 사용기준   (law.go.kr 고시 HTML, 24h 캐시)
          │
          ▼
    run_pipeline()
          │
    ┌─────┴──────┐
    ▼            ▼
 Qdrant       legal_master (RDB)
(벡터 검색)   (구조화 판정 + 변경 감지)
```

모든 소스는 적재 시점에 `chunk_id = uuid5(master_id)`를 확정해 Qdrant point ID와 1:1 연결됩니다.

### 4-2. 소스별 적재 함수

| 소스 | Qdrant | RDB |
|------|--------|-----|
| PDF | `upsert_documents()` | `execute_payload_to_rdb()` |
| 법제처 Open API | `upsert_with_ids()` | `_upsert_articles_to_rdb()` |
| 산안비 사용기준 | `upsert_with_ids()` | `upsert_to_rdb()` |

### 4-3. run_pipeline() 파라미터

```python
run_pipeline(
    data_dir="data",           # PDF 위치
    output_dir="outputs",      # 변환 결과 저장
    collection="legal_documents",
    force=False,               # True: Qdrant 컬렉션 초기화 후 전체 재인덱싱
    reconvert=False,           # True: final.md 무시하고 PDF 변환부터 재실행
    skip_law_api=False,        # True: 법제처 API 수집 건너뜀
    skip_usage_standard=False, # True: 산안비 고시 수집 건너뜀
    database_url=None,         # None이면 환경변수 DATABASE_URL 자동 사용
)
```

### 4-4. 선택적 실행

```bash
# PDF만 재인덱싱 (법제처 API, 산안비 고시 건너뜀)
uv run python -c "
from src.services.ingestion_service import run_pipeline
run_pipeline(force=True, skip_law_api=True, skip_usage_standard=True)
"

# 산안비 사용기준만 단독 실행
uv run python -m src.services.ingestion.usage_standard_scraper

# 법제처 Open API만 단독 실행
uv run python -m src.services.ingestion.law_api_scraper
```

### 4-5. 새벽 배치 Refresh (hash 기반 변경 감지)

최초 적재 이후 법령 개정이 발생하면, 변경된 청크만 선택적으로 교체합니다.

```
① 재수집 (법제처 API / law.go.kr)
② sha256(새 body) vs legal_master.hash 비교
   ├── 동일  → skip (Qdrant·RDB 건드리지 않음)
   ├── 변경  → chunk_id = uuid5(master_id) 역산
   │          qdrant.delete(chunk_id) → 재임베딩 → qdrant.upsert(id=chunk_id)
   │          legal_master UPDATE (body, hash)
   │          law_log INSERT (prev_hash, new_hash, 'updated')
   ├── 신규  → qdrant.upsert + legal_master INSERT + law_log ('added')
   └── 삭제  → qdrant.delete + legal_master DELETE + law_log ('deleted')
```

> Refresh 로직은 `src/services/refresh/` 에서 구현 예정입니다. 스키마(hash, chunk_id, law_log)는 이미 준비되어 있습니다.

---

## 6. Project Structure

```
rag/
├── main.py                          # FastAPI 루트 진입점
├── pyproject.toml                   # 의존성 정의
├── docker-compose.yaml              # Qdrant 컨테이너 설정
├── Makefile                         # vector-db / db-migrate 제어
│
├── data/                            # 법령 PDF 원본 (git 제외)
├── outputs/                         # PDF 파싱 결과 final.md (git 제외)
├── artifacts/                       # payload JSON (git 제외)
├── scripts/
│   └── seed_legal_rule_profiles.json  # 운영형 보강 룰 시드 데이터
│
├── src/
│   ├── agents/
│   │   ├── classifier_agent/
│   │   │   ├── agent.py             # 분류 에이전트 메인 로직
│   │   │   └── cli.py               # CLI 진입점
│   │   └── validator_agent/
│   │       ├── agent.py             # 검증 에이전트 메인 로직
│   │       ├── audit.py             # 카테고리 최종 판정
│   │       ├── calculator.py        # 집행 지표 계산
│   │       ├── context_retriever.py # 벡터 DB 검색
│   │       ├── parser.py            # 요청 파싱
│   │       ├── presenter.py         # 감사 결과 DTO 변환 및 사유 생성
│   │       ├── rule_matcher.py      # 허용/불가 규칙 매칭
│   │       └── cli.py               # CLI 진입점
│   ├── core/
│   │   ├── judge.py                 # LangGraph 판정 노드
│   │   ├── llm_config.py            # 전역 LLM 설정
│   │   ├── rag.py                   # 앙상블 리트리버 + Reranker
│   │   └── storage.py               # Qdrant 연동 (make_chunk_id, upsert_with_ids)
│   ├── prompts/
│   │   ├── shared_prompt.py         # JUDGE_PROMPT, REWRITE_PROMPT
│   │   └── validator_prompt.py      # CATEGORY_DECISION_PROMPT, AUDIT_REASON_SYNTHESIS_PROMPT
│   ├── repositories/
│   │   ├── legal_rules_repository.py  # PostgreSQL legal_master 조회
│   │   └── legal_rules_exporter.py    # PDF 파싱 → payload → execute_payload_to_rdb()
│   ├── schemas/
│   │   ├── classifier.py            # 분류 스키마 (CATEGORIES 포함)
│   │   ├── shared.py                # 공통 스키마
│   │   └── validator.py             # 검증 스키마
│   └── services/
│       ├── ingestion/
│       │   ├── breadcrumb.py        # Breadcrumb 주입
│       │   ├── converter.py         # PDF → Markdown (Docling)
│       │   ├── law_api_scraper.py   # 법제처 Open API 조문 수집 + RDB 적재
│       │   ├── restructure.py       # Markdown 계층 구조 재구성
│       │   ├── splitter.py          # Markdown 청크 분할
│       │   └── usage_standard_scraper.py  # 산안비 고시 HTML 파싱 + RDB 적재
│       ├── ingestion_service.py     # run_pipeline() — 전체 인덱싱 진입점
│       ├── refresh/                 # (예정) 새벽 배치 hash diff + law_log 기록
│       └── validator_service.py     # 검증 서비스 계층
│
├── tests/
│   └── agents/
│       ├── classifier_agent/
│       │   ├── cases/               # inputs.json, expected.json
│       │   ├── output/              # 평가 결과 (자동 생성)
│       │   └── test_classifier_agent.py
│       └── validator_agent/
│           ├── cases/               # inputs.json, expected.json
│           ├── output/              # 평가 결과 (자동 생성)
│           ├── test_validator_agent.py
│           └── test_validator_query.py
│
├── examples/                        # 샘플 입력 JSON
└── docs/
    ├── legal-agent.md               # 이 문서
    └── swagger(classification,validation).yaml
```

---

## 7. Development & Contribution

### Branch Strategy

| 유형 | 패턴 | 예시 |
|------|------|------|
| 기능 개발 | `feature/<name>` | `feature/refresh-pipeline` |
| 버그 수정 | `fix/<name>` | `fix/chunk-id-mismatch` |
| 긴급 수정 | `hotfix/<name>` | `hotfix/db-connection` |

PR 대상 브랜치는 항상 `main`입니다.

### Merge Request / PR Protocol

- 공통 파일 (`__init__.py`, `.gitignore`, `pyproject.toml`, `main.py`, `uv.lock`) 은 PR에서 제외합니다.
- PR에는 변경 대상 모듈만 포함합니다.
- 코드 리뷰 승인 후 머지하며, 테스트 통과 필수입니다.

### Test Suite Execution

```bash
# Classifier 평가
uv run python -m tests.agents.classifier_agent.test_classifier_agent

# Validator 평가
uv run python -m tests.agents.validator_agent.test_validator_agent

# 단건 쿼리 테스트
uv run python -m tests.agents.validator_agent.test_validator_query
uv run python -m tests.agents.validator_agent.test_validator_query \
  --question "안전모 구입 비용은 산안비 항목인가?"
```

결과 파일은 `tests/agents/<agent>/output/results_YYYYMMDD_HHMM.json` 에 저장됩니다.

### Coding Standards

모든 Python 파일 최상단에 아래 형식의 헤더 주석을 작성합니다.

```python
# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
# 수정일   : 2026-05-26
#
# [ 주요 함수 정의 ]
#
# 1. run_pipeline() : 전체 인덱싱 파이프라인
# --------------------------------------------------------------------------
```

파일명 규칙:
- Python 모듈: 소문자 + 스네이크 케이스
- 서비스 파일: `*_service.py`
- 저장소 파일: `*_repository.py`
- 테스트 파일: `test_*.py`

---

## 8. Usage

### Classifier Agent 실행

```bash
uv run python -m src.agents.classifier_agent.cli \
  --input examples/classifier_agent/sample_input.json \
  --collection legal_documents
```

### Validator Agent 실행

```bash
uv run python -m src.agents.validator_agent.cli \
  --input examples/validator_agent/sample_input.json \
  --collection legal_documents
```

### API 서버 실행

```bash
uvicorn main:app --reload
```

API 명세: `docs/swagger(classification,validation).yaml` 또는 서버 실행 후 `/docs`

---

## 9. Troubleshooting

### `ModuleNotFoundError`

```bash
uv sync
```

### Qdrant 연결 실패

```bash
make vector-db    # 컨테이너 실행
docker ps         # 상태 확인
```

### PostgreSQL 연결 실패

`.env`의 `DATABASE_URL`을 확인하고 PostgreSQL이 실행 중인지 확인합니다.

```bash
psql $DATABASE_URL -c "SELECT 1;"
```

### `legal_rag` 스키마가 없다는 오류

Flyway V2 마이그레이션이 먼저 실행되어야 합니다.

```bash
make db-migrate
```

### 전체 재인덱싱 (chunk_id 포함 최신화)

스키마 변경 또는 코드 변경 후 기존 데이터를 새로 적재할 때 사용합니다.

```bash
# 1. Flyway 마이그레이션 재실행 (스키마 재생성)
make db-migrate

# 2. Qdrant + legal_master 전체 재적재
uv run python -c "
from src.services.ingestion_service import run_pipeline
run_pipeline(force=True)
"
```

### PDF 변환 실패 — `final.md`가 없는 경우

```bash
uv run python -c "
from src.services.ingestion_service import run_pipeline
run_pipeline(force=True, reconvert=True)
"
```

### Qdrant 컬렉션 수동 초기화

```bash
uv run python -c "
from src.core.storage import reset_collection
reset_collection('legal_documents')
"
```

### 산안비 고시 HTML 수집 실패 (WAF 차단)

law.go.kr은 직접 접근 시 WAF로 차단됩니다. 캐시 파일(`.cache/usage_standard_raw.html`)을 브라우저로 직접 저장하거나 캐시를 강제 갱신합니다.

```bash
uv run python -m src.services.ingestion.usage_standard_scraper --force-refresh
```

### 법제처 API 키 오류

`LAW_API_KEY` 환경변수를 확인합니다. [open.law.go.kr](https://open.law.go.kr)에서 키를 발급받습니다.

```bash
echo $LAW_API_KEY
```
