# 산업안전보건관리비 AI 감사 시스템

건설업 산업안전보건관리비 사용내역서의 항목 분류 및 집행 적정성을 자동 검증하는 AI 에이전트 시스템입니다.

---

## 목차

1. [Project Overview](#1-project-overview)
2. [Getting Started](#2-getting-started)
3. [Database Architecture & Setup](#3-database-architecture--setup)
4. [Project Structure](#4-project-structure)
5. [Development & Contribution](#5-development--contribution)
6. [Usage](#6-usage)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Project Overview

### 소개

이 시스템은 두 개의 AI 에이전트로 구성됩니다.

- **Classifier Agent (2번 Agent)** — 사용내역서 항목을 9개 카테고리로 분류합니다.
- **Validator Agent (6번 Agent)** — 각 카테고리의 집행 내역이 법령 기준에 적합한지 검증하고 감사 의견을 생성합니다.

### 주요 기능

- 사용내역서 항목을 법령 기반으로 카테고리 자동 분류 및 이상 항목 탐지
- 카테고리별 집행 적정성 판정 (`적정`, `부적정`, `검토필요`)
- 법령 조항 인용 및 출처 기반 감사 사유 자동 생성
- BM25 + 벡터 앙상블 RAG 검색 + Cross-encoder Reranking
- PostgreSQL 법령 지식베이스 + Qdrant 벡터 DB 이중 구조

### Tech Stack

| 분류 | 기술 |
|------|------|
| Language | Python 3.11.9 |
| Package Manager | uv |
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Orchestration | LangGraph |
| Vector DB | Qdrant |
| Relational DB | PostgreSQL |
| Embedding | sentence-transformers (HuggingFace) |
| Reranking | HuggingFace Cross-encoder |
| PDF 변환 | Docling |
| Law Scraping | requests + Selenium |

---

## 2. Getting Started

### Prerequisites

- Python 3.11.9
- [uv](https://github.com/astral-sh/uv) 패키지 매니저
- Docker & Docker Compose (Qdrant 실행용)
- PostgreSQL (법령 DB)
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

# PostgreSQL (법령 지식베이스)
DATABASE_URL=postgresql://safety_user:safety_password@localhost:5432/safety

# Qdrant (벡터 DB) — 기본값: http://localhost:6333
QDRANT_URL=http://localhost:6333

# Qdrant 포트 (docker-compose용)
VECTOR_DB_PORT=6333
VECTOR_DB_GRPC_PORT=6334
```

---

## 3. Database Architecture & Setup

이 시스템은 두 종류의 DB를 사용합니다.

| DB | 역할 |
|----|------|
| **PostgreSQL** (`legal_rag` 스키마) | 법령 원문, 구조화 규칙, 운영형 보강 룰 |
| **Qdrant** | 법령 텍스트 벡터 임베딩 (시맨틱 검색) |

> **참고:** 사용내역서 업무 DB는 백엔드가 별도로 관리합니다. 여기서 구축하는 것은 Classifier/Validator가 참조하는 **법령 지식베이스**입니다.

---

### 3-1. PostgreSQL: 법령 메타데이터 및 구조화 데이터

#### 스키마 구조

`legal_rag` 스키마 아래 6개 테이블로 구성됩니다.

| 테이블 | 역할 |
|--------|------|
| `legal_source_documents` | 법령 원본 문서 목록 (고시, 해설집, 부록) |
| `legal_corpus` | 법령/해설/Q&A 원문 본문 (조문·문단 단위) |
| `legal_citations` | 조항 인용 색인 (`제7조제1항제2호` 등 구조화) |
| `legal_rules` | 기계 판단 가능한 허용/불가/한도 규칙 |
| `legal_rule_profiles` | 운영형 보강 룰 (synonym, allow_terms, disallow_terms) |
| `legal_rule_master` | 서비스 조회용 반정규화 통합 Read Model |

스키마 상세 설명 및 DBML: `docs/legal_db_explainer.md`

#### Step 1 — PostgreSQL 스키마 생성

백엔드 팀이 Flyway V2 마이그레이션으로 `legal_rag` 스키마와 6개 테이블을 생성합니다.
직접 생성하는 경우 `docs/legal_db_explainer.md`의 DBML을 참고하세요.

#### Step 2 — PDF 법령 파일 배치

아래 3개 PDF 파일을 `data/` 디렉토리에 넣습니다.

```
data/
  건설업 산업안전보건관리비 계상 및 사용기준(고용노동부고시)(제2025-11호)(20250212).pdf
  건설업 산업안전 보건관리비 해설 및 질의회시집(최종).pdf
  부록4산업안전보건관리비항목별사용불가내역.hwp.pdf
```

#### Step 3 — PDF → Markdown 변환 및 파싱

```bash
uv run python -c "
from src.services.ingestion_service import run_pipeline
run_pipeline()
"
```

변환 결과는 `outputs/<문서명>/final.md` 에 저장됩니다.

#### Step 4 — Payload 생성 (JSON + SQL)

`outputs/`의 파싱 결과와 `scripts/seed_legal_rule_profiles.json`을 읽어 PostgreSQL seed 파일을 생성합니다.

```bash
uv run python -m src.repositories.legal_rules_export \
  --outputs-dir outputs \
  --rule-config-path scripts/seed_legal_rule_profiles.json \
  --json-out artifacts/legal_rules_payload.json \
  --sql-out artifacts/legal_rules_seed.sql
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--outputs-dir` | PDF 파싱 결과 디렉토리 | `outputs` |
| `--rule-config-path` | 운영형 보강 룰 JSON | `scripts/seed_legal_rule_profiles.json` |
| `--json-out` | payload JSON 출력 경로 | `artifacts/legal_rules_payload.json` |
| `--sql-out` | seed SQL 출력 경로 | `artifacts/legal_rules_seed.sql` |
| `--apply` | 생성된 SQL을 즉시 DB에 적재 | `False` |
| `--database-url` | PostgreSQL 연결 문자열 | `.env`의 `DATABASE_URL` |
| `--cleanup` | DB 적재 후 중간 파일(json/sql) 삭제 | `False` |

#### Step 5 — SQL을 PostgreSQL에 적재

**방법 A — `--apply` 플래그 (권장)**

```bash
uv run python -m src.repositories.legal_rules_export \
  --outputs-dir outputs \
  --apply \
  --database-url "postgresql://safety_user:safety_password@localhost:5432/safety"
```

**방법 B — psql 직접 실행**

```bash
psql $DATABASE_URL -f artifacts/legal_rules_seed.sql
```

---

### 3-2. Qdrant: 벡터 임베딩 (시맨틱 검색)

#### Step 1 — Qdrant 컨테이너 실행

```bash
make vector-db
```

또는 직접:

```bash
docker compose up -d vector_db
```

Qdrant 대시보드: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

#### Step 2 — 법령 문서를 벡터 DB에 적재

`run_pipeline()`을 실행하면 PDF 변환과 Qdrant 적재가 함께 수행됩니다. PDF 파싱 결과(`final.md`)가 이미 있는 경우 벡터 DB에만 재인덱싱할 수 있습니다.

```bash
# 최초 실행 (PDF 변환 + Qdrant 적재)
uv run python -c "
from src.services.ingestion_service import run_pipeline
run_pipeline()
"

# Qdrant만 전체 재인덱싱 (final.md 재사용)
uv run python -c "
from src.services.ingestion_service import run_pipeline
run_pipeline(force=True)
"
```

#### 전체 DB 구축 순서 요약

```
1. PDF 파일을 data/ 에 배치
        ↓
2. Qdrant 컨테이너 실행
   make vector-db
        ↓
3. PDF → Markdown 변환 + Qdrant 적재
   uv run python -c "from src.services.ingestion_service import run_pipeline; run_pipeline()"
        ↓
4. Payload 생성 (JSON + SQL)
   uv run python -m src.repositories.legal_rules_export
        ↓
5. PostgreSQL 적재 (Flyway 스키마 생성 선행 필수)
   uv run python -m src.repositories.legal_rules_export --apply
```

---

## 4. Project Structure

```
rag/
├── main.py                          # 루트 진입점
├── pyproject.toml                   # 의존성 정의
├── docker-compose.yaml              # Qdrant 컨테이너 설정
├── Makefile                         # vector-db 제어 명령
│
├── data/                            # 법령 PDF 원본 (git 제외)
├── outputs/                         # PDF 파싱 결과 final.md (git 제외)
├── artifacts/                       # payload JSON, seed SQL (git 제외)
├── scripts/
│   └── seed_legal_rule_profiles.json  # 운영형 보강 룰 시드 데이터
│
├── src/
│   ├── agents/
│   │   ├── classifier_agent/
│   │   │   ├── agent.py             # 분류 에이전트 메인 로직
│   │   │   └── main.py              # CLI 진입점
│   │   └── validator_agent/
│   │       ├── agent.py             # 검증 에이전트 메인 로직
│   │       ├── audit.py             # 카테고리 최종 판정
│   │       ├── calculator.py        # 집행 지표 계산
│   │       ├── context_retriever.py # 벡터 DB 검색
│   │       ├── parser.py            # 요청 파싱
│   │       ├── presenter.py         # 감사 결과 DTO 변환 및 사유 생성
│   │       ├── rule_matcher.py      # 허용/불가 규칙 매칭
│   │       └── main.py              # CLI 진입점
│   ├── core/
│   │   ├── judge.py                 # LangGraph 판정 노드
│   │   ├── llm_config.py            # 전역 LLM 설정
│   │   ├── rag.py                   # 앙상블 리트리버 + Reranker
│   │   └── storage.py               # Qdrant 연동
│   ├── prompts/
│   │   ├── shared.py                # JUDGE_PROMPT, REWRITE_PROMPT
│   │   └── validator.py             # CATEGORY_DECISION_PROMPT, AUDIT_REASON_SYNTHESIS_PROMPT
│   ├── repositories/
│   │   ├── legal_rules_repository.py  # PostgreSQL 법령 조회
│   │   └── legal_rules_export.py      # PDF 파싱 → payload → SQL export
│   ├── schemas/
│   │   ├── classifier.py            # 분류 스키마 (CATEGORIES 포함)
│   │   ├── shared.py                # 공통 스키마
│   │   └── validator.py             # 검증 스키마
│   └── services/
│       ├── ingestion/               # PDF 변환, 마크다운 파싱, 청크 분할
│       ├── ingestion_service.py     # RAG 인덱싱 파이프라인
│       └── validator_service.py     # 검증 서비스 계층
│
├── tests/
│   ├── classifier/
│   │   ├── cases/                   # inputs.json, expected.json
│   │   ├── output/                  # 평가 결과 (자동 생성)
│   │   └── eval.py                  # Classifier 평가 스크립트
│   └── validator/
│       ├── cases/                   # inputs.json, expected.json
│       ├── output/                  # 평가 결과 (자동 생성)
│       ├── eval.py                  # Validator 평가 스크립트
│       └── query.py                 # 단건 쿼리 테스트
│
├── examples/                        # 샘플 입력 JSON
└── docs/                            # 설계 문서
    ├── legal_db_explainer.md        # PostgreSQL 스키마 상세 설명
    └── swagger(classification,validation).yaml  # API 명세
```

---

## 5. Development & Contribution

### Branch Strategy

| 유형 | 패턴 | 예시 |
|------|------|------|
| 기능 개발 | `feature/<name>` | `feature/validator-reason` |
| 버그 수정 | `fix/<name>` | `fix/source-display` |
| 긴급 수정 | `hotfix/<name>` | `hotfix/db-connection` |

PR 대상 브랜치는 항상 `main`입니다.

### Merge Request / PR Protocol

- 공통 파일 (`__init__.py`, `.gitignore`, `pyproject.toml`, `main.py`, `README.md`, `uv.lock`) 은 PR에서 제외합니다.
- PR에는 변경 대상 모듈만 포함합니다.
- 코드 리뷰 승인 후 머지하며, 테스트 통과 필수입니다.

### Test Suite Execution

```bash
# Classifier 평가
uv run python -m tests.classifier.eval --verbose

# Validator 평가
uv run python -m tests.validator.eval --verbose

# 특정 케이스만 실행 (Classifier)
uv run python -m tests.classifier.eval --id <케이스ID> --verbose

# 모델 지정 (Validator, 기본값: gpt-4o-mini)
uv run python -m tests.validator.eval --model gpt-4o --verbose
```

결과 파일은 `tests/<agent>/output/results_YYYYMMDD_HHMM.json` 에 저장됩니다.

### Coding Standards

모든 Python 파일 최상단에 아래 형식의 헤더 주석을 작성합니다.

```python
# --------------------------------------------------------------------------
# 작성자   : 송상민(ss19801)
# 작성일   : 2026-05-04
#
# [ 주요 클래스 및 함수 정의 ]
#
# 1. build_payload() : 법령/해설/프로필 통합 payload 생성
# 2. payload_to_sql() : payload를 PostgreSQL seed SQL로 변환
# 3. main() : export 및 선택적 DB 적용 진입점
# --------------------------------------------------------------------------
```

파일명 규칙:
- Python 모듈: 소문자 + 스네이크 케이스
- 서비스 파일: `*_service.py`
- 저장소 파일: `*_repository.py`
- 테스트 파일: `eval.py` 또는 `test_*.py`

---

## 6. Usage

### Classifier Agent 실행

```bash
uv run python -m src.agents.classifier_agent.main \
  --input examples/classifier_agent/sample_input.json \
  --collection legal_documents
```

### Validator Agent 실행

```bash
uv run python -m src.agents.validator_agent.main \
  --input examples/validator_agent/sample_input.json \
  --collection legal_documents
```

### API 서버 실행

```bash
uvicorn main:app --reload
```

API 명세: `docs/swagger(classification,validation).yaml` 또는 서버 실행 후 `/docs`

---

## 7. Troubleshooting

### `ModuleNotFoundError: No module named 'langchain_openai'`

의존성이 설치되지 않은 경우입니다.

```bash
uv sync
```

### Qdrant 연결 실패

```
ConnectionError: http://localhost:6333
```

Qdrant 컨테이너 실행 상태를 확인합니다.

```bash
make vector-db    # 컨테이너 실행
docker ps         # 상태 확인
```

### PostgreSQL 연결 실패

```
psycopg2.OperationalError: could not connect to server
```

`.env`의 `DATABASE_URL`을 확인하고 PostgreSQL이 실행 중인지 확인합니다.

```bash
psql $DATABASE_URL -c "SELECT 1;"
```

### `legal_rag` 스키마가 없다는 오류

Flyway V2 마이그레이션이 먼저 실행되어야 합니다. `--apply` 전에 반드시 스키마 생성이 선행되어야 합니다.

### PDF 변환 실패 — `final.md`가 없는 경우

`reconvert=True`로 PDF 변환부터 재시도합니다.

```bash
uv run python -c "
from src.services.ingestion_service import run_pipeline
run_pipeline(force=True, reconvert=True)
"
```

### 벡터 DB 전체 재인덱싱

```bash
uv run python -c "
from src.services.ingestion_service import run_pipeline
run_pipeline(force=True)
"
```

### Qdrant 컬렉션 수동 초기화

```bash
uv run python -c "
from src.core.storage import reset_collection
reset_collection('legal_documents')
"
```