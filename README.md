# AI Workspace

## 환경 설정 원칙

- 로컬 개발은 `.env`를 사용합니다. 실제 `.env`는 커밋하지 않습니다.
- 커밋되는 파일은 `.env.example`뿐이며, API key와 비밀번호는 placeholder로 둡니다.
- Kubernetes 배포 시에는 ConfigMap/Secret으로 DB, Qdrant, OpenAI 설정을 주입합니다.
- `main`에 머지되면 운영 반영 대상이 되므로, 통합 확인은 `develop`에서 먼저 진행합니다.

로컬에서 공용 DB를 사용하려면 port-forward를 열고 `.env.example`을 복사합니다.

```bash
kubectl port-forward svc/team5-postgres 5433:5432 -n skala3-finalproj-class2-team5
cp .env.example .env
```

Qdrant를 Kubernetes에 띄운 뒤 로컬에서 확인할 때는 별도 터미널에서 포워딩합니다.

```bash
kubectl port-forward svc/team5-qdrant 6333:6333 -n skala3-finalproj-class2-team5
```

주요 목적:

- 문서 이해 및 분류용 AI agent 개발
- 증빙 점검, 리포트 생성 등 도메인별 agent 개발
- OCR 추출, 전처리, 후처리 파이프라인 관리
- 프롬프트, 스키마, 예제 입력 데이터, 공통 유틸 관리

## 범위

이 레포에서 다루는 것:

- AI agent
- OCR
- LLM 프롬프트
- 벡터 검색, 임베딩, 문서 파싱
- AI 입출력 스키마
- 샘플 입력, 실험 예제

## 디렉토리 원칙

모든 구현 코드는 반드시 `src/` 아래에 둡니다.
루트에는 프로젝트 설정 파일과 공통 문서만 둡니다.

권장 구조:

```text
src/
  agents/
    report_agent/
    safety_doc_agent/
  ocr/
  prompts/
  schemas/
  services/
  repositories/
  core/
examples/
tests/
docs/
```

의미:

- `src/agents/`: 에이전트별 실행 로직과 판단 흐름
- `src/ocr/`: OCR 추출 및 전처리
- `src/prompts/`: 공통 또는 에이전트별 프롬프트
- `src/schemas/`: AI 입출력 스키마
- `src/services/`: OpenAI, Qdrant, 파일 처리 같은 서비스 계층
- `src/repositories/`: DB 접근 계층
- `src/core/`: 설정, 상수, 공통 유틸
- `examples/`: 샘플 입력 및 테스트용 예제 데이터
- `tests/`: 자동화 테스트
- `docs/`: 작업 규칙, 설계 문서

## 폴더별 파일 구성

각 폴더에는 아래 성격의 파일을 둡니다.

### `src/agents/<agent_name>/`

에이전트 하나를 실행하는 데 필요한 진입점과 도메인 로직을 둡니다.

권장 파일:

- `__init__.py`: 패키지 선언
- `cli.py`: CLI 진입점 또는 로컬 실행용 명령
- `agent.py`: 에이전트 메인 orchestration
- `workflow.py`: 여러 단계 작업 흐름 정의
- `config.py`: 에이전트 전용 설정
- `prompts.py` 또는 `prompts/`: 해당 에이전트 전용 프롬프트
- `parser.py`: 문서/입력 파싱 로직
- `audit.py`, `report.py`, `classify.py`: 도메인별 핵심 동작 모듈
- `db_flow_demo.py`: DB 연동 전 시뮬레이션용 실험 모듈

예시:

```text
src/agents/safety_doc_agent/
  __init__.py
  cli.py
  config.py
  parser.py
  audit.py
  vector_store.py
  db_flow_demo.py
```

```text
src/agents/report_agent/
  __init__.py
  agent.py
  context_builder.py
  renderer.py
```

### `src/ocr/`

OCR 추출 및 후처리 로직을 둡니다.

권장 파일:

- `extractor.py`: OCR 엔진 호출
- `preprocess.py`: 이미지 전처리
- `postprocess.py`: OCR 결과 정제
- `layout.py`: 표, 블록, 페이지 레이아웃 해석
- `types.py`: OCR 결과 스키마

예시:

```text
src/ocr/
  extractor.py
  preprocess.py
  postprocess.py
  layout.py
```

### `src/prompts/`

에이전트 간 재사용 가능한 공통 프롬프트를 둡니다.

권장 파일:

- `system_*.md`: system prompt 원문
- `*_prompt.py`: 프롬프트 조립 함수
- `templates.py`: 공통 템플릿

예시:

```text
src/prompts/
  evidence_requirement_prompt.py
  report_prompt.py
  system_report.md
```

### `src/schemas/`

AI 입력/출력, OCR 결과, DB row 변환용 스키마를 둡니다.

권장 파일:

- `evidence.py`: 증빙 관련 스키마
- `report.py`: 리포트 관련 스키마
- `ocr.py`: OCR 결과 스키마
- `common.py`: 공통 스키마

### `src/services/`

저장소, 프롬프트, 모델 호출을 조합하는 응용 서비스 계층을 둡니다.

권장 파일:

- `*_service.py`: 하나의 서비스 책임을 갖는 모듈

예시:

```text
src/services/
  evidence_requirement_service.py
  evidence_check_service.py
  report_generation_service.py
```

### `src/repositories/`

DB 접근 인터페이스와 SQL, 또는 실제 어댑터 구현을 둡니다.

권장 파일:

- `*_repository.py`: 저장소 인터페이스 또는 구현
- `postgres_queries.py`: SQL 모음
- `db.py`: 커넥션/세션 유틸

예시:

```text
src/repositories/
  evidence_repository.py
  postgres_queries.py
```

### `src/core/`

프로젝트 전역에서 재사용하는 공통 설정과 상수를 둡니다.

권장 파일:

- `config.py`: 공통 설정
- `constants.py`: 상수
- `logging.py`: 로깅 설정
- `exceptions.py`: 공통 예외

### `examples/`

샘플 입력, 시뮬레이션 JSON, 테스트용 예제 파일을 둡니다.

권장 구조:

```text
examples/
  report_agent/
    sample_input.json
  safety_doc_agent/
    db_flow_sample.json
    sample_site_docs/
```

### `tests/`

자동화 테스트를 둡니다.

권장 구조:

```text
tests/
  agents/
    test_safety_doc_agent.py
  services/
    test_evidence_requirement_service.py
  fixtures/
    sample_item.json
```

### `docs/`

작업 규칙, 프롬프트 정책, 설계 메모를 둡니다.

권장 파일:

- `architecture.md`
- `prompt-guidelines.md`
- `db-integration.md`
- `workflow.md`

## 파일명 규칙

파일명은 역할이 바로 드러나도록 짓고, 가능한 한 소문자 + 스네이크 케이스를 사용합니다.

기본 규칙:

- Python 모듈: 소문자 + 스네이크 케이스
- 디렉토리명: 소문자 + 스네이크 케이스
- 문서 파일: 소문자 + 하이픈 또는 스네이크 케이스 중 하나로 통일
- 프롬프트 원문 markdown: `system_*.md`, `user_*.md`, `*_template.md`
- 서비스 파일: `*_service.py`
- 저장소 파일: `*_repository.py`
- 테스트 파일: `test_*.py`
- 예제 JSON: `sample_*.json`, `*_sample.json`

좋은 예시:

- `evidence_requirement_service.py`
- `evidence_repository.py`
- `db_flow_demo.py`
- `report_prompt.py`
- `test_report_agent.py`
- `sample_item.json`
- `system_report.md`

피해야 하는 예시:

- `ReportAgent.py`
- `myFile.py`
- `test.py`
- `final_final_real.py`
- `new folder/`

### 에이전트 폴더 내부 파일명 예시

`safety_doc_agent`

- `cli.py`
- `config.py`
- `parser.py`
- `audit.py`
- `vector_store.py`
- `db_flow_demo.py`

`report_agent`

- `agent.py`
- `context_builder.py`
- `renderer.py`
- `schemas.py`
- `llm.py`

### 프롬프트 파일명 예시

- `evidence_requirement_prompt.py`
- `report_generation_prompt.py`
- `system_report.md`
- `report_draft_template.md`

### 문서 파일명 예시

- `architecture.md`
- `db-integration.md`
- `agent-naming-rules.md`

### 샘플/실험 파일명 예시

- `db_flow_sample.json`
- `sample_input.json`
- `sample_site_docs/`

## 새 에이전트 추가 체크리스트

새 에이전트를 만들 때는 최소한 아래를 맞춥니다.

1. `src/agents/<agent_name>/` 폴더 생성
2. 실행 진입 파일 추가
3. 필요한 프롬프트 위치 결정
4. 입력/출력 스키마 정의
5. 서비스 계층 연결
6. 예제 입력 `examples/<agent_name>/` 추가
7. 테스트 파일 추가
8. README 또는 docs 문서 갱신

## 작업 규칙

- 새 기능 디렉토리를 루트에 직접 만들지 않습니다.
- 새 에이전트는 `src/agents/<agent_name>/` 아래에 추가합니다.
- OCR 관련 코드는 `src/ocr/` 아래에만 둡니다.
- 프롬프트 파일은 `src/prompts/` 또는 에이전트 내부 `prompts/`에 둡니다.
- 샘플 데이터는 `examples/` 아래에 둡니다.
- 개인 실험 파일, 임시 노트, 로컬 산출물은 커밋하지 않습니다.

## 브랜치 규칙

- 기능 개발: `feature/<name>`
- 버그 수정: `fix/<name>`
- 긴급 수정: `hotfix/<name>`

예시:

- `feature/report-agent`
- `feature/safelee-agent`
- `fix/prompt-parser`

PR 대상은 기본적으로 `main`입니다.

## 개발 환경

- Python: `3.11.9`
- 패키지 관리: `uv`

설치:

```bash
uv python install 3.11.9
uv sync
```

`uv.lock`은 팀 공통 의존성 버전 고정을 위해 함께 커밋합니다.

## Agent 예시

- `report_agent`: 입력 데이터 기반 리포트 초안 생성
- `safety_doc_agent`: 산업안전보건관리비 증빙서류 점검
