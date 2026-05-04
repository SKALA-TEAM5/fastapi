# AI Workspace

이 레포는 FastAPI 본체 전체를 관리하는 저장소가 아니라, AI agent와 OCR 관련 코드만 관리하는 작업 공간입니다.

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

이 레포에서 다루지 않는 것:

- 서비스 전체 FastAPI 애플리케이션 구조
- 프론트엔드
- 범용 백오피스 기능

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

앞으로 에이전트가 늘어나더라도 디렉토리 규칙은 동일하게 유지합니다.
