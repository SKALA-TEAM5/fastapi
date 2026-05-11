# Safety Doc Agent

`safety_doc_agent`는 산업안전보건관리비 사용내역서 항목을 기준으로 필요한 증빙자료를 추론하고,
현재 업로드된 증빙과 비교해 충족 여부를 점검하는 에이전트입니다.

## 역할

- 가이드 문서를 파싱하고 Qdrant에 인덱싱
- 사용내역서 상세 항목별 필수 증빙자료 추론
- Postgres의 증빙 연결 정보와 비교해 충족 여부 계산
- 결과를 validation log와 requirement 형태로 저장

## 관련 파일

- CLI: `src/agents/safety_doc_agent/cli.py`
- 설정: `src/agents/safety_doc_agent/config.py`
- 벡터 저장소: `src/agents/safety_doc_agent/vector_store.py`
- 프롬프트: `src/prompts/safety_doc_agent_evidence_requirement_prompt.py`
- 저장소: `src/repositories/safety_doc_agent_postgres_evidence_repository.py`
- 스키마: `src/schemas/safety_doc_agent_evidence.py`
- 서비스:
  `src/services/safety_doc_agent_evidence_requirement_service.py`
  `src/services/safety_doc_agent_evidence_check_service.py`

## 실행 환경

- Python `3.11.9`
- 패키지 관리: `uv`
- 외부 서비스:
  - OpenAI API
  - Qdrant
  - PostgreSQL

설치 예시:

```bash
uv python install 3.11.9
uv sync
```

## 환경변수

최소 필요 값:

- `OPENAI_API_KEY`

주요 설정:

- `OPENAI_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `LANGSMITH_TRACING`
- `LANGSMITH_PROJECT`
- `LANGSMITH_WORKSPACE_ID`
- `QDRANT_PATH`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`
- `SERVICE_APP_USER`
- `SERVICE_APP_PASSWORD`

기본값은 `src/agents/safety_doc_agent/config.py`에서 확인할 수 있습니다.

## 명령어

엔트리포인트:

```bash
uv run safety-doc-agent --help
```

### 1. 가이드 인덱싱

가이드 마크다운을 파싱한 뒤 Qdrant 컬렉션에 적재합니다.

```bash
uv run safety-doc-agent ingest \
  --guide '/path/to/safety-guide.md' \
  --collection local_safety_doc_agent_guide
```

이 명령은 다음을 수행합니다.

- 가이드 마크다운 파싱
- `data/parsed_guide.json` 생성
- Qdrant 컬렉션에 청크 임베딩 저장

### 2. DB 기반 증빙 판단 실행

로컬 Postgres에서 사용내역서 항목과 연결된 증빙 정보를 읽어,
필수 증빙 추론과 충족 여부 계산을 수행합니다.

```bash
uv run safety-doc-agent run-db-flow --item-id 1
```

저장 없이 입력과 출력만 확인하려면:

```bash
uv run safety-doc-agent run-db-flow --item-id 1 --dry-run
```

출력에는 아래 정보가 포함됩니다.

- `db_target`
- `input_from_db_views`
- `ai_response`
- `saved_requirements`
- `requirements_after_save`
- `evidence_status`

## 동작 흐름

1. Postgres에서 대상 사용내역서 항목과 증빙 타입 목록을 조회합니다.
2. 항목 정보를 바탕으로 OpenAI에 필수 증빙자료를 추론 요청합니다.
3. 추론 결과를 active requirement 형태로 저장합니다.
4. 연결된 증빙 파일과 비교해 충족 여부를 계산합니다.
5. 결과를 validation log와 함께 후속 검토에 활용합니다.

## 참고

- 현재 흐름은 `DB view 기반 입력 + OpenAI 판단 + evidence requirement 저장/검증` 기준입니다.
- Qdrant는 원격 서버 모드와 로컬 파일 모드를 모두 지원합니다.
- 실제 배포 전에는 DB 계정 권한과 Qdrant 컬렉션명을 환경별로 분리하는 것을 권장합니다.
