# Safety Doc Agent

`safety_doc_agent`는 산업안전보건관리비 사용내역서 항목을 기준으로 필요한 증빙자료를 추론하고,
현재 업로드된 증빙과 비교해 충족 여부를 점검하는 에이전트입니다.

## 역할

- 사용내역서 상세 항목별 필수 증빙자료 추론
- Postgres의 증빙 연결 정보와 비교해 충족 여부 계산
- 결과를 `agent_logs`와 `evidence_requirements`에 저장

## 관련 파일

- CLI: `src/agents/safety_doc_agent/cli.py`
- 설정: `src/agents/safety_doc_agent/config.py`
- 프롬프트: `src/prompts/safety_doc_agent_evidence_requirement_prompt.py`
- 저장소: `src/repositories/safety_doc_agent_postgres_evidence_repository.py`
- 스키마: `src/schemas/safety_doc_agent_evidence.py`
- 참고자료 벡터 도구: `src/tools/safety_doc_reference_vector.py`
- 서비스:
  `src/services/safety_doc_agent_evidence_requirement_service.py`
  `src/services/safety_doc_agent_evidence_check_service.py`

## 실행 환경

- Python `3.11.9`
- 패키지 관리: `uv`
- 외부 서비스:
  - OpenAI API
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
- `LANGSMITH_TRACING`
- `LANGSMITH_PROJECT`
- `LANGSMITH_WORKSPACE_ID`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`
- `SERVICE_APP_USER`
- `SERVICE_APP_PASSWORD`

기본값은 `src/agents/safety_doc_agent/config.py`에서 확인할 수 있습니다.

참고자료 벡터 도구를 사용할 때만 추가로 사용합니다.

- `OPENAI_EMBEDDING_MODEL`
- `SAFETY_DOC_QDRANT_PATH`
- `SAFETY_DOC_QDRANT_URL`
- `SAFETY_DOC_QDRANT_API_KEY`

## 명령어

엔트리포인트:

```bash
uv run safety-doc-agent --help
```

### DB 기반 증빙 판단 실행

로컬 Postgres에서 사용내역서 항목과 연결된 증빙 정보를 읽어,
필수 증빙 추론과 충족 여부 계산을 수행합니다.

```bash
uv run safety-doc-agent check-missing-evidence --item-id 1
```

저장 없이 입력과 출력만 확인하려면:

```bash
uv run safety-doc-agent check-missing-evidence --item-id 1 --dry-run
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
3. 추론 결과를 `service.evidence_requirements`의 active requirement 형태로 저장합니다.
4. 연결된 증빙 파일과 비교해 충족 여부를 계산합니다.
5. 결과를 `service.agent_logs`에 `agent_type_code = 'safety-doc'`로 기록합니다.

`agent_logs.result_code`는 누락 증빙이 없으면 `success`, 누락 증빙이 있으면 후속 확인이 필요하다는 의미의 `hil`로 저장합니다.

## 오케스트레이션 연동

오케스트레이터에서는 CLI를 subprocess로 호출하지 않고 함수형 진입점을 직접 호출합니다.

```python
from src.agents.safety_doc_agent.agent import check_missing_evidence

result = check_missing_evidence(item_id=1)
```

저장 없이 AI 입력/출력만 확인하려면:

```python
result = check_missing_evidence(item_id=1, dry_run=True)
```

테스트나 별도 worker에서는 `settings`, `repository`, `openai_client`를 주입해 같은 흐름을 재사용할 수 있습니다.

## 참고자료 벡터 도구

증빙 판단 agent의 DB 실행 경로와 분리된 Qdrant 참고자료 도구입니다.
가이드나 내부 기준 문서를 벡터DB로 추출해 두고, 프롬프트/룰 개선이나 디버깅 시 검색할 수 있습니다.

```bash
uv run python -m tools.safety_doc_reference_vector build \
  --source '/path/to/safety-guide.md' \
  --collection safety_doc_reference
```

```bash
uv run python -m tools.safety_doc_reference_vector search \
  --collection safety_doc_reference \
  --query '안전난간 설치 사진과 거래명세서 필요 여부'
```

## 참고

- 현재 흐름은 `DB view 기반 입력 + OpenAI 판단 + evidence requirement 저장/검증` 기준입니다.
- 실제 배포 전에는 DB 계정 권한을 환경별로 분리하는 것을 권장합니다.
- 오케스트레이터가 저장하는 배치 실행 로그에는 원본 파일명과 `storage_key`를 포함하지 않습니다.

## Prometheus 지표

`/metrics`에서 다음 Safety Doc 전용 지표를 확인할 수 있습니다.

- `safety_doc_runs_total{mode,result}`: 단건/배치 실행 결과
- `safety_doc_inference_duration_seconds{mode,model}`: LLM 추론 시간
- `safety_doc_llm_failures_total{mode}`: LLM 호출 실패 횟수
- `safety_doc_reference_failures_total{mode}`: 참고자료 검색 실패 횟수
- `safety_doc_missing_evidence_total{evidence_type}`: 유형별 누락 증빙 수
- `safety_doc_batch_size`: 배치당 항목 수
- `safety_doc_confidence{mode}`: 모델이 반환한 confidence 분포
- `safety_doc_tokens_total{model,type}`: 입력·출력 등 토큰 사용량

Grafana에서는 실행 성공/HIL/실패율, P95 추론 시간, 참고자료 검색 실패,
증빙 유형별 누락 추이와 모델별 토큰 사용량을 우선 대시보드로 구성합니다.
