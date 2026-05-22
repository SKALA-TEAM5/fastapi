# Report Agent

산업안전보건관리비 증빙 검토 결과를 `ReportDraft` JSON 보고서로 만드는 agent 모듈입니다.

## 역할

`ReportAgent`는 이미 수집·검증된 데이터를 보고서 구조로 정리합니다.

```text
DB rows
  + classifier result
  + validator result
  + OCR/vision/evidence validation logs
  -> ReportContext
  -> ReportAgent
  -> ReportDraft JSON
  -> UI edit
  -> DOCX export
```

운영 흐름에서는 Spring Backend가 FastAPI의 report agent 엔드포인트를 호출합니다. FastAPI는 기존 DB 테이블에서 `ReportContext`를 조립한 뒤 `ReportAgent`를 실행하고, 결과 `ReportDraft` JSON을 반환합니다. CLI는 로컬 샘플 실행과 디버깅 용도로만 사용합니다.

## 코드 구성과 책임

이 모듈은 역할별로 파일을 분리합니다. 보고서 판단과 문장 작성, JSON 입출력 책임을 명확히 나누는 것이 원칙입니다.

| 파일                                           | 역할                                                                                                                                                                                               |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schemas.py`                                 | 보고서 agent가 주고받는 데이터 구조를 정의합니다.`ReportContext`는 입력, `ReportDraft`는 화면 편집 및 렌더링용 출력입니다. classifier/validator 결과와 법령 citation 구조도 여기서 정의합니다. |
| `context_builder.py`                         | DB row를 `ReportContext`로 조립하는 경계 예시입니다. 실제 DB 구현은 하지 않고, FastAPI나 worker가 구현해야 할 repository 인터페이스를 정의합니다.                                                |
| `agent.py`                                   | 보고서 초안을 만드는 핵심 파일입니다. 입력 데이터를 표, 이슈, 보완사항 구조로 바꾸고 classifier/validator 결과를 반영합니다. LLM 결과가 있으면 허용된 문장 필드만 병합합니다.                      |
| `cli.py`                                     | `ReportContext` JSON 파일을 읽어 `ReportDraft` JSON 보고서 파일을 생성하는 로컬 실행 진입점입니다.                                                                                             |
| `llm.py`                                     | OpenAI API 호출 어댑터입니다. LLM에게 전체 보고서를 맡기지 않고 결론, 종합 의견, 필요 조치 같은 문장 필드만 JSON 패치로 요청합니다.                                                                |
| `src/api/routers/report_agent.py`            | Spring Backend가 호출하는 `POST /api/v1/agents/report/run` FastAPI 엔드포인트입니다.                                                                                                             |
| `src/repositories/report_repository.py`      | 기존 Postgres 테이블에서 보고서 입력 row를 읽어 `ReportContext` 조립에 사용할 저장소 구현입니다.                                                                                                 |
| `src/prompts/report_agent_system.md`         | LLM의 기본 역할과 금지사항을 정의합니다. 근거 없는 법령 생성, 금액/판정 변경을 금지합니다.                                                                                                         |
| `src/prompts/report_agent_draft_template.md` | LLM에게 넘기는 작업 프롬프트입니다. 어떤 JSON 필드만 반환해야 하는지 정의합니다.                                                                                                                   |
| `templates/report_template.json`             | 웹 화면과 DOCX 추출기가 같은 보고서 형식을 재현할 수 있도록 섹션, 표, 반복 행 구조를 정의하는 구조 기반 JSON 템플릿입니다.                                                                         |
| `examples/report_agent/sample_input.json`    | 최소 샘플 `ReportContext`입니다. agent와 JSON 생성 CLI 동작 확인에 사용합니다.                                                                                                                   |

전체 흐름은 다음과 같습니다.

```text
schemas.py
  데이터 구조 정의

context_builder.py
  DB row -> ReportContext

repositories/report_repository.py
  Postgres -> ReportContext input rows

agent.py
  ReportContext -> ReportDraft

llm.py
  ReportDraft의 문장 필드만 보강

api/routers/report_agent.py
  HTTP request -> ReportDraft JSON response

cli.py
  ReportContext JSON -> ReportDraft JSON 파일 (로컬 실행용)
```

## API 실행 흐름

보고서 생성은 Spring Backend가 FastAPI를 호출하는 방식으로 실행합니다.

```text
Frontend
  POST /projects/{projectId}/agents/report/run

Spring Backend
  권한 확인
  runId 생성
  FastAPI 호출

FastAPI
  POST /api/v1/agents/report/run
  DB row -> ReportContext
  ReportAgent.generate(context)
  agent_logs 기록
  ReportDraft JSON 반환

Frontend
  response.result.reportDraft를 편집 화면에 표시
  /api/report-docx로 DOCX 추출
```

FastAPI 엔드포인트:

```http
POST /api/v1/agents/report/run
```

요청 예시:

```json
{
  "run_id": "11111111-1111-1111-1111-111111111111",
  "project_id": 1,
  "usage_statement_id": 2008,
  "report_written_date": "2026-05-22",
  "report_period_label": "2026년 05월",
  "reviewer": {
    "name": "홍길동",
    "department": "안전관리팀",
    "title": "담당자"
  }
}
```

응답 예시:

```json
{
  "run_id": "11111111-1111-1111-1111-111111111111",
  "agent_type": "report",
  "status": "completed",
  "log_ids": [123],
  "result": {
    "reportDraft": {
      "layout_version": "safety_cost_report_v1",
      "report_no": "AR-20260522-1-2008",
      "report_sections": []
    }
  }
}
```

`context` 필드를 직접 전달하면 DB 조회 없이 `ReportContext`를 그대로 사용합니다. 이 경로는 테스트와 디버깅용입니다.

## 입력

`ReportContext`는 다음 데이터가 조합된 입력입니다.

- 프로젝트 기본정보: `projects`
- 사용내역서: `usage_statements`
- 집행 요약: `usage_statement_summaries`
- 집행 항목: `usage_statement_items`
- 증빙 파일: `files`, `evidence_file_links`
- 증빙 요구 충족 여부: `evidence_requirements`
- OCR/vision/증빙 검증 결과: `validation_logs`
- 조치 요청: `action_requests`
- classifier 결과: `classification_result`
- 법령 validator 결과: `legal_validation_result`

생성된 report 실행 로그는 기존 `agent_logs`에 `agent_type_code = 'report'`로 기록합니다.

집행 항목별 classifier/validator 결과는 `UsageStatementItemContext`에 붙입니다.

```python
UsageStatementItemContext(
    ...,
    classification_result=ClassificationResultContext(...),
    legal_validation_result=LegalValidationResultContext(...),
)
```

`legal_validation_result.citations`는 보고서의 법령 근거와 감사 추적용 citation으로 보존됩니다.

## 판정 기준

보고서 agent는 새로운 법령 판단을 하지 않습니다.

- 카테고리 유지/변경/검토필요: classifier 결과 사용
- 적정/검토필요/부적정: validator 결과 또는 `validation_logs.result_code` 사용
- 법령 근거: `legal_validation_result.citations` 또는 `validation_logs.details`에 있는 값만 사용
- 금액과 건수: `ReportContext`의 입력값을 기준으로 집계

입력에 없는 영수증 번호, 파일명, 법령 조항, 금액 차이는 생성하지 않습니다.

## LLM 사용

`ReportAgent()`는 항상 LLM을 호출합니다. 총평(`overall_opinion`)은 반드시 LLM 응답으로 작성되어야 하므로 `OPENAI_API_KEY`가 없거나 LLM 호출이 실패하면 보고서 생성을 실패시킵니다. 감사 보고서 본문에 들어갈 수 있도록 총평은 최소 350자 이상이어야 합니다.

```python
from src.agents.report_agent.agent import ReportAgent

draft = ReportAgent().generate(context)
```

기본 모델은 `gpt-5.2-mini`이며, `OPENAI_REPORT_MODEL` 환경변수로 바꿀 수 있습니다.

LLM이 수정할 수 있는 필드는 문장 필드로 제한됩니다.

- `conclusion`
- `overall_opinion`
- `issue_details[].agent_conclusion`
- `issue_details[].required_action`
- `supplement_actions[].action`

LLM은 다음 필드를 바꿀 수 없습니다.

- 금액
- 건수
- 판정
- 카테고리
- 법령 근거
- citation
- 담당자
- 완료 기한

## 출력

`ReportDraft`는 화면 편집, 저장, API 응답에 사용하는 구조화 JSON 보고서입니다.

`report_sections`는 `templates/report_template.json`을 `ReportDraft` 값으로 채운 결과입니다. 웹 화면과 DOCX 추출기는 이 필드를 기준으로 표지, 기본 정보, 집행 요약, 상세 내역, 종합 의견을 같은 순서와 제목으로 렌더링해야 합니다.

템플릿은 절대 좌표 대신 문서 구조를 저장합니다.

- `sections`: 보고서 섹션 순서, 제목, 종류
- `tables`: 섹션 내부 표 구조
- `headers`, `rows`: 고정 표 머리글과 행
- `repeat_for`, `row_template`: 배열 필드를 반복 렌더링하는 행 템플릿
- `repeat_mode: tables`: 상세 이슈처럼 항목마다 별도 표가 필요한 영역
- `{{draft.site_name}}`, `{{item.amount|money}}`: 값 바인딩과 표시 필터

주요 섹션:

- 표지/기본 정보
- 집행 금액 요약
- 집행 항목 분류별 요약
- 증빙 유형별 검증 현황
- 세금 및 정산 결과
- 항목별 적정성 검토 결과
- 부적정/검토 필요 상세 내역
- 보완 필요 사항
- 종합 의견
- `report_sections`: 실제 보고서 형식 렌더링용 섹션/표/문단 구조

프론트엔드에서는 `ReportDraft`를 편집 대상으로 두는 것이 권장됩니다. 사용자가 수정한 draft도 같은 JSON 구조로 저장합니다.

## 로컬 JSON 보고서 생성

CLI는 운영 호출 경로가 아니라 샘플 입력으로 agent 동작을 확인하는 로컬 실행 도구입니다. 샘플 입력으로 JSON 보고서를 생성하려면 다음 명령을 사용합니다.

```bash
uv run python -m src.agents.report_agent.cli \
  examples/report_agent/sample_input.json
```

테스트로 생성한 보고서 JSON은 기본적으로 `examples/report_agent/sample_report_llm_output.json`에 저장됩니다. 출력 파일명만 지정하면 `examples/report_agent/<파일명>`으로 저장됩니다.

실행 환경에는 `OPENAI_API_KEY`가 반드시 있어야 합니다. `.env`를 사용하는 로컬 환경에서는 먼저 `set -a; source .env; set +a`로 환경변수를 로드한 뒤 실행합니다.

## 증빙 유형 집계

증빙 유형별 검증 현황은 `evidence_type_code` 기준으로 집계합니다.

단, `other` 또는 `other_document`는 세부 서류명으로 분리합니다.

예:

```text
기타 서류(지급대장)
기타 서류(건강검진 계약서)
```

세부명은 `EvidenceFileContext.evidence_detail_name`을 우선 사용하고, 없으면 파일명에서 `지급대장`, `건강검진 계약서`, `계약서`, `선임계`, `설치확인서` 등을 추출합니다.

## 운영 원칙

- report agent는 판단 agent가 아닙니다.
- classifier/validator/OCR/vision 결과를 사실로 받아 보고서 구조와 문장을 생성합니다.
- LLM은 보고서 문장 보강에만 사용합니다.
- 최종 확정은 SHE 담당자 또는 결재자가 수행합니다.
- 불확실하거나 입력이 부족한 경우 `needs_human_review`에 남깁니다.
