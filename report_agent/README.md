# Report Agent

산업안전보건관리비 증빙 검토 결과를 보고서 초안으로 만드는 agent 모듈입니다. 이 모듈의 산출물은 화면에서 편집 가능한 `ReportDraft` JSON이며, DOCX/PDF 파일은 별도 renderer가 생성합니다.

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
  -> DOCX/PDF renderer
```

이 agent는 DB를 직접 조회하지 않습니다. FastAPI 또는 worker가 DB와 다른 agent 결과를 모아 `ReportContext`를 만든 뒤 호출해야 합니다.

## 코드 구성과 책임

이 모듈은 역할별로 파일을 분리합니다. 보고서 판단, 문장 작성, DOCX 렌더링 책임을 한 파일에 섞지 않는 것이 원칙입니다.

| 파일 | 역할 |
| --- | --- |
| `schemas.py` | 보고서 agent가 주고받는 데이터 구조를 정의합니다. `ReportContext`는 입력, `ReportDraft`는 화면 편집 및 렌더링용 출력입니다. classifier/validator 결과와 법령 citation 구조도 여기서 정의합니다. |
| `context_builder.py` | DB row를 `ReportContext`로 조립하는 경계 예시입니다. 실제 DB 구현은 하지 않고, FastAPI나 worker가 구현해야 할 repository 인터페이스를 정의합니다. |
| `agent.py` | 보고서 초안을 만드는 핵심 파일입니다. 입력 데이터를 표, 이슈, 보완사항 구조로 바꾸고 classifier/validator 결과를 반영합니다. LLM 결과가 있으면 허용된 문장 필드만 병합합니다. |
| `llm.py` | OpenAI API 호출 어댑터입니다. LLM에게 전체 보고서를 맡기지 않고 결론, 종합 의견, 필요 조치 같은 문장 필드만 JSON 패치로 요청합니다. |
| `renderer.py` | `ReportDraft`를 DOCX 파일로 렌더링합니다. v2 샘플 보고서의 11개 표와 종합 의견 문단을 채우고, 6번 상세 내역 표를 이슈 개수에 맞게 늘리거나 줄입니다. |
| `prompts/system.md` | LLM의 기본 역할과 금지사항을 정의합니다. 근거 없는 법령 생성, 금액/판정 변경을 금지합니다. |
| `prompts/report_draft.md` | LLM에게 넘기는 작업 프롬프트입니다. 어떤 JSON 필드만 반환해야 하는지 정의합니다. |
| `template_mapping.md` | v2 DOCX 샘플의 표와 본문이 `ReportDraft`의 어떤 필드와 연결되는지 정리한 유지보수용 문서입니다. 실행 코드는 아닙니다. |
| `examples/sample_input.json` | 최소 샘플 `ReportContext`입니다. agent와 renderer 동작 확인에 사용합니다. |

전체 흐름은 다음과 같습니다.

```text
schemas.py
  데이터 구조 정의

context_builder.py
  DB row -> ReportContext

agent.py
  ReportContext -> ReportDraft

llm.py
  ReportDraft의 문장 필드만 보강

renderer.py
  ReportDraft -> DOCX
```

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

`ReportAgent()`는 기본적으로 `OPENAI_API_KEY`가 있으면 LLM을 호출합니다. API 키가 없거나 호출이 실패하면 LLM 없이 만든 초안을 반환하고, 실패 사유를 `needs_human_review`에 남깁니다.

```python
from agent.report_agent.agent import ReportAgent

draft = ReportAgent().generate(context)
```

LLM을 끄려면 다음처럼 호출합니다.

```python
draft = ReportAgent(use_default_llm=False).generate(context)
```

기본 모델은 `gpt-5.2`이며, `OPENAI_REPORT_MODEL` 환경변수로 바꿀 수 있습니다.

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

`ReportDraft`는 화면 편집과 렌더링을 위한 구조화 JSON입니다.

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

프론트엔드에서는 `ReportDraft`를 편집 대상으로 두는 것이 권장됩니다. 사용자가 수정한 draft를 저장한 뒤 renderer에 넘겨 DOCX/PDF를 생성합니다.

## 증빙 유형 집계

증빙 유형별 검증 현황은 `evidence_type_code` 기준으로 집계합니다.

단, `other` 또는 `other_document`는 세부 서류명으로 분리합니다.

예:

```text
기타 서류(지급대장)
기타 서류(건강검진 계약서)
```

세부명은 `EvidenceFileContext.evidence_detail_name`을 우선 사용하고, 없으면 파일명에서 `지급대장`, `건강검진 계약서`, `계약서`, `선임계`, `설치확인서` 등을 추출합니다.

## DOCX 렌더링

`renderer.py`는 `ReportDraft`를 DOCX로 렌더링합니다.

렌더러 동작:

- 표 목록 행은 데이터 개수에 맞춰 늘리거나 줄임
- 6번 상세 내역은 이슈 수에 맞춰 `6.1`, `6.2`, `6.3`... 블록과 표를 늘리거나 줄임
- 추가 행은 기존 행 스타일을 복제해 표 음영/테두리 유지
- 샘플 파일이 없으면 기본 12표 DOCX를 생성

## FastAPI 연결 예시

권장 API 흐름:

```text
POST /reports/drafts
GET /reports/drafts/{draft_id}
PATCH /reports/drafts/{draft_id}
POST /reports/drafts/{draft_id}/render/docx
POST /reports/drafts/{draft_id}/render/pdf
```

`POST /reports/drafts`에서 할 일:

1. DB에서 프로젝트/사용내역서/항목/파일/검증 로그 조회
2. classifier/validator 결과를 항목별로 붙임
3. `ReportContext` 생성
4. `ReportAgent().generate(context)` 호출
5. `ReportDraft` 저장 및 반환

`PATCH /reports/drafts/{draft_id}`에서는 사용자가 화면에서 수정한 문장/표 데이터를 저장합니다. 원본 validator 결과와 citation은 별도 보존하는 것이 좋습니다.

## 운영 원칙

- report agent는 판단 agent가 아닙니다.
- classifier/validator/OCR/vision 결과를 사실로 받아 보고서 구조와 문장을 생성합니다.
- LLM은 보고서 문장 보강에만 사용합니다.
- 최종 확정은 SHE 담당자 또는 결재자가 수행합니다.
- 불확실하거나 입력이 부족한 경우 `needs_human_review`에 남깁니다.
