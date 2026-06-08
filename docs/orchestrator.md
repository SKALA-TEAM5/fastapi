# Orchestrator

## 목적

Orchestrator는 개별 Agent가 아니라, 서비스의 AI Review 업무 단계를 제어하는 실행 관리자다.

역할은 다음과 같다.

- 현재 DB 상태를 스캔한다.
- 실행 가능한 Agent를 결정한다.
- Agent 실행 전제 조건을 보장한다.
- `agent_logs`를 기준으로 화면 상태를 계산한다.
- 보완이 필요하면 HIL 상태와 TODO 생성을 연결한다.

Orchestrator는 장시간 실행되며 대기하지 않는다. 버튼 클릭이나 업로드 완료 같은 명확한 사용자 동작마다 실행되고, 진행 상태는 DB에 남긴다.

## API Prefix

FastAPI API prefix:

```text
/api/v1/orchestrator
```

Spring Backend가 프론트 요청을 받아 권한을 확인한 뒤 FastAPI Orchestrator API를 호출하는 구조를 권장한다.

## 필수 API

| Method   | FastAPI Path                                                                                | 역할                                                                             |
| -------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `POST` | `/api/v1/orchestrator/usage-statements/parse`                                             | 사용내역서 업로드 후 OCR/Parse와 `classi` 실행                                 |
| `POST` | `/api/v1/orchestrator/usage-statements/classify`                                          | 저장된 세부항목 수정 후 원본 파일 재파싱 없이 `classi` 재분류                  |
| `POST` | `/api/v1/orchestrator/usage-statements/evidence`                                          | 증빙 검증 버튼.`safety-doc`, 조건부 `link`, 조건부 `vision` 실행 대상 결정 |
| `POST` | `/api/v1/orchestrator/usage-statements/legal`                                             | SHE 담당자 법령 검토 실행 조건 확인 및 `legal` 실행 대상 결정                  |
| `POST` | `/api/v1/orchestrator/usage-statements/report`                                            | `legal` 성공 후 `report` 실행 대상 결정                                      |
| `GET`  | `/api/v1/orchestrator/projects/{project_id}/usage-statements/{usage_statement_id}/status` | 화면 상태, 버튼 활성화 조건, Agent 로그 요약 조회                                |
| `GET`  | `/api/v1/orchestrator/projects/{project_id}/dashboard`                                    | SHE 대시보드용 실행 상태, HIL Agent, 토큰 사용량 요약 조회                       |

## 서비스 흐름

### 1. 사용내역서 업로드

```text
사용내역서 업로드
-> Spring Backend 권한 확인
-> FastAPI Orchestrator parse 호출
-> OCR/Parse
-> classi 실행
-> 잘못 분류된 세부내역을 올바른 CAT_01~CAT_09로 이동
-> agent_logs에 classi 결과 저장
-> UI는 이동 내역이 있으면 팝업 표시 후 세부내역 탭으로 이동
```

classi 결과로 항목 이동이 발생하면 `agent_logs.details`는 아래 형태를 가져야 한다.

```json
{
  "event": "classification_updated",
  "summary": "세부내역 1건을 올바른 항목으로 이동했습니다.",
  "payload": {
    "changed_count": 1,
    "changes": [
      {
        "row_id": 1,
        "item_name": "안전모 구입",
        "before": { "category_code": "CAT_02" },
        "after": { "category_code": "CAT_03" },
        "reason": "품목명이 보호구에 해당합니다."
      }
    ]
  }
}
```

### 2. 증빙 검증

```text
프로젝트 담당자가 증빙 서류 업로드
-> "증빙 검증" 버튼 클릭
-> Orchestrator가 classi success/success 여부 확인
-> safety-doc 실행
-> 영수증 또는 세금계산서가 있으면 link 실행
-> 현장사진이 있으면 vision 실행
-> 보완 필요 시 result_code = hil
-> 보완 TODO와 빨간 배경 표시
```

조건:

- `classi`가 `status_code=success`, `result_code=success`가 아니면 후속 Agent 실행 금지
- `link`는 영수증 또는 세금계산서가 없으면 실행하지 않고 로그도 만들지 않는다.
- `vision`은 현장사진이 없으면 실행하지 않고 로그도 만들지 않는다.

### 3. 보완 후 재검증

```text
프로젝트 담당자가 보완 서류 업로드
-> "보완 완료 후 재검증" 버튼 클릭
-> Orchestrator가 현재 파일 상태 재스캔
-> safety-doc 재실행
-> 조건이 맞으면 link/vision 재실행
-> 모두 success면 SHE legal 실행 가능
```

보완이 며칠 걸릴 수 있으므로 Orchestrator는 대기하지 않는다. DB 상태만 남기고 종료한다.

### 4. SHE 법령 검토

```text
SHE 담당자가 legal 실행
-> legal Agent 실행
-> 보완 필요 시 result_code = hil
-> 프로젝트 담당자가 보완
-> SHE 담당자가 legal 재실행
```

### 5. 보고서 초안

```text
legal success/success 또는 success/hil
-> report Agent 실행
-> 보고서 초안 생성
-> SHE 담당자가 수정
```

legal이 `success/hil`인 경우에도 보고서 초안은 생성한다. 이때 report는 legal의 보완·검토 필요 항목을 보고서 상세 이슈의 조치 사항과 종합 의견에 반영한다.

## agent_logs 규칙

`agent_logs`는 Agent별 최신 상태를 저장하는 기준 테이블로 사용한다.

상태 코드:

```text
pending
running
success
fail
canceled
```

결과 코드:

```text
success
hil
fail
```

조합 규칙:

| status_code  | result_code | 의미                                |
| ------------ | ----------- | ----------------------------------- |
| `pending`  | `null`    | 실행 대기                           |
| `running`  | `null`    | 실행 중                             |
| `success`  | `success` | 실행 성공, 검증 적정                |
| `success`  | `hil`     | 실행 성공, 보완 또는 사람 검토 필요 |
| `fail`     | `fail`    | 실행 실패                           |
| `canceled` | `fail`    | 실행 취소                           |

## 코드 배치

README 구조에 맞춰 역할별로 배치한다.

```text
src/api/routers/orchestrator.py
src/schemas/orchestrator.py
src/services/orchestrator_service.py
src/repositories/orchestrator_repository.py
```

역할:

| 파일                                            | 역할                                                  |
| ----------------------------------------------- | ----------------------------------------------------- |
| `src/api/routers/orchestrator.py`             | FastAPI endpoint 정의                                 |
| `src/schemas/orchestrator.py`                 | Request/Response DTO                                  |
| `src/services/orchestrator_service.py`        | 업무 흐름 제어                                        |
| `src/repositories/orchestrator_repository.py` | DB 상태 조회,`agent_logs`, `action_requests` 저장 |

## 실제 Agent 연결 상태

| Agent | 연결 상태 | 설명 |
|---|---|---|
| `classi` | 연결됨 | 최초 업로드는 `parse_usage_statement()`로 OCR/Parse와 분류 실행, 세부항목 수정 후에는 저장된 DB 항목 기준 재분류 |
| `safety-doc` | 연결됨 | 사용내역서 세부항목별 `check_missing_evidence(item_id)` 실행 |
| `link` | 연결됨 | 영수증/거래명세표/세금계산서 파일이 있을 때 `run_link_pipeline()` 실행 |
| `vision` | 미구현 | 현장사진 파일이 있으면 실행 대상은 되지만, 실제 Vision Agent 구현체가 아직 없어 `fail/fail` 로그 기록 |
| `legal` | 연결됨 | 기존 validator agent를 실행하고 `agent_logs.details.payload.results[]`에 report가 읽을 항목별 법령 판정 저장 |
| `report` | 연결됨 | `ReportAgent`와 `build_report_context()`로 보고서 초안 생성 |

## 보완 TODO 연결

`safety-doc`, `link` 결과가 `result_code=hil`이면 Orchestrator가 `action_requests`에 item별 보완 TODO를 생성한다.

재검증 시에는 같은 Agent가 만든 기존 open/in_progress TODO를 먼저 닫고, 현재 검증 결과 기준으로 다시 생성한다.

```text
증빙 검증 실행
-> 기존 [safety-doc]/[link] TODO closed 처리
-> Agent 실행
-> result_code=hil이면 action_requests open 생성
-> result_code=success이면 새 TODO 생성 없음
```

증빙 검증 API는 TODO 요청자 기록을 위해 `requested_by_user_id`를 받을 수 있다.

```json
{
  "project_id": 1,
  "usage_statement_id": 20,
  "requested_by_user_id": 3
}
```

`requested_by_user_id`가 없으면 `agent_logs`는 기록하지만 `action_requests`는 생성하지 않는다.

## legal -> report 데이터 계약

`legal`은 실행 성공 시 `agent_logs`에 다음 형태로 결과를 남긴다.

```json
{
  "event": "legal_completed",
  "summary": "법령 검토 결과 보고서 반영 대상 2건",
  "payload": {
    "category_results": [],
    "results": [
      {
        "item_id": 123,
        "category_code": "CAT_02",
        "status": "검토필요",
        "reason": "판정 사유",
        "citations": [
          {
            "legal_basis": "제7조제1항제2호",
            "summary": "근거 요약"
          }
        ]
      }
    ]
  }
}
```

`report`는 `results[].item_id`를 우선 사용하고, 항목 ID가 없으면 `category_code` 기준 결과를 fallback으로 사용한다.

## 현재 TODO

- `vision` Agent 구현체 추가 후 Orchestrator 연결
