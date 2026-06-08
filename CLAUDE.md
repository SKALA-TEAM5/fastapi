SCHEMA는 확정이기 때문에, 모든 코드를 수정할 때도 Legal RDB(PostGreSQL)의 SCHEMA는 건드리지 말기

# ⚠️ Legal Agent 실제 작업 위치

- Legal agent 관련 실제 소스는 /Users/ss19801/Documents/Projects/final/rag 에 있음
- 이 fastapi 프로젝트의 legal 관련 파일은 건드리지 말 것
- Legal agent 작업은 rag 프로젝트의 CLAUDE.md 기준으로 진행

# 작업 범위 제한 (현재 세션)

- API 엔드포인트 변경은 나중에 별도로 진행 — 지금은 손대지 말 것
- legal_db(PostgreSQL) 변경 없이 작업할 것 — exporter/repository 코드와 payload JSON만 수정

# 파일 범위 제한

- /Users/ss19801/Documents/Projects/final/rag 에 있는 모든 파일이 내가 구현한 파일임
- 이 파일들이 fastapi에 merge되어 있으므로, fastapi에서 아래 경로의 파일들만 건드릴 것:
  - src/agents/classifier_agent/
  - src/agents/validator_agent/
  - src/core/
  - src/prompts/shared_prompt.py, src/prompts/validator_prompt.py
  - src/repositories/legal_rules_repository.py
  - src/repositories/legal_rules_exporter.py
  - src/schemas/classifier.py, src/schemas/shared.py, src/schemas/validator.py
  - src/services/ingestion/
  - src/services/ingestion_service.py
  - src/services/validator_service.py
  - src/services/refresh/
  - scripts/seed_legal_rule_profiles.json
  - artifacts/legal_rules_payload.json
- 위 목록 외 파일(OCR, 매칭 서비스, 세금계산서, API 라우터 등)은 절대 건드리지 말 것

앞으로 rag쪽의 파일은 안건드려도 됩니다.

# 상태 코드 정리 (2026-05-22 기준)

프로젝트 (projects.project_status_code)
active : 진행 중
completed  : 완료
suspended : 중단

사용내역서 (usage_statements.status_code)
draft : 작성 중
upload_completed : 제출 완료
supplement_required : 보완요청
review_completed : 검토 완료

파일 (files.status_code)
draft : 업로드 직후, agent 미처리
success  : agent 처리 완료, 이상 없음
fail : agent 처리 완료, 문제 있음

에이전트 (agent_logs.status_code / result_code)
status_code 실행 상태
pending :  대기
running :  실행 중
success :  실행 완료
fail       :  실행 실패
canceled : 취소

result_code 판단 결과
success : 이상 없음
hil : 이슈 발견, 사람 확인 필요
fail : 실패

agent_type_code 에이전트 종류
orchestrator : 오케스트레이터
classi : 카테고리 분류 검증
safety-doc : 필수 서류 판단
link : 금액·날짜 비교
vision : 현장사진 안전시설 확인
legal : 법령 기준 검증
report : 보고서 생성
