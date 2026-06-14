# AI Agent Observability

FastAPI의 `/metrics` 엔드포인트를 Prometheus가 수집하고 Grafana에서 Agent별 실행 상태를 조회합니다.

## 공통 Agent 지표

`classi`, `safety-doc`, `link`, `vision`, `legal`, `report`, `chatbot`을 대상으로 합니다.
Chatbot은 SSE 스트리밍 시작부터 완료까지의 시간을 측정하고 중복 요청은 `skipped`로 기록합니다.

- `ai_agent_runs_total{agent,result}`: Agent별 실행 결과
- `ai_agent_run_duration_seconds{agent}`: Agent별 실행 시간
- `ai_agent_runs_in_progress{agent}`: 현재 실행 중인 Agent 수
- `ai_agent_review_items_total{agent}`: 생성된 HIL/TODO 항목 수
- `ai_agent_tokens_total{agent,model,type}`: 모델별 토큰 사용량

`result`는 `success`, `hil`, `fail`, `skipped`, `canceled`, `unknown` 중 하나입니다.
라벨에는 프로젝트 ID, 사용자 ID, 파일명 같은 고카디널리티 값이나 개인정보를 넣지 않습니다.

Safety Doc 내부 추론과 참고자료 검색 상태는 `docs/safety-doc-agent.md`의 전용 지표를 함께 사용합니다.

## Prometheus 수집 예시

```yaml
scrape_configs:
  - job_name: fastapi-ai-workspace
    metrics_path: /metrics
    static_configs:
      - targets: ["team5-fastapi:8001"]
```

## Grafana 주요 쿼리

Agent별 분당 실행 수:

```promql
sum by (agent, result) (rate(ai_agent_runs_total[5m]))
```

Agent별 P95 실행 시간:

```promql
histogram_quantile(
  0.95,
  sum by (le, agent) (rate(ai_agent_run_duration_seconds_bucket[5m]))
)
```

Agent별 실패율:

```promql
sum by (agent) (rate(ai_agent_runs_total{result="fail"}[10m]))
/
clamp_min(sum by (agent) (rate(ai_agent_runs_total[10m])), 0.001)
```

Agent별 HIL 발생량:

```promql
sum by (agent) (increase(ai_agent_runs_total{result="hil"}[1h]))
```

Agent별 토큰 사용량:

```promql
sum by (agent, type) (increase(ai_agent_tokens_total[1h]))
```

## 권장 알림

- Agent 실패율이 10분 동안 20% 초과
- Agent P95 실행 시간이 10분 동안 120초 초과
- `ai_agent_runs_in_progress`가 15분 이상 1 이상 유지
- Vision 또는 Legal Agent가 15분 동안 연속 실패
- Safety Doc 참고자료 검색 실패가 10분 동안 3회 이상 발생

Grafana 대시보드는 `monitoring/grafana/dashboards/ai-agent-overview.json`을 import해 사용할 수 있습니다.
Prometheus 알림 규칙은 `monitoring/prometheus/ai-agent-alerts.yml`을 rule file로 등록합니다.
