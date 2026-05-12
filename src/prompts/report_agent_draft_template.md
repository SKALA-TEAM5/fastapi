ReportContext와 LLM 없이 먼저 만든 ReportDraft가 제공됩니다.

다음 JSON만 반환하세요.

```json
{
  "conclusion": "string",
  "overall_opinion": "string",
  "issue_details": [
    {
      "no": 1,
      "agent_conclusion": "string",
      "required_action": "string"
    }
  ],
  "supplement_actions": [
    {
      "no": 1,
      "action": "string"
    }
  ]
}
```

작성 기준:
- 한국어 공문/감사 보고서 문체를 사용합니다.
- 금액, 건수, 판정은 입력 ReportDraft의 값을 유지합니다.
- 법령 근거는 입력에 있는 경우에만 언급합니다.
- `issue_details[].no`와 `supplement_actions[].no`는 입력 ReportDraft에 있는 번호만 사용합니다.
- `legal_basis`, `legal_citations`, 금액, 판정, 카테고리, 담당자, 기한은 변경하지 않습니다.
- `required_action`은 입력의 `required_action_fact` 또는 기존 `required_action`을 바탕으로 문장만 다듬습니다.
- 담당자 최종 확인이 필요하다는 문장을 유지합니다.

`overall_opinion` 작성 기준:
- 짧은 요약문이 아니라 실제 감사·검토 보고서의 종합 의견 본문으로 작성합니다.
- 350자 이상 700자 이하의 하나의 완성된 문단으로 작성합니다.
- 다음 내용을 모두 포함합니다.
  1. 검토 목적과 검토 범위
  2. 검토에 사용한 기준과 근거 자료의 성격
  3. 주요 확인 결과와 증빙상 미비점
  4. 해당 미비점이 정산·사후 감사에 미칠 수 있는 영향
  5. 보완 제출, 정정, 담당자 최종 확인 필요성
- 입력에 없는 법령명, 조항, 영수증 번호, 세금계산서 번호, 파일명은 만들지 않습니다.
- `ReportDraft`의 금액, 건수, 판정, 항목명, 검토 대상 기간은 그대로 사용합니다.
