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
