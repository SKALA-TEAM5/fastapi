from datetime import datetime

from src.repositories.report_repository import _group_agent_logs_by_item


def test_safety_doc_agent_log_uses_item_level_evidence_status():
    grouped = _group_agent_logs_by_item(
        [
            {
                "id": 4,
                "agent_type_code": "safety-doc",
                "result_code": "hil",
                "reason": "필수 증빙 누락 항목 1건",
                "model_name": "gpt-4.1-mini",
                "created_at": datetime(2026, 6, 13, 20, 33, 6),
                "details": {
                    "event": "safety_doc_completed",
                    "payload": {
                        "todos": [
                            {
                                "usage_statement_item_id": 14,
                                "title": "보호구 착용 상태 사진",
                                "reason": "필수 증빙 누락: 보호구 착용 상태 사진",
                            }
                        ],
                        "item_results": [
                            {
                                "item_id": 13,
                                "result": {
                                    "evidence_status": {
                                        "status": "OK",
                                        "missing_evidences": [],
                                    }
                                },
                            },
                            {
                                "item_id": 14,
                                "result": {
                                    "evidence_status": {
                                        "status": "MISSING",
                                        "missing_evidences": ["wearing_photo"],
                                    }
                                },
                            },
                        ],
                    },
                },
            }
        ]
    )

    assert grouped[13][0]["result_code"] == "appropriate"
    assert grouped[14][0]["result_code"] == "needs_review"
    assert grouped[14][1]["result_code"] == "needs_review"
