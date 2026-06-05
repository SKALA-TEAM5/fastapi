import json


def test_vision_log_details_do_not_contain_circular_reference():
    body = {
        "status_code": "success",
        "result_code": "hil",
        "reason": "보완 필요",
        "todos": [{"file_id": 52, "reason": "검토 필요"}],
        "details": {
            "summary": "현장사진 검토 완료",
            "results": [{"file_id": 52, "status": "needs_review"}],
        },
    }

    source_details = body.get("details")
    details = dict(source_details) if isinstance(source_details, dict) else {}
    details["payload"] = dict(details.get("payload") or {})
    vision_response = {key: value for key, value in body.items() if key != "details"}
    details["payload"]["vision_response"] = vision_response

    encoded = json.dumps(details)

    assert "vision_response" in encoded
    assert "현장사진 검토 완료" in encoded
