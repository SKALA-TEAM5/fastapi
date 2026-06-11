import json

from src.services.orchestrator_service import _vision_file_details


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


def test_vision_file_details_reads_body_details_results():
    body = {
        "status_code": "success",
        "result_code": "hil",
        "reason": "보완 필요",
        "details": {
            "summary": "현장사진 검토 완료",
            "results": [
                {
                    "file_id": 52,
                    "original_filename": "site-photo.jpg",
                    "is_appropriate": False,
                    "message": "안전모 미착용이 1건 확인되었습니다.",
                    "result": {
                        "image_width": 1280,
                        "image_height": 720,
                        "detections": [
                            {
                                "label": "안전모 미착용",
                                "confidence": 0.88,
                                "bbox_xyxy": [120, 80, 300, 260],
                                "equipment": "safety_helmet",
                                "is_wearing": False,
                            }
                        ],
                    },
                }
            ],
        },
    }
    details = dict(body["details"])

    file_details = _vision_file_details(
        body=body,
        details=details,
        photos=[{"file_id": 52, "original_filename": "site-photo.jpg"}],
        usage_statement_id=1,
        reason="보완 필요",
        result_code="hil",
    )

    validation = file_details[52]["vision_validation"]
    assert validation["usage_statement_id"] == 1
    assert validation["result_code"] == "hil"
    assert validation["reason"] == "안전모 미착용이 1건 확인되었습니다."
    assert validation["image_width"] == 1280
    assert validation["image_height"] == 720
    assert validation["detections"][0]["bbox_xyxy"] == [120, 80, 300, 260]
