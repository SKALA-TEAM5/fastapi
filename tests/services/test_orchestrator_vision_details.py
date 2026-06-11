import json

from src.services.orchestrator_service import (
    _is_vision_allowed_file_context,
    _target_equipment_from_item_context,
    _vision_file_details,
    _vision_file_statuses,
)


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


def test_vision_file_details_use_file_level_success_result_code():
    body = {
        "status_code": "success",
        "result_code": "hil",
        "reason": "보완 필요",
        "details": {
            "results": [
                {
                    "file_id": 52,
                    "is_appropriate": True,
                    "message": "안전모가 1건 확인되었습니다.",
                    "result": {"is_appropriate": True, "detections": []},
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
    assert validation["result_code"] == "success"
    assert validation["is_appropriate"] is True


def test_vision_file_statuses_use_each_file_result():
    body = {
        "status_code": "success",
        "result_code": "hil",
        "details": {
            "results": [
                {"file_id": 52, "is_appropriate": True, "result": {"is_appropriate": True}},
                {"file_id": 53, "is_appropriate": None, "result": {"is_appropriate": None}},
            ],
        },
    }

    statuses = _vision_file_statuses(body=body, details=dict(body["details"]))

    assert statuses == {52: "success", 53: "fail"}


def test_vision_allows_only_linked_allowed_category_contexts_by_default():
    assert not _is_vision_allowed_file_context(None)
    assert not _is_vision_allowed_file_context({"category_code": "CAT_01"})
    assert _is_vision_allowed_file_context({"category_code": "CAT_02"})
    assert _is_vision_allowed_file_context({"category_code": "CAT_03"})


def test_target_equipment_mapping_is_category_specific():
    assert _target_equipment_from_item_context(
        {"category_code": "CAT_02", "usage_statement_item_name": "안전망 설치"}
    ) == "safety_net"
    assert _target_equipment_from_item_context(
        {"category_code": "CAT_02", "usage_statement_item_name": "안전모 구입"}
    ) is None
    assert _target_equipment_from_item_context(
        {"category_code": "CAT_03", "usage_statement_item_name": "안전띠 구입"}
    ) == "safety_belt"
    assert _target_equipment_from_item_context(
        {"category_code": "CAT_03", "usage_statement_item_name": "안전망 설치"}
    ) is None
