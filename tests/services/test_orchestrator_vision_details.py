import json

from src.services import orchestrator_service as orchestrator
from src.services.orchestrator_service import (
    _is_vision_allowed_file_context,
    _run_vision_agent,
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

    encoded = json.dumps(details, ensure_ascii=False)

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


def test_run_vision_agent_posts_only_linked_allowed_photos(monkeypatch):
    calls: list[tuple[str, dict]] = []
    posted: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "status_code": "success",
                "result_code": "hil",
                "reason": "현장사진 검토 보완 필요",
                "todos": [{"file_id": 52, "reason": "안전모 미착용 확인"}],
                "usage": {"input_tokens": 7, "output_tokens": 3},
                "model_name": "vision-test",
                "details": {
                    "summary": "현장사진 검토 완료",
                    "results": [
                        {
                            "file_id": 52,
                            "is_appropriate": False,
                            "message": "안전모 미착용 확인",
                            "result": {"detections": [{"label": "안전모 미착용"}]},
                        }
                    ],
                },
            }

    monkeypatch.setattr(orchestrator, "VISION_AGENT_BASE_URL", "http://vision.local")
    monkeypatch.setattr(
        orchestrator,
        "list_evidence_files_by_type",
        lambda project_id, evidence_type_codes: [
            {
                "id": 52,
                "original_filename": "allowed.jpg",
                "storage_key": "photos/allowed.jpg",
                "uploaded_evidence_type_code": "site_photo",
                "mime_type": "image/jpeg",
                "size_bytes": 123,
            },
            {
                "id": 53,
                "original_filename": "ignored.jpg",
                "storage_key": "photos/ignored.jpg",
                "uploaded_evidence_type_code": "site_photo",
                "mime_type": "image/jpeg",
                "size_bytes": 456,
            },
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_evidence_file_todo_context_index",
        lambda usage_statement_id: {
            52: {"category_code": "CAT_02", "usage_statement_item_name": "안전모 구입"},
            53: {"category_code": "CAT_01", "usage_statement_item_name": "교육비"},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "create_presigned_file_url",
        lambda storage_key: f"https://minio.local/{storage_key}",
    )

    def fake_post(url, *, json, timeout):
        posted["url"] = url
        posted["json"] = json
        posted["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(orchestrator.requests, "post", fake_post)
    monkeypatch.setattr(orchestrator, "upsert_agent_log", lambda **kwargs: calls.append(("upsert", kwargs)))
    monkeypatch.setattr(
        orchestrator,
        "update_file_statuses_by_id",
        lambda **kwargs: calls.append(("statuses_by_id", kwargs)),
    )
    monkeypatch.setattr(
        orchestrator,
        "update_file_details",
        lambda **kwargs: calls.append(("details_by_id", kwargs)),
    )
    monkeypatch.setattr(orchestrator, "_record_agent_usage", lambda **kwargs: calls.append(("usage", kwargs)))

    result = _run_vision_agent(project_id=5, usage_statement_id=3, requested_by_user_id=9)

    assert result["status_code"] == "success"
    assert result["result_code"] == "hil"
    assert result["todos"][0]["category_code"] == "CAT_02"
    assert result["details"]["payload"]["photos"][0]["file_id"] == 52
    assert "details" not in result["details"]["payload"]["vision_response"]
    assert [photo["file_id"] for photo in posted["json"]["photos"]] == [52]
    assert posted["json"]["photos"][0]["presigned_url"] == "https://minio.local/photos/allowed.jpg"

    statuses_call = next(kwargs for name, kwargs in calls if name == "statuses_by_id")
    assert statuses_call["statuses_by_file_id"] == {52: "fail"}
    details_call = next(kwargs for name, kwargs in calls if name == "details_by_id")
    assert details_call["details_by_file_id"][52]["vision_validation"]["reason"] == "안전모 미착용 확인"
    completed_log = [kwargs for name, kwargs in calls if name == "upsert"][-1]
    assert completed_log["agent_type_code"] == "vision"
    assert completed_log["details"]["payload"]["todos"][0]["usage_statement_item_name"] == "안전모 구입"
    usage_call = next(kwargs for name, kwargs in calls if name == "usage")
    assert usage_call["token"] == 10
    assert usage_call["requested_by_user_id"] == 9


def test_run_vision_agent_config_missing_marks_target_photos_failed(monkeypatch):
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(orchestrator, "VISION_AGENT_BASE_URL", "")
    monkeypatch.setattr(
        orchestrator,
        "list_evidence_files_by_type",
        lambda project_id, evidence_type_codes: [{"id": 52, "storage_key": "photos/allowed.jpg"}],
    )
    monkeypatch.setattr(
        orchestrator,
        "_evidence_file_todo_context_index",
        lambda usage_statement_id: {52: {"category_code": "CAT_03", "usage_statement_item_name": "안전띠 구입"}},
    )
    monkeypatch.setattr(orchestrator, "create_presigned_file_url", lambda storage_key: "https://minio.local/photo")
    monkeypatch.setattr(orchestrator, "upsert_agent_log", lambda **kwargs: calls.append(("upsert", kwargs)))
    monkeypatch.setattr(orchestrator, "update_file_statuses", lambda **kwargs: calls.append(("statuses", kwargs)))

    result = _run_vision_agent(project_id=5, usage_statement_id=3)

    assert result["status_code"] == "fail"
    assert result["result_code"] == "fail"
    assert "VISION_AGENT_BASE_URL" in result["reason"]
    statuses_call = next(kwargs for name, kwargs in calls if name == "statuses")
    assert statuses_call["file_ids"] == [52]
    assert statuses_call["status_code"] == "fail"
    failed_log = [kwargs for name, kwargs in calls if name == "upsert"][-1]
    assert failed_log["status_code"] == "fail"
    assert failed_log["details"]["event"] == "agent_config_missing"
