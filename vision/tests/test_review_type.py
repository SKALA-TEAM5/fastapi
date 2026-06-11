from src.main import _review_type, _unsupported_target_result
from src.core.config import Settings
from src.schemas.vision import Detection, VisionReviewPhoto
from src.services.vision_detection_service import VisionDetectionService


def make_photo(evidence_type_code: str, filename: str = "photo.jpg") -> VisionReviewPhoto:
    return VisionReviewPhoto(
        file_id=1,
        original_filename=filename,
        storage_key=f"projects/1/{filename}",
        evidence_type_code=evidence_type_code,
        mime_type="image/jpeg",
        size_bytes=100,
        presigned_url="http://localhost:9000/safety-files/photo.jpg",
    )


def test_wearing_photo_uses_ppe_model():
    assert _review_type(make_photo("wearing_photo")) == "ppe"


def test_item_photo_uses_ppe_model():
    assert _review_type(make_photo("item_photo")) == "ppe"


def test_safety_net_filename_uses_safety_net_model():
    assert _review_type(make_photo("site_photo", "safety-net.jpg")) == "safety-net"


def test_unsupported_target_result_is_appropriate_without_detections():
    photo = make_photo("site_photo")

    result = _unsupported_target_result(photo)

    assert result["review_type"] == "unsupported"
    assert result["status"] == "appropriate"
    assert result["is_appropriate"] is True
    assert result["result"]["detections"] == []


def test_target_ppe_review_uses_only_target_equipment():
    service = VisionDetectionService(Settings())
    detections = [
        Detection(
            class_id=3,
            class_code="07",
            equipment="safety_helmet",
            label="안전모 착용",
            is_wearing=True,
            needs_review=False,
            box_color="blue",
            confidence=0.91,
            bbox_xyxy=[10, 20, 100, 200],
        ),
        Detection(
            class_id=2,
            class_code="05",
            equipment="safety_shoes",
            label="안전화 착용",
            is_wearing=True,
            needs_review=False,
            box_color="blue",
            confidence=0.93,
            bbox_xyxy=[30, 220, 120, 300],
        ),
    ]

    review = service._build_target_review("safety_helmet", [d for d in detections if d.equipment == "safety_helmet"])

    assert review.equipment == "safety_helmet"
    assert review.status == "detected"
    assert review.is_appropriate is True
    assert "안전모가 1건 확인" in review.reason


def test_target_ppe_review_treats_not_wearing_as_target_presence():
    service = VisionDetectionService(Settings())
    detections = [
        Detection(
            class_id=3,
            class_code="07",
            equipment="safety_helmet",
            label="안전모 착용",
            is_wearing=True,
            needs_review=False,
            box_color="blue",
            confidence=0.91,
            bbox_xyxy=[10, 20, 100, 200],
        ),
        Detection(
            class_id=4,
            class_code="08",
            equipment="safety_helmet",
            label="안전모 미착용",
            is_wearing=False,
            needs_review=False,
            box_color="red",
            confidence=0.88,
            bbox_xyxy=[110, 20, 200, 200],
        ),
    ]

    review = service._build_target_review("safety_helmet", detections)

    assert review.status == "detected"
    assert review.is_appropriate is True
    assert "안전모가 2건 확인" in review.reason
