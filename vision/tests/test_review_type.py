from src.main import _review_type
from src.schemas.vision import VisionReviewPhoto


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
