from pydantic import BaseModel, Field
from typing import Any, Literal, Optional


class Detection(BaseModel):
    class_id: int
    class_code: str
    equipment: Optional[str] = None
    label: Optional[str] = None
    is_wearing: Optional[bool] = None
    needs_review: bool = False
    box_color: str
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: list[float]


class EquipmentReview(BaseModel):
    equipment: str
    equipment_label: str
    status: str
    is_appropriate: Optional[bool]
    confidence: Optional[float]
    reason: str


class SafetyNetReview(BaseModel):
    model_name: str
    status: str
    is_appropriate: Optional[bool]
    confidence: Optional[float]
    reason: str
    raw_label: Optional[str] = None


class SourceDetectionRequest(BaseModel):
    source_uri: str
    source_id: Optional[str] = None


class PpeDetectionResponse(BaseModel):
    source_id: Optional[str] = None
    source_uri: Optional[str] = None
    source_image_url: Optional[str] = None
    model_name: str
    image_width: int
    image_height: int
    status: str
    is_appropriate: Optional[bool]
    message: str
    annotated_image_path: Optional[str] = None
    annotated_image_url: Optional[str] = None
    reviews: list[EquipmentReview]
    detections: list[Detection]


class SafetyNetDetectionResponse(BaseModel):
    source_id: Optional[str] = None
    source_uri: Optional[str] = None
    source_image_url: Optional[str] = None
    model_name: str
    image_width: int
    image_height: int
    status: str
    is_appropriate: Optional[bool]
    message: str
    annotated_image_path: Optional[str] = None
    annotated_image_url: Optional[str] = None
    safety_net_review: SafetyNetReview


class DetectionResponse(BaseModel):
    source_id: Optional[str] = None
    source_uri: Optional[str] = None
    source_image_url: Optional[str] = None
    model_name: str
    image_width: int
    image_height: int
    overall_status: str
    is_appropriate: Optional[bool]
    message: str
    ppe_status: str
    ppe_is_appropriate: Optional[bool]
    ppe_message: str
    safety_net_review: SafetyNetReview
    annotated_image_path: Optional[str] = None
    annotated_image_url: Optional[str] = None
    reviews: list[EquipmentReview]
    detections: list[Detection]


class VisionReviewPhoto(BaseModel):
    file_id: int
    original_filename: str
    storage_key: str
    evidence_type_code: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    presigned_url: str
    usage_statement_item_id: Optional[int] = None
    usage_statement_item_name: Optional[str] = None
    category_code: Optional[str] = None
    category_name: Optional[str] = None
    target_equipment: Optional[str] = None


class VisionReviewRequest(BaseModel):
    project_id: int
    usage_statement_id: int
    photos: list[VisionReviewPhoto]


class VisionReviewTodo(BaseModel):
    file_id: int
    reason: str
    usage_statement_item_id: Optional[int] = None
    category_code: Optional[str] = None
    category_name: Optional[str] = None
    usage_statement_item_name: Optional[str] = None
    target_equipment: Optional[str] = None


class VisionReviewResponse(BaseModel):
    status_code: Literal["success", "fail"]
    result_code: Literal["success", "hil", "fail"]
    reason: str
    token: int = 0
    model_name: str
    todos: list[VisionReviewTodo]
    details: dict[str, Any]
