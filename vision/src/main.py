import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from src.core.config import settings
from src.schemas.vision import (
    DetectionResponse,
    PpeDetectionResponse,
    SafetyNetDetectionResponse,
    SourceDetectionRequest,
    VisionReviewPhoto,
    VisionReviewRequest,
    VisionReviewResponse,
    VisionReviewTodo,
)
from src.services.vision_detection_service import VisionDetectionService
from src.vision.annotation import save_annotated_image
from src.vision.image_loader import load_rgb_image, load_rgb_image_from_uri


app = FastAPI(title="Safety Vision API", version="0.1.0")
vision_service = VisionDetectionService(settings)
settings.input_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/vision-results", StaticFiles(directory=str(settings.output_dir)), name="vision-results")
app.mount("/vision-files", StaticFiles(directory=str(settings.input_dir)), name="vision-files")

PPE_TARGET_EQUIPMENT = {"safety_helmet", "safety_shoes", "safety_belt"}
SAFETY_NET_TARGET_EQUIPMENT = "safety_net"
TARGET_EQUIPMENT_LABELS = {
    "safety_helmet": "안전모",
    "safety_shoes": "안전화",
    "safety_belt": "안전벨트",
    "safety_net": "안전망",
}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_name": settings.model_name,
        "model_path": str(settings.model_path),
        "model_exists": settings.model_path.exists(),
        "model_is_file": settings.model_path.is_file(),
        "safety_net_model_name": settings.safety_net_model_name,
        "safety_net_model_path": str(settings.safety_net_model_path),
        "safety_net_model_exists": settings.safety_net_model_path.exists(),
        "safety_net_model_is_file": settings.safety_net_model_path.is_file(),
        "device": settings.model_device,
        "public_base_url": settings.public_base_url or None,
    }


@app.get("/labels")
def labels() -> dict[int, str]:
    return settings.class_names


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)) -> DetectionResponse:
    image = await _load_upload_image(file)

    try:
        response = vision_service.detect(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    output_path = _upload_annotated_output_path(file.filename, "combined")
    save_annotated_image(
        image,
        response.detections,
        output_path,
        response.safety_net_review,
    )
    response.annotated_image_path = str(output_path)
    response.annotated_image_url = _annotated_image_url(output_path)
    return response


@app.post("/detect/source", response_model=DetectionResponse)
async def detect_source(request: SourceDetectionRequest) -> DetectionResponse:
    image = _load_source_image(request.source_uri)

    try:
        response = vision_service.detect(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response.source_id = request.source_id
    response.source_uri = request.source_uri
    response.source_image_url = _source_image_url(request.source_uri)
    output_path = _annotated_output_path(request, "combined")
    save_annotated_image(
        image,
        response.detections,
        output_path,
        response.safety_net_review,
    )
    response.annotated_image_path = str(output_path)
    response.annotated_image_url = _annotated_image_url(output_path)
    return response


@app.post("/detect/ppe", response_model=PpeDetectionResponse)
async def detect_ppe(file: UploadFile = File(...)) -> PpeDetectionResponse:
    image = await _load_upload_image(file)

    try:
        response = vision_service.detect_ppe(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return response


@app.post("/detect/ppe/source", response_model=PpeDetectionResponse)
async def detect_ppe_source(request: SourceDetectionRequest) -> PpeDetectionResponse:
    image = _load_source_image(request.source_uri)

    try:
        response = vision_service.detect_ppe(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response.source_id = request.source_id
    response.source_uri = request.source_uri
    response.source_image_url = _source_image_url(request.source_uri)
    return response


@app.post("/detect/safety-net", response_model=SafetyNetDetectionResponse)
async def detect_safety_net(file: UploadFile = File(...)) -> SafetyNetDetectionResponse:
    image = await _load_upload_image(file)

    try:
        response = vision_service.detect_safety_net(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    output_path = _upload_annotated_output_path(file.filename, "safety-net")
    save_annotated_image(image, [], output_path, response.safety_net_review)
    response.annotated_image_path = str(output_path)
    response.annotated_image_url = _annotated_image_url(output_path)
    return response


@app.post("/detect/safety-net/source", response_model=SafetyNetDetectionResponse)
async def detect_safety_net_source(request: SourceDetectionRequest) -> SafetyNetDetectionResponse:
    image = _load_source_image(request.source_uri)

    try:
        response = vision_service.detect_safety_net(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response.source_id = request.source_id
    response.source_uri = request.source_uri
    response.source_image_url = _source_image_url(request.source_uri)
    output_path = _annotated_output_path(request, "safety-net")
    save_annotated_image(image, [], output_path, response.safety_net_review)
    response.annotated_image_path = str(output_path)
    response.annotated_image_url = _annotated_image_url(output_path)
    return response


@app.post(
    "/vision/review",
    response_model=VisionReviewResponse,
    summary="Orchestrator 전용 현장사진 검토",
    description=(
        "FastAPI Orchestrator가 여러 현장사진의 presigned URL을 전달하면 "
        "Vision Agent가 PPE/안전망 검토 결과와 보완 필요 todos를 반환합니다. "
        "부적정, 검토 필요, 판단 불가는 result_code=hil로 내려가고, "
        "이미지 다운로드 실패나 모델 오류만 result_code=fail로 내려갑니다."
    ),
)
async def vision_review(request: VisionReviewRequest) -> VisionReviewResponse:
    results: list[dict[str, Any]] = []
    todos: list[VisionReviewTodo] = []
    model_name = f"{settings.model_name},{settings.safety_net_model_name}"

    for photo in request.photos:
        try:
            if not _is_supported_target_equipment(photo.target_equipment):
                result = _target_missing_result(photo)
            else:
                image = _load_source_image(photo.presigned_url)
                result = _review_target_photo(photo, image)
        except Exception as exc:
            results.append(
                {
                    "file_id": photo.file_id,
                    "original_filename": photo.original_filename,
                    "storage_key": photo.storage_key,
                    **_photo_context(photo),
                    "status": "error",
                    "is_appropriate": None,
                    "message": str(exc),
                }
            )
            return VisionReviewResponse(
                status_code="fail",
                result_code="fail",
                reason=f"현장사진 검토 중 오류가 발생했습니다: {photo.original_filename}",
                model_name=model_name,
                todos=todos,
                details={
                    "summary": "현장사진 검토 실패",
                    "project_id": request.project_id,
                    "usage_statement_id": request.usage_statement_id,
                    "results": results,
                },
            )

        results.append(result)
        if result["is_appropriate"] is not True:
            todos.append(_todo_from_result(photo, result))

    if todos:
        result_code = "hil"
        reason = f"현장사진 {len(request.photos)}건 중 {len(todos)}건 보완 필요"
    else:
        result_code = "success"
        reason = f"현장사진 {len(request.photos)}건 모두 적정"

    return VisionReviewResponse(
        status_code="success",
        result_code=result_code,
        reason=reason,
        model_name=model_name,
        todos=todos,
        details={
            "summary": "현장사진 검토 완료",
            "project_id": request.project_id,
            "usage_statement_id": request.usage_statement_id,
            "results": results,
        },
    )


def _todo_from_result(photo: VisionReviewPhoto, result: dict[str, Any]) -> VisionReviewTodo:
    return VisionReviewTodo(
        file_id=photo.file_id,
        reason=str(result.get("message") or "현장사진 검토가 필요합니다."),
        usage_statement_item_id=photo.usage_statement_item_id,
        category_code=photo.category_code,
        category_name=photo.category_name,
        usage_statement_item_name=photo.usage_statement_item_name,
        target_equipment=photo.target_equipment,
    )


async def _load_upload_image(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    try:
        return load_rgb_image(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _load_source_image(source_uri: str):
    try:
        return load_rgb_image_from_uri(source_uri, settings.input_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _annotated_output_path(request: SourceDetectionRequest, result_type: str) -> Path:
    base_name = request.source_id or _source_name(request.source_uri)
    return _named_annotated_output_path(base_name, result_type)


def _upload_annotated_output_path(filename: str | None, result_type: str) -> Path:
    return _named_annotated_output_path(Path(filename or "uploaded").stem, result_type)


def _named_annotated_output_path(base_name: str, result_type: str) -> Path:
    safe_name = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", base_name).strip("._")
    if not safe_name:
        safe_name = "uploaded"
    return settings.output_dir / "annotated" / result_type / f"{safe_name}.annotated.jpg"


def _source_name(source_uri: str) -> str:
    parsed = urlparse(source_uri)
    if parsed.scheme in ("http", "https"):
        return Path(parsed.path).stem
    return Path(source_uri).stem


def _annotated_image_url(output_path: Path) -> str:
    relative_path = output_path.relative_to(settings.output_dir)
    return _public_url(f"/vision-results/{relative_path.as_posix()}")


def _source_image_url(source_uri: str) -> str:
    parsed = urlparse(source_uri)
    if parsed.scheme in ("http", "https"):
        return source_uri
    return _public_url(f"/vision-files/{Path(source_uri).name}")


def _public_url(path: str) -> str:
    if not settings.public_base_url:
        return path
    return f"{settings.public_base_url}{path}"


def _review_type(photo: VisionReviewPhoto) -> str:
    evidence_type_code = (photo.evidence_type_code or "").lower()
    if evidence_type_code in {"wearing_photo", "item_photo"}:
        return "ppe"

    text = " ".join(
        value or ""
        for value in (
            evidence_type_code,
            photo.storage_key,
            photo.original_filename,
        )
    ).lower()

    if any(keyword in text for keyword in ("safety-net", "safety_net", "safetynet", "안전망")):
        return "safety-net"

    if any(keyword in text for keyword in ("ppe", "protective", "helmet", "안전모", "안전화", "안전벨트")):
        return "ppe"

    return "combined"


def _is_supported_target_equipment(target_equipment: str | None) -> bool:
    target = (target_equipment or "").strip()
    return target in PPE_TARGET_EQUIPMENT or target == SAFETY_NET_TARGET_EQUIPMENT


def _review_target_photo(photo: VisionReviewPhoto, image: Any) -> dict[str, Any]:
    target_equipment = (photo.target_equipment or "").strip()
    if target_equipment in PPE_TARGET_EQUIPMENT:
        response = vision_service.detect_target_ppe(image, target_equipment)
        source_id = str(photo.file_id)
        response.source_id = source_id
        response.source_uri = photo.presigned_url
        response.source_image_url = _source_image_url(photo.presigned_url)
        return _photo_result(photo, "ppe", response.status, response.is_appropriate, response.message, response)

    if target_equipment == SAFETY_NET_TARGET_EQUIPMENT:
        response = vision_service.detect_safety_net(image)
        source_id = str(photo.file_id)
        response.source_id = source_id
        response.source_uri = photo.presigned_url
        response.source_image_url = _source_image_url(photo.presigned_url)
        output_path = _named_annotated_output_path(source_id, "safety-net")
        save_annotated_image(image, [], output_path, response.safety_net_review)
        response.annotated_image_path = str(output_path)
        response.annotated_image_url = _annotated_image_url(output_path)
        return _safety_net_target_result(photo, response)

    return _target_missing_result(photo)


def _review_photo(photo: VisionReviewPhoto, image: Any, review_type: str) -> dict[str, Any]:
    if review_type == "ppe":
        response = vision_service.detect_ppe(image)
        source_id = str(photo.file_id)
        response.source_id = source_id
        response.source_uri = photo.presigned_url
        response.source_image_url = _source_image_url(photo.presigned_url)
        return _photo_result(photo, review_type, response.status, response.is_appropriate, response.message, response)

    if review_type == "safety-net":
        response = vision_service.detect_safety_net(image)
        source_id = str(photo.file_id)
        response.source_id = source_id
        response.source_uri = photo.presigned_url
        response.source_image_url = _source_image_url(photo.presigned_url)
        output_path = _named_annotated_output_path(source_id, "safety-net")
        save_annotated_image(image, [], output_path, response.safety_net_review)
        response.annotated_image_path = str(output_path)
        response.annotated_image_url = _annotated_image_url(output_path)
        return _photo_result(photo, review_type, response.status, response.is_appropriate, response.message, response)

    response = vision_service.detect(image)
    source_id = str(photo.file_id)
    response.source_id = source_id
    response.source_uri = photo.presigned_url
    response.source_image_url = _source_image_url(photo.presigned_url)
    return _photo_result(
        photo,
        review_type,
        response.overall_status,
        response.is_appropriate,
        response.message,
        response,
    )


def _photo_result(
    photo: VisionReviewPhoto,
    review_type: str,
    status: str,
    is_appropriate: bool | None,
    message: str,
    response: Any,
) -> dict[str, Any]:
    payload = response.model_dump()
    return {
        "file_id": photo.file_id,
        "original_filename": photo.original_filename,
        "storage_key": photo.storage_key,
        "evidence_type_code": photo.evidence_type_code,
        **_photo_context(photo),
        "review_type": review_type,
        "status": status,
        "is_appropriate": is_appropriate,
        "message": message,
        "model_name": payload.get("model_name"),
        "source_image_url": payload.get("source_image_url"),
        "annotated_image_url": payload.get("annotated_image_url"),
        "result": payload,
    }


def _photo_context(photo: VisionReviewPhoto) -> dict[str, Any]:
    return {
        "usage_statement_item_id": photo.usage_statement_item_id,
        "usage_statement_item_name": photo.usage_statement_item_name,
        "category_code": photo.category_code,
        "category_name": photo.category_name,
        "target_equipment": photo.target_equipment,
    }


def _target_missing_result(photo: VisionReviewPhoto) -> dict[str, Any]:
    reason = "검증 대상 보호구를 특정할 수 없습니다."
    return {
        "file_id": photo.file_id,
        "original_filename": photo.original_filename,
        "storage_key": photo.storage_key,
        "evidence_type_code": photo.evidence_type_code,
        **_photo_context(photo),
        "review_type": "target_missing",
        "status": "needs_review",
        "is_appropriate": None,
        "message": reason,
        "model_name": None,
        "source_image_url": None,
        "annotated_image_url": None,
        "result": {
            "model_name": None,
            "image_width": None,
            "image_height": None,
            "status": "needs_review",
            "is_appropriate": None,
            "message": reason,
            "reviews": [],
            "detections": [],
        },
    }


def _safety_net_target_result(
    photo: VisionReviewPhoto,
    response: SafetyNetDetectionResponse,
) -> dict[str, Any]:
    review = response.safety_net_review
    equipment_label = TARGET_EQUIPMENT_LABELS[SAFETY_NET_TARGET_EQUIPMENT]
    status = (
        "appropriate"
        if review.is_appropriate is True
        else "not_appropriate"
        if review.is_appropriate is False
        else "needs_review"
    )
    payload = response.model_dump()
    payload["reviews"] = [
        {
            "equipment": SAFETY_NET_TARGET_EQUIPMENT,
            "equipment_label": equipment_label,
            "status": review.status,
            "is_appropriate": review.is_appropriate,
            "confidence": review.confidence,
            "reason": review.reason,
        }
    ]
    payload["detections"] = []
    return {
        "file_id": photo.file_id,
        "original_filename": photo.original_filename,
        "storage_key": photo.storage_key,
        "evidence_type_code": photo.evidence_type_code,
        **_photo_context(photo),
        "review_type": "safety-net",
        "status": status,
        "is_appropriate": review.is_appropriate,
        "message": review.reason,
        "model_name": payload.get("model_name"),
        "source_image_url": payload.get("source_image_url"),
        "annotated_image_url": payload.get("annotated_image_url"),
        "result": payload,
    }
