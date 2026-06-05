from functools import lru_cache
from typing import Any, Optional, Tuple

from PIL import Image

from src.core.config import Settings
from src.schemas.vision import (
    Detection,
    DetectionResponse,
    EquipmentReview,
    PpeDetectionResponse,
    SafetyNetDetectionResponse,
    SafetyNetReview,
)


class VisionDetectionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @lru_cache(maxsize=1)
    def load_model(self) -> Any:
        if not self.settings.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {self.settings.model_path}")

        from ultralytics import YOLO

        return YOLO(str(self.settings.model_path))

    @lru_cache(maxsize=1)
    def load_safety_net_model(self) -> Any:
        if not self.settings.safety_net_model_path.is_file():
            raise FileNotFoundError(f"model file not found: {self.settings.safety_net_model_path}")

        from ultralytics import YOLO

        return YOLO(str(self.settings.safety_net_model_path))

    def detect_ppe(self, image: Image.Image) -> PpeDetectionResponse:
        detections = self._detect_ppe_boxes(image)
        reviews = self._build_reviews(detections)
        status, is_appropriate, message = self._build_ppe_result(reviews)

        return PpeDetectionResponse(
            model_name=self.settings.model_name,
            image_width=image.width,
            image_height=image.height,
            status=status,
            is_appropriate=is_appropriate,
            message=message,
            reviews=reviews,
            detections=detections,
        )

    def detect_safety_net(self, image: Image.Image) -> SafetyNetDetectionResponse:
        review = self._classify_safety_net(image)
        return SafetyNetDetectionResponse(
            model_name=self.settings.safety_net_model_name,
            image_width=image.width,
            image_height=image.height,
            status=review.status,
            is_appropriate=review.is_appropriate,
            message=review.reason,
            safety_net_review=review,
        )

    def detect(self, image: Image.Image) -> DetectionResponse:
        ppe_response = self.detect_ppe(image)
        safety_net_response = self.detect_safety_net(image)
        overall_status, is_appropriate, message = self._build_overall_result(
            ppe_response.status,
            ppe_response.is_appropriate,
            ppe_response.message,
            safety_net_response.safety_net_review,
        )

        return DetectionResponse(
            model_name=self.settings.model_name,
            image_width=image.width,
            image_height=image.height,
            overall_status=overall_status,
            is_appropriate=is_appropriate,
            message=message,
            ppe_status=ppe_response.status,
            ppe_is_appropriate=ppe_response.is_appropriate,
            ppe_message=ppe_response.message,
            safety_net_review=safety_net_response.safety_net_review,
            reviews=ppe_response.reviews,
            detections=ppe_response.detections,
        )

    def _detect_ppe_boxes(self, image: Image.Image) -> list[Detection]:
        predict_kwargs: dict[str, Any] = {
            "conf": self.settings.model_conf,
            "iou": self.settings.model_iou,
            "verbose": False,
        }

        if self.settings.model_device != "auto":
            predict_kwargs["device"] = self.settings.model_device

        model = self.load_model()
        results = model.predict(image, **predict_kwargs)
        boxes = results[0].boxes
        detections: list[Detection] = []

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                class_code = self.settings.class_names.get(class_id, str(class_id))
                class_label = self.settings.class_labels.get(class_code, {})
                is_wearing = class_label.get("is_wearing")
                confidence = round(float(box.conf.item()), 4)
                needs_review = bool(
                    is_wearing is True
                    and confidence < self.settings.review_confidence_threshold
                )
                xyxy = [round(float(value), 2) for value in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_code=class_code,
                        equipment=class_label.get("equipment"),
                        label=class_label.get("label"),
                        is_wearing=is_wearing,
                        needs_review=needs_review,
                        box_color=self._box_color(is_wearing, needs_review),
                        confidence=confidence,
                        bbox_xyxy=xyxy,
                    )
                )

        return detections

    def _build_reviews(self, detections: list[Detection]) -> list[EquipmentReview]:
        reviews: list[EquipmentReview] = []

        for equipment in ("safety_helmet", "safety_shoes", "safety_belt"):
            matches = [detection for detection in detections if detection.equipment == equipment]
            if not matches:
                equipment_label = self.settings.equipment_labels[equipment]
                reviews.append(
                    EquipmentReview(
                        equipment=equipment,
                        equipment_label=equipment_label,
                        status="unknown",
                        is_appropriate=None,
                        confidence=None,
                        reason=f"{equipment_label} 항목이 확인되지 않았습니다.",
                    )
                )
                continue

            not_wearing_matches = [detection for detection in matches if detection.is_wearing is False]
            if not_wearing_matches:
                best = max(not_wearing_matches, key=lambda detection: detection.confidence)
                reviews.append(
                    EquipmentReview(
                        equipment=equipment,
                        equipment_label=self.settings.equipment_labels[equipment],
                        status="not_wearing",
                        is_appropriate=False,
                        confidence=best.confidence,
                        reason=f"{best.label} 탐지 결과를 기준으로 부적정 판단했습니다.",
                    )
                )
                continue

            needs_review_matches = [
                detection
                for detection in matches
                if detection.needs_review
            ]
            if needs_review_matches:
                best = max(needs_review_matches, key=lambda detection: detection.confidence)
                reviews.append(
                    EquipmentReview(
                        equipment=equipment,
                        equipment_label=self.settings.equipment_labels[equipment],
                        status="needs_review",
                        is_appropriate=None,
                        confidence=best.confidence,
                        reason=(
                            f"{best.label} confidence가 "
                            f"{self.settings.review_confidence_threshold:.2f} 미만이라 검토가 필요합니다."
                        ),
                    )
                )
                continue

            valid_wearing_matches = [
                detection
                for detection in matches
                if detection.is_wearing is True
            ]
            best = max(valid_wearing_matches, key=lambda detection: detection.confidence)
            reviews.append(
                EquipmentReview(
                    equipment=equipment,
                    equipment_label=self.settings.equipment_labels[equipment],
                    status="wearing",
                    is_appropriate=True,
                    confidence=best.confidence,
                    reason=f"{best.label} 탐지 결과를 기준으로 판단했습니다.",
                )
            )

        return reviews

    def _classify_safety_net(self, image: Image.Image) -> SafetyNetReview:
        model = self.load_safety_net_model()
        predict_kwargs: dict[str, Any] = {"verbose": False}
        if self.settings.model_device != "auto":
            predict_kwargs["device"] = self.settings.model_device

        result = model.predict(image, **predict_kwargs)[0]
        probs = result.probs
        if probs is None:
            return SafetyNetReview(
                model_name=self.settings.safety_net_model_name,
                status="unclear",
                is_appropriate=None,
                confidence=None,
                raw_label=None,
                reason="안전망 설치 여부를 판단하기 어렵습니다.",
            )

        class_id = int(probs.top1)
        confidence = round(float(probs.top1conf), 4)
        raw_label = str(model.names.get(class_id, class_id))

        if confidence < self.settings.safety_net_confidence_threshold:
            return SafetyNetReview(
                model_name=self.settings.safety_net_model_name,
                status="unclear",
                is_appropriate=None,
                confidence=confidence,
                raw_label=raw_label,
                reason="이 사진에서는 안전망 설치 여부를 판단하기 어렵습니다.",
            )

        if raw_label == "installed":
            return SafetyNetReview(
                model_name=self.settings.safety_net_model_name,
                status="installed",
                is_appropriate=True,
                confidence=confidence,
                raw_label=raw_label,
                reason="안전망 설치 상태로 판단되었습니다.",
            )

        if raw_label == "missing":
            return SafetyNetReview(
                model_name=self.settings.safety_net_model_name,
                status="missing",
                is_appropriate=False,
                confidence=confidence,
                raw_label=raw_label,
                reason="안전망 미설치 상태로 판단되어 부적정입니다.",
            )

        return SafetyNetReview(
            model_name=self.settings.safety_net_model_name,
            status="unclear",
            is_appropriate=None,
            confidence=confidence,
            raw_label=raw_label,
            reason="이 사진에서는 안전망 설치 여부를 판단하기 어렵습니다.",
        )

    def _build_ppe_result(
        self, reviews: list[EquipmentReview]
    ) -> Tuple[str, Optional[bool], str]:
        not_wearing = [review for review in reviews if review.status == "not_wearing"]
        needs_review = [review for review in reviews if review.status == "needs_review"]
        unknown = [review for review in reviews if review.status == "unknown"]
        messages: list[str] = []

        if not_wearing:
            labels = ", ".join(review.equipment_label for review in not_wearing)
            messages.append(f"{labels} 항목이 부적정으로 판단되었습니다.")

        if needs_review:
            labels = ", ".join(review.equipment_label for review in needs_review)
            messages.append(f"{labels} 항목은 검토가 필요합니다.")

        if unknown:
            labels = ", ".join(review.equipment_label for review in unknown)
            messages.append(f"{labels} 항목이 확인되지 않았습니다.")

        if not_wearing:
            return "not_appropriate", False, " ".join(messages)

        if needs_review or unknown:
            return "needs_review", None, " ".join(messages)

        return "appropriate", True, "안전모, 안전화, 안전벨트가 모두 적정으로 판단되었습니다."

    def _build_overall_result(
        self,
        ppe_status: str,
        ppe_is_appropriate: Optional[bool],
        ppe_message: str,
        safety_net_review: SafetyNetReview,
    ) -> Tuple[str, Optional[bool], str]:
        failed_messages: list[str] = []
        review_messages: list[str] = []

        if ppe_is_appropriate is False:
            failed_messages.append(ppe_message)
        elif ppe_is_appropriate is None:
            review_messages.append(ppe_message)

        if safety_net_review.is_appropriate is False:
            failed_messages.append(safety_net_review.reason)
        elif safety_net_review.is_appropriate is None:
            review_messages.append(safety_net_review.reason)

        if failed_messages:
            return "not_appropriate", False, " ".join(failed_messages)

        if review_messages:
            return "needs_review", None, " ".join(review_messages)

        return (
            "appropriate",
            True,
            "보호구 착용 상태와 안전망 설치 상태가 모두 적정으로 판단되었습니다.",
        )

    def _box_color(self, is_wearing: object, needs_review: bool) -> str:
        if needs_review:
            return "yellow"
        if is_wearing is False:
            return "red"
        return "blue"
