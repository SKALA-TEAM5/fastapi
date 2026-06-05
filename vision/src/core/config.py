import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("VISION_MODEL_PATH", "vision/models/ppe-detector.pt"))
    safety_net_model_path: Path = Path(os.getenv("SAFETY_NET_MODEL_PATH", "vision/models/safety-net-classifier.pt"))
    model_name: str = os.getenv("VISION_MODEL_NAME", "ppe-detector")
    safety_net_model_name: str = os.getenv("SAFETY_NET_MODEL_NAME", "safety-net-classifier")
    model_conf: float = float(os.getenv("VISION_MODEL_CONF", "0.35"))
    model_iou: float = float(os.getenv("VISION_MODEL_IOU", "0.45"))
    model_device: str = os.getenv("VISION_MODEL_DEVICE", "auto")
    review_confidence_threshold: float = float(os.getenv("VISION_REVIEW_CONF", "0.50"))
    safety_net_confidence_threshold: float = float(os.getenv("SAFETY_NET_REVIEW_CONF", "0.70"))
    input_dir: Path = Path(os.getenv("VISION_INPUT_DIR", "volumes/files"))
    output_dir: Path = Path(os.getenv("VISION_OUTPUT_DIR", "volumes/vision_results"))
    public_base_url: str = os.getenv("VISION_PUBLIC_BASE_URL", "").rstrip("/")
    class_names: dict[int, str] = field(
        default_factory=lambda: {
            0: "01",
            1: "02",
            2: "05",
            3: "07",
            4: "08",
        }
    )
    class_labels: dict[str, dict[str, Union[str, bool]]] = field(
        default_factory=lambda: {
            "01": {"equipment": "safety_belt", "label": "안전벨트 착용", "is_wearing": True},
            "02": {"equipment": "safety_belt", "label": "안전벨트 미착용", "is_wearing": False},
            "05": {"equipment": "safety_shoes", "label": "안전화 착용", "is_wearing": True},
            "06": {"equipment": "safety_shoes", "label": "안전화 미착용", "is_wearing": False},
            "07": {"equipment": "safety_helmet", "label": "안전모 착용", "is_wearing": True},
            "08": {"equipment": "safety_helmet", "label": "안전모 미착용", "is_wearing": False},
        }
    )
    equipment_labels: dict[str, str] = field(
        default_factory=lambda: {
            "safety_helmet": "안전모",
            "safety_shoes": "안전화",
            "safety_belt": "안전벨트",
        }
    )


settings = Settings()
