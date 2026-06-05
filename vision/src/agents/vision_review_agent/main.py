import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import settings
from src.services.vision_detection_service import VisionDetectionService
from src.vision.annotation import save_annotated_image
from src.vision.image_loader import load_rgb_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TaskName = str


def iter_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.stem.endswith("-test")
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def review_images(
    input_dir: Path,
    output_dir: Path,
    task: str,
    safety_net_head_count: int,
    safety_net_tail_count: int,
) -> dict:
    service = VisionDetectionService(settings)
    image_paths = iter_image_paths(input_dir)
    results = []
    task_by_file = classify_image_tasks(
        image_paths,
        task,
        safety_net_head_count=safety_net_head_count,
        safety_net_tail_count=safety_net_tail_count,
    )

    for image_path in image_paths:
        image = load_rgb_image(image_path.read_bytes())
        image_task = task_by_file[image_path.name]
        if image_task == "ppe":
            response = service.detect_ppe(image)
            annotated_path = None
        else:
            response = service.detect_safety_net(image)
            annotated_path = output_dir / "annotated" / "safety-net" / f"{image_path.stem}.annotated.jpg"
            save_annotated_image(image, [], annotated_path, response.safety_net_review)

        response.source_uri = str(image_path)
        response.source_image_url = f"/vision-files/{image_path.name}"
        if annotated_path is not None:
            response.annotated_image_path = str(annotated_path)
            response.annotated_image_url = f"/vision-results/annotated/{image_task}/{annotated_path.name}"
        result = {
            "source_file": str(image_path),
            "task": image_task,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            **response.model_dump(),
        }
        results.append(result)
        write_json(output_dir / image_task / f"{image_path.stem}.vision.json", result)

    summary = {
        "task": task,
        "task_by_file": task_by_file,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": settings.model_name,
        "model_path": str(settings.model_path),
        "safety_net_model_name": settings.safety_net_model_name,
        "safety_net_model_path": str(settings.safety_net_model_path),
        "total_images": len(image_paths),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    return summary


def classify_image_tasks(
    image_paths: list[Path],
    task: str,
    safety_net_head_count: int,
    safety_net_tail_count: int,
) -> dict[str, TaskName]:
    if task != "auto":
        return {image_path.name: task for image_path in image_paths}

    last_safety_net_index = len(image_paths) - safety_net_tail_count
    task_by_file = {}
    for index, image_path in enumerate(image_paths):
        task_by_file[image_path.name] = classify_image_task_by_name(
            image_path.name,
            index=index,
            last_safety_net_index=last_safety_net_index,
            safety_net_head_count=safety_net_head_count,
        )
    return task_by_file


def classify_image_task_by_name(
    filename: str,
    index: int,
    last_safety_net_index: int,
    safety_net_head_count: int,
) -> TaskName:
    stem = Path(filename).stem
    if filename.startswith("ppe__") or re.fullmatch(r"ppe\d+", stem):
        return "ppe"
    if filename.startswith("safety-net__") or re.fullmatch(r"safety-net\d+", stem):
        return "safety-net"

    is_safety_net_head = index < safety_net_head_count
    is_safety_net_tail = index >= last_safety_net_index
    return "safety-net" if is_safety_net_head or is_safety_net_tail else "ppe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vision review for images and save JSON results.")
    parser.add_argument("--input-dir", type=Path, default=settings.input_dir)
    parser.add_argument("--output-dir", type=Path, default=settings.output_dir)
    parser.add_argument("--task", choices=("ppe", "safety-net", "auto"), default="ppe")
    parser.add_argument("--safety-net-head-count", type=int, default=5)
    parser.add_argument("--safety-net-tail-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = review_images(
        args.input_dir,
        args.output_dir,
        args.task,
        args.safety_net_head_count,
        args.safety_net_tail_count,
    )
    print(f"reviewed {summary['total_images']} image(s)")
    print(f"results saved to {summary['output_dir']}")


if __name__ == "__main__":
    main()
