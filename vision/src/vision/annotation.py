from pathlib import Path
from typing import Union

from PIL import Image, ImageDraw, ImageFont

from typing import Optional

from src.schemas.vision import Detection, SafetyNetReview


BOX_COLORS = {
    "blue": (37, 99, 235),
    "red": (220, 38, 38),
    "yellow": (202, 138, 4),
}
MIN_ANNOTATED_WIDTH = 960

FONT_CANDIDATES = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
)


def scaled_size(image: Image.Image, ratio: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, round(image.width * ratio)))


def load_label_font(
    image: Image.Image,
    ratio: float = 0.028,
    minimum: int = 18,
    maximum: int = 90,
) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    font_size = scaled_size(image, ratio=ratio, minimum=minimum, maximum=maximum)

    for font_path in FONT_CANDIDATES:
        if Path(font_path).is_file():
            return ImageFont.truetype(font_path, font_size)

    return ImageFont.load_default()


def save_annotated_image(
    image: Image.Image,
    detections: list[Detection],
    output_path: Path,
    safety_net_review: Optional[SafetyNetReview] = None,
) -> None:
    annotated, scale_x, scale_y = prepare_annotated_canvas(image)
    draw = ImageDraw.Draw(annotated)
    font = load_label_font(annotated)
    line_width = scaled_size(annotated, ratio=0.003, minimum=2, maximum=10)
    padding_x = scaled_size(annotated, ratio=0.012, minimum=7, maximum=28)
    padding_y = scaled_size(annotated, ratio=0.008, minimum=5, maximum=20)
    placed_labels: list[tuple[float, float, float, float]] = []

    if safety_net_review is not None:
        safety_net_font = load_label_font(annotated, ratio=0.044, minimum=34, maximum=110)
        _draw_safety_net_banner(
            image=annotated,
            draw=draw,
            font=safety_net_font,
            review=safety_net_review,
            padding_x=max(padding_x, scaled_size(annotated, ratio=0.018, minimum=12, maximum=36)),
            padding_y=max(padding_y, scaled_size(annotated, ratio=0.012, minimum=8, maximum=26)),
        )

    for detection in detections:
        color = BOX_COLORS.get(detection.box_color, (107, 114, 128))
        x1, y1, x2, y2 = scale_bbox(detection.bbox_xyxy, scale_x, scale_y)
        label = detection.label or detection.class_code
        suffix = " 검토" if detection.needs_review else ""
        text = f"{label} {detection.confidence:.2f}{suffix}"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
        text_height = bottom - top
        text_width = right - left
        text_box_width = text_width + padding_x * 2
        text_box_height = text_height + padding_y * 2
        text_x = max(0, min(x1, annotated.width - text_box_width))
        text_y = y1 - text_box_height - line_width

        if text_y < 0:
            text_y = min(annotated.height - text_box_height, y1 + line_width)

        label_box = (
            text_x,
            text_y,
            text_x + text_box_width,
            text_y + text_box_height,
        )
        while any(_overlaps(label_box, placed) for placed in placed_labels):
            next_y = label_box[3] + line_width
            if next_y + text_box_height > annotated.height:
                break
            label_box = (
                label_box[0],
                next_y,
                label_box[2],
                next_y + text_box_height,
            )

        draw.rounded_rectangle(
            label_box,
            radius=max(6, line_width),
            fill=color,
        )
        draw.text(
            (label_box[0] + padding_x, label_box[1] + padding_y),
            text,
            fill=(255, 255, 255),
            font=font,
        )
        placed_labels.append(label_box)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)


def prepare_annotated_canvas(image: Image.Image) -> tuple[Image.Image, float, float]:
    if image.width >= MIN_ANNOTATED_WIDTH:
        return image.copy(), 1.0, 1.0

    scale = MIN_ANNOTATED_WIDTH / image.width
    target_size = (MIN_ANNOTATED_WIDTH, round(image.height * scale))
    return image.resize(target_size, Image.Resampling.LANCZOS), scale, scale


def scale_bbox(bbox_xyxy: list[float], scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y


def _draw_safety_net_banner(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont],
    review: SafetyNetReview,
    padding_x: int,
    padding_y: int,
) -> None:
    color = _safety_net_color(review.status)
    text = _safety_net_text(review)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    margin = scaled_size(image, ratio=0.018, minimum=7, maximum=24)
    banner_width = min(image.width - margin * 2, text_width + padding_x * 2)
    banner_height = text_height + padding_y * 2

    x1 = margin
    y1 = margin
    x2 = x1 + banner_width
    y2 = y1 + banner_height
    text_x = x1 + padding_x - left
    text_y = y1 + (banner_height - text_height) / 2 - top

    draw.rounded_rectangle((x1, y1, x2, y2), radius=max(8, margin // 2), fill=color)
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)


def _safety_net_color(status: str) -> tuple[int, int, int]:
    if status == "missing":
        return BOX_COLORS["red"]
    if status == "unclear":
        return BOX_COLORS["yellow"]
    return BOX_COLORS["blue"]


def _safety_net_text(review: SafetyNetReview) -> str:
    confidence = f" {review.confidence:.2f}" if review.confidence is not None else ""
    if review.status == "installed":
        return f"안전망 설치됨{confidence}"
    if review.status == "missing":
        return f"안전망 미설치{confidence}"
    return f"안전망 판단 어려움{confidence}"


def _overlaps(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )
