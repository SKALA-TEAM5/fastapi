import io
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image


def load_rgb_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("invalid image file") from exc


def load_rgb_image_from_uri(source_uri: str, base_dir: Optional[Path] = None) -> Image.Image:
    parsed = urlparse(source_uri)

    if parsed.scheme in ("http", "https"):
        request = Request(source_uri, headers={"User-Agent": "safety-vision/0.1"})
        try:
            with urlopen(request, timeout=10) as response:
                return load_rgb_image(response.read())
        except Exception as exc:
            raise ValueError(f"failed to read image url: {source_uri}") from exc

    path = Path(source_uri)
    if not path.is_file() and base_dir is not None and not path.is_absolute():
        path = base_dir / path.name

    if not path.is_file():
        raise ValueError(f"image file not found: {source_uri}")

    return load_rgb_image(path.read_bytes())
