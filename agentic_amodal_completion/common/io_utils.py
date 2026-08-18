import json
import os
import re
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageDraw


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(text: str, fallback: str = "item") -> str:
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", str(text).strip().lower()).strip("_")
    return token or fallback


def read_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def write_json(data: Any, path: str) -> str:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def write_text(text: str, path: str) -> str:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(str(text).strip() + "\n")
    return path


def save_mask(mask: np.ndarray, path: str) -> str:
    ensure_dir(os.path.dirname(path))
    arr = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)
    return path


def load_mask(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    mask_l = Image.fromarray(((mask > 0).astype(np.uint8)) * 255, mode="L")
    color_layer = Image.new("RGBA", base.size, color + (int(255 * alpha),))
    transparent = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer = Image.composite(color_layer, transparent, mask_l)
    return Image.alpha_composite(base, layer).convert("RGB")


def mask_to_white_bg_crop(image: Image.Image, mask: np.ndarray, save_path: str) -> str:
    """Apply mask to image (non-mask → white), crop to mask bounding box, save as RGB."""
    img = image.convert("RGB")
    arr = np.array(img)
    bin_mask = (mask > 127) if mask.max() > 1 else (mask > 0)

    # white out non-mask region
    arr[~bin_mask] = 255

    # crop to bounding box
    ys, xs = np.where(bin_mask)
    if len(xs) == 0:
        Image.fromarray(arr).save(save_path)
        return save_path
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    cropped = Image.fromarray(arr).crop((x1, y1, x2 + 1, y2 + 1))

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    cropped.save(save_path)
    return save_path


def draw_bboxes(image: Image.Image, objects: list[dict]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for item in objects:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{item.get('id','')} {item.get('name','')}", fill=(255, 0, 0))
    return out
