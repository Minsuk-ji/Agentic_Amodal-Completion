"""Segment the target object from the instruction."""
import os
import re

import numpy as np
from PIL import Image

from ..config import AmodalConfig
from ..common.io_utils import ensure_dir, save_mask
from ..common.mask_ops import mask_bbox
from ..types import ObjectItem
from .sa2va import segment_with_sa2va


def segment_target(
    image_path: str,
    out_dir: str,
    config: AmodalConfig,
    instruction: str = "",
) -> ObjectItem:
    """Segment the target from the instruction and return ObjectItem."""
    ensure_dir(out_dir)
    target_name = _extract_target_name(instruction)

    mask = segment_with_sa2va(image_path, target_name, config, keep_on_gpu=False)
    if mask is None or int((mask > 0).sum()) == 0:
        raise RuntimeError(f"SA2VA could not segment target: {target_name!r}")

    mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask
    bbox = mask_bbox(mask_uint8)
    if bbox is None:
        raise RuntimeError(f"Target mask has no valid bounding box: {target_name!r}")

    mask_path = save_mask(mask_uint8, os.path.join(out_dir, "02_M_target.png"))
    return ObjectItem(
        id="target",
        name=target_name,
        bbox=[int(v) for v in bbox],
        mask_path=mask_path,
    )


def _extract_target_name(instruction: str) -> str:
    m = re.search(
        r"complete\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+in\b|\s+from\b|\s+of\b|\s*$)",
        instruction, re.I,
    )
    if m:
        return m.group(1).strip().lower()
    return instruction.strip().lower()
