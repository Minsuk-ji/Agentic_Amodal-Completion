"""Pure image/mask processing: no VLM calls, no model inference.

Builds the target-clean composite, checks whether the target mask touches the image
boundary, applies adaptive canvas expansion, and assembles the final inpaint mask
(dilated occluder mask ∪ canvas-expansion region).
"""
import os

import numpy as np
from PIL import Image

from ..config import AmodalConfig
from ..common.io_utils import load_mask, overlay_mask, save_mask
from ..common.mask_ops import clean_target_image, dilate_mask, empty_mask, resize_mask, union_masks
from ..types import BoundaryExpansion, ObjectItem


# ── Target and occluder image/mask builders ─────────────────────────────────

def build_target_clean(
    image_path: str,
    target: ObjectItem,
    out_dir: str,
    occ_mask_path: str = "",
) -> str:
    """Produce target-only image: target pixels visible, occluded areas white.

    If occ_mask_path is given, subtracts the refined occluder mask from the
    target mask so occluded pixels are correctly shown as white.
    """
    image = Image.open(image_path).convert("RGB")
    mask = load_mask(target.mask_path)
    if occ_mask_path and os.path.exists(occ_mask_path):
        occ = load_mask(occ_mask_path)
        if occ.shape == mask.shape:
            mask = np.where(occ > 0, 0, mask).astype(np.uint8)
    clean = clean_target_image(image, mask)
    path = os.path.join(out_dir, "04_I_target_clean.png")
    clean.save(path)
    return path


def build_occluder_mask_from_target(
    image_path: str,
    target: ObjectItem,
    out_dir: str,
) -> tuple[str, str]:
    """Occluder mask = pixels inside target bbox that are NOT part of the target mask."""
    image = Image.open(image_path).convert("RGB")
    target_mask = load_mask(target.mask_path)
    h, w = target_mask.shape
    x0, y0, x1, y1 = target.bbox

    bbox_region = np.zeros((h, w), dtype=bool)
    bbox_region[y0:y1, x0:x1] = True
    occ = (bbox_region & (target_mask == 0)).astype(np.uint8) * 255

    mask_path = save_mask(occ, os.path.join(out_dir, "05_M_occ.png"))
    overlay_path = os.path.join(out_dir, "06_M_occ_overlay.png")
    overlay_mask(image, occ, color=(255, 0, 0), alpha=0.45).save(overlay_path)
    return mask_path, overlay_path


# ── Canvas expansion image/mask builders ────────────────────────────────────

def compute_expansion_directions(mask_path: str, threshold: int = 10) -> list[str]:
    """Return which edges the mask touches within `threshold` pixels."""
    mask = load_mask(mask_path)
    h, w = mask.shape
    directions: list[str] = []
    if (mask[:threshold, :] > 0).any():
        directions.append("top")
    if (mask[h - threshold:, :] > 0).any():
        directions.append("bottom")
    if (mask[:, :threshold] > 0).any():
        directions.append("left")
    if (mask[:, w - threshold:] > 0).any():
        directions.append("right")
    return directions


def apply_canvas_expansion(image_path: str, expansion: BoundaryExpansion, out_path: str) -> str:
    """Paste the image onto a white canvas expanded by the given pixel amounts."""
    expansion = _resolve_pixel_expansion(expansion, Image.open(image_path).size)
    image = Image.open(image_path).convert("RGB")
    if not expansion.need_expand or (expansion.top + expansion.bottom + expansion.left + expansion.right) == 0:
        image.save(out_path)
        return out_path
    width, height = image.size
    canvas = Image.new("RGB", (width + expansion.left + expansion.right, height + expansion.top + expansion.bottom), (255, 255, 255))
    canvas.paste(image, (expansion.left, expansion.top))
    canvas.save(out_path)
    return out_path


def build_canvas_expansion_mask(original_size: tuple[int, int], expansion: BoundaryExpansion, out_path: str) -> str:
    """White (=inpaint) where the canvas was extended, black elsewhere."""
    expansion = _resolve_pixel_expansion(expansion, original_size)
    w, h = original_size
    new_w = w + expansion.left + expansion.right
    new_h = h + expansion.top + expansion.bottom
    mask = np.ones((new_h, new_w), dtype=np.uint8) * 255
    mask[expansion.top: expansion.top + h, expansion.left: expansion.left + w] = 0
    if not expansion.need_expand:
        mask[:] = 0
    return save_mask(mask, out_path)


def expand_mask_to_canvas(
    mask_path: str,
    original_size: tuple[int, int],
    expansion: BoundaryExpansion,
    out_path: str,
) -> str:
    """Place an existing mask in the correct position on the expanded canvas."""
    expansion = _resolve_pixel_expansion(expansion, original_size)
    mask = load_mask(mask_path)
    w, h = original_size
    if mask.shape[:2] != (h, w):
        mask = resize_mask(mask, original_size)
    new_w = w + expansion.left + expansion.right
    new_h = h + expansion.top + expansion.bottom
    canvas = np.zeros((new_h, new_w), dtype=np.uint8)
    canvas[expansion.top: expansion.top + h, expansion.left: expansion.left + w] = mask
    return save_mask(canvas, out_path)


def build_final_inpaint_mask(
    occ_mask_path: str,
    canvas_expansion_mask_path: str,
    out_path: str,
    config: AmodalConfig,
) -> str:
    """Final inpaint mask = dilated occluder mask ∪ canvas expansion area."""
    occ = load_mask(occ_mask_path)
    canvas = load_mask(canvas_expansion_mask_path)
    occ_dilated = dilate_mask(occ, config.inpaint_dilation_kernel, config.inpaint_dilation_iterations)
    if canvas.shape != occ_dilated.shape:
        canvas = resize_mask(canvas, (occ_dilated.shape[1], occ_dilated.shape[0]))
    final = ((occ_dilated > 0) | (canvas > 0)).astype(np.uint8) * 255
    return save_mask(final, out_path)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_pixel_expansion(expansion: BoundaryExpansion, image_size: tuple[int, int]) -> BoundaryExpansion:
    """Fall back to ratio-based pixel computation when pixel fields are all zero."""
    if expansion.left + expansion.right + expansion.top + expansion.bottom > 0:
        return expansion
    w, h = image_size
    expansion.left = int(round(w * float(expansion.left_ratio)))
    expansion.right = int(round(w * float(expansion.right_ratio)))
    expansion.top = int(round(h * float(expansion.top_ratio)))
    expansion.bottom = int(round(h * float(expansion.bottom_ratio)))
    expansion.need_expand = (expansion.left + expansion.right + expansion.top + expansion.bottom) > 0
    return expansion
