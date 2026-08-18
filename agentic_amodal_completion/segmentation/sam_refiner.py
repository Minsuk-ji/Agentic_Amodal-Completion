"""SAMRefiner adapter: refines coarse Sa2VA masks with iterative SAM prompting
(point + box + mask inputs derived from the coarse mask).

Requires the SAMRefiner repo and a SAM ViT-H checkpoint — see README for setup.
Paths are read from `config.sam_refiner_root` / `config.sam_checkpoint`
(env vars `AGENTIC_AMODAL_SAM_REFINER_ROOT` / `AGENTIC_AMODAL_SAM_CKPT`).
"""
import os
import sys
from typing import Optional

import numpy as np

from ..config import AmodalConfig
from ..common.device import gpu_cleanup

_sam = None
_sam_device: Optional[str] = None
_sam_ckpt: Optional[str] = None


def _ensure_paths(config: AmodalConfig) -> None:
    sam_pkg = os.path.join(config.sam_refiner_root, "segment-anything")
    for p in (config.sam_refiner_root, sam_pkg):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_sam(config: AmodalConfig, device: str):
    global _sam, _sam_device, _sam_ckpt
    if _sam is not None and _sam_device == device and _sam_ckpt == config.sam_checkpoint:
        return _sam
    if _sam is not None:
        del _sam
        _sam = None
        gpu_cleanup()

    _ensure_paths(config)
    from segment_anything import sam_model_registry

    print(f"[SAMRefiner] Loading SAM {config.sam_model_type} on {device}")
    sam = sam_model_registry[config.sam_model_type](checkpoint=config.sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    _sam = sam
    _sam_device = device
    _sam_ckpt = config.sam_checkpoint
    return _sam


def unload_sam() -> None:
    global _sam, _sam_device, _sam_ckpt
    if _sam is None:
        return
    del _sam
    _sam = None
    _sam_device = None
    _sam_ckpt = None
    gpu_cleanup()


def refine_mask(
    image_path: str,
    coarse_mask: np.ndarray,
    config: AmodalConfig,
) -> np.ndarray:
    """Refine a single coarse uint8 mask using SAMRefiner.

    Returns a uint8 (0/255) mask of the same spatial size.
    Falls back to the original coarse mask on any error.
    """
    import torch

    _ensure_paths(config)
    from sam_refiner import sam_refiner

    device = f"cuda:{config.default_gpu_id}" if torch.cuda.is_available() else "cpu"
    sam = _load_sam(config, device)

    binary = (coarse_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return coarse_mask

    try:
        refined_list, _, _ = sam_refiner(
            image_path,
            [binary],
            sam,
            iters=config.sam_refiner_iters,
        )
        refined = refined_list[0].astype(np.uint8) * 255
        print(f"[SAMRefiner] Refined: {binary.sum()} → {(refined > 0).sum()} px")
        return refined
    except Exception as ex:
        print(f"[SAMRefiner] Failed ({ex}), using coarse mask")
        return coarse_mask


def refine_masks(
    image_path: str,
    masks: dict[str, np.ndarray],
    config: AmodalConfig,
) -> dict[str, np.ndarray]:
    """Refine all occluder masks in a {name: uint8_mask} dict."""
    if not masks:
        return masks
    refined = {}
    for name, mask in masks.items():
        print(f"[SAMRefiner] Refining '{name}'…")
        refined[name] = refine_mask(image_path, mask, config)
    unload_sam()
    return refined
