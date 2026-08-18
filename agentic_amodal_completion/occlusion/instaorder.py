"""InstaOrder-based occlusion order filtering.

Requires the InstaOrder repo and checkpoint — see README for setup. Paths are read from
`config.instaorder_dir` / `config.instaorder_ckpt`.
"""
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from ..config import AmodalConfig

_DATA_MEAN = [0.485, 0.456, 0.406]
_DATA_STD  = [0.229, 0.224, 0.225]

_model = None
_model_key = None


def _load_model(config: AmodalConfig):
    global _model, _model_key
    key = (config.instaorder_dir, config.instaorder_ckpt)
    if _model is not None and _model_key == key:
        return _model

    instaorder_dir = os.path.abspath(config.instaorder_dir)
    saved = {m: sys.modules.pop(m) for m in ["utils", "models", "inference"] if m in sys.modules}
    sys.path.insert(0, instaorder_dir)
    try:
        import yaml
        from models.supervised_order import InstaOrderNet_o

        cfg_path = os.path.join(instaorder_dir, "experiments/InstaOrder/InstaOrderNet_o/config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        model = InstaOrderNet_o(cfg["model"], dist_model=False)
        ckpt = torch.load(config.instaorder_ckpt, map_location="cpu")
        state = {k.replace("module.", "", 1): v for k, v in ckpt["state_dict"].items()}
        model.model.module.load_state_dict(state, strict=True)
        device = f"cuda:{config.default_gpu_id}"
        model.model.to(device).eval()
        print(f"[InstaOrder] Loaded from {config.instaorder_ckpt}")
    finally:
        for m in ["utils", "models", "inference"]:
            sys.modules.pop(m, None)
        for m, v in saved.items():
            sys.modules[m] = v

    _model = model
    _model_key = key
    return _model


def unload_model():
    global _model, _model_key
    if _model is not None:
        del _model
        _model = None
        _model_key = None
        torch.cuda.empty_cache()


def _to_rgb_tensor(pil_img: Image.Image, size: int, device: str) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB").resize((size, size), Image.BILINEAR)).astype(np.float32) / 255.0
    arr = (arr - np.array(_DATA_MEAN)) / np.array(_DATA_STD)
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float().to(device)


def _to_mask_tensor(mask_arr: np.ndarray, size: int, device: str) -> torch.Tensor:
    m = cv2.resize(mask_arr.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0).float().to(device)


def filter_occluders(
    image_pil: Image.Image,
    target_mask_bin: np.ndarray,
    entity_masks: dict[str, np.ndarray],
    config: AmodalConfig,
) -> dict[str, np.ndarray]:
    """
    Returns confirmed occluders: entities where P(entity>target) > 0.5 AND > P(target>entity).
    entity_masks: {name: uint8 binary mask (0/1 or 0/255)}
    target_mask_bin: uint8 binary mask (0 or 1)
    """
    model = _load_model(config)
    device = f"cuda:{config.default_gpu_id}"
    size = config.instaorder_input_size

    rgb_t = _to_rgb_tensor(image_pil, size, device)
    tgt_t = _to_mask_tensor(target_mask_bin, size, device)

    confirmed = {}
    for name, mask_raw in entity_masks.items():
        entity_bin = (mask_raw > 127).astype(np.uint8) if mask_raw.max() > 1 else mask_raw.astype(np.uint8)
        if entity_bin.sum() == 0:
            print(f"[InstaOrder] {name}: empty mask, skip")
            continue
        ent_t = _to_mask_tensor(entity_bin, size, device)
        with torch.no_grad():
            o1 = torch.sigmoid(model.model(torch.cat([ent_t, tgt_t, rgb_t], dim=1)))
            if not config.ablation_unidirectional_instaorder:
                o2 = torch.sigmoid(model.model(torch.cat([tgt_t, ent_t, rgb_t], dim=1)))
                p_eo = ((o1[0, 1] + o2[0, 0]) / 2).item()
                p_te = ((o1[0, 0] + o2[0, 1]) / 2).item()
            else:
                p_eo = o1[0, 1].item()
                p_te = o1[0, 0].item()
        is_occ = p_eo > 0.5 and p_eo > p_te
        print(f"[InstaOrder] {name}: P(e>t)={p_eo:.3f} P(t>e)={p_te:.3f} → {'OCCLUDER' if is_occ else 'not occluder'}")
        if is_occ:
            confirmed[name] = entity_bin

    if config.aggressive_model_unload:
        unload_model()

    return confirmed
