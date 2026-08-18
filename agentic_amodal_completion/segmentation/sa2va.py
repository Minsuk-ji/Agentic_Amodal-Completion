"""Sa2VA grounded segmentation: turns a natural-language object reference into a mask.

Used for (1) segmenting the target object named in the user's instruction, (2) segmenting
every scene entity detected by the VLM planning stage in a single forward pass, and (3)
re-segmenting the target inside a completed image during boundary verification.
"""
import json
import re
from typing import Optional, Union

import numpy as np
from PIL import Image

from ..config import AmodalConfig
from ..common.device import configure_huggingface_paths, gpu_cleanup, normalize_device, set_current_cuda_device


_sa2va = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "max_new_tokens": None,
    "max_dynamic_patch": None,
    "model_id": None,
}


def clear_sa2va_cache() -> None:
    old_model = _sa2va.get("model")
    old_tokenizer = _sa2va.get("tokenizer")
    _sa2va.update(
        {
            "model": None,
            "tokenizer": None,
            "device": None,
            "max_new_tokens": None,
            "max_dynamic_patch": None,
            "model_id": None,
        }
    )
    try:
        del old_model
        del old_tokenizer
    except Exception:
        pass
    gpu_cleanup()


def _resolve_device(config: AmodalConfig) -> str:
    import torch

    requested = normalize_device(config.sa2va_device or None, fallback_gpu_id=config.default_gpu_id, allow_auto=True)
    if requested == "auto":
        requested = f"cuda:{config.default_gpu_id}" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return requested


def load_sa2va_model(
    config: AmodalConfig,
    max_new_tokens: Optional[int] = None,
    max_dynamic_patch: Optional[int] = None,
):
    configure_huggingface_paths()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(config)
    tokens = max(8, int(max_new_tokens if max_new_tokens is not None else config.sa2va_max_new_tokens))
    patches = max(1, int(max_dynamic_patch if max_dynamic_patch is not None else config.sa2va_max_dynamic_patch))

    if _sa2va["model"] is not None:
        if (
            _sa2va["model_id"] == config.sa2va_model_id
            and _sa2va["device"] == device
            and _sa2va["max_new_tokens"] == tokens
            and _sa2va["max_dynamic_patch"] == patches
        ):
            return _sa2va["model"], _sa2va["tokenizer"]
        clear_sa2va_cache()

    print(f"[Sa2VA] Loading {config.sa2va_model_id} on {device}")
    set_current_cuda_device(device, fallback_gpu_id=config.default_gpu_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.sa2va_model_id,
        torch_dtype="auto",
        device_map={"": device},
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.sa2va_model_id, trust_remote_code=True)
    if hasattr(model, "preparing_for_generation"):
        model.preparing_for_generation(tokenizer=tokenizer, max_new_tokens=tokens, torch_dtype=torch.bfloat16)
    if hasattr(model, "max_dynamic_patch"):
        model.max_dynamic_patch = patches

    _sa2va.update(
        {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "max_new_tokens": tokens,
            "max_dynamic_patch": patches,
            "model_id": config.sa2va_model_id,
        }
    )
    return model, tokenizer


def offload_sa2va_to_cpu() -> None:
    model = _sa2va.get("model")
    if model is None:
        return
    if str(_sa2va.get("device") or "").startswith("cuda"):
        print("[Sa2VA] Offload to CPU")
        model.to("cpu")
        _sa2va["device"] = "cpu"
        gpu_cleanup()


def ensure_sa2va_on_device(config: AmodalConfig) -> None:
    model = _sa2va.get("model")
    if model is None:
        return
    target = _resolve_device(config)
    current = str(_sa2va.get("device") or "")
    if current != target:
        set_current_cuda_device(target, fallback_gpu_id=config.default_gpu_id)
        model.to(target)
        _sa2va["device"] = target


def _cuda_index() -> Optional[int]:
    device = str(_sa2va.get("device") or "")
    if not device.startswith("cuda:"):
        return None
    try:
        return int(device.split(":", 1)[1])
    except Exception:
        return None


def _predict_forward(image: Image.Image, instruction: str, config: AmodalConfig):
    import torch

    model, tokenizer = load_sa2va_model(config)
    prev_cuda_device = None
    target_cuda_device = _cuda_index()
    if target_cuda_device is not None and torch.cuda.is_available():
        prev_cuda_device = torch.cuda.current_device()
        if prev_cuda_device != target_cuda_device:
            torch.cuda.set_device(target_cuda_device)
    try:
        return model.predict_forward(image=image, text=instruction, tokenizer=tokenizer)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "outofmemoryerror" in msg:
            clear_sa2va_cache()
            model, tokenizer = load_sa2va_model(
                config,
                max_new_tokens=config.sa2va_oom_retry_max_new_tokens,
                max_dynamic_patch=config.sa2va_oom_retry_max_dynamic_patch,
            )
            return model.predict_forward(image=image, text=instruction, tokenizer=tokenizer)
        if "same device" in msg or "found at least two devices" in msg:
            clear_sa2va_cache()
            model, tokenizer = load_sa2va_model(config)
            return model.predict_forward(image=image, text=instruction, tokenizer=tokenizer)
        raise
    finally:
        if prev_cuda_device is not None and torch.cuda.is_available() and prev_cuda_device != torch.cuda.current_device():
            torch.cuda.set_device(prev_cuda_device)


def ask_sa2va(image: Union[Image.Image, str], question: str, config: AmodalConfig, keep_on_gpu: bool = False) -> str:
    ensure_sa2va_on_device(config)
    load_sa2va_model(config)
    image_pil = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
    result = _predict_forward(image_pil, f"<image>{question}", config)
    if not keep_on_gpu:
        offload_sa2va_to_cpu()
    return str(result.get("prediction", "")).strip()


def segment_with_sa2va(
    image: Union[Image.Image, str, np.ndarray],
    text_prompt: str,
    config: AmodalConfig,
    keep_on_gpu: bool = False,
) -> Optional[np.ndarray]:
    ensure_sa2va_on_device(config)
    load_sa2va_model(config)
    if isinstance(image, str):
        image_pil = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image).convert("RGB")
    else:
        image_pil = image.convert("RGB")

    instruction = f"<image>Please segment {text_prompt}."
    try:
        result = _predict_forward(image_pil, instruction, config)
        prediction = str(result.get("prediction", ""))
        if "[SEG]" not in prediction or "prediction_masks" not in result:
            return None
        pred_mask = result["prediction_masks"][0]
        mask = np.squeeze(np.array(pred_mask))
        return mask > 0.5 if mask.dtype != bool else mask
    finally:
        if not keep_on_gpu:
            offload_sa2va_to_cpu()


def segment_to_uint8(image: Union[Image.Image, str, np.ndarray], text_prompt: str, config: AmodalConfig) -> Optional[np.ndarray]:
    mask = segment_with_sa2va(image, text_prompt, config=config)
    if mask is None:
        return None
    return mask.astype(np.uint8) * 255


def segment_all_objects(
    image: Union[Image.Image, str],
    names: list[str],
    config: AmodalConfig,
) -> dict[str, np.ndarray]:
    """Segment all objects in ONE forward pass (image encoded once).

    Returns {name: uint8_mask} for each object that was successfully segmented.
    Falls back to per-object calls for any names that didn't get a mask.
    """
    if not names:
        return {}

    ensure_sa2va_on_device(config)
    load_sa2va_model(config)

    if isinstance(image, str):
        image_pil = Image.open(image).convert("RGB")
    else:
        image_pil = image.convert("RGB")

    # Ask model to output [SEG] after each object name, in order.
    names_str = ", ".join(names)
    instruction = (
        f"<image>Segment each object and output [SEG] right after its name: {names_str}."
    )

    try:
        result = _predict_forward(image_pil, instruction, config)
    finally:
        offload_sa2va_to_cpu()

    prediction = str(result.get("prediction", ""))
    raw_masks = result.get("prediction_masks", [])

    if not raw_masks:
        return {}

    # Map each returned mask to an object name by parsing the output text.
    # Split on [SEG] — the text segment before each [SEG] tells us which object it belongs to.
    name_to_mask = _map_masks_to_names(prediction, names, raw_masks)

    # Fall back to individual calls for any names that didn't get a mask.
    missing = [n for n in names if n not in name_to_mask]
    if missing:
        for name in missing:
            mask = segment_with_sa2va(image_pil, name, config, keep_on_gpu=False)
            if mask is not None and (mask > 0).sum() > 0:
                name_to_mask[name] = mask.astype(np.uint8) * 255

    return name_to_mask


def _map_masks_to_names(prediction: str, names: list[str], raw_masks: list) -> dict[str, np.ndarray]:
    """Associate each [SEG] mask with an object name by parsing the output text."""
    import re

    # Split on [SEG] tokens — parts[i] is the text BEFORE the i-th [SEG].
    parts = re.split(r"\[SEG\]", prediction, flags=re.IGNORECASE)
    result: dict[str, np.ndarray] = {}

    for i, part in enumerate(parts[:-1]):
        if i >= len(raw_masks):
            break
        # Find the name that appears closest (rightmost) before this [SEG].
        best_name: Optional[str] = None
        best_pos = -1
        for name in names:
            pos = part.lower().rfind(name.lower())
            if pos > best_pos:
                best_pos = pos
                best_name = name

        if best_name is None or best_name in result:
            continue

        mask_arr = np.squeeze(np.array(raw_masks[i]))
        mask_bool = (mask_arr > 0.5) if mask_arr.dtype != bool else mask_arr
        if mask_bool.sum() > 0:
            result[best_name] = mask_bool.astype(np.uint8) * 255

    return result


def discover_object_names(image: Union[Image.Image, str], config: AmodalConfig) -> list[str]:
    question = (
        "List the distinct visible objects in this image for segmentation. "
        "Return only JSON in this format: {\"objects\":[\"object name\", ...]}. "
        "Use concise specific object names. Do not include background-only regions."
    )
    text = ask_sa2va(image, question, config, keep_on_gpu=False)
    names = _parse_object_names(text)
    clean = []
    seen = set()
    for name in names:
        token = re.sub(r"\s+", " ", str(name).strip().lower())
        if not token or token in seen:
            continue
        seen.add(token)
        clean.append(token)
        if len(clean) >= config.max_objects:
            break
    return clean


def _parse_object_names(text: str) -> list[str]:
    raw = str(text).strip()
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except Exception:
                data = None
        else:
            data = None
    if isinstance(data, dict):
        values = data.get("objects") or data.get("object_names") or data.get("items")
        if isinstance(values, list):
            return [str(v.get("name", v)) if isinstance(v, dict) else str(v) for v in values]
    if isinstance(data, list):
        return [str(v.get("name", v)) if isinstance(v, dict) else str(v) for v in data]

    # Fallback for numbered/bulleted SA2VA answers.
    lines = re.split(r"[\n,;]+", raw)
    names = []
    for line in lines:
        item = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
        item = re.sub(r"^(objects?:|visible objects?:)", "", item, flags=re.I).strip()
        item = item.strip("\"'[]{} ")
        if item:
            names.append(item)
    return names
