"""VLM-based completion check: ask if target is fully visible, expand if not."""
import base64
import json
import os
import re
import tempfile

from PIL import Image

from ..config import AmodalConfig
from ..common.io_utils import save_mask, write_json
from ..canvas.preprocess import compute_expansion_directions
from ..segmentation.sa2va import clear_sa2va_cache, segment_to_uint8
from ..types import BoundaryVerification


def check_boundary_completion(
    completed_path: str,
    target_name: str,
    out_dir: str,
    config: AmodalConfig,
    retry_index: int,
) -> tuple[BoundaryVerification, str]:
    """
    1. VLM decides true/false: is the target fully completed?
    2. If false, SA2VA segments the target → compute_expansion_directions → expand by image_size * initial_ratio.
    """
    # ── Step 1: VLM true/false ────────────────────────────────────────────────
    is_complete = _ask_vlm_completion(completed_path, target_name, config)

    if is_complete:
        verification = BoundaryVerification(
            boundary_complete=True,
            expand_pixels={},
            reason="target fully completed",
        )
        path = write_json(verification.to_dict(), os.path.join(out_dir, f"15_boundary_verification_{retry_index:02d}.json"))
        print(f"[Verifier] complete=True")
        return verification, path

    # ── Step 2: SA2VA segments target → directions ────────────────────────────
    mask = segment_to_uint8(completed_path, target_name, config)

    if config.aggressive_model_unload:
        clear_sa2va_cache()

    if mask is None or int((mask > 0).sum()) == 0:
        verification = BoundaryVerification(
            boundary_complete=True,
            expand_pixels={},
            reason="target not found in completed image",
        )
        path = write_json(verification.to_dict(), os.path.join(out_dir, f"15_boundary_verification_{retry_index:02d}.json"))
        return verification, path

    save_mask(mask, os.path.join(out_dir, f"15_boundary_verification_{retry_index:02d}_mask.png"))

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    save_mask(mask, tmp.name)
    directions = compute_expansion_directions(tmp.name, threshold=config.boundary_touch_threshold)
    os.unlink(tmp.name)

    if not directions:
        verification = BoundaryVerification(
            boundary_complete=True,
            expand_pixels={},
            reason="VLM flagged incomplete but mask not near any edge — stopping",
        )
        path = write_json(verification.to_dict(), os.path.join(out_dir, f"15_boundary_verification_{retry_index:02d}.json"))
        print(f"[Verifier] VLM=incomplete but mask not at edge, stopping")
        return verification, path

    # ── Step 3: expand by image_size * initial_ratio (same as plan_canvas_expansion) ──
    w, h = Image.open(completed_path).size
    ratio = config.boundary_initial_expand_ratio

    expand_pixels = {
        "top":    int(round(h * ratio)) if "top"    in directions else 0,
        "bottom": int(round(h * ratio)) if "bottom" in directions else 0,
        "left":   int(round(w * ratio)) if "left"   in directions else 0,
        "right":  int(round(w * ratio)) if "right"  in directions else 0,
    }

    verification = BoundaryVerification(
        boundary_complete=False,
        expand_pixels=expand_pixels,
        reason=f"mask touches {directions}: expanding by {expand_pixels}",
    )
    path = write_json(verification.to_dict(), os.path.join(out_dir, f"15_boundary_verification_{retry_index:02d}.json"))
    print(f"[Verifier] complete=False, expanding {directions}")
    return verification, path


def _ask_vlm_completion(completed_path: str, target_name: str, config: AmodalConfig) -> bool:
    """Ask VLM if the target is fully visible. Returns True/False. Falls back to code on error."""
    import requests

    try:
        img_b64 = base64.b64encode(open(completed_path, "rb").read()).decode()
        text = (
            f'Is the "{target_name}" in this image fully visible with no parts cut off at the edges? '
            f'Return ONLY JSON: {{"complete": true}} or {{"complete": false}}'
        )
        resp = requests.post(
            f"http://{config.open_llm_host}:{config.open_llm_port}/v1/chat/completions",
            json={
                "model": config.open_llm_model,
                "max_tokens": 20,
                "messages": [
                    {"role": "system", "content": "You are a vision assistant. Return only JSON."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": text},
                    ]},
                ],
            },
            timeout=30,
        )
        raw = re.sub(r"<think>.*?</think>", "", resp.json()["choices"][0]["message"]["content"], flags=re.DOTALL).strip()
        s, e = raw.find("{"), raw.rfind("}")
        if s >= 0 and e > s:
            return bool(json.loads(raw[s:e+1]).get("complete", False))

    except Exception as ex:
        print(f"[Verifier] VLM check failed: {ex}, falling back to code")
        return _code_based_completion(completed_path, target_name, config)

    return False


def _code_based_completion(completed_path: str, target_name: str, config: AmodalConfig) -> bool:
    """Fallback: check if target mask touches any edge."""
    mask = segment_to_uint8(completed_path, target_name, config)
    if config.aggressive_model_unload:
        clear_sa2va_cache()
    if mask is None or int((mask > 0).sum()) == 0:
        return True
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    save_mask(mask, tmp.name)
    directions = compute_expansion_directions(tmp.name, threshold=config.boundary_touch_threshold)
    os.unlink(tmp.name)
    return len(directions) == 0
