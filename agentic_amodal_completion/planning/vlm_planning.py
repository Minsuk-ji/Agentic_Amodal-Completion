"""VLM-based planning: scene-entity detection, canvas expansion, inpainting prompt.

All calls go to an OpenAI-compatible chat-completions endpoint (a local vLLM server or
the OpenAI API), configured via `config.open_llm_host`/`open_llm_port`/`open_llm_model`.
"""
import os
import re

from PIL import Image

from ..config import AmodalConfig
from ..common.io_utils import write_json
from ..canvas.preprocess import compute_expansion_directions
from ..types import BoundaryExpansion, ObjectItem  # noqa: F401 ObjectItem kept for callers


# ── VLM entity detection ────────────────────────────────────────────────────

def detect_entities(
    image_path: str,
    target_name: str,
    config: AmodalConfig,
) -> list[str]:
    """Ask VLM to list all entities relevant for occlusion analysis. Returns list excluding target."""
    import base64
    import requests

    img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    text = (
        f'Detect semantically meaningful entities in this image for occlusion-aware scene analysis. '
        f'The target object is "{target_name}".\n'
        f'Include: 1. Independent object instances (things). '
        f'2. Non-instance regions (stuff) only if they occlude an object or define its visible extent.\n'
        f'Do not include fine-grained parts. Return a comma-separated list only.'
    )
    try:
        resp = requests.post(
            f"http://{config.open_llm_host}:{config.open_llm_port}/v1/chat/completions",
            json={
                "model": config.open_llm_model,
                "max_tokens": 150,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "You are a vision assistant. Return only a comma-separated list of entity names."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": text},
                    ]},
                ],
            },
            timeout=30,
        )
        raw = re.sub(r"<think>.*?</think>", "", resp.json()["choices"][0]["message"]["content"], flags=re.DOTALL).strip()
        all_entities = [e.strip().lower() for e in raw.split(",") if e.strip()]
        entities = [e for e in all_entities if e != target_name.lower()]
        print(f"[Planning] Entities detected: {entities}")
        return entities
    except Exception as ex:
        print(f"[Planning] Entity detection failed: {ex}")
        return []


# ── VLM canvas expansion (true/false + bbox boundary check) ─────────────────

def plan_canvas_expansion_vlm(
    image_path: str,
    target,
    out_dir: str,
    config: AmodalConfig,
) -> tuple["BoundaryExpansion", str]:
    """Ask VLM if target is fully visible. If not, check bbox boundary touch → 15% expand."""
    import base64
    import io
    import json as _json
    import requests
    import numpy as np
    from PIL import ImageDraw

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    if not target.mask_path or not os.path.exists(target.mask_path):
        exp = BoundaryExpansion(need_expand=False, reason="no target mask")
        path = write_json(exp.to_dict(), os.path.join(out_dir, "08_canvas_expansion_plan.json"))
        return exp, path

    mask_arr = np.array(Image.open(target.mask_path).convert("L"))
    ys, xs = np.where(mask_arr > 127)
    if len(xs) == 0:
        exp = BoundaryExpansion(need_expand=False, reason="empty target mask")
        path = write_json(exp.to_dict(), os.path.join(out_dir, "08_canvas_expansion_plan.json"))
        return exp, path

    tx1, ty1, tx2, ty2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    annot = image.copy()
    draw = ImageDraw.Draw(annot)
    lw = max(2, int(min(w, h) * 0.004))
    draw.rectangle([tx1, ty1, tx2, ty2], outline=(255, 0, 0), width=lw)
    buf = io.BytesIO()
    annot.save(buf, format="PNG")
    annot_b64 = base64.b64encode(buf.getvalue()).decode()

    text = (
        f'The red bounding box marks the visible region of the "{target.name}" in this image.\n'
        f'Image size: {w}x{h} pixels. '
        f'Bounding box: left={tx1}, top={ty1}, right={tx2}, bottom={ty2}.\n\n'
        f'Is the complete body of the "{target.name}" fully visible in the image, '
        f'or is any part of its body cut off by the image boundary?\n'
        f'Answer in JSON:\n'
        f'{{"is_complete": true, "reason": ""}}\n'
        f'Return only JSON.'
    )
    is_complete = False
    reason = ""
    try:
        resp = requests.post(
            f"http://{config.open_llm_host}:{config.open_llm_port}/v1/chat/completions",
            json={
                "model": config.open_llm_model,
                "max_tokens": 80,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "You are a vision assistant. Return only valid JSON."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{annot_b64}"}},
                        {"type": "text", "text": text},
                    ]},
                ],
            },
            timeout=30,
        )
        raw = re.sub(r"<think>.*?</think>", "", resp.json()["choices"][0]["message"]["content"], flags=re.DOTALL).strip()
        parsed = _json.loads(re.sub(r"```json|```", "", raw).strip())
        is_complete = bool(parsed.get("is_complete", False))
        reason = parsed.get("reason", "")
        print(f"[Planning] Canvas expansion VLM: is_complete={is_complete}, reason={reason!r}")
    except Exception as ex:
        print(f"[Planning] Canvas expansion VLM failed: {ex}, assuming incomplete")

    top = right = bottom = left = 0
    if not is_complete:
        tw, th = w * 0.02, h * 0.02
        ratio = config.boundary_initial_expand_ratio
        if tx1 <= tw:          left   = int(w * ratio)
        if tx2 >= w - tw:      right  = int(w * ratio)
        if ty1 <= th:          top    = int(h * ratio)
        if ty2 >= h - th:      bottom = int(h * ratio)

    need = (left + right + top + bottom) > 0
    exp = BoundaryExpansion(
        need_expand=need,
        top=top, bottom=bottom, left=left, right=right,
        top_ratio=top / h if h else 0.0,
        bottom_ratio=bottom / h if h else 0.0,
        left_ratio=left / w if w else 0.0,
        right_ratio=right / w if w else 0.0,
        reason=f"VLM: is_complete={is_complete}. {reason}",
    )
    path = write_json(exp.to_dict(), os.path.join(out_dir, "08_canvas_expansion_plan.json"))
    return exp, path


# ── Occluder detection (legacy) ───────────────────────────────────────────────

def ask_has_occluder(
    image_path: str,
    target: ObjectItem,
    save_dir: str,
    config: AmodalConfig,
) -> tuple[bool, list[str], str]:
    """Ask VLM whether any object occludes the target.

    Returns (has_occluder, occluder_names, saved_json_path).
    """
    import base64
    import io
    import requests
    import numpy as np
    from PIL import ImageDraw

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # annotate target bbox on image
    img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    annot_b64 = img_b64
    if target.mask_path and os.path.exists(target.mask_path):
        mask_arr = np.array(Image.open(target.mask_path).convert("L"))
        ys, xs = np.where(mask_arr > 127)
        if len(xs) > 0:
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            annot = image.copy()
            draw = ImageDraw.Draw(annot)
            lw = max(2, int(min(w, h) * 0.005))
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=lw)
            annot.save(os.path.join(save_dir, "03_occ_annot.png"))
            buf = io.BytesIO()
            annot.save(buf, format="PNG")
            annot_b64 = base64.b64encode(buf.getvalue()).decode()

    text = (
        f'The red bounding box marks the visible region of the "{target.name}". '
        f'Are there any objects in this image that are occluding or blocking the {target.name}, '
        f'meaning they are in front of it and hiding part of its body? '
        f'If yes, reply with a comma-separated list of those object names only. '
        f'If no objects are occluding it, reply with "none". '
        f'Reply with object names or "none" only, no other text.'
    )

    try:
        resp = requests.post(
            f"http://{config.open_llm_host}:{config.open_llm_port}/v1/chat/completions",
            json={
                "model": config.open_llm_model,
                "max_tokens": 80,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "You are a vision assistant specializing in occlusion analysis."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{annot_b64}"}},
                        {"type": "text", "text": text},
                    ]},
                ],
            },
            timeout=30,
        )
        raw = re.sub(r"<think>.*?</think>", "",
                     resp.json()["choices"][0]["message"]["content"],
                     flags=re.DOTALL).strip()
        print(f"[Planning] Occluder VLM raw: {raw!r}")
    except Exception as ex:
        print(f"[Planning] Occluder VLM failed: {ex}, assuming no occluder")
        raw = "none"

    if raw.lower().rstrip(".") == "none" or not raw:
        has_occluder = False
        names: list[str] = []
    else:
        has_occluder = True
        target_lower = target.name.lower()
        seen: set[str] = set()
        names = []
        for token in raw.split(","):
            n = token.strip().strip('"\'').lower()
            if not n or n == "none" or target_lower in n or n in seen:
                continue
            seen.add(n)
            names.append(n)
        has_occluder = len(names) > 0

    result = {"has_occluder": has_occluder, "occluder_names": names, "vlm_raw": raw}
    print(f"[Planning] Occluder detection: has={has_occluder}, names={names}")
    path = write_json(result, os.path.join(save_dir, "03_occluder_plan.json"))
    return has_occluder, names, path


# ── Canvas expansion planning ────────────────────────────────────────────────

def plan_canvas_expansion(
    image_path: str,
    target_clean_path: str,
    target: ObjectItem,
    out_dir: str,
    config: AmodalConfig,
) -> tuple[BoundaryExpansion, str]:
    directions = compute_expansion_directions(target.mask_path, threshold=config.boundary_touch_threshold)

    if not directions:
        exp = BoundaryExpansion(need_expand=False, reason="target mask not near any edge")
        path = write_json(exp.to_dict(), os.path.join(out_dir, "08_canvas_expansion_plan.json"))
        return exp, path

    image = Image.open(image_path)
    w, h = image.size
    ratio = config.boundary_initial_expand_ratio

    top    = int(round(h * ratio)) if "top"    in directions else 0
    bottom = int(round(h * ratio)) if "bottom" in directions else 0
    left   = int(round(w * ratio)) if "left"   in directions else 0
    right  = int(round(w * ratio)) if "right"  in directions else 0

    exp = BoundaryExpansion(
        need_expand=True,
        top=top, bottom=bottom, left=left, right=right,
        top_ratio=top / h if h else 0.0,
        bottom_ratio=bottom / h if h else 0.0,
        left_ratio=left / w if w else 0.0,
        right_ratio=right / w if w else 0.0,
        reason=(
            f"mask touches {directions}: "
            f"top={top}px bottom={bottom}px left={left}px right={right}px "
            f"(image-size based, ratio={ratio})"
        ),
    )
    path = write_json(exp.to_dict(), os.path.join(out_dir, "08_canvas_expansion_plan.json"))
    return exp, path


# ── Inpainting description ───────────────────────────────────────────────────

def generate_description(target: ObjectItem, out_dir: str, config: AmodalConfig) -> tuple[str, str, str]:
    """Use VLM to generate an inpainting prompt for the hidden region.

    Passes the original scene image + target_clean image (white holes = occluded parts).
    Asks the VLM to describe what should be filled into those white regions.

    Returns (prompt, "", saved_path).
    """
    import base64
    import requests

    image_path       = os.path.join(out_dir, "00_input.png")
    target_clean_path = os.path.join(out_dir, "04_I_target_clean.png")
    prompt_text = target.name  # fallback

    if config.ablation_plain_prompt:
        print(f"[Ablation] Using plain label as inpaint prompt: '{prompt_text}'")
        path = os.path.join(out_dir, "13_T_star.txt")
        with open(path, "w") as f:
            f.write(prompt_text + "\n")
        return prompt_text, "", path

    if os.path.exists(image_path) and os.path.exists(target_clean_path):
        try:
            orig_b64  = base64.b64encode(open(image_path,        "rb").read()).decode()
            clean_b64 = base64.b64encode(open(target_clean_path, "rb").read()).decode()

            text = (
                f'Image 1 is the original scene. '
                f'Image 2 shows the "{target.name}" with white regions where it is hidden by other objects. '
                f'Look at the VISIBLE (non-white) parts of the {target.name} in Image 1 and Image 2. '
                f'Write an inpainting prompt so a model can fill the hidden (white) areas consistently. '
                f'Format: "A {target.name}, [material, color, texture, surface patterns of the {target.name}]" '
                f'Example format: "A red-brick clock tower with ornate golden details, large circular clock faces, and a dome-shaped roof." '
                f'Max 25 words. Return only the prompt text, no JSON, no labels.'
            )
            resp = requests.post(
                f"http://{config.open_llm_host}:{config.open_llm_port}/v1/chat/completions",
                json={
                    "model": config.open_llm_model,
                    "max_tokens": 100,
                    "messages": [
                        {"role": "system", "content": "You are an image inpainting prompt writer. Return only the prompt text, nothing else."},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{orig_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{clean_b64}"}},
                            {"type": "text", "text": text},
                        ]},
                    ],
                },
                timeout=15,
            )
            raw = re.sub(
                r"<think>.*?</think>", "",
                resp.json()["choices"][0]["message"]["content"],
                flags=re.DOTALL,
            ).strip().strip('"\'')
            if raw:
                prompt_text = raw
            print(f"[Planning] Inpaint prompt: '{prompt_text}'")
        except Exception as ex:
            print(f"[Planning] Description generation failed: {ex}, falling back to '{target.name}'")

    path = os.path.join(out_dir, "13_T_star.txt")
    with open(path, "w") as f:
        f.write(prompt_text + "\n")
    return prompt_text, "", path
