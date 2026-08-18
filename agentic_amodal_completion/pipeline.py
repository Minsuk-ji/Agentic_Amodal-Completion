"""Top-level pipeline orchestration — wires all six stages together.

Stage order: segmentation → occlusion → canvas (boundary check + expansion) →
planning (prompt generation) → inpainting → verification (retry loop).
"""
import os
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .config import AmodalConfig
from .common.io_utils import ensure_dir, load_mask, mask_to_white_bg_crop, overlay_mask, save_mask, write_json
from .segmentation.target import segment_target
from .segmentation.sa2va import segment_all_objects, segment_to_uint8
from .segmentation.sam_refiner import refine_mask, refine_masks
from .occlusion.instaorder import filter_occluders
from .canvas.preprocess import (
    apply_canvas_expansion,
    build_canvas_expansion_mask,
    build_final_inpaint_mask,
    build_target_clean,
    expand_mask_to_canvas,
)
from .planning.vlm_planning import detect_entities, generate_description, plan_canvas_expansion, plan_canvas_expansion_vlm
from .inpainting.flux_inpaint import run_flux_inpaint
from .verification.verifier import check_boundary_completion
from .types import BoundaryExpansion, PipelineResult


def run_amodal_pipeline(image_path: str, instruction: str, config: AmodalConfig) -> PipelineResult:
    start = time.time()
    save_dir = ensure_dir(os.path.join(config.save_dir, config.image_index))
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    input_copy = os.path.join(save_dir, "00_input.png")
    image.save(input_copy)

    # ── Step 1: Segment target → visible mask ────────────────────────────────
    target = segment_target(input_copy, save_dir, config, instruction=instruction)
    if target.mask_path and os.path.exists(target.mask_path):
        coarse = np.array(Image.open(target.mask_path).convert("L"))
        refined = refine_mask(input_copy, coarse, config)
        Image.fromarray(refined).save(target.mask_path)

    target_bin = np.zeros((h, w), dtype=np.uint8)
    if target.mask_path and os.path.exists(target.mask_path):
        target_bin = (np.array(Image.open(target.mask_path).convert("L")) > 127).astype(np.uint8)

    # ── Step 2: VLM entity detection ─────────────────────────────────────────
    if config.ablation_no_vlm_entity:
        entities = []
        print("[Ablation] Skipping VLM entity detection → empty occluder list")
    else:
        entities = detect_entities(input_copy, target.name, config)

    # ── Step 3: SA2VA segmentation ────────────────────────────────────────────
    entity_masks = segment_all_objects(image, entities, config) if entities else {}

    # ── Step 3.5: SAMRefiner — refine entity masks ────────────────────────────
    if entity_masks:
        entity_masks = refine_masks(input_copy, entity_masks, config)

    # ── Step 4: InstaOrder occluder filtering ─────────────────────────────────
    confirmed_occluders = filter_occluders(image, target_bin, entity_masks, config)

    # ── Step 5: Build occluder union mask ─────────────────────────────────────
    occ_mask_path = _build_occ_mask(confirmed_occluders, (w, h), save_dir)

    # ── Step 6: Build target-clean ────────────────────────────────────────────
    target_clean_path = build_target_clean(input_copy, target, save_dir, occ_mask_path)

    # ── Step 7: VLM canvas expansion (true/false + bbox boundary check) ───────
    if config.ablation_no_adaptive_canvas:
        expansion, expansion_path = plan_canvas_expansion(input_copy, target_clean_path, target, save_dir, config)
        print("[Ablation] Using fixed bbox-based canvas expansion, no VLM + no retry loop")
    else:
        expansion, expansion_path = plan_canvas_expansion_vlm(input_copy, target, save_dir, config)

    # ── Step 8: Build expanded canvas + masks ────────────────────────────────
    expanded_input_path = os.path.join(save_dir, "09_I_input.png")
    apply_canvas_expansion(target_clean_path, expansion, expanded_input_path)

    canvas_mask_path = os.path.join(save_dir, "10_M_canvas_expansion.png")
    build_canvas_expansion_mask(image.size, expansion, canvas_mask_path)

    expanded_occ_path = os.path.join(save_dir, "11_M_occ_expanded_canvas.png")
    expand_mask_to_canvas(occ_mask_path, image.size, expansion, expanded_occ_path)

    inpaint_mask_path = os.path.join(save_dir, "12_M_inpaint.png")
    build_final_inpaint_mask(expanded_occ_path, canvas_mask_path, inpaint_mask_path, config)

    # ── Step 9: Generate inpainting description ───────────────────────────────
    description, negative_prompt, description_path = generate_description(target, save_dir, config)

    # ── Step 10: Flux Fill inpaint ────────────────────────────────────────────
    completed_path: Optional[str] = None
    completed_target_rgba_path: Optional[str] = None
    boundary_verifications: list[dict] = []

    if config.run_inpainting:
        completed_path = run_flux_inpaint(
            expanded_input_path, inpaint_mask_path, description,
            os.path.join(save_dir, "14_I_comp.png"), config,
            negative_prompt=negative_prompt,
        )
        boundary_expand_fn = (lambda **kw: (kw["completed_path"], [], kw["expansion"])) \
            if config.ablation_no_adaptive_canvas else _boundary_expand_loop
        completed_path, boundary_verifications, expansion = boundary_expand_fn(
            completed_path=completed_path,
            image_size=image.size,
            target_clean_path=target_clean_path,
            occ_mask_path=occ_mask_path,
            expansion=expansion,
            description=description,
            negative_prompt=negative_prompt,
            target_name=target.name,
            save_dir=save_dir,
            config=config,
        )
        completed_target_rgba_path = _save_target_rgba(completed_path, target.name, save_dir, config)

    result = PipelineResult(
        input_image=input_copy,
        instruction=instruction,
        save_dir=save_dir,
        occluder_plan_path=os.path.join(save_dir, "03_occluder_plan.json"),
        target_clean_path=expanded_input_path,
        occluder_mask_path=expanded_occ_path,
        inpaint_mask_path=inpaint_mask_path,
        description_path=description_path,
        completed_path=completed_path,
        completed_target_rgba_path=completed_target_rgba_path,
        debug={
            "elapsed_seconds": round(time.time() - start, 3),
            "target": target.to_dict(),
            "confirmed_occluders": list(confirmed_occluders.keys()),
            "canvas_expansion_path": expansion_path,
            "canvas_expansion": expansion.to_dict(),
            "boundary_verifications": boundary_verifications,
            "run_inpainting": config.run_inpainting,
            "inpaint_backend": config.inpaint_backend,
        },
    )
    write_json(result.to_dict(), os.path.join(save_dir, "result.json"))
    return result


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_occ_mask(
    confirmed_occluders: dict,
    image_size: tuple[int, int],
    save_dir: str,
) -> str:
    w, h = image_size
    occ_union = np.zeros((h, w), dtype=np.uint8)
    for mask_bin in confirmed_occluders.values():
        m = mask_bin if mask_bin.shape == (h, w) else cv2.resize(
            mask_bin.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        occ_union = np.maximum(occ_union, (m > 0).astype(np.uint8) * 255)
    print(f"[Pipeline] Occ mask pixels: {(occ_union > 0).sum()}")
    return save_mask(occ_union, os.path.join(save_dir, "05_M_occ.png"))


def _boundary_expand_loop(
    *,
    completed_path: str,
    image_size: tuple[int, int],
    target_clean_path: str,
    occ_mask_path: str,
    expansion: BoundaryExpansion,
    description: str,
    negative_prompt: str = "",
    target_name: str,
    save_dir: str,
    config: AmodalConfig,
) -> tuple[str, list[dict], BoundaryExpansion]:
    verifications: list[dict] = []
    current_completed = completed_path
    current_expansion = BoundaryExpansion(**expansion.to_runtime_dict())

    for retry_idx in range(int(config.max_final_boundary_retries) + 1):
        verification, vpath = check_boundary_completion(
            current_completed, target_name, save_dir, config, retry_idx
        )
        item = verification.to_dict()
        item["path"] = vpath
        item["completed_path"] = current_completed
        verifications.append(item)

        if verification.boundary_complete or not any(v > 0 for v in verification.expand_pixels.values()):
            break
        if retry_idx >= int(config.max_final_boundary_retries):
            break

        current_expansion = _apply_pixel_expansion(current_expansion, verification.expand_pixels, image_size)

        expand_dir = ensure_dir(os.path.join(save_dir, f"expand_{retry_idx + 1:02d}"))
        expand_input = os.path.join(expand_dir, "09_I_input.png")
        apply_canvas_expansion(target_clean_path, current_expansion, expand_input)

        expand_canvas_mask = os.path.join(expand_dir, "10_M_canvas_expansion.png")
        build_canvas_expansion_mask(image_size, current_expansion, expand_canvas_mask)

        expand_occ = os.path.join(expand_dir, "11_M_occ_expanded_canvas.png")
        expand_mask_to_canvas(occ_mask_path, image_size, current_expansion, expand_occ)

        expand_inpaint = os.path.join(expand_dir, "12_M_inpaint.png")
        build_final_inpaint_mask(expand_occ, expand_canvas_mask, expand_inpaint, config)

        current_completed = run_flux_inpaint(
            expand_input, expand_inpaint, description,
            os.path.join(expand_dir, "14_I_comp.png"), config,
            negative_prompt=negative_prompt,
        )

    return current_completed, verifications, current_expansion


def _apply_pixel_expansion(
    expansion: BoundaryExpansion,
    expand_pixels: dict[str, int],
    image_size: tuple[int, int],
) -> BoundaryExpansion:
    w, h = image_size
    out = BoundaryExpansion(**expansion.to_runtime_dict())
    out.top += expand_pixels.get("top", 0)
    out.bottom += expand_pixels.get("bottom", 0)
    out.left += expand_pixels.get("left", 0)
    out.right += expand_pixels.get("right", 0)
    out.top_ratio = out.top / max(1, h)
    out.bottom_ratio = out.bottom / max(1, h)
    out.left_ratio = out.left / max(1, w)
    out.right_ratio = out.right / max(1, w)
    out.need_expand = (out.top + out.bottom + out.left + out.right) > 0
    out.reason = (out.reason + f" | expanded by {expand_pixels}").strip(" |")
    return out


def _save_target_rgba(
    completed_path: str,
    target_name: str,
    save_dir: str,
    config: AmodalConfig,
) -> Optional[str]:
    mask = segment_to_uint8(completed_path, target_name, config)
    if mask is None:
        write_json(
            {"target_name": target_name, "error": "SA2VA returned no target mask."},
            os.path.join(save_dir, "16_I_comp_target_rgba_error.json"),
        )
        return None

    save_mask(mask, os.path.join(save_dir, "16_I_comp_target_mask.png"))
    image = Image.open(completed_path).convert("RGB")
    alpha = Image.fromarray(((mask > 0).astype("uint8")) * 255, mode="L").resize(image.size, Image.NEAREST)
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    out_path = os.path.join(save_dir, "16_I_comp_target_rgba.png")
    rgba.save(out_path)

    # white-bg RGB crop (used for evaluation and BoN scoring)
    mask_to_white_bg_crop(image, mask, os.path.join(save_dir, "16_I_comp_target_crop.png"))

    return out_path
