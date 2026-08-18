import os
from dataclasses import dataclass


PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))

# Third-party model repos (InstaOrder, FLUX-Controlnet-Inpainting, SAMRefiner) are not
# vendored in this repository — clone them under `external/` (see README) or point the
# corresponding *_dir / *_root env vars at wherever you keep them.
EXTERNAL_DIR = os.getenv("AGENTIC_AMODAL_EXTERNAL_DIR", os.path.join(REPO_DIR, "external"))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass
class AmodalConfig:
    save_dir: str = "results"
    image_index: str = "amodal_000"
    seed: int = 42
    default_gpu_id: int = int(os.getenv("AGENTIC_AMODAL_GPU_ID", "0"))

    # VLM planning: either a local OpenAI-compatible server (e.g. vLLM serving a Qwen VL
    # model) or the OpenAI API.
    open_llm: bool = False
    open_llm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    open_llm_host: str = "0.0.0.0"
    open_llm_port: str = "8000"
    gpt_model: str = "gpt-4o-mini"

    # Sa2VA grounded segmentation (target + scene entities).
    sa2va_model_id: str = os.getenv("AGENTIC_AMODAL_SA2VA_MODEL", "ByteDance/Sa2VA-8B")
    sa2va_device: str = os.getenv("AGENTIC_AMODAL_SA2VA_DEVICE", "")
    sa2va_max_new_tokens: int = int(os.getenv("AGENTIC_AMODAL_SA2VA_MAX_NEW_TOKENS", "256"))
    sa2va_max_dynamic_patch: int = int(os.getenv("AGENTIC_AMODAL_SA2VA_MAX_DYNAMIC_PATCH", "4"))
    sa2va_oom_retry_max_new_tokens: int = int(os.getenv("AGENTIC_AMODAL_SA2VA_OOM_TOKENS", "96"))
    sa2va_oom_retry_max_dynamic_patch: int = int(os.getenv("AGENTIC_AMODAL_SA2VA_OOM_PATCH", "3"))
    max_objects: int = int(os.getenv("AGENTIC_AMODAL_MAX_OBJECTS", "20"))

    # Mask post-processing.
    inpaint_dilation_kernel: int = int(os.getenv("AGENTIC_AMODAL_DILATION_KERNEL", "4"))
    inpaint_dilation_iterations: int = int(os.getenv("AGENTIC_AMODAL_DILATION_ITERS", "0"))
    max_refine_steps: int = int(os.getenv("AGENTIC_AMODAL_MAX_REFINE_STEPS", "2"))
    max_final_boundary_retries: int = int(os.getenv("AGENTIC_AMODAL_MAX_FINAL_BOUNDARY_RETRIES", "3"))
    boundary_touch_threshold: int = int(os.getenv("AGENTIC_AMODAL_BOUNDARY_THRESHOLD", "10"))
    boundary_initial_expand_ratio: float = float(os.getenv("AGENTIC_AMODAL_INITIAL_EXPAND_RATIO", "0.15"))
    boundary_min_expand_ratio: float = float(os.getenv("AGENTIC_AMODAL_MIN_EXPAND_RATIO", "0.3"))

    # SAMRefiner — iterative SAM-prompted refinement of coarse Sa2VA masks.
    sam_refiner_root: str = os.getenv("AGENTIC_AMODAL_SAM_REFINER_ROOT", os.path.join(EXTERNAL_DIR, "SAMRefiner"))
    sam_checkpoint: str = os.getenv(
        "AGENTIC_AMODAL_SAM_CKPT", os.path.join(EXTERNAL_DIR, "checkpoints", "sam_vit_h_4b8939.pth")
    )
    sam_model_type: str = os.getenv("AGENTIC_AMODAL_SAM_MODEL_TYPE", "vit_h")
    sam_refiner_iters: int = int(os.getenv("AGENTIC_AMODAL_SAM_REFINER_ITERS", "5"))

    # Flux inpainting.
    run_inpainting: bool = _env_bool("AGENTIC_AMODAL_RUN_INPAINT", True)
    inpaint_backend: str = os.getenv("AGENTIC_AMODAL_INPAINT_BACKEND", "flux_fill")
    flux_controlnet_id: str = os.getenv(
        "AGENTIC_AMODAL_FLUX_CONTROLNET",
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
    )
    flux_base_id: str = os.getenv("AGENTIC_AMODAL_FLUX_BASE", "black-forest-labs/FLUX.1-dev")
    flux_fill_id: str = os.getenv("AGENTIC_AMODAL_FLUX_FILL", "black-forest-labs/FLUX.1-Fill-dev")
    flux_module_dir: str = os.getenv(
        "AGENTIC_AMODAL_FLUX_MODULE_DIR",
        os.path.join(EXTERNAL_DIR, "FLUX-Controlnet-Inpainting"),
    )
    flux_device: str = os.getenv("AGENTIC_AMODAL_FLUX_DEVICE", "")
    flux_cpu_offload: bool = _env_bool("AGENTIC_AMODAL_FLUX_CPU_OFFLOAD", True)
    flux_size: int = int(os.getenv("AGENTIC_AMODAL_FLUX_SIZE", "768"))
    flux_steps: int = int(os.getenv("AGENTIC_AMODAL_FLUX_STEPS", "28"))
    flux_controlnet_conditioning_scale: float = float(os.getenv("AGENTIC_AMODAL_FLUX_CN_SCALE", "0.9"))
    flux_guidance_scale: float = float(os.getenv("AGENTIC_AMODAL_FLUX_GUIDANCE", "3.5"))
    flux_true_guidance_scale: float = float(os.getenv("AGENTIC_AMODAL_FLUX_TRUE_GUIDANCE", "1.0"))
    flux_fill_steps: int = int(os.getenv("AGENTIC_AMODAL_FLUX_FILL_STEPS", "50"))
    flux_fill_guidance_scale: float = float(os.getenv("AGENTIC_AMODAL_FLUX_FILL_GUIDANCE", "30.0"))
    flux_fill_max_sequence_length: int = int(os.getenv("AGENTIC_AMODAL_FLUX_FILL_MAX_SEQ", "512"))
    flux_negative_prompt: str = os.getenv("AGENTIC_AMODAL_FLUX_NEGATIVE", "")
    aggressive_model_unload: bool = _env_bool("AGENTIC_AMODAL_AGGRESSIVE_UNLOAD", True)

    # InstaOrder occlusion-order filtering.
    instaorder_dir: str = os.getenv("AGENTIC_AMODAL_INSTAORDER_DIR", os.path.join(EXTERNAL_DIR, "InstaOrder"))
    instaorder_ckpt: str = os.getenv(
        "AGENTIC_AMODAL_INSTAORDER_CKPT",
        os.path.join(EXTERNAL_DIR, "InstaOrder", "InstaOrder_ckpt", "InstaOrder_InstaOrderNet_o.pth.tar"),
    )
    instaorder_input_size: int = int(os.getenv("AGENTIC_AMODAL_INSTAORDER_SIZE", "256"))

    # Ablation flags — toggle individual stages off/to a naive baseline to reproduce the
    # ablation study (see README "Method" section).
    ablation_no_vlm_entity: bool = False        # Skip VLM entity detection → empty occluder mask
    ablation_unidirectional_instaorder: bool = False  # Unidirectional InstaOrder (baseline style)
    ablation_no_adaptive_canvas: bool = False   # Fixed bbox-based expansion, no VLM + no retry loop
    ablation_plain_prompt: bool = False         # Use plain label as inpaint prompt, no VLM
