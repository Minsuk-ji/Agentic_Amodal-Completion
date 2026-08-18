"""Flux-based inpainting backend: FLUX.1 ControlNet-Inpainting or FLUX.1-Fill.

`flux_fill` loads `black-forest-labs/FLUX.1-Fill-dev` directly from the Hugging Face Hub.
`controlnet` additionally requires the alimama-creative/FLUX-Controlnet-Inpainting repo's
module files on disk — see README and `config.flux_module_dir`.
"""
import os
import sys
from typing import Optional

from PIL import Image

from ..config import AmodalConfig
from ..common.device import gpu_cleanup, normalize_device, set_current_cuda_device


_pipe = None
_pipe_key = None


def unload_flux_pipeline() -> None:
    global _pipe, _pipe_key
    if _pipe is None:
        return
    del _pipe
    _pipe = None
    _pipe_key = None
    gpu_cleanup()


def load_flux_pipeline(config: AmodalConfig):
    global _pipe, _pipe_key
    backend = _normalize_backend(config.inpaint_backend)
    key = (
        backend,
        config.flux_fill_id,
        config.flux_controlnet_id,
        config.flux_base_id,
        config.flux_module_dir,
        config.flux_device,
        config.flux_cpu_offload,
    )
    if _pipe is not None and _pipe_key == key:
        return _pipe
    if _pipe is not None:
        unload_flux_pipeline()

    import torch

    device = normalize_device(config.flux_device or None, fallback_gpu_id=config.default_gpu_id)
    set_current_cuda_device(device, fallback_gpu_id=config.default_gpu_id)
    print(f"[FluxInpaint] Loading {backend} pipeline on {device}")
    if backend == "flux_fill":
        from diffusers import FluxFillPipeline

        pipe = FluxFillPipeline.from_pretrained(config.flux_fill_id, torch_dtype=torch.bfloat16)
    else:
        from diffusers.utils import check_min_version

        module_dir = config.flux_module_dir
        required = ["controlnet_flux.py", "pipeline_flux_controlnet_inpaint.py", "transformer_flux.py"]
        if not module_dir or not all(os.path.exists(os.path.join(module_dir, name)) for name in required):
            raise FileNotFoundError(
                "Flux ControlNet Inpainting module files were not found. "
                f"Expected {required} under: {module_dir}. "
                "Clone alimama-creative/FLUX-Controlnet-Inpainting under `external/` or pass "
                "--flux_module_dir /path/to/FLUX-Controlnet-Inpainting."
            )
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        from controlnet_flux import FluxControlNetModel
        from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
        from transformer_flux import FluxTransformer2DModel

        check_min_version("0.30.2")
        controlnet = FluxControlNetModel.from_pretrained(config.flux_controlnet_id, torch_dtype=torch.bfloat16)
        transformer = FluxTransformer2DModel.from_pretrained(
            config.flux_base_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        pipe = FluxControlNetInpaintingPipeline.from_pretrained(
            config.flux_base_id,
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe.transformer.to(torch.bfloat16)
        pipe.controlnet.to(torch.bfloat16)
    if config.flux_cpu_offload:
        if not hasattr(pipe, "enable_model_cpu_offload"):
            raise RuntimeError("This pipeline does not support enable_model_cpu_offload().")
        pipe.enable_model_cpu_offload(gpu_id=_cuda_index(device))
        print(f"[FluxInpaint] Enabled model CPU offload to {device}")
    else:
        pipe.to(device)
    _pipe = pipe
    _pipe_key = key
    return _pipe


def run_flux_inpaint(
    image_path: str,
    mask_path: str,
    prompt: str,
    output_path: str,
    config: AmodalConfig,
    negative_prompt: str = "",
    seed: Optional[int] = None,
) -> str:
    import torch

    pipe = load_flux_pipeline(config)
    device = normalize_device(config.flux_device or None, fallback_gpu_id=config.default_gpu_id)
    set_current_cuda_device(device, fallback_gpu_id=config.default_gpu_id)
    size = _target_size(Image.open(image_path).size, config.flux_size)
    image = Image.open(image_path).convert("RGB").resize(size, Image.LANCZOS)
    backend = _normalize_backend(config.inpaint_backend)
    _seed = seed if seed is not None else config.seed
    if backend == "flux_fill":
        mask = Image.open(mask_path).convert("L").resize(size, Image.NEAREST)
        generator = torch.Generator("cpu").manual_seed(_seed)
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=size[1],
            width=size[0],
            guidance_scale=config.flux_fill_guidance_scale,
            num_inference_steps=config.flux_fill_steps,
            max_sequence_length=config.flux_fill_max_sequence_length,
            generator=generator,
        ).images[0]
    else:
        mask = Image.open(mask_path).convert("RGB").resize(size, Image.NEAREST)
        generator = torch.Generator(device=device).manual_seed(_seed)
        result = pipe(
            prompt=prompt,
            height=size[1],
            width=size[0],
            control_image=image,
            control_mask=mask,
            num_inference_steps=config.flux_steps,
            generator=generator,
            controlnet_conditioning_scale=config.flux_controlnet_conditioning_scale,
            guidance_scale=config.flux_guidance_scale,
            negative_prompt=negative_prompt or config.flux_negative_prompt,
            true_guidance_scale=config.flux_true_guidance_scale,
        ).images[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    if config.aggressive_model_unload:
        unload_flux_pipeline()
    return output_path


def _target_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    width, height = size
    if width == height:
        return _multiple_of_16(max_side), _multiple_of_16(max_side)
    scale = float(max_side) / float(max(width, height))
    new_w = _multiple_of_16(width * scale)
    new_h = _multiple_of_16(height * scale)
    return max(16, new_w), max(16, new_h)


def _multiple_of_16(value: float) -> int:
    return int(round(float(value) / 16.0) * 16)


def _cuda_index(device: str) -> int:
    if str(device).startswith("cuda:"):
        return int(str(device).split(":", 1)[1])
    return 0


def _normalize_backend(value: str) -> str:
    raw = str(value or "flux_fill").strip().lower().replace("-", "_")
    if raw in {"controlnet", "control_net"}:
        return "controlnet"
    return "flux_fill"
