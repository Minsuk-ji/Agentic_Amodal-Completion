import argparse
import json

from .config import AmodalConfig
from .pipeline import run_amodal_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic amodal completion pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--save_dir", default="results")
    parser.add_argument("--image_index", default="amodal_000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default_gpu_id", type=int, default=0)
    parser.add_argument("--no_inpaint", action="store_true")

    parser.add_argument("--use_open_llm", action="store_true")
    parser.add_argument("--open_llm_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--open_llm_host", default="0.0.0.0")
    parser.add_argument("--open_llm_port", default="8000")
    parser.add_argument("--gpt_model", default="gpt-4o-mini")

    parser.add_argument("--sa2va_model_id", default="ByteDance/Sa2VA-8B")
    parser.add_argument("--sa2va_device", default="")

    parser.add_argument("--dilation_kernel", type=int, default=4)
    parser.add_argument("--dilation_iters", type=int, default=0)
    parser.add_argument("--max_refine_steps", type=int, default=2)
    parser.add_argument("--max_final_boundary_retries", type=int, default=3)
    parser.add_argument("--boundary_touch_threshold", type=int, default=10)

    parser.add_argument("--inpaint_backend", choices=["controlnet", "flux_fill"], default="flux_fill")
    parser.add_argument("--flux_module_dir", default="")
    parser.add_argument("--flux_controlnet_id", default="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha")
    parser.add_argument("--flux_base_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--flux_fill_id", default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--flux_device", default="")
    parser.add_argument("--disable_flux_cpu_offload", action="store_true")
    parser.add_argument("--flux_size", type=int, default=768)
    parser.add_argument("--flux_fill_steps", type=int, default=50)
    parser.add_argument("--flux_fill_guidance_scale", type=float, default=30.0)
    args = parser.parse_args()

    defaults = AmodalConfig()
    config = AmodalConfig(
        save_dir=args.save_dir,
        image_index=args.image_index,
        seed=args.seed,
        default_gpu_id=args.default_gpu_id,
        open_llm=args.use_open_llm,
        open_llm_model=args.open_llm_model,
        open_llm_host=args.open_llm_host,
        open_llm_port=args.open_llm_port,
        gpt_model=args.gpt_model,
        sa2va_model_id=args.sa2va_model_id,
        sa2va_device=args.sa2va_device,
        inpaint_dilation_kernel=args.dilation_kernel,
        inpaint_dilation_iterations=args.dilation_iters,
        max_refine_steps=args.max_refine_steps,
        max_final_boundary_retries=args.max_final_boundary_retries,
        boundary_touch_threshold=args.boundary_touch_threshold,
        run_inpainting=not args.no_inpaint,
        inpaint_backend=args.inpaint_backend,
        flux_module_dir=args.flux_module_dir or defaults.flux_module_dir,
        flux_controlnet_id=args.flux_controlnet_id,
        flux_base_id=args.flux_base_id,
        flux_fill_id=args.flux_fill_id,
        flux_device=args.flux_device,
        flux_cpu_offload=not args.disable_flux_cpu_offload,
        flux_size=args.flux_size,
        flux_fill_steps=args.flux_fill_steps,
        flux_fill_guidance_scale=args.flux_fill_guidance_scale,
    )

    result = run_amodal_pipeline(args.image, args.instruction, config)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
