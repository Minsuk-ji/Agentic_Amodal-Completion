#!/usr/bin/env bash
# Runs the pipeline end-to-end on the bundled example image.
#
# Prerequisites:
#   1. A vLLM (or other OpenAI-compatible) server serving the planning VLM, e.g.:
#        vllm serve Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000 --trust-remote-code
#   2. External model repos/checkpoints under `external/` (see README "Setup").
set -euo pipefail

cd "$(dirname "$0")/.."

python -m agentic_amodal_completion.main \
  --image examples/images/surfboard.jpg \
  --instruction "complete the surfboard in the image" \
  --save_dir results \
  --image_index surfboard_completion \
  --seed 42 \
  --default_gpu_id "${GPU_ID:-0}" \
  --use_open_llm \
  --open_llm_model "${OPEN_LLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}" \
  --open_llm_host "${OPEN_LLM_HOST:-0.0.0.0}" \
  --open_llm_port "${OPEN_LLM_PORT:-8000}" \
  --inpaint_backend "${INPAINT_BACKEND:-flux_fill}" \
  --flux_fill_id "${FLUX_FILL_ID:-black-forest-labs/FLUX.1-Fill-dev}" \
  --flux_fill_steps "${FLUX_FILL_STEPS:-50}" \
  --flux_fill_guidance_scale "${FLUX_FILL_GUIDANCE_SCALE:-30}" \
  --flux_fill_max_sequence_length "${FLUX_FILL_MAX_SEQUENCE_LENGTH:-512}"
