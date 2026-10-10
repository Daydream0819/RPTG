#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
CODE_DIR="${ROOT}/code"
MODEL_PATH="${ROOT}/hf_cache"
OUTPUT_DIR="${ROOT}/results/rptg_qualitative_stress_w15_adg"

cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 "${CODE_DIR}/generation/generate_qualitative_stress_prompts.py" \
  --output_dir "${OUTPUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --methods cfg cfg_rescale adg adg_norm apg fdg rptg rptg_hr \
  --prompt_ids pink_dog food_fruit_bowl architecture_cathedral animal_red_fox \
  --cfg_scale 15 \
  --num_steps 20 \
  --shift 3.0 \
  --frequency_cutoff 0.25 \
  --high_radial_scale 0.25 \
  --low_cfg_scale 1.0 \
  --cfg_rescale_factor 0.7 \
  --adg_max_angle 1.0471975512 \
  --apg_r 1.5 \
  --apg_beta -0.2 \
  --fdg_kernel_size 5 \
  --fdg_sigma 1.0 \
  --make_grid

echo "ADG qualitative outputs saved to ${OUTPUT_DIR}"
