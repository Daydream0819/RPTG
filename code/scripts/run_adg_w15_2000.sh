#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
CODE_DIR="${ROOT}/code"
MODEL_PATH="${ROOT}/hf_cache"
META_DIR="${ROOT}/output"
RESULT_DIR="${ROOT}/results/sd3_adg_w15_2000"

cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  --metadata_path "${META_DIR}/experiment_metadata.json" \
  --real_images_dir "${META_DIR}/real_images_for_fid" \
  --output_dir "${RESULT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --methods adg \
  --cfg_scale 15 \
  --num_steps 20 \
  --shift 3.0 \
  --base_seed 42 \
  --adg_max_angle 1.0471975512

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 "${CODE_DIR}/evaluation/evaluate_coco_innovation_benchmark.py" \
  --metadata_path "${META_DIR}/experiment_metadata.json" \
  --real_images_dir "${META_DIR}/real_images_for_fid" \
  --generated_root "${RESULT_DIR}" \
  --methods adg \
  --clip_batch_size 16 \
  --fid_batch_size 32 \
  --fid_num_workers 0

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 "${CODE_DIR}/evaluation/evaluate_preference_metrics.py" \
  --metadata_path "${META_DIR}/experiment_metadata.json" \
  --generated_root "${RESULT_DIR}" \
  --methods adg

echo "ADG w=15 quantitative outputs saved to ${RESULT_DIR}"
