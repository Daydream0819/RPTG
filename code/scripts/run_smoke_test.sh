#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp}"
CODE_DIR="${CODE_DIR:-${ROOT}/code}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/hf_cache}"
COCO_ANNO="${COCO_ANNO:-${ROOT}/coco2017_data/annotations/captions_val2017.json}"
COCO_IMG_DIR="${COCO_IMG_DIR:-${ROOT}/coco2017_data/val2017}"
OUTPUT_META="${OUTPUT_META:-${ROOT}/output_smoke}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/results/rptg_smoke}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

cd "${ROOT}"

python3 "${CODE_DIR}/data/prepare_coco_evaluation_data.py" \
  --coco_anno_path "${COCO_ANNO}" \
  --coco_img_dir "${COCO_IMG_DIR}" \
  --output_dir "${OUTPUT_META}" \
  --num_prompts 16 \
  --seed 42

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  --metadata_path "${OUTPUT_META}/experiment_metadata.json" \
  --real_images_dir "${OUTPUT_META}/real_images_for_fid" \
  --output_dir "${RESULT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --methods cfg cfg_rescale adg apg fdg rptg rptg_hr \
  --cfg_scale 15 \
  --num_steps 20 \
  --shift 3.0 \
  --base_seed 42 \
  --frequency_cutoff 0.25 \
  --high_radial_scale 0.25 \
  --low_cfg_scale 1.0 \
  --cfg_rescale_factor 0.7 \
  --apg_r 1.5 \
  --apg_beta -0.2 \
  --fdg_kernel_size 17 \
  --fdg_sigma 4.0 \
  --max_samples 8

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python3 "${CODE_DIR}/evaluation/evaluate_coco_innovation_benchmark.py" \
  --metadata_path "${OUTPUT_META}/experiment_metadata.json" \
  --real_images_dir "${OUTPUT_META}/real_images_for_fid" \
  --generated_root "${RESULT_DIR}" \
  --methods cfg cfg_rescale adg apg fdg rptg rptg_hr \
  --clip_batch_size 16 \
  --fid_batch_size 32 \
  --fid_num_workers 0

echo "Smoke-test results saved to ${RESULT_DIR}/benchmark_results.csv"
