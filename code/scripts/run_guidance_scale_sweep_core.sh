#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp}"
CODE_DIR="${CODE_DIR:-${ROOT}/code}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/hf_cache}"
COCO_ANNO="${COCO_ANNO:-${ROOT}/coco2017_data/annotations/captions_val2017.json}"
COCO_IMG_DIR="${COCO_IMG_DIR:-${ROOT}/coco2017_data/val2017}"
OUTPUT_META="${OUTPUT_META:-${ROOT}/output_sweep_1000}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

METHODS=(cfg apg fdg fdg_plus_apg rptg_hr)
SCALES=(3 5 7.5 10 15)

cd "${ROOT}"

python3 "${CODE_DIR}/data/prepare_coco_evaluation_data.py" \
  --coco_anno_path "${COCO_ANNO}" \
  --coco_img_dir "${COCO_IMG_DIR}" \
  --output_dir "${OUTPUT_META}" \
  --num_prompts "${NUM_PROMPTS}" \
  --seed 42

for SCALE in "${SCALES[@]}"; do
  TAG="${SCALE//./p}"
  RESULT_DIR="${RESULT_ROOT}/scale_sweep_core_w${TAG}_${NUM_PROMPTS}"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
    --metadata_path "${OUTPUT_META}/experiment_metadata.json" \
    --real_images_dir "${OUTPUT_META}/real_images_for_fid" \
    --output_dir "${RESULT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --methods "${METHODS[@]}" \
    --cfg_scale "${SCALE}" \
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
    --fdg_sigma 4.0

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python3 "${CODE_DIR}/evaluation/evaluate_coco_innovation_benchmark.py" \
    --metadata_path "${OUTPUT_META}/experiment_metadata.json" \
    --real_images_dir "${OUTPUT_META}/real_images_for_fid" \
    --generated_root "${RESULT_DIR}" \
    --methods "${METHODS[@]}" \
    --clip_batch_size 16 \
    --fid_batch_size 32 \
    --fid_num_workers 0
done

echo "Guidance-scale sweep results saved under ${RESULT_ROOT}/scale_sweep_core_w*_${NUM_PROMPTS}"

