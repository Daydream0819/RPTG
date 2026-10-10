#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
CODE_DIR="${ROOT}/code"
MODEL_PATH="${ROOT}/hf_cache"
COCO_ANNO="${ROOT}/coco2017_data/annotations/captions_val2017.json"
COCO_IMG_DIR="${ROOT}/coco2017_data/val2017"
OUTPUT_META="${ROOT}/output_sweep_1000"
RESULT_ROOT="${ROOT}/results"

cd "${ROOT}"

if [[ ! -f "${OUTPUT_META}/experiment_metadata.json" ]]; then
  echo "Preparing 1000 COCO prompts for ADG guidance-scale sweep..."
  python3 "${CODE_DIR}/data/prepare_coco_evaluation_data.py" \
    --coco_anno_path "${COCO_ANNO}" \
    --coco_img_dir "${COCO_IMG_DIR}" \
    --output_dir "${OUTPUT_META}" \
    --num_prompts 1000 \
    --seed 42
fi

COMMON_GENERATE_ARGS=(
  --metadata_path "${OUTPUT_META}/experiment_metadata.json"
  --real_images_dir "${OUTPUT_META}/real_images_for_fid"
  --model_path "${MODEL_PATH}"
  --methods adg
  --num_steps 20
  --shift 3.0
  --base_seed 42
  --adg_max_angle 1.0471975512
)

COMMON_EVAL_ARGS=(
  --metadata_path "${OUTPUT_META}/experiment_metadata.json"
  --real_images_dir "${OUTPUT_META}/real_images_for_fid"
  --methods adg
  --clip_batch_size 16
  --fid_batch_size 32
  --fid_num_workers 0
)

PARALLEL_SCALES=(3 5 7.5 10 12.5)
SCALES=(3 5 7.5 10 12.5 15)
GPUS=(0 1 2 3 4)
PIDS=()

for i in "${!PARALLEL_SCALES[@]}"; do
  scale="${PARALLEL_SCALES[$i]}"
  gpu="${GPUS[$i]}"
  tag="${scale//./p}"
  echo "Launching ADG w=${scale} on GPU ${gpu}..."
  CUDA_VISIBLE_DEVICES="${gpu}" python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
    "${COMMON_GENERATE_ARGS[@]}" \
    --output_dir "${RESULT_ROOT}/adg_scale_sweep_w${tag}_1000" \
    --cfg_scale "${scale}" &
  PIDS+=("$!")
done

wait "${PIDS[@]}"

echo "Launching ADG w=15 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/adg_scale_sweep_w15_1000" \
  --cfg_scale 15

for scale in "${SCALES[@]}"; do
  tag="${scale//./p}"
  echo "Evaluating ADG w=${scale}..."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 "${CODE_DIR}/evaluation/evaluate_coco_innovation_benchmark.py" \
    "${COMMON_EVAL_ARGS[@]}" \
    --generated_root "${RESULT_ROOT}/adg_scale_sweep_w${tag}_1000"
done

echo "ADG scale-sweep outputs saved under ${RESULT_ROOT}/adg_scale_sweep_w*_1000"
