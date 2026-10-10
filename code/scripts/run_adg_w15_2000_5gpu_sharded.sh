#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
CODE_DIR="${ROOT}/code"
MODEL_PATH="${ROOT}/hf_cache"
COCO_ANNO="${ROOT}/coco2017_data/annotations/captions_val2017.json"
COCO_IMG_DIR="${ROOT}/coco2017_data/val2017"
META_DIR="${ROOT}/output"
RESULT_ROOT="${ROOT}/results"
PART_ROOT="${RESULT_ROOT}/sd3_adg_w15_2000_parts"
FINAL_DIR="${RESULT_ROOT}/sd3_adg_w15_2000"

cd "${ROOT}"

if [[ ! -f "${META_DIR}/experiment_metadata.json" ]]; then
  echo "Preparing 2000 COCO prompts for ADG w=15..."
  python3 "${CODE_DIR}/data/prepare_coco_evaluation_data.py" \
    --coco_anno_path "${COCO_ANNO}" \
    --coco_img_dir "${COCO_IMG_DIR}" \
    --output_dir "${META_DIR}" \
    --num_prompts 2000 \
    --seed 42
fi

COMMON_GENERATE_ARGS=(
  --metadata_path "${META_DIR}/experiment_metadata.json"
  --real_images_dir "${META_DIR}/real_images_for_fid"
  --model_path "${MODEL_PATH}"
  --methods adg
  --cfg_scale 15
  --num_steps 20
  --shift 3.0
  --base_seed 42
  --adg_max_angle 1.0471975512
)

COMMON_EVAL_ARGS=(
  --metadata_path "${META_DIR}/experiment_metadata.json"
  --real_images_dir "${META_DIR}/real_images_for_fid"
  --methods adg
  --clip_batch_size 16
  --fid_batch_size 32
  --fid_num_workers 0
)

echo "[1/3] Launching ADG w=15 generation on 5 GPUs with prompt sharding..."
mkdir -p "${PART_ROOT}"
PIDS=()
for PART in 0 1 2 3 4; do
  START=$((PART * 400))
  END=$(((PART + 1) * 400))
  PART_DIR="${PART_ROOT}/part${PART}"
  echo "Launching part ${PART}: prompts [${START}, ${END}) on GPU ${PART}..."
  CUDA_VISIBLE_DEVICES="${PART}" python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
    "${COMMON_GENERATE_ARGS[@]}" \
    --output_dir "${PART_DIR}" \
    --start_index "${START}" \
    --end_index "${END}" &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done
if [ "${FAILED}" -ne 0 ]; then
  echo "At least one ADG generation shard failed." >&2
  exit 1
fi

echo "[2/3] Merging shards into ${FINAL_DIR}..."
mkdir -p "${FINAL_DIR}"
cp "${META_DIR}/experiment_metadata.json" "${FINAL_DIR}/experiment_metadata.json"
rm -rf "${FINAL_DIR:?}/adg"
mkdir -p "${FINAL_DIR}/adg"
for PART in 0 1 2 3 4; do
  cp -n "${PART_ROOT}/part${PART}/adg/"*.jpg "${FINAL_DIR}/adg/"
done
COUNT=$(find "${FINAL_DIR}/adg" -maxdepth 1 -name "*.jpg" | wc -l)
echo "adg: ${COUNT} images"

echo "[3/3] Evaluating ADG w=15 results..."
CUDA_VISIBLE_DEVICES=0 python3 "${CODE_DIR}/evaluation/evaluate_coco_innovation_benchmark.py" \
  "${COMMON_EVAL_ARGS[@]}" \
  --generated_root "${FINAL_DIR}"

CUDA_VISIBLE_DEVICES=0 python3 "${CODE_DIR}/evaluation/evaluate_preference_metrics.py" \
  --metadata_path "${META_DIR}/experiment_metadata.json" \
  --generated_root "${FINAL_DIR}" \
  --methods adg

echo "ADG w=15 5-GPU quantitative outputs saved to ${FINAL_DIR}"
