#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
CODE_DIR="${ROOT}/code"
MODEL_PATH="${ROOT}/hf_cache"
OUTPUT_META="${ROOT}/output_sweep_1000"
RESULT_ROOT="${ROOT}/results"
FINAL_DIR="${RESULT_ROOT}/scale_sweep_w15_1000"

METHODS=(cfg apg fdg rptg_apg rptg_hr_apg)
COMMON_GENERATE_ARGS=(
  --metadata_path "${OUTPUT_META}/experiment_metadata.json"
  --real_images_dir "${OUTPUT_META}/real_images_for_fid"
  --model_path "${MODEL_PATH}"
  --methods "${METHODS[@]}"
  --cfg_scale 15
  --num_steps 20
  --shift 3.0
  --base_seed 42
  --frequency_cutoff 0.25
  --high_radial_scale 0.25
  --low_cfg_scale 1.0
  --cfg_rescale_factor 0.7
  --apg_r 1.5
  --apg_beta -0.2
  --fdg_kernel_size 17
  --fdg_sigma 4.0
)
COMMON_EVAL_ARGS=(
  --metadata_path "${OUTPUT_META}/experiment_metadata.json"
  --real_images_dir "${OUTPUT_META}/real_images_for_fid"
  --methods "${METHODS[@]}"
  --clip_batch_size 16
  --fid_batch_size 32
  --fid_num_workers 0
)

cd "${ROOT}"

echo "[1/3] Launching w=15 generation on 5 GPUs with prompt sharding..."
CUDA_VISIBLE_DEVICES=0 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/scale_sweep_w15_1000_part0" \
  --start_index 0 \
  --end_index 200 &
PID_0=$!

CUDA_VISIBLE_DEVICES=1 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/scale_sweep_w15_1000_part1" \
  --start_index 200 \
  --end_index 400 &
PID_1=$!

CUDA_VISIBLE_DEVICES=2 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/scale_sweep_w15_1000_part2" \
  --start_index 400 \
  --end_index 600 &
PID_2=$!

CUDA_VISIBLE_DEVICES=3 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/scale_sweep_w15_1000_part3" \
  --start_index 600 \
  --end_index 800 &
PID_3=$!

CUDA_VISIBLE_DEVICES=4 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/scale_sweep_w15_1000_part4" \
  --start_index 800 \
  --end_index 1000 &
PID_4=$!

wait "${PID_0}" "${PID_1}" "${PID_2}" "${PID_3}" "${PID_4}"

echo "[2/3] Merging shard outputs into ${FINAL_DIR}..."
mkdir -p "${FINAL_DIR}"
cp -n "${OUTPUT_META}/experiment_metadata.json" "${FINAL_DIR}/experiment_metadata.json"
for METHOD in "${METHODS[@]}"; do
  mkdir -p "${FINAL_DIR}/${METHOD}"
  for PART in 0 1 2 3 4; do
    cp -n "${RESULT_ROOT}/scale_sweep_w15_1000_part${PART}/${METHOD}/"*.jpg \
      "${FINAL_DIR}/${METHOD}/"
  done
  COUNT=$(find "${FINAL_DIR}/${METHOD}" -maxdepth 1 -name "*.jpg" | wc -l)
  echo "${METHOD}: ${COUNT} images"
done

echo "[3/3] Evaluating merged w=15 results..."
CUDA_VISIBLE_DEVICES=0 python3 "${CODE_DIR}/evaluation/evaluate_coco_innovation_benchmark.py" \
  "${COMMON_EVAL_ARGS[@]}" \
  --generated_root "${FINAL_DIR}"

echo "Done. Results saved to ${FINAL_DIR}/benchmark_results.csv"
