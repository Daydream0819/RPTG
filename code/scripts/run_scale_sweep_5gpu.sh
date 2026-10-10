#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
CODE_DIR="${ROOT}/code"
MODEL_PATH="${ROOT}/hf_cache"
COCO_ANNO="${ROOT}/coco2017_data/annotations/captions_val2017.json"
COCO_IMG_DIR="${ROOT}/coco2017_data/val2017"
OUTPUT_META="${ROOT}/output_sweep_1000"
RESULT_ROOT="${ROOT}/results"

METHODS=(cfg apg fdg rptg_apg rptg_hr_apg)
COMMON_GENERATE_ARGS=(
  --metadata_path "${OUTPUT_META}/experiment_metadata.json"
  --real_images_dir "${OUTPUT_META}/real_images_for_fid"
  --model_path "${MODEL_PATH}"
  --methods "${METHODS[@]}"
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

echo "[1/3] Preparing 1000 COCO prompts for guidance-scale sweep..."
python3 "${CODE_DIR}/data/prepare_coco_evaluation_data.py" \
  --coco_anno_path "${COCO_ANNO}" \
  --coco_img_dir "${COCO_IMG_DIR}" \
  --output_dir "${OUTPUT_META}" \
  --num_prompts 1000 \
  --seed 42

echo "[2/3] Launching generation on 5 GPUs..."
CUDA_VISIBLE_DEVICES=0 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/scale_sweep_w3_1000" \
  --cfg_scale 3 &
PID_W3=$!

CUDA_VISIBLE_DEVICES=1 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/scale_sweep_w5_1000" \
  --cfg_scale 5 &
PID_W5=$!

CUDA_VISIBLE_DEVICES=2 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/scale_sweep_w7p5_1000" \
  --cfg_scale 7.5 &
PID_W7P5=$!

CUDA_VISIBLE_DEVICES=3 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/scale_sweep_w10_1000" \
  --cfg_scale 10 &
PID_W10=$!

CUDA_VISIBLE_DEVICES=4 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/scale_sweep_w12p5_1000" \
  --cfg_scale 12.5 &
PID_W12P5=$!

wait "${PID_W3}" "${PID_W5}" "${PID_W7P5}" "${PID_W10}" "${PID_W12P5}"

echo "[3/4] Running the matched w=15 sweep point on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 "${CODE_DIR}/generation/generate_coco_innovation_benchmark.py" \
  "${COMMON_GENERATE_ARGS[@]}" \
  --output_dir "${RESULT_ROOT}/scale_sweep_w15_1000" \
  --cfg_scale 15

echo "[4/4] Evaluating each scale on GPU 0..."
for SCALE_DIR in \
  scale_sweep_w3_1000 \
  scale_sweep_w5_1000 \
  scale_sweep_w7p5_1000 \
  scale_sweep_w10_1000 \
  scale_sweep_w12p5_1000 \
  scale_sweep_w15_1000
do
  echo "Evaluating ${SCALE_DIR}..."
  CUDA_VISIBLE_DEVICES=0 python3 "${CODE_DIR}/evaluation/evaluate_coco_innovation_benchmark.py" \
    "${COMMON_EVAL_ARGS[@]}" \
    --generated_root "${RESULT_ROOT}/${SCALE_DIR}"
done

echo "Done. Results are saved under ${RESULT_ROOT}/scale_sweep_w*_1000/benchmark_results.csv"
