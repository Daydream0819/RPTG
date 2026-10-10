#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
CODE_DIR="${ROOT}/code"
MODEL_PATH="${ROOT}/hf_cache"
COCO_ANNO="${ROOT}/coco2017_data/annotations/captions_val2017.json"
COCO_IMG_DIR="${ROOT}/coco2017_data/val2017"
OUTPUT_META="${ROOT}/output_gamma_1000"
RESULT_ROOT="${ROOT}/results"
RESULT_DIR="${RESULT_ROOT}/rptg_high_radial_sweep_w15_1000"
PART_ROOT="${RESULT_ROOT}/rptg_high_radial_sweep_w15_1000_parts"

METHODS=(
  rptg_hr_g0
  rptg_hr_g0p1
  rptg_hr_g0p25
  rptg_hr_g0p5
  rptg_hr_g0p75
  rptg_hr_g1
)
GAMMAS=(0 0.1 0.25 0.5 0.75 1.0)
GPUS=(0 1 2 3 4 5)

cd "${ROOT}"

echo "[1/3] Preparing 1000 COCO prompts for gamma_H sweep..."
python3 "${CODE_DIR}/data/prepare_coco_evaluation_data.py" \
  --coco_anno_path "${COCO_ANNO}" \
  --coco_img_dir "${COCO_IMG_DIR}" \
  --output_dir "${OUTPUT_META}" \
  --num_prompts 1000 \
  --seed 42

echo "[2/3] Generating RPTG-HR gamma_H sweep at w=15 on 6 GPUs..."
mkdir -p "${PART_ROOT}"

PIDS=()
for IDX in "${!GAMMAS[@]}"; do
  GAMMA="${GAMMAS[$IDX]}"
  GPU="${GPUS[$IDX]}"
  METHOD="${METHODS[$IDX]}"
  PART_DIR="${PART_ROOT}/${METHOD}"

  echo "Launching ${METHOD} (gamma_H=${GAMMA}) on GPU ${GPU}..."
  CUDA_VISIBLE_DEVICES="${GPU}" python3 "${CODE_DIR}/generation/generate_rptg_high_radial_sweep.py" \
    --metadata_path "${OUTPUT_META}/experiment_metadata.json" \
    --real_images_dir "${OUTPUT_META}/real_images_for_fid" \
    --output_dir "${PART_DIR}" \
    --model_path "${MODEL_PATH}" \
    --high_radial_scales "${GAMMA}" \
    --variant stabilized \
    --cfg_scale 15 \
    --num_steps 20 \
    --shift 3.0 \
    --base_seed 42 \
    --frequency_cutoff 0.25 \
    --apg_r 1.5 \
    --apg_beta -0.2 &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done
if [ "${FAILED}" -ne 0 ]; then
  echo "At least one gamma_H generation job failed." >&2
  exit 1
fi

echo "Collecting method directories into ${RESULT_DIR}..."
mkdir -p "${RESULT_DIR}"
cp "${OUTPUT_META}/experiment_metadata.json" "${RESULT_DIR}/experiment_metadata.json"
for METHOD in "${METHODS[@]}"; do
  PART_DIR="${PART_ROOT}/${METHOD}"
  if [ ! -d "${PART_DIR}/${METHOD}" ]; then
    echo "Missing generated method directory: ${PART_DIR}/${METHOD}" >&2
    exit 1
  fi
  rm -rf "${RESULT_DIR:?}/${METHOD}"
  cp -a "${PART_DIR}/${METHOD}" "${RESULT_DIR}/${METHOD}"
done

python3 - "${METHODS[@]}" <<PY
import json
import os
import sys

result_dir = "${RESULT_DIR}"
part_root = "${PART_ROOT}"
methods = sys.argv[1:]
summary = {
    "experiment": "rptg_high_radial_sweep_w15_1000_6gpu",
    "result_dir": result_dir,
    "part_root": part_root,
    "num_prompts": 1000,
    "cfg_scale": 15,
    "gamma_H_values": [0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "methods": methods,
    "parts": {},
}
for method in methods:
    config_path = os.path.join(part_root, method, "generation_config.json")
    with open(config_path, "r", encoding="utf-8") as file:
        summary["parts"][method] = json.load(file)
with open(os.path.join(result_dir, "generation_config.json"), "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)
PY

echo "[3/3] Evaluating CLIP, FID, and artifact metrics..."
CUDA_VISIBLE_DEVICES=0 python3 "${CODE_DIR}/evaluation/evaluate_coco_innovation_benchmark.py" \
  --metadata_path "${OUTPUT_META}/experiment_metadata.json" \
  --real_images_dir "${OUTPUT_META}/real_images_for_fid" \
  --generated_root "${RESULT_DIR}" \
  --methods "${METHODS[@]}" \
  --clip_batch_size 16 \
  --fid_batch_size 32 \
  --fid_num_workers 0

echo "Done. Results:"
echo "${RESULT_DIR}/benchmark_results.csv"
