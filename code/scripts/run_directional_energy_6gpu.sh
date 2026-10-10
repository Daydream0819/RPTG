#!/usr/bin/env bash
set -euo pipefail

METADATA_PATH="${METADATA_PATH:-/root/autodl-tmp/output/experiment_metadata.json}"
CODE_DIR="${CODE_DIR:-/root/autodl-tmp/code}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/hf_cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/results/rptg_directional_energy_6gpu}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
NUM_STEPS="${NUM_STEPS:-20}"
SHIFT="${SHIFT:-3.0}"
FREQUENCY_CUTOFF="${FREQUENCY_CUTOFF:-0.25}"
BASE_SEED="${BASE_SEED:-42}"

mkdir -p "${OUTPUT_ROOT}/logs"

SCALES=(3 5 7.5 10 12.5 15)
GPUS=(0 1 2 3 4 5)

for i in "${!SCALES[@]}"; do
  scale="${SCALES[$i]}"
  gpu="${GPUS[$i]}"
  scale_dir="${OUTPUT_ROOT}/w${scale}"
  log_path="${OUTPUT_ROOT}/logs/w${scale}.log"

  echo "Launching w=${scale} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=1 python3 "${CODE_DIR}/generation/measure_rptg_directional_energy.py" \
    --metadata_path "${METADATA_PATH}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${scale_dir}" \
    --cfg_scales "${scale}" \
    --num_steps "${NUM_STEPS}" \
    --shift "${SHIFT}" \
    --frequency_cutoff "${FREQUENCY_CUTOFF}" \
    --base_seed "${BASE_SEED}" \
    --max_samples "${MAX_SAMPLES}" \
    > "${log_path}" 2>&1 &
done

wait

python3 - <<'PY'
import csv
import glob
import os

output_root = os.environ.get("OUTPUT_ROOT", "/root/autodl-tmp/results/rptg_directional_energy_6gpu")
paths = sorted(glob.glob(os.path.join(output_root, "w*", "directional_energy_summary.csv")))
if not paths:
    raise SystemExit(f"No summary CSV files found under {output_root}")

rows = []
fieldnames = None
for path in paths:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        rows.extend(reader)

merged_path = os.path.join(output_root, "directional_energy_summary.csv")
with open(merged_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Merged {len(paths)} files into {merged_path}")
PY

echo "Done. Merged summary: ${OUTPUT_ROOT}/directional_energy_summary.csv"
