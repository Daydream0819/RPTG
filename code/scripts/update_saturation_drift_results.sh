#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp}"
CODE_DIR="${CODE_DIR:-${ROOT}/code}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results}"

infer_meta_dir() {
  local result_dir="$1"
  local name
  name="$(basename "${result_dir}")"
  case "${name}" in
    *high_radial*)
      echo "${ROOT}/output_gamma_1000"
      ;;
    *scale_sweep*|adg_scale_sweep_*)
      echo "${ROOT}/output_sweep_1000"
      ;;
    *)
      echo "${ROOT}/output"
      ;;
  esac
}

update_one() {
  local result_dir="$1"
  local meta_dir
  meta_dir="$(infer_meta_dir "${result_dir}")"
  local metadata_path="${meta_dir}/experiment_metadata.json"
  local real_images_dir="${meta_dir}/real_images_for_fid"

  if [[ ! -f "${result_dir}/benchmark_results.csv" ]]; then
    echo "Skip ${result_dir}: benchmark_results.csv not found"
    return 0
  fi
  if [[ ! -f "${metadata_path}" ]]; then
    echo "Skip ${result_dir}: metadata not found at ${metadata_path}"
    return 0
  fi
  if [[ ! -d "${real_images_dir}" ]]; then
    echo "Skip ${result_dir}: real images not found at ${real_images_dir}"
    return 0
  fi

  echo "Updating saturation drift: ${result_dir}"
  python3 "${CODE_DIR}/utils/add_saturation_drift_to_benchmark_csv.py" \
    --metadata_path "${metadata_path}" \
    --real_images_dir "${real_images_dir}" \
    --generated_root "${result_dir}"
}

if [[ "$#" -gt 0 ]]; then
  for result_dir in "$@"; do
    update_one "${result_dir}"
  done
else
  while IFS= read -r csv_path; do
    update_one "$(dirname "${csv_path}")"
  done < <(find "${RESULT_ROOT}" -mindepth 2 -maxdepth 2 -name benchmark_results.csv | sort)
fi
