import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


SATURATION_DRIFT_FIELD = "saturation_drift_mean"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add saturation_drift_mean to an existing benchmark_results.csv "
            "without recomputing CLIP, FID, or preference metrics."
        )
    )
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--real_images_dir", type=Path, required=True)
    parser.add_argument("--generated_root", type=Path, required=True)
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=None,
        help="Defaults to <generated_root>/benchmark_results.csv.",
    )
    parser.add_argument(
        "--json_path",
        type=Path,
        default=None,
        help="Defaults to <generated_root>/benchmark_results.json when it exists.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Defaults to methods already present in the benchmark CSV.",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not write .before_sat_drift backup files.",
    )
    return parser.parse_args()


def read_metadata(path):
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, list):
        raise ValueError("metadata JSON must contain a list")
    return metadata


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def mean_saturation(path):
    image = Image.open(path).convert("HSV")
    saturation = np.asarray(image, dtype=np.float32)[:, :, 1] / 255.0
    return float(saturation.mean())


def build_real_saturation_cache(metadata, real_images_dir):
    cache = {}
    for item in tqdm(metadata, desc="Real saturation"):
        index = item["generated_index"]
        real_path = real_images_dir / f"{index}.jpg"
        if not real_path.exists():
            raise FileNotFoundError(real_path)
        cache[str(index)] = mean_saturation(real_path)
    return cache


def compute_method_saturation_drift(method, metadata, generated_root, real_saturation):
    method_dir = generated_root / method
    if not method_dir.exists():
        raise FileNotFoundError(method_dir)

    values = []
    for item in tqdm(metadata, desc=f"{method} saturation drift"):
        index = str(item["generated_index"])
        generated_path = method_dir / f"{index}.jpg"
        if not generated_path.exists():
            continue
        generated_sat = mean_saturation(generated_path)
        values.append(abs(generated_sat - real_saturation[index]))

    if not values:
        raise RuntimeError(f"No images found for method {method} under {method_dir}")
    return float(np.mean(values))


def insert_field_after(fieldnames, new_field, after_field):
    fieldnames = list(fieldnames)
    if new_field in fieldnames:
        return fieldnames
    if after_field in fieldnames:
        index = fieldnames.index(after_field) + 1
        return fieldnames[:index] + [new_field] + fieldnames[index:]
    return fieldnames + [new_field]


def backup(path):
    backup_path = path.with_suffix(path.suffix + ".before_sat_drift")
    shutil.copy2(path, backup_path)
    return backup_path


def update_json(json_path, drift_by_method):
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        for row in payload:
            method = row.get("method")
            if method in drift_by_method:
                row[SATURATION_DRIFT_FIELD] = drift_by_method[method]
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return json_path


def main():
    args = parse_args()
    csv_path = args.csv_path or args.generated_root / "benchmark_results.csv"
    json_path = args.json_path or args.generated_root / "benchmark_results.json"

    metadata = read_metadata(args.metadata_path)
    rows = read_csv(csv_path)
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")

    methods = args.methods or [row["method"] for row in rows]
    real_saturation = build_real_saturation_cache(metadata, args.real_images_dir)
    drift_by_method = {
        method: compute_method_saturation_drift(
            method, metadata, args.generated_root, real_saturation
        )
        for method in methods
    }

    for row in rows:
        method = row.get("method")
        if method in drift_by_method:
            row[SATURATION_DRIFT_FIELD] = f"{drift_by_method[method]:.12f}"

    fieldnames = insert_field_after(rows[0].keys(), SATURATION_DRIFT_FIELD, "saturation_mean")

    if not args.no_backup:
        print(f"CSV backup: {backup(csv_path)}")
        if json_path.exists():
            print(f"JSON backup: {backup(json_path)}")

    write_csv(csv_path, rows, fieldnames)
    updated_json = update_json(json_path, drift_by_method)
    print(f"Updated CSV: {csv_path}")
    if updated_json is not None:
        print(f"Updated JSON: {updated_json}")
    for method, value in drift_by_method.items():
        print(f"{method}: {SATURATION_DRIFT_FIELD}={value:.6f}")


if __name__ == "__main__":
    main()
