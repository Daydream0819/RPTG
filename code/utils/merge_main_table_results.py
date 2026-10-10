import argparse
import csv
from pathlib import Path


DEFAULT_FIELDS = [
    "method",
    "clip_mean",
    "fid",
    "pickscore_mean",
    "imagereward_mean",
    "hpsv2_mean",
    "saturation_mean",
    "saturation_drift_mean",
    "contrast_mean",
    "brightness_drift_mean",
    "color_shift_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge benchmark and preference CSV files into one main-table CSV."
    )
    parser.add_argument("--benchmark_csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--preference_csvs", type=Path, nargs="*", default=[])
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def merge_rows(paths, rows_by_method):
    for path in paths:
        for row in read_csv(path):
            method = row.get("method")
            if not method:
                continue
            rows_by_method.setdefault(method, {"method": method}).update(row)


def write_csv(path, rows_by_method, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for method in rows_by_method:
            row = rows_by_method[method]
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    args = parse_args()
    rows_by_method = {}
    merge_rows(args.benchmark_csvs, rows_by_method)
    merge_rows(args.preference_csvs, rows_by_method)
    write_csv(args.output_csv, rows_by_method, args.fields)
    print(f"Wrote merged main-table CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
