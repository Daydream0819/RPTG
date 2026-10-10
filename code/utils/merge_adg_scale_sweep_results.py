import argparse
import csv
from pathlib import Path


DEFAULT_RESULT_ROOT = Path("/root/autodl-tmp/results")
DEFAULT_SCALES = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0]


def scale_tag(scale):
    return f"{scale:g}".replace(".", "p")


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("No rows to write")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge ADG guidance-scale sweep benchmark_results.csv files."
    )
    parser.add_argument("--result_root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--scales", type=float, nargs="+", default=DEFAULT_SCALES
    )
    parser.add_argument(
        "--base_csv",
        type=Path,
        default=None,
        help="Optional existing merged sweep CSV to append ADG rows to.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("/root/autodl-tmp/results/adg_scale_sweep_merged.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    if args.base_csv is not None:
        rows.extend(read_csv(args.base_csv))

    for scale in args.scales:
        csv_path = (
            args.result_root
            / f"adg_scale_sweep_w{scale_tag(scale)}_1000"
            / "benchmark_results.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        scale_rows = read_csv(csv_path)
        for row in scale_rows:
            if row.get("method") != "adg":
                continue
            row = dict(row)
            row["w"] = f"{scale:g}"
            row["display_method"] = "adg"
            rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, rows)
    print(f"Wrote merged ADG sweep CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
