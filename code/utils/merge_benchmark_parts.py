import argparse
import json
import os
import shutil


DEFAULT_METHODS = [
    "cfg",
    "cfg_rescale",
    "apg",
    "fdg",
    "fdg_plus_apg",
    "rptg",
    "rptg_hr",
    "rptg_apg",
    "rptg_hr_apg",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge sharded SD3 benchmark outputs into one directory."
    )
    parser.add_argument(
        "--part_dirs",
        nargs="+",
        required=True,
        help="Part output directories produced by generate_coco_innovation_benchmark.py.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Merged benchmark output directory.",
    )
    parser.add_argument(
        "--metadata_path",
        required=True,
        help="Full experiment metadata JSON used for evaluation.",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images in the merged output directory.",
    )
    return parser.parse_args()


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, list):
        raise ValueError("metadata JSON must contain a list")
    return metadata


def copy_method_images(part_dirs, output_dir, methods, overwrite):
    copied = {method: 0 for method in methods}
    skipped = {method: 0 for method in methods}

    for method in methods:
        merged_method_dir = os.path.join(output_dir, method)
        os.makedirs(merged_method_dir, exist_ok=True)

        for part_dir in part_dirs:
            part_method_dir = os.path.join(part_dir, method)
            if not os.path.isdir(part_method_dir):
                raise FileNotFoundError(part_method_dir)

            for filename in sorted(os.listdir(part_method_dir)):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                src = os.path.join(part_method_dir, filename)
                dst = os.path.join(merged_method_dir, filename)
                if os.path.exists(dst) and not overwrite:
                    skipped[method] += 1
                    continue
                shutil.copy2(src, dst)
                copied[method] += 1

    return copied, skipped


def validate_merged(output_dir, metadata, methods):
    expected = {f"{item['generated_index']}.jpg" for item in metadata}
    summary = {}
    for method in methods:
        method_dir = os.path.join(output_dir, method)
        actual = {
            filename
            for filename in os.listdir(method_dir)
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        }
        summary[method] = {
            "expected": len(expected),
            "actual": len(actual),
            "missing": len(expected - actual),
            "extra": len(actual - expected),
        }
    return summary


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata_path)
    os.makedirs(args.output_dir, exist_ok=True)

    copied, skipped = copy_method_images(
        args.part_dirs,
        args.output_dir,
        args.methods,
        args.overwrite,
    )
    with open(
        os.path.join(args.output_dir, "experiment_metadata.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    summary = validate_merged(args.output_dir, metadata, args.methods)
    merge_report = {
        "part_dirs": args.part_dirs,
        "output_dir": args.output_dir,
        "methods": args.methods,
        "copied": copied,
        "skipped_existing": skipped,
        "validation": summary,
    }
    with open(
        os.path.join(args.output_dir, "merge_report.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(merge_report, file, indent=2, ensure_ascii=False)

    print(json.dumps(merge_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
