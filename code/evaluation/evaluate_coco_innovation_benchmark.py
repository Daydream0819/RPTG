import argparse
import csv
import json
import os
import sys

import numpy as np
import open_clip
import torch
from cleanfid import fid
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


DEFAULT_METHODS = [
    "cfg",
    "cfg_rescale",
    "adg",
    "apg",
    "fdg",
    "fdg_plus_apg",
    "rptg",
    "rptg_hr",
    "rptg_apg",
    "rptg_hr_apg",
]


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the SD3 COCO benchmark."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="/root/autodl-tmp/output/experiment_metadata.json",
    )
    parser.add_argument(
        "--real_images_dir",
        type=str,
        default="/root/autodl-tmp/output/real_images_for_fid",
    )
    parser.add_argument(
        "--generated_root",
        type=str,
        default="/root/autodl-tmp/results/sd3_rptg_w15_main",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--clip_batch_size", type=int, default=16)
    parser.add_argument("--clip_model", type=str, default="ViT-H-14")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s32b_b79k")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument("--fid_num_workers", type=int, default=0)
    parser.add_argument(
        "--fid_use_dataparallel",
        action="store_true",
        help="Enable clean-fid DataParallel. Disabled by default to avoid NCCL errors on multi-GPU servers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


RESULT_FIELDS = [
    "method",
    "clip_mean",
    "clip_std",
    "clip_count",
    "saturation_mean",
    "saturation_drift_mean",
    "contrast_mean",
    "brightness_mean",
    "brightness_drift_mean",
    "color_shift_mean",
    "fid",
    "missing_images",
    "extra_images",
]


def save_results(results, generated_root):
    json_path = os.path.join(generated_root, "benchmark_results.json")
    csv_path = os.path.join(generated_root, "benchmark_results.csv")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    return json_path, csv_path


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, list):
        raise ValueError("metadata JSON must contain a list")
    return metadata


def validate_image_directory(image_dir, metadata):
    expected = {f"{item['generated_index']}.jpg" for item in metadata}
    actual = {
        name
        for name in os.listdir(image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    }
    return sorted(expected - actual), sorted(actual - expected)


def validate_method_images(method_dir, metadata):
    present_items = []
    missing_indices = []
    expected_filenames = set()

    for item in metadata:
        generated_index = str(item["generated_index"])
        filename = f"{generated_index}.jpg"
        expected_filenames.add(filename)
        image_path = os.path.join(method_dir, filename)
        if os.path.exists(image_path):
            present_items.append((item, image_path))
        else:
            missing_indices.append(generated_index)

    actual_filenames = {
        name
        for name in os.listdir(method_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    }
    extra_filenames = sorted(actual_filenames - expected_filenames)
    return present_items, missing_indices, extra_filenames


def compute_artifact_metrics(items, real_images_dir):
    saturation_values = []
    saturation_drifts = []
    contrast_values = []
    brightness_values = []
    brightness_drifts = []
    color_shifts = []

    for item, generated_path in tqdm(items, desc="Artifact metrics"):
        generated_image = Image.open(generated_path).convert("RGB")
        real_path = os.path.join(real_images_dir, f"{item['generated_index']}.jpg")
        real_image = Image.open(real_path).convert("RGB")

        generated_rgb = np.asarray(generated_image, dtype=np.float32) / 255.0
        real_rgb = np.asarray(real_image, dtype=np.float32) / 255.0
        generated_gray = (
            np.asarray(generated_image.convert("L"), dtype=np.float32) / 255.0
        )
        real_gray = np.asarray(real_image.convert("L"), dtype=np.float32) / 255.0
        generated_saturation = (
            np.asarray(generated_image.convert("HSV"), dtype=np.float32)[:, :, 1]
            / 255.0
        )
        real_saturation = (
            np.asarray(real_image.convert("HSV"), dtype=np.float32)[:, :, 1] / 255.0
        )

        saturation_values.append(float(generated_saturation.mean()))
        saturation_drifts.append(
            float(abs(generated_saturation.mean() - real_saturation.mean()))
        )
        contrast_values.append(float(generated_gray.std()))
        brightness_values.append(float(generated_gray.mean()))
        brightness_drifts.append(float(abs(generated_gray.mean() - real_gray.mean())))
        color_shifts.append(
            float(
                np.linalg.norm(
                    generated_rgb.mean(axis=(0, 1)) - real_rgb.mean(axis=(0, 1))
                )
            )
        )

    return {
        "saturation_mean": float(np.mean(saturation_values)),
        "saturation_drift_mean": float(np.mean(saturation_drifts)),
        "contrast_mean": float(np.mean(contrast_values)),
        "brightness_mean": float(np.mean(brightness_values)),
        "brightness_drift_mean": float(np.mean(brightness_drifts)),
        "color_shift_mean": float(np.mean(color_shifts)),
    }


@torch.no_grad()
def compute_clip_score(model, preprocess, tokenizer, device, items, batch_size, description):
    scores = []
    for start in tqdm(range(0, len(items), batch_size), desc=description):
        batch = items[start : start + batch_size]
        images = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for _, path in batch]
        ).to(device)
        texts = tokenizer([item["caption"] for item, _ in batch]).to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores.extend(torch.sum(image_features * text_features, dim=-1).float().cpu().tolist())

    score_tensor = torch.tensor(scores, dtype=torch.float32)
    return {
        "clip_mean": score_tensor.mean().item(),
        "clip_std": score_tensor.std(unbiased=False).item(),
        "clip_count": len(scores),
    }


def run_evaluation():
    args = parse_cli_args()
    metadata = load_metadata(args.metadata_path)
    if not os.path.isdir(args.real_images_dir):
        raise FileNotFoundError(args.real_images_dir)

    real_missing, real_extra = validate_image_directory(args.real_images_dir, metadata)
    if real_missing or real_extra:
        raise RuntimeError(
            "The real-image directory does not exactly match metadata: "
            f"{len(real_missing)} missing, {len(real_extra)} extra."
        )

    print(f"Loading CLIP model: {args.clip_model}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
    )
    model = model.to(args.device).eval()
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    results = []
    for method_name in args.methods:
        method_dir = os.path.join(args.generated_root, method_name)
        if not os.path.isdir(method_dir):
            print(f"Skipping missing directory: {method_dir}")
            continue

        items, missing, extra = validate_method_images(method_dir, metadata)
        if not items:
            print(f"Skipping {method_name}: no generated images")
            continue
        if missing:
            print(f"Warning: {method_name} is missing {len(missing)} images.")
        if extra:
            print(f"Warning: {method_name} has {len(extra)} extra images.")

        clip_result = compute_clip_score(
            model,
            preprocess,
            tokenizer,
            args.device,
            items,
            args.clip_batch_size,
            f"CLIP {method_name}",
        )
        artifact_result = compute_artifact_metrics(items, args.real_images_dir)

        fid_score = None
        if not args.skip_fid and not missing and not extra:
            print(f"Computing FID: {method_name}")
            fid_score = float(
                fid.compute_fid(
                    args.real_images_dir,
                    method_dir,
                    batch_size=args.fid_batch_size,
                    num_workers=args.fid_num_workers,
                    device=args.device,
                    use_dataparallel=args.fid_use_dataparallel,
                )
            )
        elif args.skip_fid:
            print(f"Skipping FID by request: {method_name}")

        result = {
            "method": method_name,
            **clip_result,
            **artifact_result,
            "fid": fid_score,
            "missing_images": len(missing),
            "extra_images": len(extra),
        }
        results.append(result)
        print(
            f"{method_name}: CLIP={result['clip_mean']:.4f}, "
            f"FID={fid_score if fid_score is not None else 'SKIPPED'}"
        )
        save_results(results, args.generated_root)

    json_path, csv_path = save_results(results, args.generated_root)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    run_evaluation()
