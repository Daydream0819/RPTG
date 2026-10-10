import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.guidance_methods import (
    sample_anchor_apg,
    sample_anchor_adg,
    sample_cfg_rescale,
    sample_fdg_laplacian,
    sample_fdg_plus_apg,
    sample_rptg,
    sample_rptg_apg_stabilized,
    sample_rptg_hr,
    sample_rptg_hr_apg_stabilized,
    sample_standard_cfg,
)
from models.latent_io import decode_sd3_latent_to_pil
from models.sd3_model import SD3TransformerWrapper


METHOD_NAMES = [
    "cfg",
    "cfg_rescale",
    "adg",
    "apg",
    "fdg",
    "fdg_plus_apg",
    "rptg",
    "rptg_hr",
]

ALL_METHOD_NAMES = METHOD_NAMES + [
    "adg_norm",
    "rptg_apg",
    "rptg_hr_apg",
]


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Generate the SD3 COCO benchmark for CFG, APG, FDG, and RPTG."
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
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/results/sd3_rptg_w15_main",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/hf_cache",
    )
    parser.add_argument("--cfg_scale", type=float, default=15.0)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--frequency_cutoff", type=float, default=0.25)
    parser.add_argument("--high_radial_scale", type=float, default=0.25)
    parser.add_argument("--low_cfg_scale", type=float, default=1.0)
    parser.add_argument("--cfg_rescale_factor", type=float, default=0.7)
    parser.add_argument("--adg_max_angle", type=float, default=1.0471975512)
    parser.add_argument("--apg_r", type=float, default=1.5)
    parser.add_argument("--apg_beta", type=float, default=-0.2)
    parser.add_argument(
        "--fdg_guidance_scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "FDG pyramid scales from high to low frequency. "
            "Default: cfg_scale low_cfg_scale."
        ),
    )
    parser.add_argument("--fdg_kernel_size", type=int, default=5)
    parser.add_argument("--fdg_sigma", type=float, default=1.0)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHOD_NAMES,
        choices=ALL_METHOD_NAMES,
        help="Methods to generate. Defaults to the seven main-table methods.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit to use all metadata entries.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Inclusive global metadata start index for prompt sharding.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Exclusive global metadata end index for prompt sharding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_metadata(path, max_samples, start_index, end_index):
    with open(path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, list):
        raise ValueError("metadata JSON must contain a list")

    required_fields = {"caption", "generated_index"}
    for index, item in enumerate(metadata):
        missing = required_fields - set(item)
        if missing:
            raise ValueError(
                f"metadata item {index} is missing fields: {sorted(missing)}"
            )

    if start_index < 0:
        raise ValueError("--start_index must be non-negative")
    if end_index is None:
        end_index = len(metadata)
    if end_index < start_index:
        raise ValueError("--end_index must be greater than or equal to --start_index")
    if end_index > len(metadata):
        raise ValueError(
            f"--end_index={end_index} exceeds metadata length {len(metadata)}"
        )

    indexed_metadata = list(enumerate(metadata))
    indexed_metadata = indexed_metadata[start_index:end_index]
    if max_samples is not None:
        indexed_metadata = indexed_metadata[:max_samples]
    return indexed_metadata


def validate_real_images(metadata, real_images_dir):
    missing = []
    for _, item in metadata:
        image_path = os.path.join(real_images_dir, f"{item['generated_index']}.jpg")
        if not os.path.exists(image_path):
            missing.append(image_path)
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} real images are missing. First entries:\n{preview}"
        )


def experiment_signature(args, metadata, methods):
    return {
        "num_samples": len(metadata),
        "global_start_index": args.start_index,
        "global_end_index": args.end_index,
        "methods": methods,
        "cfg_scale": args.cfg_scale,
        "num_steps": args.num_steps,
        "shift": args.shift,
        "base_seed": args.base_seed,
        "seed_rule": "base_seed + metadata list index",
        "low_cfg_scale": args.low_cfg_scale,
        "frequency_cutoff": args.frequency_cutoff,
        "high_radial_scale": args.high_radial_scale,
        "cfg_rescale_factor": args.cfg_rescale_factor,
        "adg_max_angle": args.adg_max_angle,
        "apg_r": args.apg_r,
        "apg_beta": args.apg_beta,
        "fdg_guidance_scales": args.fdg_guidance_scales,
        "fdg_kernel_size": args.fdg_kernel_size,
        "fdg_sigma": args.fdg_sigma,
        "jpeg_quality": args.jpeg_quality,
    }


def validate_or_write_signature(output_dir, signature, methods):
    os.makedirs(output_dir, exist_ok=True)
    signature_path = os.path.join(output_dir, "experiment_signature.json")
    if os.path.exists(signature_path):
        with open(signature_path, "r", encoding="utf-8") as file:
            existing = json.load(file)
        if existing != signature:
            raise RuntimeError(
                "Output directory contains a different experiment signature. "
                "Use a new --output_dir to avoid mixing runs."
            )
        return

    existing_images = []
    for method_name in methods:
        method_dir = os.path.join(output_dir, method_name)
        if os.path.isdir(method_dir):
            existing_images.extend(os.listdir(method_dir))
    if existing_images:
        raise RuntimeError(
            "Existing method outputs were found without an experiment signature. "
            "Use a clean --output_dir."
        )

    with open(signature_path, "w", encoding="utf-8") as file:
        json.dump(signature, file, indent=2, ensure_ascii=False)


def build_methods(wrapper, initial_noise, context, args):
    common = {
        "wrapper": wrapper,
        "z_T": initial_noise,
        "context": context,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "shift": args.shift,
    }
    fdg = {
        **common,
        "low_cfg_scale": args.low_cfg_scale,
        "guidance_scales": args.fdg_guidance_scales,
        "kernel_size": args.fdg_kernel_size,
        "sigma": args.fdg_sigma,
    }
    apg = {
        "r": args.apg_r,
        "beta": args.apg_beta,
    }

    return {
        "cfg": lambda: sample_standard_cfg(**common),
        "cfg_rescale": lambda: sample_cfg_rescale(
            **common, rescale_factor=args.cfg_rescale_factor
        ),
        "adg": lambda: sample_anchor_adg(
            **common, max_turn_angle=args.adg_max_angle
        ),
        "adg_norm": lambda: sample_anchor_adg(
            **common,
            max_turn_angle=args.adg_max_angle,
            preserve_text_norm=True,
        ),
        "apg": lambda: sample_anchor_apg(**common, **apg),
        "fdg": lambda: sample_fdg_laplacian(**fdg),
        "fdg_plus_apg": lambda: sample_fdg_plus_apg(**fdg, **apg),
        "rptg": lambda: sample_rptg(
            **common,
            cutoff_ratio=args.frequency_cutoff,
            high_radial_scale=0.0,
        ),
        "rptg_hr": lambda: sample_rptg_hr(
            **common,
            cutoff_ratio=args.frequency_cutoff,
            high_radial_scale=args.high_radial_scale,
        ),
        "rptg_apg": lambda: sample_rptg_apg_stabilized(
            **common,
            cutoff_ratio=args.frequency_cutoff,
            high_radial_scale=0.0,
            **apg,
        ),
        "rptg_hr_apg": lambda: sample_rptg_hr_apg_stabilized(
            **common,
            cutoff_ratio=args.frequency_cutoff,
            high_radial_scale=args.high_radial_scale,
            **apg,
        ),
    }


def run_generation():
    args = parse_cli_args()
    metadata = load_metadata(
        args.metadata_path,
        args.max_samples,
        args.start_index,
        args.end_index,
    )
    methods_to_run = args.methods
    validate_real_images(metadata, args.real_images_dir)
    signature = experiment_signature(args, metadata, methods_to_run)
    validate_or_write_signature(args.output_dir, signature, methods_to_run)

    for method_name in methods_to_run:
        os.makedirs(os.path.join(args.output_dir, method_name), exist_ok=True)

    with open(
        os.path.join(args.output_dir, "experiment_metadata.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump([item for _, item in metadata], file, indent=2, ensure_ascii=False)

    wrapper = SD3TransformerWrapper(
        device=args.device,
        dtype=torch.bfloat16,
        model_path=args.model_path,
    )

    runtimes = {method_name: 0.0 for method_name in methods_to_run}
    generated_counts = {method_name: 0 for method_name in methods_to_run}
    skipped_counts = {method_name: 0 for method_name in methods_to_run}

    for local_index, (global_index, item) in enumerate(tqdm(metadata, desc="Prompts")):
        prompt = item["caption"]
        generated_index = str(item["generated_index"])
        pending_methods = [
            method_name
            for method_name in methods_to_run
            if not os.path.exists(
                os.path.join(args.output_dir, method_name, f"{generated_index}.jpg")
            )
        ]
        if not pending_methods:
            for method_name in methods_to_run:
                skipped_counts[method_name] += 1
            continue

        context = wrapper.encode_prompt(prompt)
        generator = torch.Generator(device=wrapper.device).manual_seed(
            args.base_seed + global_index
        )
        initial_noise = torch.randn(
            (1, 16, 128, 128),
            generator=generator,
            device=wrapper.device,
            dtype=wrapper.dtype,
        )
        methods = build_methods(wrapper, initial_noise, context, args)

        for method_name in methods_to_run:
            output_path = os.path.join(
                args.output_dir, method_name, f"{generated_index}.jpg"
            )
            if os.path.exists(output_path):
                skipped_counts[method_name] += 1
                continue

            print(
                f"\n[{local_index + 1}/{len(metadata)} | global {global_index}] "
                f"{generated_index} | {method_name}"
            )
            start_time = time.time()
            latent = methods[method_name]()
            runtimes[method_name] += time.time() - start_time
            image = decode_sd3_latent_to_pil(wrapper, latent)
            image.save(output_path, format="JPEG", quality=args.jpeg_quality)
            generated_counts[method_name] += 1

            del latent, image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del context, initial_noise, methods
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_config = {
        "metadata_path": args.metadata_path,
        "real_images_dir": args.real_images_dir,
        **signature,
        "generated_counts": generated_counts,
        "skipped_counts": skipped_counts,
        "runtimes_seconds": runtimes,
    }
    with open(
        os.path.join(args.output_dir, "generation_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_config, file, indent=2, ensure_ascii=False)

    print(f"Generation complete: {args.output_dir}")


if __name__ == "__main__":
    run_generation()
