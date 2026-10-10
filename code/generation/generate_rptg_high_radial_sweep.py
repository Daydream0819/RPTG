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

from generate_coco_innovation_benchmark import (
    load_metadata,
    validate_or_write_signature,
    validate_real_images,
)
from models.guidance_methods import (
    sample_rptg_hr,
    sample_rptg_hr_apg_stabilized,
)
from models.latent_io import decode_sd3_latent_to_pil
from models.sd3_model import SD3TransformerWrapper


def format_scale_tag(value):
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Generate an RPTG-HR high-frequency radial-weight sweep."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="/root/autodl-tmp/output_gamma_1000/experiment_metadata.json",
    )
    parser.add_argument(
        "--real_images_dir",
        type=str,
        default="/root/autodl-tmp/output_gamma_1000/real_images_for_fid",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/results/rptg_high_radial_sweep_w15_1000",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/hf_cache",
    )
    parser.add_argument(
        "--high_radial_scales",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 1.0],
        help="Values of gamma_H to evaluate.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="stabilized",
        choices=("stabilized", "pure"),
        help=(
            "stabilized uses RPTG-HR with APG-style clipping and reverse "
            "momentum; pure uses the bare RPTG-HR update."
        ),
    )
    parser.add_argument("--method_prefix", type=str, default="rptg_hr_g")
    parser.add_argument("--cfg_scale", type=float, default=15.0)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--frequency_cutoff", type=float, default=0.25)
    parser.add_argument("--apg_r", type=float, default=1.5)
    parser.add_argument("--apg_beta", type=float, default=-0.2)
    parser.add_argument("--jpeg_quality", type=int, default=95)
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


def build_sweep_methods(args):
    seen = set()
    methods = []
    for scale in args.high_radial_scales:
        name = f"{args.method_prefix}{format_scale_tag(scale)}"
        if name in seen:
            raise ValueError(f"Duplicate method name from gamma_H={scale}: {name}")
        seen.add(name)
        methods.append((name, float(scale)))
    return methods


def experiment_signature(args, metadata, methods):
    return {
        "experiment": "rptg_high_radial_sweep",
        "num_samples": len(metadata),
        "global_start_index": args.start_index,
        "global_end_index": args.end_index,
        "methods": [name for name, _ in methods],
        "method_high_radial_scales": {name: scale for name, scale in methods},
        "variant": args.variant,
        "cfg_scale": args.cfg_scale,
        "num_steps": args.num_steps,
        "shift": args.shift,
        "base_seed": args.base_seed,
        "seed_rule": "base_seed + metadata list index",
        "frequency_cutoff": args.frequency_cutoff,
        "apg_r": args.apg_r,
        "apg_beta": args.apg_beta,
        "jpeg_quality": args.jpeg_quality,
    }


def sample_one(wrapper, initial_noise, context, high_radial_scale, args):
    common = {
        "wrapper": wrapper,
        "z_T": initial_noise,
        "context": context,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "shift": args.shift,
        "cutoff_ratio": args.frequency_cutoff,
        "high_radial_scale": high_radial_scale,
    }
    if args.variant == "pure":
        return sample_rptg_hr(**common)
    return sample_rptg_hr_apg_stabilized(
        **common,
        r=args.apg_r,
        beta=args.apg_beta,
    )


def run_generation():
    args = parse_cli_args()
    metadata = load_metadata(
        args.metadata_path,
        args.max_samples,
        args.start_index,
        args.end_index,
    )
    methods = build_sweep_methods(args)
    validate_real_images(metadata, args.real_images_dir)

    signature = experiment_signature(args, metadata, methods)
    validate_or_write_signature(
        args.output_dir,
        signature,
        [name for name, _ in methods],
    )

    for method_name, _ in methods:
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

    runtimes = {method_name: 0.0 for method_name, _ in methods}
    generated_counts = {method_name: 0 for method_name, _ in methods}
    skipped_counts = {method_name: 0 for method_name, _ in methods}

    for local_index, (global_index, item) in enumerate(tqdm(metadata, desc="Prompts")):
        prompt = item["caption"]
        generated_index = str(item["generated_index"])
        pending = [
            (method_name, scale)
            for method_name, scale in methods
            if not os.path.exists(
                os.path.join(args.output_dir, method_name, f"{generated_index}.jpg")
            )
        ]
        if not pending:
            for method_name, _ in methods:
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

        for method_name, scale in methods:
            output_path = os.path.join(
                args.output_dir, method_name, f"{generated_index}.jpg"
            )
            if os.path.exists(output_path):
                skipped_counts[method_name] += 1
                continue

            print(
                f"\n[{local_index + 1}/{len(metadata)} | global {global_index}] "
                f"{generated_index} | {method_name} | gamma_H={scale:g}"
            )
            start_time = time.time()
            latent = sample_one(wrapper, initial_noise, context, scale, args)
            runtimes[method_name] += time.time() - start_time
            image = decode_sd3_latent_to_pil(wrapper, latent)
            image.save(output_path, format="JPEG", quality=args.jpeg_quality)
            generated_counts[method_name] += 1

            del latent, image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del context, initial_noise
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
