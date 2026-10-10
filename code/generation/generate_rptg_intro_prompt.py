import argparse
import json
import os
import re
import sys
import time

import torch
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.guidance_methods import (
    sample_anchor_adg,
    sample_anchor_apg,
    sample_fdg_laplacian,
    sample_rptg_directional_ablation,
    sample_rptg_hr_apg_stabilized,
    sample_standard_cfg,
)
from models.latent_io import decode_sd3_latent_to_pil
from models.sd3_model import SD3TransformerWrapper


DEFAULT_PROMPT = (
    "A fluffy white Samoyed dog wearing a red scarf, sitting in fresh snow "
    "under bright winter sunlight, detailed fur, sparkling snow, shallow "
    "depth of field, photorealistic"
)

COMPACT_METHODS = ["cfg", "radial", "tangential", "rptg_hr"]
FULL_METHODS = ["cfg", "adg", "apg", "fdg", "radial", "tangential", "rptg_hr"]

METHOD_LABELS = {
    "cfg": "CFG",
    "adg": "ADG",
    "apg": "APG",
    "fdg": "FDG",
    "radial": "Radial",
    "tangential": "Tangential",
    "rptg_hr": "RPTG-HR",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a single-prompt qualitative panel for the RPTG intro figure."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/results/rptg_intro_prompt_w15",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/hf_cache",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--prompt_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20270625)
    parser.add_argument("--cfg_scale", type=float, default=15.0)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--frequency_cutoff", type=float, default=0.25)
    parser.add_argument("--high_radial_scale", type=float, default=0.25)
    parser.add_argument(
        "--radial_band",
        choices=["low", "high", "all"],
        default="low",
        help=(
            "Frequency band used for the Radial diagnostic column. "
            "Use low for the intro mechanism figure; use all for full-frequency ablation."
        ),
    )
    parser.add_argument(
        "--radial_direction",
        choices=["full_radial", "positive_radial", "negative_radial"],
        default="positive_radial",
        help=(
            "Radial component used for the Radial diagnostic column. "
            "positive_radial highlights amplitude growth."
        ),
    )
    parser.add_argument("--low_cfg_scale", type=float, default=1.0)
    parser.add_argument("--adg_max_angle", type=float, default=1.0471975512)
    parser.add_argument("--apg_r", type=float, default=1.5)
    parser.add_argument("--apg_beta", type=float, default=-0.2)
    parser.add_argument("--fdg_kernel_size", type=int, default=5)
    parser.add_argument("--fdg_sigma", type=float, default=1.0)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--thumb_size", type=int, default=384)
    parser.add_argument("--label_height", type=int, default=42)
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument(
        "--preset",
        choices=["compact", "full"],
        default="compact",
        help=(
            "compact: CFG/Radial/Tangential/RPTG-HR. "
            "full: adds ADG/APG/FDG baselines."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=list(METHOD_LABELS),
        help="Optional explicit method list. Overrides --preset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:64] or "prompt"


def make_initial_noise(wrapper, seed):
    generator = torch.Generator(device=wrapper.device).manual_seed(seed)
    return torch.randn(
        (1, 16, 128, 128),
        generator=generator,
        device=wrapper.device,
        dtype=wrapper.dtype,
    )


def build_sampler(method, wrapper, initial_noise, context, args):
    common = {
        "wrapper": wrapper,
        "z_T": initial_noise,
        "context": context,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "shift": args.shift,
    }
    if method == "cfg":
        return lambda: sample_standard_cfg(**common)
    if method == "adg":
        return lambda: sample_anchor_adg(
            **common,
            max_turn_angle=args.adg_max_angle,
        )
    if method == "apg":
        return lambda: sample_anchor_apg(
            **common,
            r=args.apg_r,
            beta=args.apg_beta,
        )
    if method == "fdg":
        return lambda: sample_fdg_laplacian(
            **common,
            low_cfg_scale=args.low_cfg_scale,
            guidance_scales=None,
            kernel_size=args.fdg_kernel_size,
            sigma=args.fdg_sigma,
        )
    if method == "radial":
        return lambda: sample_rptg_directional_ablation(
            **common,
            cutoff_ratio=args.frequency_cutoff,
            band=args.radial_band,
            direction=args.radial_direction,
        )
    if method == "tangential":
        return lambda: sample_rptg_directional_ablation(
            **common,
            cutoff_ratio=args.frequency_cutoff,
            band="all",
            direction="tangential",
        )
    if method == "rptg_hr":
        return lambda: sample_rptg_hr_apg_stabilized(
            **common,
            cutoff_ratio=args.frequency_cutoff,
            high_radial_scale=args.high_radial_scale,
            r=args.apg_r,
            beta=args.apg_beta,
        )
    raise ValueError(f"Unknown method: {method}")


def save_config(args, methods, prompt_id):
    os.makedirs(args.output_dir, exist_ok=True)
    config = {
        "prompt_id": prompt_id,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "methods": methods,
        "method_labels": {method: METHOD_LABELS[method] for method in methods},
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "num_steps": args.num_steps,
        "shift": args.shift,
        "frequency_cutoff": args.frequency_cutoff,
        "high_radial_scale": args.high_radial_scale,
        "radial_band": args.radial_band,
        "radial_direction": args.radial_direction,
        "low_cfg_scale": args.low_cfg_scale,
        "adg_max_angle": args.adg_max_angle,
        "apg_r": args.apg_r,
        "apg_beta": args.apg_beta,
        "fdg_kernel_size": args.fdg_kernel_size,
        "fdg_sigma": args.fdg_sigma,
        "note": (
            "Radial and Tangential are diagnostic components. "
            "The default radial column uses low-frequency positive radial updates "
            "to highlight amplitude growth. "
            "RPTG-HR uses the APG-stabilized implementation from the main table."
        ),
    }
    path = os.path.join(args.output_dir, "intro_prompt_config.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
    return path


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_panel(output_dir, methods, prompt_id, args):
    thumb = args.thumb_size
    pad = args.padding
    label_h = args.label_height
    width = len(methods) * thumb + (len(methods) + 1) * pad
    height = label_h + thumb + 2 * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(20, bold=True)

    for col, method in enumerate(methods):
        x = pad + col * (thumb + pad)
        label = METHOD_LABELS[method]
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (x + (thumb - (bbox[2] - bbox[0])) / 2, 10),
            label,
            fill=(20, 20, 20),
            font=font,
        )
        image_path = os.path.join(output_dir, method, f"{prompt_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb, thumb), "white")
        tile.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
        canvas.paste(tile, (x, label_h + pad))

    panel_path = os.path.join(output_dir, "intro_prompt_panel.jpg")
    canvas.save(panel_path, format="JPEG", quality=args.jpeg_quality)
    return panel_path


def run_generation():
    args = parse_args()
    methods = args.methods
    if methods is None:
        methods = COMPACT_METHODS if args.preset == "compact" else FULL_METHODS
    prompt_id = args.prompt_id or slugify(args.prompt)
    config_path = save_config(args, methods, prompt_id)

    for method in methods:
        os.makedirs(os.path.join(args.output_dir, method), exist_ok=True)

    wrapper = SD3TransformerWrapper(
        device=args.device,
        dtype=torch.bfloat16,
        model_path=args.model_path,
    )
    context = wrapper.encode_prompt(args.prompt, negative_prompt=args.negative_prompt)
    initial_noise = make_initial_noise(wrapper, args.seed)

    runtimes = {}
    for method in methods:
        output_path = os.path.join(args.output_dir, method, f"{prompt_id}.jpg")
        if os.path.exists(output_path):
            print(f"Skip existing: {output_path}")
            continue
        print(f"Generating {METHOD_LABELS[method]} -> {output_path}")
        sampler = build_sampler(method, wrapper, initial_noise, context, args)
        start_time = time.time()
        latent = sampler()
        runtimes[method] = time.time() - start_time
        image = decode_sd3_latent_to_pil(wrapper, latent)
        image.save(output_path, format="JPEG", quality=args.jpeg_quality)
        del latent, image, sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    panel_path = make_panel(args.output_dir, methods, prompt_id, args)
    runtime_path = os.path.join(args.output_dir, "intro_prompt_runtimes.json")
    with open(runtime_path, "w", encoding="utf-8") as file:
        json.dump(runtimes, file, indent=2)

    print(f"Config: {config_path}")
    print(f"Panel: {panel_path}")
    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    run_generation()
