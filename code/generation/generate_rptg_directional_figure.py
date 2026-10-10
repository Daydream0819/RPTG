import argparse
import json
import os
import sys
import time

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.guidance_methods import (
    sample_rptg_directional_ablation,
    sample_rptg_hr_apg_stabilized,
    sample_standard_cfg,
)
from models.latent_io import decode_sd3_latent_to_pil
from models.sd3_model import SD3TransformerWrapper


DEFAULT_PROMPTS = [
    "A fluffy white Samoyed dog wearing a red scarf, sitting in fresh snow under bright winter sunlight, detailed fur, sparkling snow, shallow depth of field, photorealistic",
    "A dramatic alpine lake at sunset, snow-covered mountains reflected in crystal clear water, orange clouds, deep blue sky, cinematic landscape photography, ultra detailed",
    "A clean product poster on a white desk with large readable text saying RPTG, blue and orange geometric shapes, studio lighting, sharp typography, high resolution",
    "A glossy red sports car parked on a wet city street at night, neon signs reflected on the car body and pavement, sharp headlights, cinematic lighting, photorealistic",
]


METHOD_SPECS = [
    {
        "name": "cfg",
        "label": "CFG",
        "kind": "cfg",
    },
    {
        "name": "low_full_radial",
        "label": "Low Full Radial",
        "kind": "directional",
        "band": "low",
        "direction": "full_radial",
    },
    {
        "name": "low_pos_radial",
        "label": "Low +Radial",
        "kind": "directional",
        "band": "low",
        "direction": "positive_radial",
    },
    {
        "name": "low_neg_radial",
        "label": "Low -Radial",
        "kind": "directional",
        "band": "low",
        "direction": "negative_radial",
    },
    {
        "name": "low_tangential",
        "label": "Low Tangential",
        "kind": "directional",
        "band": "low",
        "direction": "tangential",
    },
    {
        "name": "high_full_radial",
        "label": "High Full Radial",
        "kind": "directional",
        "band": "high",
        "direction": "full_radial",
    },
    {
        "name": "high_pos_radial",
        "label": "High +Radial",
        "kind": "directional",
        "band": "high",
        "direction": "positive_radial",
    },
    {
        "name": "high_neg_radial",
        "label": "High -Radial",
        "kind": "directional",
        "band": "high",
        "direction": "negative_radial",
    },
    {
        "name": "high_tangential",
        "label": "High Tangential",
        "kind": "directional",
        "band": "high",
        "direction": "tangential",
    },
    {
        "name": "full_radial",
        "label": "Full Radial",
        "kind": "directional",
        "band": "all",
        "direction": "full_radial",
    },
    {
        "name": "full_tangential",
        "label": "Full Tangential",
        "kind": "directional",
        "band": "all",
        "direction": "tangential",
    },
    {
        "name": "rptg_hr_apg",
        "label": "RPTG-HR (APG opt.)",
        "kind": "rptg_hr_apg",
    },
]


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Generate the RPTG frequency-direction ablation figure."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/results/rptg_directional_figure_w15",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/hf_cache",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default=None,
        help="Optional JSON file containing a list of prompts. Defaults to four paper prompts.",
    )
    parser.add_argument("--cfg_scale", type=float, default=15.0)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--base_seed", type=int, default=20270618)
    parser.add_argument("--frequency_cutoff", type=float, default=0.25)
    parser.add_argument("--high_radial_scale", type=float, default=0.25)
    parser.add_argument("--apg_r", type=float, default=1.5)
    parser.add_argument("--apg_beta", type=float, default=-0.2)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--grid_thumb_size", type=int, default=256)
    parser.add_argument("--grid_padding", type=int, default=14)
    parser.add_argument("--grid_label_height", type=int, default=54)
    parser.add_argument("--grid_prompt_width", type=int, default=58)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_prompts(path):
    if path is None:
        return DEFAULT_PROMPTS
    with open(path, "r", encoding="utf-8") as file:
        prompts = json.load(file)
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError("--prompts_json must contain a JSON list of strings")
    if not prompts:
        raise ValueError("--prompts_json must contain at least one prompt")
    return prompts


def make_initial_noise(wrapper, seed):
    generator = torch.Generator(device=wrapper.device).manual_seed(seed)
    return torch.randn(
        (1, 16, 128, 128),
        generator=generator,
        device=wrapper.device,
        dtype=wrapper.dtype,
    )


def build_sampler(method_spec, wrapper, initial_noise, context, args):
    common = {
        "wrapper": wrapper,
        "z_T": initial_noise,
        "context": context,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "shift": args.shift,
    }
    if method_spec["kind"] == "cfg":
        return lambda: sample_standard_cfg(**common)
    if method_spec["kind"] == "directional":
        return lambda: sample_rptg_directional_ablation(
            **common,
            cutoff_ratio=args.frequency_cutoff,
            band=method_spec["band"],
            direction=method_spec["direction"],
        )
    if method_spec["kind"] == "rptg_hr_apg":
        return lambda: sample_rptg_hr_apg_stabilized(
            **common,
            cutoff_ratio=args.frequency_cutoff,
            high_radial_scale=args.high_radial_scale,
            r=args.apg_r,
            beta=args.apg_beta,
        )
    raise ValueError(f"Unknown method kind: {method_spec['kind']}")


def save_run_config(args, prompts):
    os.makedirs(args.output_dir, exist_ok=True)
    config = {
        "prompts": prompts,
        "methods": METHOD_SPECS,
        "cfg_scale": args.cfg_scale,
        "num_steps": args.num_steps,
        "shift": args.shift,
        "base_seed": args.base_seed,
        "seed_rule": "base_seed + prompt_index",
        "frequency_cutoff": args.frequency_cutoff,
        "high_radial_scale": args.high_radial_scale,
        "apg_r": args.apg_r,
        "apg_beta": args.apg_beta,
    }
    with open(
        os.path.join(args.output_dir, "directional_figure_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def fit_text(draw, text, max_width, font):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_grid(output_dir, prompts, args):
    thumb = args.grid_thumb_size
    pad = args.grid_padding
    label_h = args.grid_label_height
    prompt_w = args.grid_prompt_width
    cols = len(METHOD_SPECS)
    rows = len(prompts)

    width = prompt_w + cols * thumb + (cols + 1) * pad
    height = label_h + rows * thumb + (rows + 1) * pad
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
        small_font = ImageFont.truetype("Arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for col, spec in enumerate(METHOD_SPECS):
        x0 = prompt_w + pad + col * (thumb + pad)
        label_lines = fit_text(draw, spec["label"], thumb, font)
        y = 8
        for line in label_lines[:2]:
            bbox = draw.textbbox((0, 0), line, font=font)
            draw.text((x0 + (thumb - (bbox[2] - bbox[0])) / 2, y), line, fill="black", font=font)
            y += 21

    for row, prompt in enumerate(prompts):
        y0 = label_h + pad + row * (thumb + pad)
        prompt_label = f"P{row + 1}"
        bbox = draw.textbbox((0, 0), prompt_label, font=font)
        draw.text(
            (pad + (prompt_w - (bbox[2] - bbox[0])) / 2, y0 + 8),
            prompt_label,
            fill="black",
            font=font,
        )
        for col, spec in enumerate(METHOD_SPECS):
            image_path = os.path.join(output_dir, spec["name"], f"prompt_{row:02d}.jpg")
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            cell = Image.new("RGB", (thumb, thumb), "white")
            cell.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
            x0 = prompt_w + pad + col * (thumb + pad)
            grid.paste(cell, (x0, y0))

    grid_path = os.path.join(output_dir, "rptg_directional_grid.png")
    grid.save(grid_path)
    return grid_path


def run_generation():
    args = parse_cli_args()
    prompts = load_prompts(args.prompts_json)
    save_run_config(args, prompts)

    for spec in METHOD_SPECS:
        os.makedirs(os.path.join(args.output_dir, spec["name"]), exist_ok=True)

    wrapper = SD3TransformerWrapper(
        device=args.device,
        dtype=torch.bfloat16,
        model_path=args.model_path,
    )

    runtimes = {spec["name"]: 0.0 for spec in METHOD_SPECS}
    for prompt_index, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        context = wrapper.encode_prompt(prompt)
        seed = args.base_seed + prompt_index
        initial_noise = make_initial_noise(wrapper, seed)

        for spec in METHOD_SPECS:
            output_path = os.path.join(
                args.output_dir, spec["name"], f"prompt_{prompt_index:02d}.jpg"
            )
            if os.path.exists(output_path):
                continue
            print(f"\n[prompt {prompt_index + 1}/{len(prompts)}] {spec['label']}")
            sampler = build_sampler(spec, wrapper, initial_noise, context, args)
            start = time.time()
            latent = sampler()
            runtimes[spec["name"]] += time.time() - start
            image = decode_sd3_latent_to_pil(wrapper, latent)
            image.save(output_path, format="JPEG", quality=args.jpeg_quality)

            del latent, image, sampler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del context, initial_noise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grid_path = make_grid(args.output_dir, prompts, args)
    with open(
        os.path.join(args.output_dir, "directional_figure_runtimes.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(runtimes, file, indent=2)
    print(f"Generation complete: {args.output_dir}")
    print(f"Grid saved to: {grid_path}")


if __name__ == "__main__":
    run_generation()
