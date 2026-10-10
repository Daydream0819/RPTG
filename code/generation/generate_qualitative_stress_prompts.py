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

from generate_coco_innovation_benchmark import build_methods
from models.latent_io import decode_sd3_latent_to_pil
from models.sd3_model import SD3TransformerWrapper


QUALITATIVE_PROMPTS = [
    {
        "id": "scene_neon_market",
        "category": "scene",
        "prompt": (
            "A neon-lit street market at night, colorful signs, wet pavement, "
            "cinematic lighting, highly detailed"
        ),
        "seed": 1042,
    },
    {
        "id": "object_white_teapot",
        "category": "bright object",
        "prompt": (
            "A white porcelain teapot on a white table beside a sunlit window, "
            "soft shadows, realistic photography"
        ),
        "seed": 1043,
    },
    {
        "id": "food_fruit_bowl",
        "category": "color and texture",
        "prompt": (
            "A close-up photo of a bowl of fresh fruit with strawberries, "
            "oranges, blueberries, and mint leaves, natural light"
        ),
        "seed": 1044,
    },
    {
        "id": "pink_dog",
        "category": "colorful animal",
        "prompt": (
            "A fluffy small dog with soft pastel pink fur sitting on a clean "
            "white studio floor, gentle daylight, realistic photography, "
            "detailed fur texture, soft shadows"
        ),
        "seed": 1048,
    },
    {
        "id": "portrait_blue_scarf",
        "category": "portrait",
        "prompt": (
            "A portrait of an elderly woman wearing a blue scarf, warm indoor "
            "lighting, detailed skin texture, realistic photo"
        ),
        "seed": 1045,
    },
    {
        "id": "architecture_cathedral",
        "category": "architecture",
        "prompt": (
            "A detailed photo of a gothic cathedral interior with stained glass "
            "windows and stone arches, natural light"
        ),
        "seed": 1046,
    },
    {
        "id": "animal_red_fox",
        "category": "animal",
        "prompt": (
            "A red fox standing in a snowy forest, detailed fur, soft morning "
            "light, realistic wildlife photography"
        ),
        "seed": 1047,
    },
]


METHOD_ALIASES = {
    "cfg": "cfg",
    "cfg_rescale": "cfg_rescale",
    "adg": "adg",
    "adg_norm": "adg_norm",
    "apg": "apg",
    "fdg": "fdg",
    "rptg": "rptg_apg",
    "rptg_hr": "rptg_hr_apg",
}

DISPLAY_NAMES = {
    "cfg": "CFG",
    "cfg_rescale": "CFG-Rescale",
    "adg": "ADG",
    "adg_norm": "ADG-Norm",
    "apg": "APG",
    "fdg": "FDG",
    "rptg": "RPTG",
    "rptg_hr": "RPTG-HR",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate qualitative stress-test examples for RPTG."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/results/rptg_qualitative_stress_w15",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/hf_cache",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cfg", "cfg_rescale", "apg", "fdg", "rptg", "rptg_hr"],
        choices=list(METHOD_ALIASES),
    )
    parser.add_argument(
        "--prompt_ids",
        nargs="+",
        default=None,
        help="Optional subset of prompt ids to generate.",
    )
    parser.add_argument("--cfg_scale", type=float, default=15.0)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--frequency_cutoff", type=float, default=0.25)
    parser.add_argument("--high_radial_scale", type=float, default=0.25)
    parser.add_argument("--low_cfg_scale", type=float, default=1.0)
    parser.add_argument("--cfg_rescale_factor", type=float, default=0.7)
    parser.add_argument("--adg_max_angle", type=float, default=1.0471975512)
    parser.add_argument("--apg_r", type=float, default=1.5)
    parser.add_argument("--apg_beta", type=float, default=-0.2)
    parser.add_argument("--fdg_kernel_size", type=int, default=5)
    parser.add_argument("--fdg_sigma", type=float, default=1.0)
    parser.add_argument(
        "--fdg_guidance_scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "FDG pyramid scales from high to low frequency. "
            "Default follows generate_coco_innovation_benchmark.py."
        ),
    )
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--make_grid", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def selected_prompts(prompt_ids):
    if prompt_ids is None:
        return QUALITATIVE_PROMPTS
    prompt_by_id = {item["id"]: item for item in QUALITATIVE_PROMPTS}
    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in prompt_by_id]
    if missing:
        raise ValueError(f"Unknown prompt ids: {missing}")
    return [prompt_by_id[prompt_id] for prompt_id in prompt_ids]


def write_experiment_files(args, prompts):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "qualitative_prompts.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(prompts, file, indent=2, ensure_ascii=False)

    config = {
        "methods": args.methods,
        "method_aliases": {name: METHOD_ALIASES[name] for name in args.methods},
        "cfg_scale": args.cfg_scale,
        "num_steps": args.num_steps,
        "shift": args.shift,
        "frequency_cutoff": args.frequency_cutoff,
        "high_radial_scale": args.high_radial_scale,
        "low_cfg_scale": args.low_cfg_scale,
        "cfg_rescale_factor": args.cfg_rescale_factor,
        "adg_max_angle": args.adg_max_angle,
        "apg_r": args.apg_r,
        "apg_beta": args.apg_beta,
        "fdg_kernel_size": args.fdg_kernel_size,
        "fdg_sigma": args.fdg_sigma,
        "jpeg_quality": args.jpeg_quality,
        "note": (
            "Displayed RPTG/RPTG-HR use the stabilized implementations "
            "rptg_apg/rptg_hr_apg to match the main quantitative table."
        ),
    }
    with open(
        os.path.join(args.output_dir, "qualitative_generation_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def make_contact_sheet(output_dir, methods, prompts):
    cell_size = 256
    left_margin = 190
    top_margin = 64
    caption_height = 72
    grid_width = left_margin + cell_size * len(methods)
    grid_height = top_margin + (cell_size + caption_height) * len(prompts)
    canvas = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for col, method in enumerate(methods):
        x = left_margin + col * cell_size
        draw.text((x + 8, 20), DISPLAY_NAMES[method], fill="black", font=font)

    for row, item in enumerate(prompts):
        y = top_margin + row * (cell_size + caption_height)
        draw.text((8, y + 8), item["category"], fill="black", font=font)
        draw.text((8, y + 34), f"seed={item['seed']}", fill="black", font=small_font)
        draw.text((8, y + 54), item["id"], fill="black", font=small_font)
        for col, method in enumerate(methods):
            image_path = os.path.join(output_dir, method, f"{item['id']}.jpg")
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((cell_size, cell_size), Image.LANCZOS)
            tile = Image.new("RGB", (cell_size, cell_size), "white")
            x_offset = (cell_size - image.width) // 2
            y_offset = (cell_size - image.height) // 2
            tile.paste(image, (x_offset, y_offset))
            canvas.paste(tile, (left_margin + col * cell_size, y))

    grid_path = os.path.join(output_dir, "qualitative_grid.jpg")
    canvas.save(grid_path, format="JPEG", quality=95)
    print(f"Saved grid: {grid_path}")


def main():
    args = parse_args()
    prompts = selected_prompts(args.prompt_ids)
    write_experiment_files(args, prompts)
    for method in args.methods:
        os.makedirs(os.path.join(args.output_dir, method), exist_ok=True)

    wrapper = SD3TransformerWrapper(
        device=args.device,
        dtype=torch.bfloat16,
        model_path=args.model_path,
    )

    runtimes = {method: 0.0 for method in args.methods}
    for item in tqdm(prompts, desc="Qualitative prompts"):
        prompt = item["prompt"]
        context = wrapper.encode_prompt(prompt)
        generator = torch.Generator(device=wrapper.device).manual_seed(item["seed"])
        initial_noise = torch.randn(
            (1, 16, 128, 128),
            generator=generator,
            device=wrapper.device,
            dtype=wrapper.dtype,
        )
        methods = build_methods(wrapper, initial_noise, context, args)

        for display_method in args.methods:
            output_path = os.path.join(
                args.output_dir,
                display_method,
                f"{item['id']}.jpg",
            )
            if os.path.exists(output_path):
                continue
            actual_method = METHOD_ALIASES[display_method]
            print(f"\n{item['id']} | {display_method} ({actual_method})")
            start_time = time.time()
            latent = methods[actual_method]()
            runtimes[display_method] += time.time() - start_time
            image = decode_sd3_latent_to_pil(wrapper, latent)
            image.save(output_path, format="JPEG", quality=args.jpeg_quality)
            del latent, image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del context, initial_noise, methods
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(
        os.path.join(args.output_dir, "qualitative_runtimes.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(runtimes, file, indent=2, ensure_ascii=False)

    if args.make_grid:
        make_contact_sheet(args.output_dir, args.methods, prompts)

    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
