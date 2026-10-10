import argparse
import csv
import json
import os
import sys

import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from models.guidance_methods import (
    clean_anchor,
    euler_step_from_velocity,
    make_sd3_timesteps,
    velocity_from_anchor,
)
from models.sd3_model import SD3TransformerWrapper


COMPONENT_SPECS = [
    ("low_positive_radial", "low", "positive_radial"),
    ("low_negative_radial", "low", "negative_radial"),
    ("low_tangential", "low", "tangential"),
    ("high_positive_radial", "high", "positive_radial"),
    ("high_negative_radial", "high", "negative_radial"),
    ("high_tangential", "high", "tangential"),
]


def build_radial_frequency_mask(x, cutoff_ratio=0.25, band="low"):
    if not 0.0 < cutoff_ratio <= 1.0:
        raise ValueError("cutoff_ratio must be in (0, 1]")
    if band not in {"low", "high"}:
        raise ValueError("band must be 'low' or 'high'")

    height, width = x.shape[-2:]
    freq_y = torch.fft.fftfreq(height, device=x.device, dtype=torch.float32)
    freq_x = torch.fft.rfftfreq(width, device=x.device, dtype=torch.float32)
    radius = torch.sqrt(freq_y[:, None].square() + freq_x[None, :].square())
    low_mask = (radius <= 0.5 * cutoff_ratio).to(torch.float32)
    mask = low_mask if band == "low" else 1.0 - low_mask
    return mask.view(1, 1, height, width // 2 + 1)


def fft_radial_tangential_signed_components(x0_c, x0_u, cutoff_ratio=0.25):
    spectrum_c = torch.fft.rfft2(x0_c.float(), dim=(-2, -1), norm="ortho")
    spectrum_u = torch.fft.rfft2(x0_u.float(), dim=(-2, -1), norm="ortho")
    residual = spectrum_c - spectrum_u

    unit = spectrum_c / (torch.abs(spectrum_c) + 1e-8)
    radial_coeff = torch.real(residual * torch.conj(unit))
    radial = radial_coeff * unit
    tangential = residual - radial

    low_mask = build_radial_frequency_mask(x0_c, cutoff_ratio=cutoff_ratio, band="low")
    high_mask = build_radial_frequency_mask(x0_c, cutoff_ratio=cutoff_ratio, band="high")
    return spectrum_c, radial, tangential, radial_coeff, low_mask, high_mask


def select_frequency_direction_component(
    radial,
    tangential,
    radial_coeff,
    low_mask,
    high_mask,
    band,
    direction,
):
    if band not in {"low", "high"}:
        raise ValueError("band must be 'low' or 'high'")
    if direction not in {"positive_radial", "negative_radial", "tangential"}:
        raise ValueError(
            "direction must be 'positive_radial', 'negative_radial', or 'tangential'"
        )

    frequency_mask = low_mask if band == "low" else high_mask
    if direction == "positive_radial":
        direction_mask = (radial_coeff > 0).to(radial.dtype)
        return frequency_mask * direction_mask * radial
    if direction == "negative_radial":
        direction_mask = (radial_coeff < 0).to(radial.dtype)
        return frequency_mask * direction_mask * radial
    return frequency_mask * tangential


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure weighted RPTG frequency-direction component energies over "
            "guidance scales without decoding images."
        )
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="/root/autodl-tmp/output/experiment_metadata.json",
    )
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/hf_cache")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/results/rptg_directional_energy",
    )
    parser.add_argument(
        "--cfg_scales",
        type=float,
        nargs="+",
        default=[3.0, 5.0, 7.5, 10.0, 12.5, 15.0],
    )
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--frequency_cutoff", type=float, default=0.25)
    parser.add_argument("--high_radial_scale", type=float, default=0.25)
    parser.add_argument(
        "--trajectory",
        choices=["cfg", "rptg", "rptg_hr"],
        default="cfg",
        help=(
            "Trajectory used to collect residuals. The default 'cfg' measures "
            "all components along the standard CFG trajectory for each w."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument(
        "--save_step_rows",
        action="store_true",
        help="Also write per-prompt per-step component rows. This can be large.",
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

    indexed_metadata = list(enumerate(metadata))[start_index:end_index]
    if max_samples is not None:
        indexed_metadata = indexed_metadata[:max_samples]
    return indexed_metadata


def empty_stats():
    return {
        "weighted_energy_sum": 0.0,
        "unweighted_energy_sum": 0.0,
        "total_weighted_residual_energy_sum": 0.0,
        "signed_radial_coupling_sum": 0.0,
        "positive_radial_coupling_sum": 0.0,
        "absolute_radial_coupling_sum": 0.0,
        "num_coefficients": 0,
        "num_steps": 0,
        "num_prompts": 0,
    }


def add_component_stats(stats, component, total_residual, lambda_scale):
    weighted_component = lambda_scale * component
    weighted_residual = lambda_scale * total_residual
    stats["weighted_energy_sum"] += torch.sum(torch.abs(weighted_component).square()).item()
    stats["unweighted_energy_sum"] += torch.sum(torch.abs(component).square()).item()
    stats["total_weighted_residual_energy_sum"] += torch.sum(
        torch.abs(weighted_residual).square()
    ).item()
    stats["num_coefficients"] += component.numel()
    stats["num_steps"] += 1


def add_radial_coupling_stats(
    stats,
    spectrum_c,
    radial_coeff,
    frequency_mask,
    sign_mask,
    lambda_scale,
):
    # This is the first-order term 2 lambda A_c a_r from Eq. energy_change,
    # restricted to the selected signed radial component.
    coupling = (
        2.0
        * lambda_scale
        * torch.abs(spectrum_c)
        * radial_coeff
        * frequency_mask.to(radial_coeff.dtype)
        * sign_mask.to(radial_coeff.dtype)
    )
    stats["signed_radial_coupling_sum"] += torch.sum(coupling).item()
    stats["positive_radial_coupling_sum"] += torch.sum(torch.clamp(coupling, min=0.0)).item()
    stats["absolute_radial_coupling_sum"] += torch.sum(torch.abs(coupling)).item()


def build_component_tensors(radial, tangential, radial_coeff, low_mask, high_mask):
    components = {}
    for name, band, direction in COMPONENT_SPECS:
        components[name] = select_frequency_direction_component(
            radial=radial,
            tangential=tangential,
            radial_coeff=radial_coeff,
            low_mask=low_mask,
            high_mask=high_mask,
            band=band,
            direction=direction,
        )
    return components


def advance_trajectory(
    z_t,
    t_curr,
    t_next,
    v_u,
    v_c,
    x0_c,
    spectrum_c,
    radial,
    tangential,
    high_mask,
    cfg_scale,
    trajectory,
    high_radial_scale,
):
    if trajectory == "cfg":
        v_guided = v_u + cfg_scale * (v_c - v_u)
        return euler_step_from_velocity(z_t, t_curr, t_next, v_guided)

    if trajectory == "rptg":
        selected = tangential
    elif trajectory == "rptg_hr":
        selected = tangential + high_radial_scale * high_mask * radial
    else:
        raise ValueError(f"Unknown trajectory: {trajectory}")

    spectrum_guided = spectrum_c + (cfg_scale - 1.0) * selected
    x0_guided = torch.fft.irfft2(
        spectrum_guided,
        s=x0_c.shape[-2:],
        dim=(-2, -1),
        norm="ortho",
    ).to(dtype=x0_c.dtype)
    v_guided = velocity_from_anchor(z_t, t_curr, x0_guided)
    return euler_step_from_velocity(z_t, t_curr, t_next, v_guided)


@torch.no_grad()
def measure_for_scale(wrapper, metadata, args, cfg_scale, step_writer=None):
    lambda_scale = cfg_scale - 1.0
    timesteps = make_sd3_timesteps(
        args.num_steps, wrapper.device, wrapper.dtype, args.shift
    )
    stats_by_component = {name: empty_stats() for name, _, _ in COMPONENT_SPECS}

    for local_index, (global_index, item) in enumerate(
        tqdm(metadata, desc=f"w={cfg_scale:g}", leave=False)
    ):
        prompt = item["caption"]
        generated_index = str(item["generated_index"])
        context = wrapper.encode_prompt(prompt)
        generator = torch.Generator(device=wrapper.device).manual_seed(
            args.base_seed + global_index
        )
        z_t = torch.randn(
            (1, 16, 128, 128),
            generator=generator,
            device=wrapper.device,
            dtype=wrapper.dtype,
        )

        for step_index in range(args.num_steps):
            t_curr = timesteps[step_index]
            t_next = timesteps[step_index + 1]
            v_u, v_c = wrapper(z_t, t_curr, context)
            x0_u = clean_anchor(z_t, t_curr, v_u)
            x0_c = clean_anchor(z_t, t_curr, v_c)

            (
                spectrum_c,
                radial,
                tangential,
                radial_coeff,
                low_mask,
                high_mask,
            ) = fft_radial_tangential_signed_components(
                x0_c, x0_u, cutoff_ratio=args.frequency_cutoff
            )
            components = build_component_tensors(
                radial=radial,
                tangential=tangential,
                radial_coeff=radial_coeff,
                low_mask=low_mask,
                high_mask=high_mask,
            )
            total_residual = radial + tangential

            for name, band, direction in COMPONENT_SPECS:
                stats = stats_by_component[name]
                component = components[name]
                add_component_stats(
                    stats=stats,
                    component=component,
                    total_residual=total_residual,
                    lambda_scale=lambda_scale,
                )
                if direction in {"positive_radial", "negative_radial"}:
                    frequency_mask = low_mask if band == "low" else high_mask
                    sign_mask = radial_coeff > 0 if direction == "positive_radial" else radial_coeff < 0
                    add_radial_coupling_stats(
                        stats=stats,
                        spectrum_c=spectrum_c,
                        radial_coeff=radial_coeff,
                        frequency_mask=frequency_mask,
                        sign_mask=sign_mask,
                        lambda_scale=lambda_scale,
                    )

                if step_writer is not None:
                    coeff_count = component.numel()
                    weighted_energy = torch.sum(
                        torch.abs(lambda_scale * component).square()
                    ).item()
                    total_energy = torch.sum(
                        torch.abs(lambda_scale * total_residual).square()
                    ).item()
                    step_writer.writerow(
                        {
                            "cfg_scale": cfg_scale,
                            "lambda": lambda_scale,
                            "trajectory": args.trajectory,
                            "global_index": global_index,
                            "generated_index": generated_index,
                            "step_index": step_index,
                            "component": name,
                            "weighted_energy": weighted_energy,
                            "weighted_energy_mean": weighted_energy / max(coeff_count, 1),
                            "weighted_energy_ratio": weighted_energy / (total_energy + 1e-12),
                        }
                    )

            z_t = advance_trajectory(
                z_t=z_t,
                t_curr=t_curr,
                t_next=t_next,
                v_u=v_u,
                v_c=v_c,
                x0_c=x0_c,
                spectrum_c=spectrum_c,
                radial=radial,
                tangential=tangential,
                high_mask=high_mask,
                cfg_scale=cfg_scale,
                trajectory=args.trajectory,
                high_radial_scale=args.high_radial_scale,
            )

        for stats in stats_by_component.values():
            stats["num_prompts"] += 1

        del context, z_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = []
    for name, band, direction in COMPONENT_SPECS:
        stats = stats_by_component[name]
        coeff_count = max(stats["num_coefficients"], 1)
        total_energy = stats["total_weighted_residual_energy_sum"] + 1e-12
        rows.append(
            {
                "cfg_scale": cfg_scale,
                "lambda": lambda_scale,
                "trajectory": args.trajectory,
                "component": name,
                "band": band,
                "direction": direction,
                "weighted_energy_sum": stats["weighted_energy_sum"],
                "weighted_energy_mean": stats["weighted_energy_sum"] / coeff_count,
                "unweighted_energy_mean": stats["unweighted_energy_sum"] / coeff_count,
                "weighted_energy_ratio": stats["weighted_energy_sum"] / total_energy,
                "signed_radial_coupling_sum": stats["signed_radial_coupling_sum"],
                "positive_radial_coupling_sum": stats["positive_radial_coupling_sum"],
                "absolute_radial_coupling_sum": stats["absolute_radial_coupling_sum"],
                "positive_radial_coupling_mean": stats["positive_radial_coupling_sum"] / coeff_count,
                "absolute_radial_coupling_mean": stats["absolute_radial_coupling_sum"] / coeff_count,
                "num_prompts": stats["num_prompts"],
                "num_steps": stats["num_steps"],
                "num_coefficients": stats["num_coefficients"],
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_cli_args()
    os.makedirs(args.output_dir, exist_ok=True)
    metadata = load_metadata(
        args.metadata_path,
        args.max_samples,
        args.start_index,
        args.end_index,
    )
    config = {
        "metadata_path": args.metadata_path,
        "model_path": args.model_path,
        "num_samples": len(metadata),
        "global_start_index": args.start_index,
        "global_end_index": args.end_index,
        "cfg_scales": args.cfg_scales,
        "trajectory": args.trajectory,
        "num_steps": args.num_steps,
        "shift": args.shift,
        "base_seed": args.base_seed,
        "frequency_cutoff": args.frequency_cutoff,
        "high_radial_scale": args.high_radial_scale,
        "components": [name for name, _, _ in COMPONENT_SPECS],
        "energy_definition": "sum |(cfg_scale - 1) * DeltaX_component|^2",
        "coupling_definition": "2 * (cfg_scale - 1) * |X_c| * a_r",
    }
    with open(
        os.path.join(args.output_dir, "directional_energy_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    wrapper = SD3TransformerWrapper(
        device=args.device,
        dtype=torch.bfloat16,
        model_path=args.model_path,
    )

    step_file = None
    step_writer = None
    if args.save_step_rows:
        step_path = os.path.join(args.output_dir, "directional_energy_steps.csv")
        step_file = open(step_path, "w", encoding="utf-8", newline="")
        step_writer = csv.DictWriter(
            step_file,
            fieldnames=[
                "cfg_scale",
                "lambda",
                "trajectory",
                "global_index",
                "generated_index",
                "step_index",
                "component",
                "weighted_energy",
                "weighted_energy_mean",
                "weighted_energy_ratio",
            ],
        )
        step_writer.writeheader()

    try:
        all_rows = []
        for cfg_scale in args.cfg_scales:
            all_rows.extend(
                measure_for_scale(
                    wrapper=wrapper,
                    metadata=metadata,
                    args=args,
                    cfg_scale=cfg_scale,
                    step_writer=step_writer,
                )
            )
        output_csv = os.path.join(args.output_dir, "directional_energy_summary.csv")
        write_csv(output_csv, all_rows)
        print(f"Wrote summary: {output_csv}")
        if args.save_step_rows:
            print(
                "Wrote per-step rows: "
                f"{os.path.join(args.output_dir, 'directional_energy_steps.csv')}"
            )
    finally:
        if step_file is not None:
            step_file.close()


if __name__ == "__main__":
    main()
