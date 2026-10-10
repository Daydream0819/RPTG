# Code Map

This document explains every source file included in the GitHub release.

## Core Model Files

| File | Role |
|---|---|
| `code/models/sd3_model.py` | Wraps `diffusers.StableDiffusion3Pipeline`, loads local SD3/SD3.5 weights, encodes prompts, and exposes paired unconditional/conditional velocity predictions. |
| `code/models/guidance_methods.py` | Implements all guidance rules: CFG, CFG Rescale, APG, FDG, FDG+APG, ADG diagnostics, RPTG, RPTG-HR, APG-stabilized RPTG variants, and frequency-direction ablations. |
| `code/models/latent_io.py` | Decodes SD3 latent tensors through the VAE and converts them into PIL images for saving. |
| `code/models/__init__.py` | Package marker for model utilities. |

## Data Preparation

| File | Role |
|---|---|
| `code/data/prepare_coco_evaluation_data.py` | Samples a deterministic subset from COCO val2017 captions, writes `experiment_metadata.json`, and copies matching real images into `real_images_for_fid/`. |
| `code/data/__init__.py` | Package marker for data utilities. |

## Generation Scripts

| File | Role |
|---|---|
| `code/generation/generate_coco_innovation_benchmark.py` | Main benchmark generator. It creates one image per prompt per selected method and writes runtime/config metadata. |
| `code/generation/generate_qualitative_stress_prompts.py` | Generates fixed qualitative stress prompts and optional comparison grids for visual paper figures. |
| `code/generation/generate_rptg_directional_figure.py` | Generates the low/high, radial/tangential frequency-direction ablation images used for qualitative analysis. |
| `code/generation/generate_rptg_high_radial_sweep.py` | Generates RPTG-HR variants across different high-frequency radial weights `gamma_H`. |
| `code/generation/generate_rptg_intro_prompt.py` | Generates single-prompt teaser candidates comparing CFG, radial, tangential, and RPTG-HR behavior. |
| `code/generation/measure_rptg_directional_energy.py` | Measures frequency-domain radial/tangential energy and spectral drift statistics across guidance scales. |
| `code/generation/__init__.py` | Package marker for generation utilities. |

## Evaluation Scripts

| File | Role |
|---|---|
| `code/evaluation/evaluate_coco_innovation_benchmark.py` | Computes CLIP score, FID, saturation, saturation drift, contrast, brightness, brightness drift, color shift, missing images, and extra images. |
| `code/evaluation/evaluate_preference_metrics.py` | Optional evaluation for ImageReward, HPSv2, and PickScore. These metrics require extra packages and model caches. |
| `code/evaluation/__init__.py` | Package marker for evaluation utilities. |

## Plotting Scripts

| File | Role |
|---|---|
| `code/plotting/plot_guidance_scale_sweep.py` | Draws the guidance-scale robustness figure from benchmark CSV files. |
| `code/plotting/plot_guidance_scale_sweep_clean.py` | Alternative compact guidance-scale sweep plot with cleaner layout. |
| `code/plotting/plot_high_radial_sweep.py` | Merges and plots the `gamma_H` high-radial retention sweep. |
| `code/plotting/plot_directional_energy_reportlab.py` | Builds a vector PDF for directional energy and spectral drift analysis using ReportLab. |
| `code/plotting/make_rptg_directional_layouts.py` | Assembles directional ablation images into paper-ready layouts. |
| `code/plotting/plot_guidance_scale_sweep_matlab.m` | MATLAB version of the guidance-scale sweep plot. |
| `code/plotting/plot_rptg_formula_energy_matlab.m` | MATLAB version of the directional energy figure. |
| `code/plotting/__init__.py` | Package marker for plotting utilities. |

## Utility Scripts

| File | Role |
|---|---|
| `code/utils/merge_benchmark_parts.py` | Merges prompt-sharded generation folders into one complete benchmark result directory. |
| `code/utils/merge_adg_scale_sweep_results.py` | Merges separately run ADG guidance-scale CSVs. |
| `code/utils/add_saturation_drift_to_benchmark_csv.py` | Adds saturation drift to existing benchmark CSVs without recomputing CLIP or FID. |
| `code/utils/merge_main_table_results.py` | Combines benchmark metrics and preference metrics into a single paper-table CSV. |
| `code/utils/__init__.py` | Package marker for utility scripts. |

## Launch Scripts

| File | Role |
|---|---|
| `code/scripts/run_smoke_test.sh` | Minimal end-to-end run for 8 generated prompts. Use this first on a new server. |
| `code/scripts/run_main_table_w15.sh` | Reproduces the main `w=15` table for the seven core methods. |
| `code/scripts/run_guidance_scale_sweep_core.sh` | Reproduces the guidance-scale sweep for core methods over multiple CFG scales. |
| `code/scripts/run_scale_sweep_5gpu.sh` | Historical 5-GPU guidance-scale launcher with APG-stabilized RPTG variants. |
| `code/scripts/run_w15_sweep_5gpu_sharded.sh` | Historical prompt-sharded `w=15` launcher. |
| `code/scripts/run_high_radial_sweep_w15_1000.sh` | Runs the RPTG-HR `gamma_H` sweep at `w=15`. |
| `code/scripts/run_directional_energy_6gpu.sh` | Runs directional energy measurement for multiple guidance scales in parallel. |
| `code/scripts/run_rptg_intro_prompt_w15.sh` | Generates teaser prompt candidates. |
| `code/scripts/run_adg_qualitative_w15.sh` | Generates qualitative comparisons including ADG diagnostics. |
| `code/scripts/run_adg_official_qualitative_w15.sh` | Generates official-style ADG qualitative checks. |
| `code/scripts/run_adg_w15_2000.sh` | Runs ADG-only quantitative evaluation at `w=15`. |
| `code/scripts/run_adg_w15_2000_5gpu_sharded.sh` | Prompt-sharded ADG-only `w=15` run. |
| `code/scripts/run_adg_scale_sweep_1000.sh` | ADG-only guidance-scale sweep. |
| `code/scripts/update_saturation_drift_results.sh` | Batch post-processing helper for saturation drift. |

## Configuration and Documentation

| File | Role |
|---|---|
| `code/requirements.txt` | Python dependencies for generation, evaluation, and plotting. |
| `examples/paths.example.env` | Example environment variables for external model/data/result paths. |
| `docs/EXPERIMENTS.md` | Full paper-experiment reproduction commands. |
| `docs/SD3_EXPERIMENT_PROTOCOL.md` | Fixed protocol notes for dataset, model, baselines, parameters, and evaluation. |

