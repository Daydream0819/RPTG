# Reproducing the Paper Experiments

All commands assume this repository is available at:

```text
/root/autodl-tmp/code
```

and external assets are stored at:

```text
/root/autodl-tmp/hf_cache
/root/autodl-tmp/coco2017_data
/root/autodl-tmp/output
/root/autodl-tmp/results
```

You can override paths with environment variables in the launch scripts.

## 1. Prepare Environment

```bash
cd /root/autodl-tmp
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

Use a PyTorch wheel compatible with your CUDA driver.

## 2. Smoke Test

Purpose: verify model loading, COCO paths, image saving, and evaluation before a long run.

```bash
bash code/scripts/run_smoke_test.sh
```

Default behavior:

- samples 16 COCO captions;
- generates 8 prompts;
- runs `cfg`, `apg`, `fdg`, `fdg_plus_apg`, `rptg`, and `rptg_hr`;
- writes results to `/root/autodl-tmp/results/rptg_smoke`.

Smoke-test FID is not a paper result.

## 3. Main SD3 Table at High Guidance

Purpose: reproduce the main `w=15` quantitative table.

```bash
bash code/scripts/run_main_table_w15.sh
```

Default methods:

```text
cfg cfg_rescale apg fdg fdg_plus_apg rptg rptg_hr
```

Default parameters:

| Parameter | Value |
|---|---:|
| samples | 1000 |
| CFG scale | 15.0 |
| steps | 20 |
| shift | 3.0 |
| base seed | 42 |
| RPTG cutoff | 0.25 |
| RPTG-HR `gamma_H` | 0.25 |
| CFG rescale factor | 0.7 |
| APG `r` | 1.5 |
| APG `beta` | -0.2 |
| FDG kernel / sigma | 17 / 4.0 |

Outputs:

```text
/root/autodl-tmp/output_main_1000/experiment_metadata.json
/root/autodl-tmp/output_main_1000/real_images_for_fid/
/root/autodl-tmp/results/sd3_rptg_w15_main/
/root/autodl-tmp/results/sd3_rptg_w15_main/benchmark_results.csv
```

## 4. Guidance-Scale Robustness

Purpose: reproduce the sweep over guidance scales.

```bash
bash code/scripts/run_guidance_scale_sweep_core.sh
```

Default scales:

```text
3 5 7.5 10 15
```

Default methods:

```text
cfg apg fdg fdg_plus_apg rptg_hr
```

Outputs:

```text
/root/autodl-tmp/results/scale_sweep_core_w3_1000/benchmark_results.csv
/root/autodl-tmp/results/scale_sweep_core_w5_1000/benchmark_results.csv
/root/autodl-tmp/results/scale_sweep_core_w7p5_1000/benchmark_results.csv
/root/autodl-tmp/results/scale_sweep_core_w10_1000/benchmark_results.csv
/root/autodl-tmp/results/scale_sweep_core_w15_1000/benchmark_results.csv
```

Plot the sweep:

```bash
python3 code/plotting/plot_guidance_scale_sweep.py \
  --input_csvs \
    /root/autodl-tmp/results/scale_sweep_core_w3_1000/benchmark_results.csv \
    /root/autodl-tmp/results/scale_sweep_core_w5_1000/benchmark_results.csv \
    /root/autodl-tmp/results/scale_sweep_core_w7p5_1000/benchmark_results.csv \
    /root/autodl-tmp/results/scale_sweep_core_w10_1000/benchmark_results.csv \
    /root/autodl-tmp/results/scale_sweep_core_w15_1000/benchmark_results.csv \
  --output_dir /root/autodl-tmp/results/figures/guidance_scale_sweep
```

## 5. RPTG-HR High-Radial Sweep

Purpose: test sensitivity to the high-frequency radial retention weight `gamma_H`.

```bash
bash code/scripts/run_high_radial_sweep_w15_1000.sh
```

Default `gamma_H` values:

```text
0 0.1 0.25 0.5 0.75 1.0
```

Plot/table:

```bash
python3 code/plotting/plot_high_radial_sweep.py \
  --input_csv /root/autodl-tmp/results/rptg_high_radial_sweep_w15_1000/benchmark_results.csv \
  --output_dir /root/autodl-tmp/results/figures/high_radial_sweep
```

## 6. Frequency-Direction Ablation

Purpose: generate qualitative low/high, radial/tangential ablation panels.

```bash
python3 code/generation/generate_rptg_directional_figure.py \
  --output_dir /root/autodl-tmp/results/rptg_directional_figure_w15 \
  --model_path /root/autodl-tmp/hf_cache \
  --cfg_scale 15 \
  --num_steps 20 \
  --shift 3.0 \
  --base_seed 20270618 \
  --frequency_cutoff 0.25 \
  --high_radial_scale 0.25
```

Assemble layouts:

```bash
python3 code/plotting/make_rptg_directional_layouts.py \
  --input_dir /root/autodl-tmp/results/rptg_directional_figure_w15 \
  --output_dir /root/autodl-tmp/results/figures/directional_ablation
```

## 7. Directional Energy Measurement

Purpose: quantify radial and tangential spectral drift terms across scales.

```bash
bash code/scripts/run_directional_energy_6gpu.sh
```

Plot:

```bash
python3 code/plotting/plot_directional_energy_reportlab.py
```

The plotting script may need its CSV paths adjusted if you use non-default output folders.

## 8. Qualitative Stress Prompts

Purpose: generate paper visual comparisons under difficult high-CFG prompts.

```bash
python3 code/generation/generate_qualitative_stress_prompts.py \
  --output_dir /root/autodl-tmp/results/rptg_qualitative_stress_w15 \
  --model_path /root/autodl-tmp/hf_cache \
  --methods cfg cfg_rescale apg fdg fdg_plus_apg rptg rptg_hr \
  --cfg_scale 15 \
  --num_steps 20 \
  --shift 3.0 \
  --frequency_cutoff 0.25 \
  --high_radial_scale 0.25 \
  --make_grid
```

## 9. Optional Preference Metrics

Install optional packages first:

```bash
pip install ImageReward hpsv2
```

Then run:

```bash
python3 code/evaluation/evaluate_preference_metrics.py \
  --metadata_path /root/autodl-tmp/output_main_1000/experiment_metadata.json \
  --generated_root /root/autodl-tmp/results/sd3_rptg_w15_main \
  --methods cfg cfg_rescale apg fdg fdg_plus_apg rptg rptg_hr \
  --batch_size 16
```

## 10. Multi-GPU Rule

When running multiple servers or GPUs, shard by prompt index:

```bash
python3 code/generation/generate_coco_innovation_benchmark.py \
  --start_index 0 \
  --end_index 200 \
  ...
```

Use the same:

- `experiment_metadata.json`;
- model checkpoint;
- code commit;
- generation parameters;
- seed rule `base_seed + global item_index`.

Merge shards with:

```bash
python3 code/utils/merge_benchmark_parts.py \
  --part_dirs /root/autodl-tmp/results/part0 /root/autodl-tmp/results/part1 \
  --output_dir /root/autodl-tmp/results/merged_run
```

