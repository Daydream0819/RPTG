# SD3 RPTG Experiment Protocol

本文档固定第一轮 SD3 主实验的 dataset、参数、baseline 与代码入口。目标是先在单一模型上跑完整闭环，再扩展到 SDXL/FLUX/DiT 等多模型实验。

## 1. Dataset

主实验使用 MS-COCO 2017 validation captions。

- Prompt source: `captions_val2017.json`
- Real images: `val2017`
- 默认样本数: 200 captions
- 抽样方式: 从 COCO captions annotations 中用 `seed=42` 随机抽样
- 文件命名: 每个样本写入 `generated_index`，格式为 `00000.jpg`, `00001.jpg`, ...
- FID real set: 与抽样 metadata 一一对应的 COCO 原图，放入 `real_images_for_fid`

建议实验规模：

| Stage | Samples | Purpose |
|---|---:|---|
| Smoke test | 8-16 | 检查模型、路径、显存、图像是否正常 |
| Main SD3 table | 200 | 第一版论文主表和消融 |
| Stronger final run | 1000+ | 如果算力允许，用于最终稳健性补充 |

## 2. Default SD3 Settings

| Parameter | Value |
|---|---|
| Model | local Stable Diffusion 3 / SD3.5 checkpoint |
| Code wrapper | `code/models/sd3_model.py` |
| Model path default | `/root/autodl-tmp/hf_cache` |
| COCO path default | `/root/autodl-tmp/coco2017_data` |
| dtype | `torch.bfloat16` for transformer, float32 VAE |
| Resolution | 1024 x 1024 |
| Latent shape | `[1, 16, 128, 128]` |
| Solver | Euler |
| Steps | 20 |
| Time shift | 3.0 |
| Stress CFG scale | 15.0 |
| Base seed | 42 |
| Per-sample seed | `base_seed + item_index` |
| Frequency cutoff | 0.25 |
| RPTG-HR high radial weight | 0.25 |
| CFG rescale factor | 0.7 |
| APG r | 1.5 |
| APG beta | -0.2 |
| FDG split | Laplacian pyramid in clean-anchor space |
| FDG default guidance scales | `[15.0, 1.0]` from high to low frequency |
| FDG kernel size / sigma | 5 / 1.0 |

The first SD3 table should focus on the high-CFG stress setting `w=15.0`, because the paper's claim concerns suppressing CFG-induced spectral magnitude drift and over-saturation.

## 3. Main Baselines

The main table should use a small but strong set of baselines. Optional or historically useful variants can go to appendix.

| Method name in code | Paper label | Implementation |
|---|---|---|
| `cfg` | CFG | Velocity CFG: `v = v_u + w (v_c - v_u)` |
| `cfg_rescale` | CFG Rescale | Standard CFG followed by conditional-standard-deviation rescaling with factor 0.7 |
| `adg` | ADG | SD3-style angular clean-anchor guidance with a maximum turn angle of \(\pi/3\) |
| `apg` | APG | APG-style projection in clean-anchor space: reconstruct `x0_u, x0_c`, project `x0_c - x0_u` against `x0_c`, apply norm clipping and reverse momentum, then map back to velocity |
| `fdg` | FDG | Clean-anchor Laplacian pyramid; guidance scales are ordered from high to low frequency, default `[w, 1.0]` |
| `rptg` | RPTG | Full-spectrum tangential residual only; `high_radial_scale=0` |
| `rptg_hr` | RPTG-HR | Full-spectrum tangential residual plus `gamma_H` high-frequency radial residual |

Not recommended for the main table:

- CFG++: include only if the flow-matching migration is carefully justified; otherwise it distracts from the RPTG-vs-FDG/APG question.
- FDG+APG: useful as a diagnostic combined baseline, but omitted from the main table to keep the comparison focused on representative guidance-control families.

## 4. Code Entry Points

Prepare the deterministic COCO subset:

```bash
python3 code/data/prepare_coco_evaluation_data.py \
  --coco_anno_path /root/autodl-tmp/coco2017_data/annotations/captions_val2017.json \
  --coco_img_dir /root/autodl-tmp/coco2017_data/val2017 \
  --output_dir /root/autodl-tmp/output \
  --num_prompts 200 \
  --seed 42
```

Smoke-test generation:

```bash
python3 code/generation/generate_coco_innovation_benchmark.py \
  --metadata_path /root/autodl-tmp/output/experiment_metadata.json \
  --real_images_dir /root/autodl-tmp/output/real_images_for_fid \
  --output_dir /root/autodl-tmp/results/sd3_rptg_w15_smoke \
  --model_path /root/autodl-tmp/hf_cache \
  --cfg_scale 15.0 \
  --num_steps 20 \
  --shift 3.0 \
  --base_seed 42 \
  --frequency_cutoff 0.25 \
  --high_radial_scale 0.25 \
  --low_cfg_scale 1.0 \
  --fdg_guidance_scales 15.0 1.0 \
  --fdg_kernel_size 5 \
  --fdg_sigma 1.0 \
  --max_samples 8
```

Main SD3 generation:

```bash
python3 code/generation/generate_coco_innovation_benchmark.py \
  --metadata_path /root/autodl-tmp/output/experiment_metadata.json \
  --real_images_dir /root/autodl-tmp/output/real_images_for_fid \
  --output_dir /root/autodl-tmp/results/sd3_rptg_w15_main \
  --model_path /root/autodl-tmp/hf_cache \
  --cfg_scale 15.0 \
  --num_steps 20 \
  --shift 3.0 \
  --base_seed 42 \
  --frequency_cutoff 0.25 \
  --high_radial_scale 0.25 \
  --low_cfg_scale 1.0 \
  --cfg_rescale_factor 0.7 \
  --apg_r 1.5 \
  --apg_beta -0.2 \
  --fdg_guidance_scales 15.0 1.0 \
  --fdg_kernel_size 5 \
  --fdg_sigma 1.0
```

Evaluate CLIP and FID:

```bash
python3 code/evaluation/evaluate_coco_innovation_benchmark.py \
  --metadata_path /root/autodl-tmp/output/experiment_metadata.json \
  --real_images_dir /root/autodl-tmp/output/real_images_for_fid \
  --generated_root /root/autodl-tmp/results/sd3_rptg_w15_main \
  --clip_batch_size 16
```

## 5. First Paper Table

The first SD3 table should contain:

| Method | CLIP Score ↑ | FID ↓ | Saturation ↓ | Brightness Drift ↓ | Runtime / image ↓ |
|---|---:|---:|---:|---:|---:|
| CFG |  |  |  |  |  |
| CFG Rescale |  |  |  |  |  |
| ADG |  |  |  |  |  |
| APG |  |  |  |  |  |
| FDG |  |  |  |  |  |
| RPTG |  |  |  |  |  |
| RPTG-HR |  |  |  |  |  |

The evaluation script computes CLIP, FID, saturation, contrast, brightness, paired brightness drift, and paired RGB color shift.

## 6. Ablation After Main Table

After the main SD3 baseline table is complete, run direction ablations:

- low radial only
- low tangential only
- high radial only
- high tangential only
- full tangential, i.e. RPTG
- full tangential plus high radial, i.e. RPTG-HR
- `gamma_H` sweep: 0, 0.1, 0.25, 0.5, 1.0

The current cleaned code keeps only the main benchmark. Add a separate ablation script after the SD3 main table is stable, so the baseline implementation remains uncluttered.
