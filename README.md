# RPTG: Relative Phase Tangential Guidance

This repository contains the code used for the RPTG/RPTG-HR experiments in our
SD3 flow-matching guidance paper. It implements standard CFG, clean-anchor
baseline adaptations, and our frequency-domain radial/tangential guidance
methods for high-guidance text-to-image generation.

The repository is intentionally code-only. It does not include MS-COCO,
Stable Diffusion 3 / SD3.5 checkpoints, Hugging Face caches, generated images,
or paper draft files.

## What Is Included

```text
code/
  data/          # Prepare deterministic COCO metadata and real-image FID set.
  generation/    # Generate benchmark images, ablations, and qualitative panels.
  evaluation/    # Compute CLIP, FID, artifact metrics, and preference metrics.
  models/        # SD3 wrapper, latent decoding, and all guidance implementations.
  plotting/      # Plot guidance sweeps, gamma sweeps, and directional figures.
  scripts/       # Reproducible launch scripts for paper experiments.
  utils/         # Merge sharded results and post-process benchmark CSV files.
docs/
  CODE_MAP.md
  EXPERIMENTS.md
  SD3_EXPERIMENT_PROTOCOL.md
examples/
  paths.example.env
```

For a file-by-file explanation, see [docs/CODE_MAP.md](docs/CODE_MAP.md).
For commands that reproduce the paper experiments, see
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
For exact hardware and package-version reporting commands, see
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## External Assets

Prepare these outside the Git repository:

```text
/root/autodl-tmp/hf_cache
/root/autodl-tmp/coco2017_data
/root/autodl-tmp/output
/root/autodl-tmp/results
```

Expected COCO layout:

```text
/root/autodl-tmp/coco2017_data/
  annotations/captions_val2017.json
  val2017/*.jpg
```

Expected SD3/SD3.5 layout:

```text
/root/autodl-tmp/hf_cache/
  model_index.json
  transformer/
  vae/
  text_encoder*/
  tokenizer*/
```

The code loads the model locally with `local_files_only=True`.

## Installation

Use a CUDA-compatible PyTorch build for your server first, then install the
remaining dependencies:

```bash
cd /root/autodl-tmp
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

If you use a custom CUDA wheel, install PyTorch before running the final
`pip install -r` command.

## Quick Start

Run a small smoke test:

```bash
bash code/scripts/run_smoke_test.sh
```

Run the main `w=15` benchmark:

```bash
bash code/scripts/run_main_table_w15.sh
```

Run the guidance-scale robustness sweep:

```bash
bash code/scripts/run_guidance_scale_sweep_core.sh
```

All scripts use the default `/root/autodl-tmp` layout, but every path can be
overridden through environment variables. See [examples/paths.example.env](examples/paths.example.env).

## Main Paper Methods

The main table uses these seven methods:

| Code name | Paper label | Summary |
|---|---|---|
| `cfg` | CFG | Standard velocity-domain classifier-free guidance. |
| `cfg_rescale` | CFG Rescale | CFG followed by conditional-standard-deviation rescaling. |
| `adg` | ADG | SD3-style angular clean-anchor guidance. |
| `apg` | APG | Clean-anchor APG adaptation with projection, clipping, and momentum. |
| `fdg` | FDG | Clean-anchor Laplacian/frequency guidance. |
| `rptg` | RPTG | Full-spectrum tangential residual guidance. |
| `rptg_hr` | RPTG-HR | RPTG plus controlled high-frequency radial retention. |

The code also contains optional diagnostic baselines such as `fdg_plus_apg`,
`adg_norm`, `rptg_apg`, and `rptg_hr_apg`. These are not required for the main
table.

## Reproducibility Rules

- Use the same `experiment_metadata.json` for all methods in one comparison.
- Use the seed rule `base_seed + global item_index`.
- For multi-GPU generation, shard by prompt index, not by method.
- Keep model checkpoints, datasets, outputs, and generated images outside Git.
- Do not mix outputs from different signatures in the same result directory.
