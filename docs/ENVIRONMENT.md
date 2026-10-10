# Environment Reporting

The paper experiments were run on Linux GPU servers with NVIDIA GeForce RTX
5090 GPUs. Model checkpoints, COCO data, generated images, and evaluation
results are intentionally kept outside this repository.

To record an exact software snapshot on a reproduction server, run:

```bash
python3 --version
nvidia-smi
python3 - <<'PY'
import importlib.metadata as metadata
import torch

packages = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "numpy",
    "Pillow",
    "tqdm",
    "open_clip_torch",
    "clean-fid",
    "scipy",
    "reportlab",
]

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("gpu memory bytes:", torch.cuda.get_device_properties(0).total_memory)

for package in packages:
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: not installed")
PY
pip freeze > environment.freeze.txt
```

Attach `environment.freeze.txt` to private experiment records or supplementary
artifacts if exact package versions are required.

