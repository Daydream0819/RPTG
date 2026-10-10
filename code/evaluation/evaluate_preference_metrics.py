import argparse
import csv
import json
import os
import traceback

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm


PICKSCORE_CACHE = {}
HPSV2_CACHE = {}


DEFAULT_METHODS = [
    "cfg",
    "cfg_rescale",
    "adg",
    "apg",
    "fdg",
    "fdg_plus_apg",
    "rptg",
    "rptg_hr",
    "rptg_apg",
    "rptg_hr_apg",
]

DEFAULT_METRICS = ["imagereward", "hpsv2", "pickscore"]


def patch_transformers_for_imagereward():
    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils
    from transformers import BertTokenizer, BertTokenizerFast

    if not hasattr(modeling_utils, "apply_chunking_to_forward") and hasattr(
        pytorch_utils,
        "apply_chunking_to_forward",
    ):
        modeling_utils.apply_chunking_to_forward = (
            pytorch_utils.apply_chunking_to_forward
        )

    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(
            heads,
            n_heads,
            head_size,
            already_pruned_heads,
        ):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        modeling_utils.find_pruneable_heads_and_indices = (
            find_pruneable_heads_and_indices
        )

    if not hasattr(modeling_utils, "prune_linear_layer"):

        def prune_linear_layer(layer, index, dim=0):
            index = index.to(layer.weight.device)
            weight = layer.weight.index_select(dim, index).clone().detach()

            if layer.bias is not None:
                if dim == 1:
                    bias = layer.bias.clone().detach()
                else:
                    bias = layer.bias[index].clone().detach()
            else:
                bias = None

            new_size = list(layer.weight.size())
            new_size[dim] = len(index)
            new_layer = nn.Linear(
                new_size[1],
                new_size[0],
                bias=layer.bias is not None,
            ).to(layer.weight.device)

            new_layer.weight.requires_grad = False
            new_layer.weight.copy_(weight.contiguous())
            new_layer.weight.requires_grad = True

            if bias is not None:
                new_layer.bias.requires_grad = False
                new_layer.bias.copy_(bias.contiguous())
                new_layer.bias.requires_grad = True

            return new_layer

        modeling_utils.prune_linear_layer = prune_linear_layer

    for tokenizer_class in (BertTokenizer, BertTokenizerFast):
        if not hasattr(tokenizer_class, "additional_special_tokens_ids"):

            def additional_special_tokens_ids(self):
                tokens = getattr(self, "additional_special_tokens", [])
                ids = self.convert_tokens_to_ids(tokens) if tokens else []
                if ids:
                    return ids
                enc_id = self.convert_tokens_to_ids("[ENC]")
                if enc_id is not None and enc_id != getattr(self, "unk_token_id", None):
                    return [enc_id]
                return []

            tokenizer_class.additional_special_tokens_ids = property(
                additional_special_tokens_ids
            )

    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "_convert_head_mask_to_5d"):

        def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            if head_mask.dim() != 5:
                raise ValueError(
                    f"head_mask.dim != 5, instead {head_mask.dim()}"
                )
            dtype = getattr(self, "dtype", None)
            if dtype is None:
                try:
                    dtype = next(self.parameters()).dtype
                except StopIteration:
                    dtype = torch.float32
            return head_mask.to(dtype=dtype)

        PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d

    if not hasattr(PreTrainedModel, "get_head_mask"):

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                head_mask = self._convert_head_mask_to_5d(
                    head_mask,
                    num_hidden_layers,
                )
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        PreTrainedModel.get_head_mask = get_head_mask

    if not getattr(
        PreTrainedModel,
        "_imagereward_tie_weights_patched",
        False,
    ):

        def tie_weights_compat(self, *args, **kwargs):
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = []
            return None

        PreTrainedModel.tie_weights = tie_weights_compat
        PreTrainedModel._imagereward_tie_weights_patched = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ImageReward, HPSv2, and PickScore."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="/root/autodl-tmp/output_2000/experiment_metadata.json",
    )
    parser.add_argument(
        "--generated_root",
        type=str,
        default="/root/autodl-tmp/results/sd3_rptg_w15_2000_main",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        choices=DEFAULT_METRICS,
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hps_version", type=str, default="v2.1")
    parser.add_argument(
        "--pickscore_model",
        type=str,
        default="yuvalkirstain/PickScore_v1",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load metric models from the local cache without contacting the Hub.",
    )
    parser.add_argument(
        "--pickscore_mode",
        type=str,
        default="logit",
        choices=("cosine", "logit", "probability"),
        help=(
            "PickScore output scale: cosine gives raw normalized similarity, "
            "logit multiplies by the learned CLIP logit scale, and probability "
            "maps the logit score through sigmoid to a 0-1 score."
        ),
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="preference_results",
    )
    return parser.parse_args()


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, list):
        raise ValueError("metadata JSON must contain a list")
    return metadata


def validate_method_images(method_dir, metadata):
    items = []
    missing = []
    expected = set()
    for item in metadata:
        generated_index = str(item["generated_index"])
        filename = f"{generated_index}.jpg"
        expected.add(filename)
        image_path = os.path.join(method_dir, filename)
        if os.path.exists(image_path):
            items.append((item, image_path))
        else:
            missing.append(generated_index)

    actual = {
        name
        for name in os.listdir(method_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    }
    extra = sorted(actual - expected)
    return items, missing, extra


def summarize_scores(scores, prefix):
    values = np.asarray(scores, dtype=np.float64)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_count": int(values.size),
    }


def flatten_numeric_scores(values):
    if torch.is_tensor(values):
        return values.detach().float().cpu().reshape(-1).tolist()
    if isinstance(values, np.ndarray):
        return values.astype(np.float64).reshape(-1).tolist()
    if isinstance(values, (float, int, np.floating, np.integer)):
        return [float(values)]
    if isinstance(values, (list, tuple)):
        scores = []
        for value in values:
            scores.extend(flatten_numeric_scores(value))
        return scores
    return [float(values)]


def save_results(results, generated_root, output_prefix):
    json_path = os.path.join(generated_root, f"{output_prefix}.json")
    csv_path = os.path.join(generated_root, f"{output_prefix}.csv")
    fields = sorted({key for row in results for key in row})
    fields = ["method"] + [field for field in fields if field != "method"]

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


def compute_imagereward(items, device):
    try:
        patch_transformers_for_imagereward()
        import ImageReward as RM
    except ImportError as error:
        raise ImportError(
            "ImageReward or one of its dependencies could not be imported. "
            "Try: pip install image-reward git+https://github.com/openai/CLIP.git. "
            f"Original error: {error}"
        ) from error

    model = RM.load("ImageReward-v1.0")
    if hasattr(model, "to"):
        model = model.to(device)

    scores = []
    for item, image_path in tqdm(items, desc="ImageReward"):
        score = model.score(item["caption"], image_path)
        scores.extend(flatten_numeric_scores(score))
    return summarize_scores(scores, "imagereward")


def load_hpsv2_model(hps_version, device):
    try:
        import huggingface_hub
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from hpsv2.utils import hps_version_map
    except ImportError as error:
        raise ImportError("HPSv2 is not installed. Try: pip install hpsv2") from error

    cache_key = (hps_version, device)
    if cache_key in HPSV2_CACHE:
        return HPSV2_CACHE[cache_key]

    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        "laion2B-s32B-b79K",
        precision="amp",
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    checkpoint_path = huggingface_hub.hf_hub_download(
        "xswu/HPSv2",
        hps_version_map[hps_version],
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    tokenizer = get_tokenizer("ViT-H-14")
    HPSV2_CACHE[cache_key] = (model, preprocess_val, tokenizer)
    return HPSV2_CACHE[cache_key]


def compute_hpsv2(items, hps_version, batch_size, device):
    try:
        import hpsv2
    except ImportError as error:
        raise ImportError("HPSv2 is not installed. Try: pip install hpsv2") from error

    model, preprocess_val, tokenizer = load_hpsv2_model(hps_version, device)
    scores = []
    for start in tqdm(range(0, len(items), batch_size), desc="HPSv2"):
        batch = items[start : start + batch_size]
        prompts = [item["caption"] for item, _ in batch]
        image_tensors = []
        for _, image_path in batch:
            with Image.open(image_path) as image:
                image_tensors.append(preprocess_val(image.convert("RGB")))
        images = torch.stack(image_tensors).to(device=device, non_blocking=True)
        texts = tokenizer(prompts).to(device=device, non_blocking=True)

        if str(device).startswith("cuda"):
            autocast_context = torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext

            autocast_context = nullcontext()

        with autocast_context:
            outputs = model(images, texts)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            batch_scores = torch.diagonal(image_features @ text_features.T)
        scores.extend(batch_scores.float().cpu().tolist())
    return summarize_scores(scores, "hpsv2")


@torch.no_grad()
def compute_pickscore(
    items,
    model_name,
    device,
    batch_size,
    local_files_only=False,
    score_mode="probability",
):
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as error:
        raise ImportError(
            "transformers is not installed. Try: pip install transformers"
        ) from error

    cache_key = (model_name, device, local_files_only)
    if cache_key not in PICKSCORE_CACHE:
        processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).eval().to(device)
        PICKSCORE_CACHE[cache_key] = (processor, model)
    processor, model = PICKSCORE_CACHE[cache_key]

    def feature_tensor(output, name):
        if torch.is_tensor(output):
            return output
        for attribute in (
            f"{name}_embeds",
            "image_embeds",
            "text_embeds",
            "pooler_output",
            "last_hidden_state",
        ):
            value = getattr(output, attribute, None)
            if torch.is_tensor(value):
                if attribute == "last_hidden_state":
                    return value[:, 0]
                return value
        if isinstance(output, (tuple, list)):
            for value in output:
                if torch.is_tensor(value):
                    return value
        raise TypeError(f"Cannot extract tensor features from {type(output)}")

    scores = []
    for start in tqdm(range(0, len(items), batch_size), desc="PickScore"):
        batch = items[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for _, path in batch]
        prompts = [item["caption"] for item, _ in batch]

        image_inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)
        text_inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        image_embeds = feature_tensor(
            model.get_image_features(**image_inputs),
            "image",
        )
        text_embeds = feature_tensor(
            model.get_text_features(**text_inputs),
            "text",
        )
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        batch_scores = torch.sum(image_embeds * text_embeds, dim=-1)
        logit_scale = getattr(model, "logit_scale", None)
        if score_mode in ("logit", "probability") and torch.is_tensor(logit_scale):
            batch_scores = batch_scores * logit_scale.exp()
        if score_mode == "probability":
            batch_scores = torch.sigmoid(batch_scores)
        scores.extend(batch_scores.float().cpu().tolist())

    return summarize_scores(scores, "pickscore")


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata_path)

    results = []
    for method_name in args.methods:
        method_dir = os.path.join(args.generated_root, method_name)
        if not os.path.isdir(method_dir):
            print(f"Skipping missing directory: {method_dir}")
            continue

        items, missing, extra = validate_method_images(method_dir, metadata)
        if missing or extra:
            raise RuntimeError(
                f"{method_name} does not match metadata: "
                f"{len(missing)} missing, {len(extra)} extra."
            )

        row = {"method": method_name}
        print(f"\nEvaluating preference metrics: {method_name}")
        if "imagereward" in args.metrics:
            try:
                row.update(compute_imagereward(items, args.device))
            except Exception as error:
                row["imagereward_error"] = repr(error)
                print(f"ImageReward failed for {method_name}: {error}")
                traceback.print_exc()
            save_results(results + [row], args.generated_root, args.output_prefix)
        if "hpsv2" in args.metrics:
            try:
                row.update(
                    compute_hpsv2(
                        items,
                        args.hps_version,
                        args.batch_size,
                        args.device,
                    )
                )
            except Exception as error:
                row["hpsv2_error"] = repr(error)
                print(f"HPSv2 failed for {method_name}: {error}")
            save_results(results + [row], args.generated_root, args.output_prefix)
        if "pickscore" in args.metrics:
            try:
                row.update(
                    compute_pickscore(
                        items=items,
                        model_name=args.pickscore_model,
                        device=args.device,
                        batch_size=args.batch_size,
                        local_files_only=args.local_files_only,
                        score_mode=args.pickscore_mode,
                    )
                )
            except Exception as error:
                row["pickscore_error"] = repr(error)
                print(f"PickScore failed for {method_name}: {error}")
            save_results(results + [row], args.generated_root, args.output_prefix)

        results.append(row)
        save_results(results, args.generated_root, args.output_prefix)


if __name__ == "__main__":
    main()
