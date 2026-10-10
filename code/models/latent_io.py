import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def decode_sd3_latent_to_pil(wrapper, latent):
    """Decode a scaled SD3 latent tensor into a PIL image."""
    shift = wrapper.pipe.vae.config.shift_factor
    scale = wrapper.pipe.vae.config.scaling_factor
    latent = (latent / scale) + shift

    image_tensor = wrapper.vae.decode(latent.to(torch.float32)).sample
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_array = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((image_array * 255).astype(np.uint8))
