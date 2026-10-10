import torch
import torch.nn.functional as F


ADG_DEFAULT_MAX_ANGLE = torch.pi / 3.0


def make_sd3_timesteps(num_steps, device, dtype, shift=3.0):
    t_seq = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    return torch.flip((shift * t_seq) / (1.0 + (shift - 1.0) * t_seq), dims=[0])


def clean_anchor(z_t, t, velocity):
    return z_t - t * velocity


def velocity_from_anchor(z_t, t, x0):
    return (z_t - x0) / (t + 1e-6)


def euler_step_from_velocity(z_t, t_curr, t_next, velocity):
    return z_t + (t_next - t_curr) * velocity


def gaussian_low_pass(x, kernel_size=17, sigma=4.0):
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.expand(x.shape[1], 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.to(device=x.device, dtype=x.dtype)
    return F.conv2d(x, kernel_2d, groups=x.shape[1], padding=radius)


def pyrdown(x, kernel_size=5, sigma=1.0):
    low = gaussian_low_pass(x, kernel_size=kernel_size, sigma=sigma)
    return F.interpolate(
        low,
        scale_factor=0.5,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )


def pyrup(x, size, kernel_size=5, sigma=1.0):
    up = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return gaussian_low_pass(up, kernel_size=kernel_size, sigma=sigma)


def build_laplacian_pyramid(x, levels, kernel_size=5, sigma=1.0):
    """
    Build a high-to-low Laplacian pyramid.

    The returned list follows the convention used in the FDG reference code:
    early entries are high-frequency residuals and the final entry is the
    coarsest low-frequency image.
    """
    if levels < 1:
        raise ValueError("levels must be positive")

    current = x
    pyramid = []
    for _ in range(levels - 1):
        down = pyrdown(current, kernel_size=kernel_size, sigma=sigma)
        up = pyrup(
            down,
            size=current.shape[-2:],
            kernel_size=kernel_size,
            sigma=sigma,
        )
        pyramid.append(current - up)
        current = down
    pyramid.append(current)
    return pyramid


def build_image_from_pyramid(pyramid, kernel_size=5, sigma=1.0):
    image = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        image = pyrup(
            image,
            size=pyramid[i].shape[-2:],
            kernel_size=kernel_size,
            sigma=sigma,
        ) + pyramid[i]
    return image


def project_parallel_orthogonal(v0, v1):
    dtype = v0.dtype
    v0 = v0.double()
    v1 = v1.double()
    v1 = F.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def laplacian_guided_anchor(
    x0_c,
    x0_u,
    guidance_scales,
    parallel_weights=None,
    kernel_size=5,
    sigma=1.0,
):
    """
    FDG reference-style Laplacian guidance on clean anchors.

    guidance_scales are ordered from high to low frequency. parallel_weights
    controls an optional APG-style projection per pyramid level: 1 keeps the
    full residual, 0 keeps only the orthogonal residual.
    """
    levels = len(guidance_scales)
    if parallel_weights is None:
        parallel_weights = [1.0] * levels
    if len(parallel_weights) != levels:
        raise ValueError("parallel_weights must match guidance_scales length")

    cond_pyramid = build_laplacian_pyramid(
        x0_c, levels, kernel_size=kernel_size, sigma=sigma
    )
    uncond_pyramid = build_laplacian_pyramid(
        x0_u, levels, kernel_size=kernel_size, sigma=sigma
    )

    guided_pyramid = []
    for p_cond, p_uncond, scale, parallel_weight in zip(
        cond_pyramid,
        uncond_pyramid,
        guidance_scales,
        parallel_weights,
    ):
        diff = p_cond - p_uncond
        diff_parallel, diff_orthogonal = project_parallel_orthogonal(
            diff, p_cond
        )
        diff = parallel_weight * diff_parallel + diff_orthogonal
        guided_pyramid.append(p_cond + (scale - 1.0) * diff)

    return build_image_from_pyramid(
        guided_pyramid, kernel_size=kernel_size, sigma=sigma
    ).to(dtype=x0_c.dtype)


def build_radial_frequency_mask(x, cutoff_ratio=0.25, band="low"):
    if not 0.0 < cutoff_ratio <= 1.0:
        raise ValueError("cutoff_ratio must be in (0, 1]")
    if band not in {"low", "high", "all"}:
        raise ValueError("band must be 'low', 'high', or 'all'")

    height, width = x.shape[-2:]
    freq_y = torch.fft.fftfreq(height, device=x.device, dtype=torch.float32)
    freq_x = torch.fft.rfftfreq(width, device=x.device, dtype=torch.float32)
    radius = torch.sqrt(freq_y[:, None].square() + freq_x[None, :].square())
    low_mask = (radius <= 0.5 * cutoff_ratio).to(torch.float32)
    mask = low_mask if band == "low" else 1.0 - low_mask
    return mask.view(1, 1, height, width // 2 + 1)


def project_update_against_reference(update, reference, parallel_scale=0.0):
    dot_product = torch.sum(update * reference, dim=(1, 2, 3), keepdim=True)
    norm_sq = torch.sum(reference * reference, dim=(1, 2, 3), keepdim=True)
    parallel = (dot_product / (norm_sq + 1e-8)) * reference
    orthogonal = update - parallel
    return orthogonal + parallel_scale * parallel


def apg_clip_and_momentum(delta, reference, momentum_buffer, r=1.5, beta=-0.2):
    delta_norm = torch.linalg.vector_norm(delta, dim=(1, 2, 3), keepdim=True)
    radius = torch.linalg.vector_norm(reference, dim=(1, 2, 3), keepdim=True) * r
    scale = torch.clamp(radius / (delta_norm + 1e-5), max=1.0)
    delta = delta * scale
    return delta + beta * momentum_buffer


def angle_domain_guided_anchor(
    x0_c,
    x0_u,
    cfg_scale,
    max_turn_angle=ADG_DEFAULT_MAX_ANGLE,
    preserve_text_norm=False,
    eps=1e-6,
):
    """
    Official ADG-SD3-style angular update in clean-anchor space.

    This follows the official ADG SD3 implementation with the correspondence
    latent_hat = z_t - sigma * noise_pred. In our flow-matching notation,
    sigma is the normalized time t and latent_hat is the clean anchor x0.
    """
    dtype = x0_c.dtype
    x0_c_f = x0_c.float()
    x0_u_f = x0_u.float()

    dims = (1, 2, 3)
    c_norm = torch.linalg.vector_norm(x0_c_f, dim=dims, keepdim=True)
    u_norm = torch.linalg.vector_norm(x0_u_f, dim=dims, keepdim=True)
    dot = torch.sum(x0_u_f * x0_c_f, dim=dims, keepdim=True)
    cos_theta = dot / (u_norm * c_norm + eps)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)

    weight = cfg_scale - 1.0
    if weight <= 0:
        weight = 1e-3
    theta_guided = torch.clamp(weight * theta, min=-max_turn_angle, max=max_turn_angle)

    residual = x0_c_f - x0_u_f
    proj_coeff = torch.sum(residual * x0_u_f, dim=dims, keepdim=True) / (
        torch.sum(x0_u_f * x0_u_f, dim=dims, keepdim=True) + eps
    )
    perpendicular = residual - proj_coeff * x0_u_f

    sin_theta = torch.sin(theta)
    angular_scale = torch.sin(theta_guided) / (sin_theta + eps)
    perpendicular_guided = torch.where(
        sin_theta > 1e-3,
        perpendicular * angular_scale,
        perpendicular * weight,
    )
    x0_guided = torch.cos(theta_guided) * x0_c_f + perpendicular_guided

    if preserve_text_norm:
        text_norm = torch.linalg.vector_norm(x0_c_f, dim=1, keepdim=True)
        guided_norm = torch.linalg.vector_norm(x0_guided, dim=1, keepdim=True)
        x0_guided = x0_guided * text_norm / (guided_norm + eps)
    return x0_guided.to(dtype=dtype)


def fft_radial_tangential_components(x0_c, x0_u, cutoff_ratio=0.25):
    spectrum_c = torch.fft.rfft2(x0_c.float(), dim=(-2, -1), norm="ortho")
    spectrum_u = torch.fft.rfft2(x0_u.float(), dim=(-2, -1), norm="ortho")
    residual = spectrum_c - spectrum_u

    unit = spectrum_c / (torch.abs(spectrum_c) + 1e-8)
    radial_coeff = torch.real(residual * torch.conj(unit))
    radial = radial_coeff * unit
    tangential = residual - radial

    high_mask = build_radial_frequency_mask(x0_c, cutoff_ratio=cutoff_ratio, band="high")
    return spectrum_c, radial, tangential, high_mask


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
    if band not in {"low", "high", "all"}:
        raise ValueError("band must be 'low', 'high', or 'all'")
    if direction not in {"full_radial", "positive_radial", "negative_radial", "tangential"}:
        raise ValueError(
            "direction must be 'full_radial', 'positive_radial', 'negative_radial', or 'tangential'"
        )

    if band == "low":
        frequency_mask = low_mask
    elif band == "high":
        frequency_mask = high_mask
    else:
        frequency_mask = torch.ones_like(low_mask)
    if direction == "full_radial":
        return frequency_mask * radial
    if direction == "positive_radial":
        direction_mask = (radial_coeff > 0).to(radial.dtype)
        return frequency_mask * direction_mask * radial
    if direction == "negative_radial":
        direction_mask = (radial_coeff < 0).to(radial.dtype)
        return frequency_mask * direction_mask * radial
    return frequency_mask * tangential


@torch.no_grad()
def sample_standard_cfg(wrapper, z_T, context, num_steps=20, cfg_scale=15.0, shift=3.0):
    """Standard classifier-free guidance in the SD3 velocity output space."""
    z_t = z_T.clone()
    timesteps = make_sd3_timesteps(num_steps, wrapper.device, wrapper.dtype, shift)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        v_u, v_c = wrapper(z_t, t_curr, context)
        v_guided = v_u + cfg_scale * (v_c - v_u)
        z_t = euler_step_from_velocity(z_t, t_curr, t_next, v_guided)

    return z_t


@torch.no_grad()
def sample_cfg_rescale(
    wrapper,
    z_T,
    context,
    num_steps=20,
    cfg_scale=15.0,
    shift=3.0,
    rescale_factor=0.7,
):
    """
    CFG Rescale adapted to the SD3 velocity output.

    This follows the standard rescale-noise-CFG idea: match the guided output
    standard deviation to the conditional output and interpolate with the raw
    CFG output by rescale_factor.
    """
    if not 0.0 <= rescale_factor <= 1.0:
        raise ValueError("rescale_factor must be in [0, 1]")

    z_t = z_T.clone()
    timesteps = make_sd3_timesteps(num_steps, wrapper.device, wrapper.dtype, shift)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        v_u, v_c = wrapper(z_t, t_curr, context)
        v_cfg = v_u + cfg_scale * (v_c - v_u)
        std_c = torch.std(v_c, dim=(1, 2, 3), keepdim=True)
        std_cfg = torch.std(v_cfg, dim=(1, 2, 3), keepdim=True)
        v_rescaled = v_cfg * (std_c / (std_cfg + 1e-6))
        v_guided = rescale_factor * v_rescaled + (1.0 - rescale_factor) * v_cfg
        z_t = euler_step_from_velocity(z_t, t_curr, t_next, v_guided)

    return z_t


@torch.no_grad()
def sample_anchor_apg(
    wrapper,
    z_T,
    context,
    num_steps=20,
    cfg_scale=15.0,
    shift=3.0,
    r=1.5,
    beta=-0.2,
    parallel_scale=0.0,
):
    """
    APG adapted according to its denoised-prediction requirement.

    APG should act on denoised predictions. In flow matching, the denoised
    prediction is the clean anchor x0 = z_t - t v. We therefore project the
    clean-anchor residual against x0_c, apply APG clipping and reverse momentum,
    then map the corrected anchor back to velocity.
    """
    z_t = z_T.clone()
    timesteps = make_sd3_timesteps(num_steps, wrapper.device, wrapper.dtype, shift)
    momentum_buffer = torch.zeros_like(z_t)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        v_u, v_c = wrapper(z_t, t_curr, context)
        x0_u = clean_anchor(z_t, t_curr, v_u)
        x0_c = clean_anchor(z_t, t_curr, v_c)

        residual = x0_c - x0_u
        selected = project_update_against_reference(
            residual, x0_c, parallel_scale=parallel_scale
        )
        delta_x0 = (cfg_scale - 1.0) * selected
        momentum_buffer = apg_clip_and_momentum(
            delta_x0, x0_c, momentum_buffer, r=r, beta=beta
        )
        x0_guided = x0_c + momentum_buffer
        v_guided = velocity_from_anchor(z_t, t_curr, x0_guided)
        z_t = euler_step_from_velocity(z_t, t_curr, t_next, v_guided)

    return z_t


@torch.no_grad()
def sample_anchor_adg(
    wrapper,
    z_T,
    context,
    num_steps=20,
    cfg_scale=15.0,
    shift=3.0,
    max_turn_angle=ADG_DEFAULT_MAX_ANGLE,
    preserve_text_norm=False,
):
    """
    Angle-Domain Guidance adapted to flow-matching clean anchors.

    The original ADG update is defined on denoised predictions. For SD3 flow
    matching, the analogous denoised prediction is the clean anchor
    x0 = z_t - t v. After the ADG angular update, the corrected anchor is
    mapped back to a compatible velocity for the Euler step.
    """
    z_t = z_T.clone()
    timesteps = make_sd3_timesteps(num_steps, wrapper.device, wrapper.dtype, shift)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        v_u, v_c = wrapper(z_t, t_curr, context)
        x0_u = clean_anchor(z_t, t_curr, v_u)
        x0_c = clean_anchor(z_t, t_curr, v_c)
        x0_guided = angle_domain_guided_anchor(
            x0_c=x0_c,
            x0_u=x0_u,
            cfg_scale=cfg_scale,
            max_turn_angle=max_turn_angle,
            preserve_text_norm=preserve_text_norm,
        )
        v_guided = velocity_from_anchor(z_t, t_curr, x0_guided)
        z_t = euler_step_from_velocity(z_t, t_curr, t_next, v_guided)

    return z_t


@torch.no_grad()
def sample_fdg_laplacian(
    wrapper,
    z_T,
    context,
    num_steps=20,
    cfg_scale=15.0,
    low_cfg_scale=1.0,
    shift=3.0,
    guidance_scales=None,
    kernel_size=5,
    sigma=1.0,
):
    """
    FDG-style Laplacian pyramid guidance in clean-anchor space.

    guidance_scales follow the FDG convention from high to low frequency.
    By default, we use a two-level pyramid with high scale cfg_scale and low
    scale low_cfg_scale.
    """
    if guidance_scales is None:
        guidance_scales = [cfg_scale, low_cfg_scale]

    z_t = z_T.clone()
    timesteps = make_sd3_timesteps(num_steps, wrapper.device, wrapper.dtype, shift)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        v_u, v_c = wrapper(z_t, t_curr, context)
        x0_u = clean_anchor(z_t, t_curr, v_u)
        x0_c = clean_anchor(z_t, t_curr, v_c)
        x0_guided = laplacian_guided_anchor(
            x0_c,
            x0_u,
            guidance_scales=guidance_scales,
            parallel_weights=None,
            kernel_size=kernel_size,
            sigma=sigma,
        )
        v_guided = velocity_from_anchor(z_t, t_curr, x0_guided)
        z_t = euler_step_from_velocity(z_t, t_curr, t_next, v_guided)

    return z_t


@torch.no_grad()
def sample_fdg_plus_apg(
    wrapper,
    z_T,
    context,
    num_steps=20,
    cfg_scale=15.0,
    low_cfg_scale=1.0,
    shift=3.0,
    guidance_scales=None,
    kernel_size=5,
    sigma=1.0,
    r=1.5,
    beta=-0.2,
):
    """
    Strong combined baseline: FDG pyramid guidance with per-level projection.

    This is not a separate cited method. It is a deliberately strong baseline
    to test whether RPTG is merely FDG plus global APG.
    """
    if guidance_scales is None:
        guidance_scales = [cfg_scale, low_cfg_scale]

    z_t = z_T.clone()
    timesteps = make_sd3_timesteps(num_steps, wrapper.device, wrapper.dtype, shift)
    momentum_buffer = torch.zeros_like(z_t)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        v_u, v_c = wrapper(z_t, t_curr, context)
        x0_u = clean_anchor(z_t, t_curr, v_u)
        x0_c = clean_anchor(z_t, t_curr, v_c)
        x0_fdg_apg = laplacian_guided_anchor(
            x0_c,
            x0_u,
            guidance_scales=guidance_scales,
            parallel_weights=[0.0] * len(guidance_scales),
            kernel_size=kernel_size,
            sigma=sigma,
        )
        selected = x0_fdg_apg - x0_c
        momentum_buffer = apg_clip_and_momentum(
            selected, x0_c, momentum_buffer, r=r, beta=beta
        )
        x0_guided = x0_c + momentum_buffer
        v_guided = velocity_from_anchor(z_t, t_curr, x0_guided)
        z_t = euler_step_from_velocity(z_t, t_curr, t_next, v_guided)

    return z_t


@torch.no_grad()
def sample_rptg(
    wrapper,
    z_T,
    context,
    num_steps=20,
    cfg_scale=15.0,
    shift=3.0,
    cutoff_ratio=0.25,
    high_radial_scale=0.0,
):
    """
    Relative-Phase Tangential Guidance in clean-anchor Fourier space.

    high_radial_scale=0 gives pure RPTG. A positive value gives RPTG-HR by
    restoring a controlled portion of the high-frequency radial residual.
    """
    z_t = z_T.clone()
    timesteps = make_sd3_timesteps(num_steps, wrapper.device, wrapper.dtype, shift)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        v_u, v_c = wrapper(z_t, t_curr, context)
        x0_u = clean_anchor(z_t, t_curr, v_u)
        x0_c = clean_anchor(z_t, t_curr, v_c)

        spectrum_c, radial, tangential, high_mask = fft_radial_tangential_components(
            x0_c, x0_u, cutoff_ratio=cutoff_ratio
        )
        selected = tangential + high_radial_scale * high_mask * radial
        spectrum_guided = spectrum_c + (cfg_scale - 1.0) * selected
        x0_guided = torch.fft.irfft2(
            spectrum_guided,
            s=x0_c.shape[-2:],
            dim=(-2, -1),
            norm="ortho",
        ).to(dtype=x0_c.dtype)
        v_guided = velocity_from_anchor(z_t, t_curr, x0_guided)
        z_t = euler_step_from_velocity(z_t, t_curr, t_next, v_guided)

    return z_t


def sample_rptg_hr(
    wrapper,
    z_T,
    context,
    num_steps=20,
    cfg_scale=15.0,
    shift=3.0,
    cutoff_ratio=0.25,
    high_radial_scale=0.25,
):
    return sample_rptg(
        wrapper=wrapper,
        z_T=z_T,
        context=context,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        shift=shift,
        cutoff_ratio=cutoff_ratio,
        high_radial_scale=high_radial_scale,
    )


@torch.no_grad()
def sample_rptg_directional_ablation(
    wrapper,
    z_T,
    context,
    num_steps=20,
    cfg_scale=15.0,
    shift=3.0,
    cutoff_ratio=0.25,
    band="low",
    direction="positive_radial",
):
    """
    Diagnostic RPTG ablation that keeps one frequency-direction component.

    This is intended for mechanism figures, not as a final guidance method.
    The retained component is extrapolated from x0_c with the same CFG
    strength (cfg_scale - 1) used by RPTG.
    """
    z_t = z_T.clone()
    timesteps = make_sd3_timesteps(num_steps, wrapper.device, wrapper.dtype, shift)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
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
            x0_c, x0_u, cutoff_ratio=cutoff_ratio
        )
        selected = select_frequency_direction_component(
            radial=radial,
            tangential=tangential,
            radial_coeff=radial_coeff,
            low_mask=low_mask,
            high_mask=high_mask,
            band=band,
            direction=direction,
        )
        spectrum_guided = spectrum_c + (cfg_scale - 1.0) * selected
        x0_guided = torch.fft.irfft2(
            spectrum_guided,
            s=x0_c.shape[-2:],
            dim=(-2, -1),
            norm="ortho",
        ).to(dtype=x0_c.dtype)
        v_guided = velocity_from_anchor(z_t, t_curr, x0_guided)
        z_t = euler_step_from_velocity(z_t, t_curr, t_next, v_guided)

    return z_t


@torch.no_grad()
def sample_rptg_apg_stabilized(
    wrapper,
    z_T,
    context,
    num_steps=20,
    cfg_scale=15.0,
    shift=3.0,
    cutoff_ratio=0.25,
    high_radial_scale=0.0,
    r=1.5,
    beta=-0.2,
):
    """
    RPTG with APG-style clipping and reverse momentum.

    This keeps RPTG's per-frequency local radial/tangential geometry intact.
    Unlike APG and FDG+APG, it does not apply an additional global projection
    against x0_c. It only applies APG's engineering stabilizers to the RPTG
    clean-anchor update.
    """
    z_t = z_T.clone()
    timesteps = make_sd3_timesteps(num_steps, wrapper.device, wrapper.dtype, shift)
    momentum_buffer = torch.zeros_like(z_t)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        v_u, v_c = wrapper(z_t, t_curr, context)
        x0_u = clean_anchor(z_t, t_curr, v_u)
        x0_c = clean_anchor(z_t, t_curr, v_c)

        spectrum_c, radial, tangential, high_mask = fft_radial_tangential_components(
            x0_c, x0_u, cutoff_ratio=cutoff_ratio
        )
        selected = tangential + high_radial_scale * high_mask * radial
        delta_spectrum = (cfg_scale - 1.0) * selected
        delta_x0 = torch.fft.irfft2(
            delta_spectrum,
            s=x0_c.shape[-2:],
            dim=(-2, -1),
            norm="ortho",
        ).to(dtype=x0_c.dtype)

        momentum_buffer = apg_clip_and_momentum(
            delta_x0, x0_c, momentum_buffer, r=r, beta=beta
        )
        x0_guided = x0_c + momentum_buffer
        v_guided = velocity_from_anchor(z_t, t_curr, x0_guided)
        z_t = euler_step_from_velocity(z_t, t_curr, t_next, v_guided)

    return z_t


def sample_rptg_hr_apg_stabilized(
    wrapper,
    z_T,
    context,
    num_steps=20,
    cfg_scale=15.0,
    shift=3.0,
    cutoff_ratio=0.25,
    high_radial_scale=0.25,
    r=1.5,
    beta=-0.2,
):
    return sample_rptg_apg_stabilized(
        wrapper=wrapper,
        z_T=z_T,
        context=context,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        shift=shift,
        cutoff_ratio=cutoff_ratio,
        high_radial_scale=high_radial_scale,
        r=r,
        beta=beta,
    )
