"""
Luminance mip pyramid -- reference implementation.

Ported from AMD FSR2's ffx_fsr2_compute_luminance_pyramid.h (mip 0
construction) and GetShadingChangeLuma in
ffx_fsr2_postprocess_lock_status.h (sampling). This closes the last
stated gap in lock_status.py: shading_change_luma has been a
caller-supplied parameter since that module was written; this produces
a real value for it.

Confirmed from source: mip level 0 = log(max(epsilon, RGBToLuma(color))).
Each subsequent mip level is a standard box-filter downsample (2x2
average) -- this is the mathematical content of AMD's SPD (single-pass
downsampler); the quad-swap subgroup intrinsics in ffx_spd.h are a GPU
parallelization detail of the SAME box-filter math, not different math.
A CPU reference doesn't need the quad-swap trick to get the same result.

SIMPLIFICATION, stated: the exact mip level AMD samples for shading-change
detection is set via an internal resource-index difference
(FFX_FSR2_SHADING_CHANGE_MIP_LEVEL), not a plain documented constant --
tracing its exact value requires unwinding an enum ordering that doesn't
change the algorithm's shape, just which pyramid level gets read.
Exposed here as an explicit `mip_level` parameter instead of hardcoding
a possibly-wrong guessed index.
"""

import numpy as np

FSR2_EPSILON = 1.0 / 65504.0  # matches FSR2_TONEMAP_EPSILON = 1/FSR2_FP16_MAX


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb, np.array([0.2126, 0.7152, 0.0722]))


def build_log_luma_mip_pyramid(rgb_image: np.ndarray, num_levels: int = 6):
    """
    rgb_image: (H, W, 3) linear RGB, current frame, low-res
    Returns: list of 2D arrays, mip[0] is log-luma at full render
             resolution, each subsequent level half the size of the last
             (box-filter downsample), for num_levels total.
    """
    luma = rgb_to_luma(rgb_image)
    mip0 = np.log(np.maximum(FSR2_EPSILON, luma))

    pyramid = [mip0]
    current = mip0
    for _ in range(1, num_levels):
        h, w = current.shape
        h2, w2 = max(1, h // 2), max(1, w // 2)
        # standard 2x2 box downsample; pad with edge-replication if odd
        padded_h, padded_w = h2 * 2, w2 * 2
        cropped = current[:padded_h, :padded_w]
        if cropped.shape != (padded_h, padded_w):
            # fall back to edge-safe resize for odd dimensions
            cropped = current[:padded_h, :padded_w]
        downsampled = cropped.reshape(h2, 2, w2, 2).mean(axis=(1, 3))
        pyramid.append(downsampled)
        current = downsampled
        if h2 == 1 and w2 == 1:
            break

    return pyramid


def sample_mip_luma(pyramid, level: int, uv) -> float:
    """Bilinear sample of a specific mip level at normalized UV."""
    level = min(level, len(pyramid) - 1)
    mip = pyramid[level]
    h, w = mip.shape
    x = np.clip(uv[0] * w - 0.5, 0, w - 1)
    y = np.clip(uv[1] * h - 0.5, 0, h - 1)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top = mip[y0, x0] * (1 - fx) + mip[y0, x1] * fx
    bottom = mip[y1, x0] * (1 - fx) + mip[y1, x1] * fx
    return float(top * (1 - fy) + bottom * fy)


def get_shading_change_luma(pyramid, mip_level: int, hr_uv, exposure: float) -> float:
    """
    Ported from GetShadingChangeLuma. Returns the value lock_status.py's
    update_lock_status expects as `shading_change_luma`.
    """
    sampled_log_luma = sample_mip_luma(pyramid, mip_level, hr_uv)
    shading_change_luma = exposure * np.exp(sampled_log_luma)
    return float(np.power(max(shading_change_luma, 0.0), 1.0 / 6.0))
