"""
Upsample pass (part of Resolve/Accumulate) -- reference implementation.

Ported from AMD FSR2's ffx_fsr2_upsample.h + the RectificationBox helpers
in ffx_fsr2_common.h and Lanczos2ApproxSq in ffx_fsr2_sample.h.

This operates per-pixel (matching the shader's per-invocation structure),
not vectorized across the whole image -- that mirrors the source directly
and keeps this easy to cross-check line by line. Vectorize at the call
site once correctness is established, not here.
"""

import numpy as np
from dataclasses import dataclass, field


FSR2_UPSAMPLE_LANCZOS_WEIGHT_SCALE = 1.0 / 12.0  # ported constant, ffx_fsr2_common.h


def lanczos2_approx_sq(x2: float) -> float:
    """
    FSR1/FSR2 polynomial approximation of the Lanczos-2 kernel.
    Input is x*x (squared distance), must be clamped to <= 4 (matches
    Lanczos-2's support radius of 2).
    """
    x2 = min(x2, 4.0)
    a = (2.0 / 5.0) * x2 - 1.0
    b = (1.0 / 4.0) * x2 - 1.0
    return ((25.0 / 16.0) * a * a - (25.0 / 16.0 - 1.0)) * (b * b)


def get_upsample_lanczos_weight(src_sample_offset, kernel_weight: float) -> float:
    """Lanczos-2 sample weight for a given source-space offset."""
    ox = src_sample_offset[0] * kernel_weight
    oy = src_sample_offset[1] * kernel_weight
    return lanczos2_approx_sq(ox * ox + oy * oy)


def compute_max_kernel_weight(downscale_factor: float) -> float:
    """
    downscale_factor: render_size / display_size along one axis (< 1.0
    for upscaling). Matches AMD's DownscaleFactor() convention.
    """
    kernel_size_bias = 1.0
    kernel_weight = 1.0 + ((1.0 / downscale_factor) - 1.0) * kernel_size_bias
    return min(1.99, kernel_weight)


@dataclass
class RectificationBox:
    """Running weighted mean/variance + exact min/max over sampled colors."""
    box_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    box_vec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    aabb_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    aabb_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    box_center_weight: float = 0.0

    def add_sample(self, is_initial: bool, color_sample: np.ndarray, weight: float):
        if is_initial:
            self.aabb_min = color_sample.copy()
            self.aabb_max = color_sample.copy()
            weighted = color_sample * weight
            self.box_center = weighted.copy()
            self.box_vec = color_sample * weighted
            self.box_center_weight = weight
        else:
            self.aabb_min = np.minimum(self.aabb_min, color_sample)
            self.aabb_max = np.maximum(self.aabb_max, color_sample)
            weighted = color_sample * weight
            self.box_center += weighted
            self.box_vec += color_sample * weighted
            self.box_center_weight += weight

    def compute_variance_box_data(self, epsilon: float = 1e-6):
        """Convert running sums to (mean, stddev) and finalize the box."""
        w = self.box_center_weight if abs(self.box_center_weight) > epsilon else 1.0
        self.box_center = self.box_center / w
        self.box_vec = self.box_vec / w
        std_dev = np.sqrt(np.abs(self.box_vec - self.box_center * self.box_center))
        self.box_vec = std_dev


def compute_upsampled_color_and_weight(
    prepared_color: np.ndarray,  # (H_lr, W_lr, 3) YCoCg prepared color, low-res
    px_hr_pos,  # (x, y) integer high-res pixel position
    render_size,  # (w_lr, h_lr)
    display_size,  # (w_hr, h_hr)
    jitter,  # (jx, jy) in low-res pixel units, matches ffxFsr2GetJitterOffset scale
    reactive_factor: float,
    depth_clip_factor: float,
    is_new_sample: bool,
    hr_velocity: float,
):
    """
    Returns:
        color: (3,) upsampled YCoCg color at this pixel (weight-normalized)
        weight: float, total accumulated sample weight (0 if no valid samples)
        clipping_box: RectificationBox, finalized (mean/stddev/min/max)
    """
    w_lr, h_lr = render_size
    w_hr, h_hr = display_size
    downscale_factor = w_lr / w_hr  # assume uniform scale for x/y here

    dst_x, dst_y = px_hr_pos[0] + 0.5, px_hr_pos[1] + 0.5
    src_x, src_y = dst_x * downscale_factor, dst_y * downscale_factor
    src_input_x, src_input_y = int(np.floor(src_x)), int(np.floor(src_y))

    src_unjit_x = (src_input_x + 0.5) - jitter[0]
    src_unjit_y = (src_input_y + 0.5) - jitter[1]

    offset_tl_x = -2 if src_unjit_x > src_x else -1
    offset_tl_y = -2 if src_unjit_y > src_y else -1

    flip_row = src_unjit_y > src_y
    flip_col = src_unjit_x > src_x

    # Load the 3x3 sample grid (flip indexing per AMD's branch-avoidance trick)
    samples = {}
    for row in range(3):
        for col in range(3):
            sc = (3 - col) if flip_col else col
            sr = (3 - row) if flip_row else row
            sx = src_input_x + offset_tl_x + sc
            sy = src_input_y + offset_tl_y + sr
            cx = int(np.clip(sx, 0, w_lr - 1))
            cy = int(np.clip(sy, 0, h_lr - 1))
            samples[(row, col)] = prepared_color[cy, cx].astype(np.float64)

    base_offset_x = src_unjit_x - src_x
    base_offset_y = src_unjit_y - src_y

    kernel_reactive_factor = max(reactive_factor, float(is_new_sample))
    kernel_bias_max = compute_max_kernel_weight(downscale_factor) * (
        1.0 - kernel_reactive_factor
    )
    kernel_bias_min = max(1.0, (1.0 + kernel_bias_max) * 0.3)
    kernel_bias_factor = max(0.0, max(0.25 * depth_clip_factor, kernel_reactive_factor))
    kernel_bias = kernel_bias_max + (kernel_bias_min - kernel_bias_max) * kernel_bias_factor

    rectification_curve_bias = -2.0 + (-3.0 - -2.0) * np.clip(hr_velocity / 50.0, 0, 1)

    color_accum = np.zeros(3)
    weight_accum = 0.0
    box = RectificationBox()

    for row in range(3):
        for col in range(3):
            sc = (3 - col) if flip_col else col
            sr = (3 - row) if flip_row else row
            ox = offset_tl_x + sc
            oy = offset_tl_y + sr
            src_offset = (base_offset_x + ox, base_offset_y + oy)

            sx = src_input_x + offset_tl_x + sc
            sy = src_input_y + offset_tl_y + sr
            on_screen = 1.0 if (0 <= sx < w_lr and 0 <= sy < h_lr) else 0.0

            sample_weight = on_screen * get_upsample_lanczos_weight(src_offset, kernel_bias)
            color_accum += samples[(row, col)] * sample_weight
            weight_accum += sample_weight

            offset_sq = src_offset[0] ** 2 + src_offset[1] ** 2
            box_sample_weight = np.exp(rectification_curve_bias * offset_sq)
            box.add_sample(row == 0 and col == 0, samples[(row, col)], box_sample_weight)

    box.compute_variance_box_data()

    epsilon = 1e-6
    if weight_accum > epsilon:
        color_accum = color_accum / weight_accum
        weight_accum = weight_accum * FSR2_UPSAMPLE_LANCZOS_WEIGHT_SCALE
        color_accum = np.clip(color_accum, box.aabb_min, box.aabb_max)  # Deringing
    else:
        weight_accum = 0.0

    return color_accum, weight_accum, box
