"""
Accumulate pass -- reference implementation.

Ported from AMD FSR2's ffx_fsr2_accumulate.h: Accumulate() (the actual
temporal blend) and RectifyHistory() (history clamping against the
rectification box built in upsample.py, to prevent ghosting).

SCOPE NOTE (deliberate, not a silent gap): the lock/luma-instability
system is not yet ported. Both functions below accept
`lock_contribution` and `luma_instability_factor` as explicit parameters
with default 0.0 (= "no lock system active yet") rather than computing
them internally. When the lock system is ported, callers pass real
values through; nothing here needs to change. Also skipping the
HDR tonemap branch (FFX_FSR2_OPTION_HDR_COLOR_INPUT) -- assuming SDR
input for v1.
"""

import numpy as np

FSR2_EPSILON = 1e-6


def accumulate(
    history_color: np.ndarray,  # (3,) YCoCg, previous accumulated color
    accumulation: float,  # running accumulated weight (scalar, broadcast in source)
    upsampled_color: np.ndarray,  # (3,) YCoCg, from compute_upsampled_color_and_weight
    upsampled_weight: float,
):
    """
    Blend the new upsampled sample into the running history color.

    Returns:
        new_history_color: (3,) blended YCoCg color
        new_accumulation: float, updated accumulated weight
    """
    new_accumulation = max(FSR2_EPSILON, accumulation + upsampled_weight)
    alpha = upsampled_weight / new_accumulation
    new_history_color = history_color * (1.0 - alpha) + upsampled_color * alpha
    return new_history_color, new_accumulation


def rectify_history(
    history_color: np.ndarray,  # (3,) YCoCg
    accumulation: np.ndarray,  # (3,) broadcast accumulation, or scalar
    box_center: np.ndarray,  # (3,) from RectificationBox.compute_variance_box_data()
    box_vec: np.ndarray,  # (3,) stddev, from same
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    downscale_factor: float,  # render/display ratio, same as upsample.py
    hr_velocity: float,
    depth_clip_factor: float,
    accumulation_mask: float,
    dilated_reactive_factor: float,
    lock_contribution_this_frame: float = 0.0,  # lock system not yet ported
    luma_instability_factor: float = 0.0,  # lock system not yet ported
):
    """
    Clamp history_color into a scaled rectification box if it falls
    outside it, softened by lock/reactive contributions. Returns the
    (possibly) adjusted (history_color, accumulation).
    """
    scale_factor_influence = min(20.0, (1.0 / (downscale_factor ** 2)) ** 3.0)

    velocity_factor = np.clip(hr_velocity / 20.0, 0.0, 1.0)
    box_scale_t = max(depth_clip_factor, max(accumulation_mask, velocity_factor))
    box_scale = scale_factor_influence + (1.0 - scale_factor_influence) * box_scale_t

    scaled_box_vec = box_vec * box_scale
    box_min = box_center - scaled_box_vec
    box_max = box_center + scaled_box_vec

    box_min = np.maximum(aabb_min, box_min)
    box_max = np.minimum(aabb_max, box_max)

    if np.any(box_min > history_color) or np.any(history_color > box_max):
        clamped_history_color = np.clip(history_color, box_min, box_max)

        history_contribution = max(luma_instability_factor, lock_contribution_this_frame)

        reactive_contribution = 1.0 - dilated_reactive_factor ** 0.5
        history_contribution *= reactive_contribution
        history_contribution = np.clip(history_contribution, 0.0, 1.0)

        history_color = clamped_history_color * (1.0 - history_contribution) + (
            history_color * history_contribution
        )

        accumulation_min = np.minimum(accumulation, 0.1)
        accumulation = (
            accumulation_min * (1.0 - history_contribution)
            + accumulation * history_contribution
        )

    return history_color, accumulation
