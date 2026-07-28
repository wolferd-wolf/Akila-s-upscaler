"""
Reproject pass -- reference implementation.

Ported from AMD FSR2's ffx_fsr2_reproject.h: ComputeReprojectedUVs,
ReprojectHistoryColor, ReprojectHistoryLockStatus. This is the wire
between the Prepare pass output and everything built in upsample.py /
accumulate.py / lock_status.py -- it's what actually pulls last frame's
history color and lock state through the reprojected UV each frame.

SCOPE NOTE (stated, not hidden):
- Real HistorySample() uses a Lanczos-type bicubic filter. This port
  uses bilinear sampling (reuses sample_bilinear from prepare.py) as a
  documented simplification -- softer resampling, not wrong math, just
  lower fidelity than the real filter. Upgrading later doesn't change
  any of the surrounding logic.
- ReprojectHistoryLockStatus needs a "new lock intensity" value that in
  real FSR2 comes from a separate reactive-mask/lock-input pass not yet
  built. Taken as a caller-supplied parameter here, same pattern as
  shading_change_luma in lock_status.py.
"""

import numpy as np

from prepare import sample_bilinear
from lock_status import LOCK_LIFETIME_REMAINING

FSR2_FP16_MAX = 65504.0


def rgb_to_ycocg(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.25 * r + 0.5 * g + 0.25 * b
    co = 0.5 * r - 0.5 * b
    cg = -0.25 * r + 0.5 * g - 0.25 * b
    return np.stack([y, co, cg], axis=-1)


def ycocg_to_rgb(ycocg: np.ndarray) -> np.ndarray:
    y, co, cg = ycocg[..., 0], ycocg[..., 1], ycocg[..., 2]
    r = y + co - cg
    g = y + cg
    b = y - co - cg
    return np.stack([r, g, b], axis=-1)


def prepare_rgb(rgb: np.ndarray, exposure: float, pre_exposure: float) -> np.ndarray:
    """Undo previous frame's exposure, apply current exposure, clamp to FP16 range."""
    result = (rgb / pre_exposure) * exposure
    return np.clip(result, 0.0, FSR2_FP16_MAX)


def is_uv_inside(uv) -> bool:
    return (0.0 <= uv[0] <= 1.0) and (0.0 <= uv[1] <= 1.0)


def compute_reprojected_uvs(hr_uv, motion_vector):
    """
    hr_uv: (u, v) current high-res pixel's UV
    motion_vector: (mvx, mvy), UV-space, ADDED here (matches source
      convention in this file: fReprojectedHrUv = fHrUv + fMotionVector --
      note this is the opposite sign convention from Prepare pass's
      mv = uv_current - uv_previous; FSR2 stores/uses motion vectors with
      different sign conventions in different passes, ported as-is from
      each source file rather than "harmonized" by me, to avoid
      introducing a translation error).

    Returns: (reprojected_uv, is_existing_sample)
    """
    reprojected_uv = (hr_uv[0] + motion_vector[0], hr_uv[1] + motion_vector[1])
    is_existing_sample = is_uv_inside(reprojected_uv)
    return reprojected_uv, is_existing_sample


def reproject_history_color(
    history_buffer: np.ndarray,  # (H_hr, W_hr, 4) RGB + reactive-factor-signed-alpha
    reprojected_hr_uv,
    exposure: float,
    previous_frame_pre_exposure: float,
):
    """
    Returns:
        history_color: (3,) YCoCg
        temporal_reactive_factor: float in [0,1]
        was_in_motion_last_frame: bool
    """
    history_rgba = sample_bilinear(history_buffer, np.array([reprojected_hr_uv]))[0]
    history_rgb = history_rgba[:3]
    alpha = history_rgba[3]

    history_rgb = prepare_rgb(history_rgb, exposure, previous_frame_pre_exposure)
    history_color = rgb_to_ycocg(history_rgb)

    temporal_reactive_factor = np.clip(abs(alpha), 0.0, 1.0)
    was_in_motion_last_frame = alpha < 0.0

    return history_color, temporal_reactive_factor, was_in_motion_last_frame


def reproject_history_lock_status(
    lock_status_buffer: np.ndarray,  # (H_hr, W_hr, 2) [lifetime, temporal_luma]
    reprojected_hr_uv,
    new_lock_intensity: float,  # see SCOPE NOTE -- caller-supplied for now
):
    """
    Returns:
        is_new_lock: bool
        was_locked_prev_frame: bool
        reprojected_lock_status: (2,) sampled [lifetime, temporal_luma]
    """
    is_new_lock = new_lock_intensity > (127.0 / 255.0)

    reprojected_lock_status = sample_bilinear(
        lock_status_buffer, np.array([reprojected_hr_uv])
    )[0]

    was_locked_prev_frame = reprojected_lock_status[LOCK_LIFETIME_REMAINING] != 0.0

    return is_new_lock, was_locked_prev_frame, reprojected_lock_status
