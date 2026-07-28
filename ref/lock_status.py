"""
Lock status -- reference implementation.

Ported from AMD FSR2's ffx_fsr2_postprocess_lock_status.h: UpdateLockStatus,
the state machine that decides how much to trust history on a per-pixel
basis based on shading-change luminance and lock lifetime.

SCOPE NOTE (stated, not hidden): the real `GetShadingChangeLuma` samples
a mip level of a luminance pyramid built by a separate SPD (single-pass
downsampler) compute pass, which is NOT yet ported (it's the one file
that used wave/subgroup-quad intrinsics, deferred during the mobile
feasibility check). `update_lock_status` below takes
`shading_change_luma` as a caller-supplied input instead of computing it
internally. The state machine logic itself (lock lifetime, luminance
diff, reactive factor interaction) is a faithful 1:1 port -- only the
luma *source* is stubbed at the call boundary. When the luma pyramid
pass is built, callers switch to real sampled values with zero changes
needed here.
"""

import numpy as np

LOCK_LIFETIME_REMAINING = 0
LOCK_TEMPORAL_LUMA = 1


def min_divided_by_max(v0: float, v1: float) -> float:
    m = max(v0, v1)
    return min(v0, v1) / m if m != 0 else 0.0


def kill_lock(lock_status: np.ndarray) -> None:
    """Mutates lock_status in place, matching the source's inout semantics."""
    lock_status[LOCK_LIFETIME_REMAINING] = 0.0


def initialize_new_lock_sample() -> np.ndarray:
    return np.array([0.0, 0.0])


def update_lock_status(
    shading_change_luma: float,  # see SCOPE NOTE -- caller-supplied for now
    reactive_factor: float,
    is_new_lock: bool,
    lock_status: np.ndarray,  # (2,) [lifetime_remaining, temporal_luma], mutated in place
    accumulation_mask: float,
    depth_clip_factor: float,
):
    """
    Returns:
        new_reactive_factor: float (may be raised by this function)
        lock_status: same array, mutated in place (also returned for clarity)
        lock_contribution_this_frame: float
        luminance_diff: float
    """
    # matches source: init temporal luma on first-ever sample
    if lock_status[LOCK_TEMPORAL_LUMA] == 0.0:
        lock_status[LOCK_TEMPORAL_LUMA] = shading_change_luma

    previous_shading_change_luma = lock_status[LOCK_TEMPORAL_LUMA]
    luminance_diff = 1.0 - min_divided_by_max(
        previous_shading_change_luma, shading_change_luma
    )

    if is_new_lock:
        lock_status[LOCK_TEMPORAL_LUMA] = shading_change_luma
        lock_status[LOCK_LIFETIME_REMAINING] = (
            2.0 if lock_status[LOCK_LIFETIME_REMAINING] != 0.0 else 1.0
        )
    elif lock_status[LOCK_LIFETIME_REMAINING] <= 1.0:
        lock_status[LOCK_TEMPORAL_LUMA] = (
            lock_status[LOCK_TEMPORAL_LUMA] * 0.5 + shading_change_luma * 0.5
        )
    else:
        if luminance_diff > 0.1:
            kill_lock(lock_status)

    reactive_factor = max(
        reactive_factor, np.clip((luminance_diff - 0.1) * 10.0, 0.0, 1.0)
    )
    lock_status[LOCK_LIFETIME_REMAINING] *= 1.0 - reactive_factor
    lock_status[LOCK_LIFETIME_REMAINING] *= np.clip(1.0 - accumulation_mask, 0.0, 1.0)
    lock_status[LOCK_LIFETIME_REMAINING] *= float(depth_clip_factor < 0.1)

    lifetime_contribution = np.clip(lock_status[LOCK_LIFETIME_REMAINING] - 1.0, 0.0, 1.0)
    shading_change_contribution = np.clip(
        min_divided_by_max(lock_status[LOCK_TEMPORAL_LUMA], shading_change_luma),
        0.0,
        1.0,
    )

    lock_contribution_this_frame = np.clip(
        np.clip(lifetime_contribution * 4.0, 0.0, 1.0) * shading_change_contribution,
        0.0,
        1.0,
    )

    return reactive_factor, lock_status, lock_contribution_this_frame, luminance_diff
