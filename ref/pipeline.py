"""
Pipeline orchestration -- reference implementation.

Ported from AMD FSR2's ffx_fsr2_accumulate.h: InitParams + the outer
Accumulate(iPxHrPos) function. This is the piece that calls every other
module in this package in the right order for a single output pixel.

SCOPE NOTE -- stated dependencies not yet built, taken as explicit
parameters rather than computed internally (consistent with every
prior module in this package):
  - dilated_reactive_factor, accumulation_mask: from SampleDilatedReactiveMasks,
    part of the reactive-mask pass (not yet ported)
  - shading_change_luma: from the luma mip pyramid (SPD pass, not yet ported --
    see lock_status.py's scope note)
  - new_lock_intensity: from a separate lock-input pass (not yet ported --
    see reproject.py's scope note)

What IS fully wired end-to-end here: reprojection, history sampling,
Lanczos upsampling + rectification box construction, history clamping
(RectifyHistory), the temporal blend (Accumulate), and the lock state
machine's bookkeeping (lifetime/luma tracking) -- everything this
package has actually ported gets exercised together, in the real order,
on every call.
"""

import numpy as np

from prepare import reproject_uv, depth_clip_factor as compute_depth_clip_factor
from upsample import compute_upsampled_color_and_weight
from accumulate import accumulate as blend_accumulate, rectify_history
from lock_status import update_lock_status, initialize_new_lock_sample
from reproject import (
    compute_reprojected_uvs,
    reproject_history_color,
    reproject_history_lock_status,
    ycocg_to_rgb,
)


def get_px_hr_velocity(motion_vector, display_size) -> float:
    """length(motion_vector * display_size) -- motion vector magnitude in HR pixels."""
    return float(
        np.hypot(motion_vector[0] * display_size[0], motion_vector[1] * display_size[1])
    )


def accumulate_pixel(
    px_hr_pos,
    render_size,
    display_size,
    frame_index: int,
    motion_vector,
    prepared_color: np.ndarray,
    current_depth: np.ndarray,
    history_depth: np.ndarray,
    history_color_buffer: np.ndarray,
    lock_status_buffer: np.ndarray,
    jitter,
    near: float,
    far: float,
    fov_y_radians: float,
    exposure: float = 1.0,
    previous_frame_pre_exposure: float = 1.0,
    dilated_reactive_factor: float = 0.0,
    accumulation_mask: float = 0.0,
    shading_change_luma: float = 0.0,
    new_lock_intensity: float = 0.0,
):
    """
    Full per-pixel accumulate pipeline, mirroring AMD's InitParams +
    Accumulate(iPxHrPos) order. Returns the final RGB output color for
    this pixel plus the updated lock_status (caller stores this back
    into the lock status buffer for next frame).
    """
    w_lr, h_lr = render_size
    w_hr, h_hr = display_size

    hr_uv = ((px_hr_pos[0] + 0.5) / w_hr, (px_hr_pos[1] + 0.5) / h_hr)
    hr_velocity = get_px_hr_velocity(motion_vector, display_size)

    reprojected_hr_uv, is_existing_sample = compute_reprojected_uvs(hr_uv, motion_vector)

    lr_uv = (hr_uv[0], hr_uv[1])

    depth_clip = compute_depth_clip_factor(
        current_depth=current_depth,
        history_depth=history_depth,
        prev_uv=reproject_uv(
            np.zeros((h_lr, w_lr, 2)) + np.array(motion_vector), h_lr, w_lr
        ),
        near=near, far=far, fov_y_radians=fov_y_radians,
    )
    lr_x = int(np.clip(lr_uv[0] * w_lr, 0, w_lr - 1))
    lr_y = int(np.clip(lr_uv[1] * h_lr, 0, h_lr - 1))
    this_pixel_depth_clip = float(depth_clip[lr_y, lr_x])

    is_reset_frame = frame_index == 0
    is_new_sample = (not is_existing_sample) or is_reset_frame

    history_color = np.zeros(3)
    lock_status = initialize_new_lock_sample()
    temporal_reactive_factor = 0.0
    was_in_motion_last_frame = False
    was_locked_prev_frame = False
    is_new_lock = new_lock_intensity > (127.0 / 255.0)

    if is_existing_sample and not is_reset_frame:
        history_color, temporal_reactive_factor, was_in_motion_last_frame = (
            reproject_history_color(
                history_buffer=history_color_buffer,
                reprojected_hr_uv=reprojected_hr_uv,
                exposure=exposure,
                previous_frame_pre_exposure=previous_frame_pre_exposure,
            )
        )
        is_new_lock, was_locked_prev_frame, lock_status = reproject_history_lock_status(
            lock_status_buffer=lock_status_buffer,
            reprojected_hr_uv=reprojected_hr_uv,
            new_lock_intensity=new_lock_intensity,
        )

    this_frame_reactive_factor = max(dilated_reactive_factor, temporal_reactive_factor)

    new_reactive_factor, lock_status, lock_contribution_this_frame, luminance_diff = (
        update_lock_status(
            shading_change_luma=shading_change_luma,
            reactive_factor=this_frame_reactive_factor,
            is_new_lock=is_new_lock,
            lock_status=lock_status,
            accumulation_mask=accumulation_mask,
            depth_clip_factor=this_pixel_depth_clip,
        )
    )

    upsampled_color, upsampled_weight, clipping_box = compute_upsampled_color_and_weight(
        prepared_color=prepared_color,
        px_hr_pos=px_hr_pos,
        render_size=render_size,
        display_size=display_size,
        jitter=jitter,
        reactive_factor=this_frame_reactive_factor,
        depth_clip_factor=this_pixel_depth_clip,
        is_new_sample=is_new_sample,
        hr_velocity=hr_velocity,
    )

    accumulation = np.full(3, max(0.0, upsampled_weight))

    if is_existing_sample and not is_reset_frame:
        history_color, accumulation = rectify_history(
            history_color=history_color,
            accumulation=accumulation,
            box_center=clipping_box.box_center,
            box_vec=clipping_box.box_vec,
            aabb_min=clipping_box.aabb_min,
            aabb_max=clipping_box.aabb_max,
            downscale_factor=render_size[0] / display_size[0],
            hr_velocity=hr_velocity,
            depth_clip_factor=this_pixel_depth_clip,
            accumulation_mask=accumulation_mask,
            dilated_reactive_factor=dilated_reactive_factor,
            lock_contribution_this_frame=lock_contribution_this_frame,
            luma_instability_factor=0.0,
        )
        history_color, accumulation = blend_accumulate(
            history_color=history_color,
            accumulation=float(np.mean(accumulation)),
            upsampled_color=upsampled_color,
            upsampled_weight=upsampled_weight,
        )
    else:
        history_color = upsampled_color

    final_rgb = ycocg_to_rgb(history_color)

    return final_rgb, lock_status
