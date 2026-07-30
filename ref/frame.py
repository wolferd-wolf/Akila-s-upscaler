"""
Full-frame orchestration -- reference implementation.

This is the integration layer above pipeline.py's per-pixel
accumulate_pixel(): it runs the reactive-mask pass, depth
reconstruction/dilation pass, and luma pyramid construction over a
whole low-res frame, then calls accumulate_pixel() for every high-res
output pixel with REAL computed values instead of the caller-supplied
stand-ins used in every prior single-pixel test.

This is the first place in the project that exercises every ported
module together on a whole image across multiple frames, rather than
one pixel with hand-fed inputs.

CAVEAT, stated: prepared_color is passed through as raw linear RGB here,
not YCoCg (ComputePreparedInputColor -- the actual RGB-to-YCoCg
conversion for the low-res prepared buffer -- hasn't been separately
ported as its own module; upsample.py/accumulate.py were built assuming
YCoCg input). This integration test currently runs the pipeline in RGB
space throughout instead, which is internally consistent (upsampling
and blending RGB directly still works correctly) but does NOT match
real FSR2's actual color space per-stage. Correctness of the temporal
math is still verified; color-space fidelity to the real shader is not,
yet, and is a stated remaining gap, not a hidden one.
"""

import numpy as np

from reconstruct_depth import reconstruct_and_dilate
from reactive_mask import preprocess_reactive_masks, compute_motion_divergence
from luma_pyramid import build_log_luma_mip_pyramid, get_shading_change_luma
from pipeline import accumulate_pixel


def render_frame(
    frame_index: int,
    input_color: np.ndarray,
    current_depth: np.ndarray,
    input_motion_vectors: np.ndarray,
    reactive_mask: np.ndarray,
    transparency_composition_mask: np.ndarray,
    history_color_buffer: np.ndarray,
    lock_status_buffer: np.ndarray,
    render_size,
    display_size,
    jitter,
    near: float,
    far: float,
    fov_y_radians: float,
    exposure: float = 1.0,
    pre_exposure: float = 1.0,
    shading_change_mip_level: int = 4,
    history_depth: np.ndarray = None,
):
    """
    Renders one full frame. Returns:
        output_rgb: (H_hr, W_hr, 3) final upscaled image
        new_history_color_buffer: (H_hr, W_hr, 4) -- caller stores for next frame
        new_lock_status_buffer: (H_hr, W_hr, 2) -- caller stores for next frame
        reconstructed_prev_depth: (H_lr, W_lr) -- pass as history_depth next frame
    """
    w_lr, h_lr = render_size
    w_hr, h_hr = display_size

    if history_depth is None:
        history_depth = np.full((h_lr, w_lr), np.inf)

    dilated_depth, dilated_mv, reconstructed_prev_depth, _ = reconstruct_and_dilate(
        current_depth=current_depth,
        input_motion_vectors=input_motion_vectors,
        current_color=input_color,
        render_size=render_size,
        display_size=display_size,
        exposure=exposure,
        pre_exposure=pre_exposure,
    )

    luma_pyramid = build_log_luma_mip_pyramid(input_color, num_levels=6)

    reactive_factor_map = np.zeros((h_lr, w_lr))
    accumulation_mask_map = np.zeros((h_lr, w_lr))
    for y in range(h_lr):
        for x in range(w_lr):
            motion_divergence = compute_motion_divergence(
                (x, y), dilated_mv, render_size
            )
            reactive_factor, accumulation_mask = preprocess_reactive_masks(
                (x, y), input_color, reactive_mask, transparency_composition_mask,
                motion_divergence, render_size,
            )
            reactive_factor_map[y, x] = reactive_factor
            accumulation_mask_map[y, x] = accumulation_mask

    output_rgb = np.zeros((h_hr, w_hr, 3))
    new_lock_status_buffer = lock_status_buffer.copy()
    new_history_color_buffer = history_color_buffer.copy()

    for py in range(h_hr):
        for px in range(w_hr):
            lr_x = int(np.clip(px * w_lr / w_hr, 0, w_lr - 1))
            lr_y = int(np.clip(py * h_lr / h_hr, 0, h_lr - 1))
            motion_vector = tuple(dilated_mv[lr_y, lr_x])

            hr_uv = ((px + 0.5) / w_hr, (py + 0.5) / h_hr)
            shading_change_luma = get_shading_change_luma(
                luma_pyramid, shading_change_mip_level, hr_uv, exposure
            )

            rgb, new_lock_status = accumulate_pixel(
                px_hr_pos=(px, py),
                render_size=render_size,
                display_size=display_size,
                frame_index=frame_index,
                motion_vector=motion_vector,
                prepared_color=input_color,
                current_depth=dilated_depth,
                history_depth=history_depth,
                history_color_buffer=history_color_buffer,
                lock_status_buffer=lock_status_buffer,
                jitter=jitter,
                near=near, far=far, fov_y_radians=fov_y_radians,
                exposure=exposure, previous_frame_pre_exposure=pre_exposure,
                dilated_reactive_factor=float(reactive_factor_map[lr_y, lr_x]),
                accumulation_mask=float(accumulation_mask_map[lr_y, lr_x]),
                shading_change_luma=shading_change_luma,
                new_lock_intensity=0.0,
            )

            output_rgb[py, px] = rgb
            new_lock_status_buffer[py, px] = new_lock_status
            new_history_color_buffer[py, px, :3] = rgb
            new_history_color_buffer[py, px, 3] = 0.0

    return (
        output_rgb,
        new_history_color_buffer,
        new_lock_status_buffer,
        reconstructed_prev_depth,
    )
