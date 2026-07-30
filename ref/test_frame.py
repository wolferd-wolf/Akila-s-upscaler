import numpy as np
import pytest

from frame import render_frame

FOV_60 = np.radians(60.0)


def test_render_frame_static_scene_converges_over_multiple_frames():
    """
    The integration test: a whole synthetic image (not one hand-fed
    pixel) run through every ported module together -- reactive masks,
    depth reconstruction, luma pyramid, prepare, upsample, accumulate,
    lock status, reproject -- across several frames of a static scene,
    feeding each frame's real output forward as the next frame's input.
    This is what actually exercises the wiring between modules; the
    per-pixel unit tests elsewhere can't catch a mismatched buffer shape
    or a wrong array axis order the way a full-image run can.
    """
    w_lr, h_lr = 6, 6
    w_hr, h_hr = 12, 12

    rng = np.random.default_rng(0)
    input_color = rng.uniform(0.2, 0.6, size=(h_lr, w_lr, 3))
    current_depth = np.full((h_lr, w_lr), 0.5)
    motion_vectors = np.zeros((h_lr, w_lr, 2))
    reactive_mask = np.zeros((h_lr, w_lr))
    transparency_mask = np.zeros((h_lr, w_lr))

    history_color_buffer = np.zeros((h_hr, w_hr, 4))
    lock_status_buffer = np.zeros((h_hr, w_hr, 2))
    history_depth = None

    outputs = []
    for frame_index in range(6):
        (
            output_rgb,
            history_color_buffer,
            lock_status_buffer,
            history_depth,
        ) = render_frame(
            frame_index=frame_index,
            input_color=input_color,
            current_depth=current_depth,
            input_motion_vectors=motion_vectors,
            reactive_mask=reactive_mask,
            transparency_composition_mask=transparency_mask,
            history_color_buffer=history_color_buffer,
            lock_status_buffer=lock_status_buffer,
            render_size=(w_lr, h_lr),
            display_size=(w_hr, h_hr),
            jitter=(0.0, 0.0),
            near=0.1, far=1000.0, fov_y_radians=FOV_60,
            history_depth=history_depth,
        )
        assert output_rgb.shape == (h_hr, w_hr, 3)
        assert np.all(np.isfinite(output_rgb)), (
            f"non-finite output at frame {frame_index}"
        )
        outputs.append(output_rgb.copy())

    # after several frames of a static scene, output should have
    # stabilized -- last two frames should be close to each other,
    # not still drifting or oscillating
    diff = np.abs(outputs[-1] - outputs[-2])
    assert np.all(diff < 0.05), f"max diff between last two frames: {diff.max()}"


def test_render_frame_handles_uniform_motion_without_crashing():
    w_lr, h_lr = 6, 6
    w_hr, h_hr = 12, 12

    rng = np.random.default_rng(1)
    input_color = rng.uniform(0.2, 0.6, size=(h_lr, w_lr, 3))
    current_depth = np.full((h_lr, w_lr), 0.5)
    motion_vectors = np.full((h_lr, w_lr, 2), 0.01)  # small uniform pan
    reactive_mask = np.zeros((h_lr, w_lr))
    transparency_mask = np.zeros((h_lr, w_lr))

    history_color_buffer = np.zeros((h_hr, w_hr, 4))
    lock_status_buffer = np.zeros((h_hr, w_hr, 2))
    history_depth = None

    for frame_index in range(4):
        (
            output_rgb,
            history_color_buffer,
            lock_status_buffer,
            history_depth,
        ) = render_frame(
            frame_index=frame_index,
            input_color=input_color,
            current_depth=current_depth,
            input_motion_vectors=motion_vectors,
            reactive_mask=reactive_mask,
            transparency_composition_mask=transparency_mask,
            history_color_buffer=history_color_buffer,
            lock_status_buffer=lock_status_buffer,
            render_size=(w_lr, h_lr),
            display_size=(w_hr, h_hr),
            jitter=(0.0, 0.0),
            near=0.1, far=1000.0, fov_y_radians=FOV_60,
            history_depth=history_depth,
        )
        assert np.all(np.isfinite(output_rgb)), (
            f"non-finite output at frame {frame_index} under motion"
        )


def test_render_frame_first_frame_matches_reset_semantics():
    # frame_index=0 should behave like pipeline.py's reset-frame path --
    # no crash even with an all-zero, never-initialized history buffer.
    w_lr, h_lr = 4, 4
    w_hr, h_hr = 8, 8

    input_color = np.full((h_lr, w_lr, 3), 0.5)
    current_depth = np.full((h_lr, w_lr), 0.5)
    motion_vectors = np.zeros((h_lr, w_lr, 2))
    reactive_mask = np.zeros((h_lr, w_lr))
    transparency_mask = np.zeros((h_lr, w_lr))
    history_color_buffer = np.zeros((h_hr, w_hr, 4))
    lock_status_buffer = np.zeros((h_hr, w_hr, 2))

    output_rgb, _, _, _ = render_frame(
        frame_index=0,
        input_color=input_color,
        current_depth=current_depth,
        input_motion_vectors=motion_vectors,
        reactive_mask=reactive_mask,
        transparency_composition_mask=transparency_mask,
        history_color_buffer=history_color_buffer,
        lock_status_buffer=lock_status_buffer,
        render_size=(w_lr, h_lr),
        display_size=(w_hr, h_hr),
        jitter=(0.0, 0.0),
        near=0.1, far=1000.0, fov_y_radians=FOV_60,
    )
    assert np.all(np.isfinite(output_rgb))
