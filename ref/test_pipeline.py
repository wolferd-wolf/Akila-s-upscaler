import numpy as np
import pytest

from pipeline import accumulate_pixel, get_px_hr_velocity


FOV_60 = np.radians(60.0)


def test_get_px_hr_velocity_zero_motion():
    v = get_px_hr_velocity((0.0, 0.0), (1920, 1080))
    assert v == 0.0


def test_get_px_hr_velocity_scales_with_display_size():
    v_small = get_px_hr_velocity((0.01, 0.0), (100, 100))
    v_large = get_px_hr_velocity((0.01, 0.0), (1000, 1000))
    assert v_large > v_small


def test_accumulate_pixel_reset_frame_uses_upsampled_color_only():
    # frame_index == 0 -> is_reset_frame -> no history blend, output
    # should just be the upsampled color for this static uniform scene.
    h_lr, w_lr = 8, 8
    h_hr, w_hr = 16, 16
    constant_color = np.array([0.3, 0.0, 0.0])  # YCoCg, gray-ish
    prepared = np.tile(constant_color, (h_lr, w_lr, 1))
    depth = np.full((h_lr, w_lr), 0.5)
    history_color_buf = np.zeros((h_hr, w_hr, 4))
    lock_status_buf = np.zeros((h_hr, w_hr, 2))

    rgb, lock_status = accumulate_pixel(
        px_hr_pos=(8, 8), render_size=(w_lr, h_lr), display_size=(w_hr, h_hr),
        frame_index=0, motion_vector=(0.0, 0.0),
        prepared_color=prepared, current_depth=depth, history_depth=depth,
        history_color_buffer=history_color_buf, lock_status_buffer=lock_status_buf,
        jitter=(0.0, 0.0), near=0.1, far=1000.0, fov_y_radians=FOV_60,
    )
    assert np.all(np.isfinite(rgb))
    # constant_color in YCoCg with Co=Cg=0 -> RGB should be (Y,Y,Y), gray
    assert np.allclose(rgb, rgb[0], atol=1e-3)


def test_accumulate_pixel_off_screen_reprojection_treated_as_new_sample():
    # Large motion vector pushes reprojected UV off-screen -> is_existing_sample
    # False -> should behave like a new sample (no crash, finite output),
    # even on frame_index > 0.
    h_lr, w_lr = 8, 8
    h_hr, w_hr = 16, 16
    prepared = np.random.default_rng(1).uniform(-0.3, 0.3, size=(h_lr, w_lr, 3))
    depth = np.full((h_lr, w_lr), 0.5)
    history_color_buf = np.random.default_rng(2).uniform(0, 1, size=(h_hr, w_hr, 4))
    lock_status_buf = np.zeros((h_hr, w_hr, 2))

    rgb, lock_status = accumulate_pixel(
        px_hr_pos=(8, 8), render_size=(w_lr, h_lr), display_size=(w_hr, h_hr),
        frame_index=5, motion_vector=(5.0, 5.0),  # way off-screen
        prepared_color=prepared, current_depth=depth, history_depth=depth,
        history_color_buffer=history_color_buf, lock_status_buffer=lock_status_buf,
        jitter=(0.0, 0.0), near=0.1, far=1000.0, fov_y_radians=FOV_60,
    )
    assert np.all(np.isfinite(rgb))


def test_accumulate_pixel_static_scene_stable_over_frames():
    # Simulate a few frames of a perfectly static scene (zero motion,
    # constant color/depth). Output color should converge toward the
    # constant input color and stay finite/stable, not drift or blow up.
    h_lr, w_lr = 8, 8
    h_hr, w_hr = 16, 16
    constant_color_ycocg = np.array([0.4, 0.0, 0.0])
    prepared = np.tile(constant_color_ycocg, (h_lr, w_lr, 1))
    depth = np.full((h_lr, w_lr), 0.5)

    history_color_buf = np.zeros((h_hr, w_hr, 4))
    lock_status_buf = np.zeros((h_hr, w_hr, 2))

    px = (8, 8)
    for frame_index in range(5):
        rgb, lock_status = accumulate_pixel(
            px_hr_pos=px, render_size=(w_lr, h_lr), display_size=(w_hr, h_hr),
            frame_index=frame_index, motion_vector=(0.0, 0.0),
            prepared_color=prepared, current_depth=depth, history_depth=depth,
            history_color_buffer=history_color_buf, lock_status_buffer=lock_status_buf,
            jitter=(0.0, 0.0), near=0.1, far=1000.0, fov_y_radians=FOV_60,
        )
        assert np.all(np.isfinite(rgb))
        # write this pixel's result back into the history buffer for next frame
        history_color_buf[px[1], px[0], :3] = rgb
        history_color_buf[px[1], px[0], 3] = 0.0  # reactive factor, neutral
        lock_status_buf[px[1], px[0], :] = lock_status

    # after several frames of static input, output should be close to
    # the constant gray value the input represents
    expected_gray = constant_color_ycocg[0]
    assert np.allclose(rgb, expected_gray, atol=0.05)
