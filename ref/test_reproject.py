import numpy as np
import pytest

from reproject import (
    rgb_to_ycocg,
    ycocg_to_rgb,
    prepare_rgb,
    is_uv_inside,
    compute_reprojected_uvs,
    reproject_history_color,
    reproject_history_lock_status,
    FSR2_FP16_MAX,
)


def test_ycocg_roundtrip():
    rgb = np.array([0.8, 0.3, 0.1])
    ycocg = rgb_to_ycocg(rgb)
    back = ycocg_to_rgb(ycocg)
    assert np.allclose(back, rgb, atol=1e-6)


def test_ycocg_roundtrip_batch():
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 1, size=(5, 5, 3))
    ycocg = rgb_to_ycocg(rgb)
    back = ycocg_to_rgb(ycocg)
    assert np.allclose(back, rgb, atol=1e-6)


def test_rgb_to_ycocg_gray_has_zero_chroma():
    # Pure gray (r=g=b) should have Co=Cg=0 -- no color information.
    gray = np.array([0.5, 0.5, 0.5])
    ycocg = rgb_to_ycocg(gray)
    assert np.isclose(ycocg[0], 0.5)
    assert np.isclose(ycocg[1], 0.0, atol=1e-8)
    assert np.isclose(ycocg[2], 0.0, atol=1e-8)


def test_prepare_rgb_same_exposure_is_identity():
    rgb = np.array([0.5, 0.3, 0.1])
    result = prepare_rgb(rgb, exposure=1.0, pre_exposure=1.0)
    assert np.allclose(result, rgb)


def test_prepare_rgb_clamps_to_fp16_max():
    rgb = np.array([1e10, 1e10, 1e10])
    result = prepare_rgb(rgb, exposure=1.0, pre_exposure=1.0)
    assert np.all(result <= FSR2_FP16_MAX)


def test_prepare_rgb_clamps_negative_to_zero():
    rgb = np.array([-5.0, 0.5, 0.5])
    result = prepare_rgb(rgb, exposure=1.0, pre_exposure=1.0)
    assert result[0] == 0.0


def test_is_uv_inside_center():
    assert is_uv_inside((0.5, 0.5))


def test_is_uv_inside_boundary_included():
    assert is_uv_inside((0.0, 0.0))
    assert is_uv_inside((1.0, 1.0))


def test_is_uv_inside_outside():
    assert not is_uv_inside((-0.1, 0.5))
    assert not is_uv_inside((0.5, 1.1))


def test_compute_reprojected_uvs_zero_motion():
    uv, is_existing = compute_reprojected_uvs((0.5, 0.5), (0.0, 0.0))
    assert np.allclose(uv, (0.5, 0.5))
    assert is_existing


def test_compute_reprojected_uvs_off_screen():
    uv, is_existing = compute_reprojected_uvs((0.9, 0.5), (0.5, 0.0))
    assert not is_existing


def test_reproject_history_color_constant_buffer():
    h, w = 8, 8
    color = np.array([0.4, 0.2, 0.1])
    alpha = 0.6
    history = np.zeros((h, w, 4))
    history[..., :3] = color
    history[..., 3] = alpha

    history_color, reactive, in_motion = reproject_history_color(
        history_buffer=history, reprojected_hr_uv=(0.5, 0.5),
        exposure=1.0, previous_frame_pre_exposure=1.0,
    )
    expected_ycocg = rgb_to_ycocg(color)
    assert np.allclose(history_color, expected_ycocg, atol=1e-4)
    assert np.isclose(reactive, 0.6, atol=1e-4)
    assert not in_motion  # alpha positive


def test_reproject_history_color_negative_alpha_means_in_motion():
    h, w = 8, 8
    history = np.zeros((h, w, 4))
    history[..., 3] = -0.5
    _, reactive, in_motion = reproject_history_color(
        history_buffer=history, reprojected_hr_uv=(0.5, 0.5),
        exposure=1.0, previous_frame_pre_exposure=1.0,
    )
    assert in_motion
    assert np.isclose(reactive, 0.5)  # abs() applied


def test_reproject_history_lock_status_below_threshold_not_new_lock():
    h, w = 8, 8
    lock_buf = np.zeros((h, w, 2))
    is_new, was_locked, status = reproject_history_lock_status(
        lock_status_buffer=lock_buf, reprojected_hr_uv=(0.5, 0.5),
        new_lock_intensity=0.3,  # below 127/255 ~ 0.498
    )
    assert not is_new
    assert not was_locked


def test_reproject_history_lock_status_above_threshold_is_new_lock():
    h, w = 8, 8
    lock_buf = np.zeros((h, w, 2))
    is_new, _, _ = reproject_history_lock_status(
        lock_status_buffer=lock_buf, reprojected_hr_uv=(0.5, 0.5),
        new_lock_intensity=0.9,
    )
    assert is_new


def test_reproject_history_lock_status_nonzero_lifetime_was_locked():
    h, w = 8, 8
    lock_buf = np.zeros((h, w, 2))
    lock_buf[..., 0] = 2.0  # LOCK_LIFETIME_REMAINING channel
    _, was_locked, status = reproject_history_lock_status(
        lock_status_buffer=lock_buf, reprojected_hr_uv=(0.5, 0.5),
        new_lock_intensity=0.0,
    )
    assert was_locked
    assert np.isclose(status[0], 2.0)
