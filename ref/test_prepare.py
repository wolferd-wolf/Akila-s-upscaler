import numpy as np
import pytest

from prepare import (
    pixel_uv_grid,
    reproject_uv,
    sample_bilinear,
    disocclusion_validity,
    prepare_pass,
)


def test_pixel_uv_grid_shape_and_range():
    grid = pixel_uv_grid(4, 8)
    assert grid.shape == (4, 8, 2)
    assert grid.min() > 0.0
    assert grid.max() < 1.0


def test_pixel_uv_grid_corners():
    h, w = 2, 2
    grid = pixel_uv_grid(h, w)
    # top-left pixel center
    assert np.allclose(grid[0, 0], [0.25, 0.25])
    # bottom-right pixel center
    assert np.allclose(grid[1, 1], [0.75, 0.75])


def test_reproject_uv_zero_motion_returns_current_uv():
    h, w = 8, 8
    mv = np.zeros((h, w, 2))
    prev_uv = reproject_uv(mv, h, w)
    expected = pixel_uv_grid(h, w)
    assert np.allclose(prev_uv, expected)


def test_reproject_uv_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        reproject_uv(np.zeros((4, 4, 2)), 8, 8)


def test_reproject_uv_known_shift():
    # current pixel moved +0.1 in u since previous frame means
    # mv = uv_current - uv_previous = 0.1, so prev_uv = current - 0.1
    h, w = 4, 4
    mv = np.zeros((h, w, 2))
    mv[..., 0] = 0.1
    prev_uv = reproject_uv(mv, h, w)
    current = pixel_uv_grid(h, w)
    assert np.allclose(prev_uv[..., 0], current[..., 0] - 0.1)
    assert np.allclose(prev_uv[..., 1], current[..., 1])


def test_sample_bilinear_constant_buffer_returns_constant():
    buf = np.full((8, 8), 3.5)
    uv = np.array([[0.5, 0.5], [0.1, 0.9], [0.99, 0.01]])
    result = sample_bilinear(buf, uv)
    assert np.allclose(result, 3.5)


def test_sample_bilinear_gradient_interpolates():
    # buffer where value == column index; sampling mid-pixel should
    # interpolate smoothly, not just nearest-neighbor snap.
    w = 10
    buf = np.tile(np.arange(w, dtype=np.float64), (10, 1))
    uv = np.array([[0.55, 0.5]])  # roughly column 5.5 before clamping logic
    result = sample_bilinear(buf, uv)
    assert 4.0 <= result[0] <= 6.0


def test_sample_bilinear_out_of_range_clamps_to_edge():
    buf = np.zeros((4, 4))
    buf[0, 0] = 9.0
    uv = np.array([[-0.5, -0.5]])  # far outside [0,1], should clamp to corner
    result = sample_bilinear(buf, uv)
    assert np.isclose(result[0], 9.0)


def test_disocclusion_validity_static_scene_all_valid():
    h, w = 8, 8
    depth = np.random.default_rng(0).uniform(0.1, 1.0, size=(h, w))
    mv = np.zeros((h, w, 2))
    prev_uv = reproject_uv(mv, h, w)
    validity = disocclusion_validity(depth, depth, prev_uv)
    assert validity.all()


def test_disocclusion_validity_rejects_off_screen_reprojection():
    h, w = 8, 8
    depth = np.full((h, w), 0.5)
    mv = np.zeros((h, w, 2))
    mv[..., 0] = -2.0  # pushes prev_uv way outside [0,1]
    prev_uv = reproject_uv(mv, h, w)
    validity = disocclusion_validity(depth, depth, prev_uv)
    assert not validity.any()


def test_disocclusion_validity_rejects_depth_mismatch():
    h, w = 8, 8
    current_depth = np.full((h, w), 0.5)
    history_depth = np.full((h, w), 0.9)  # large mismatch -> disoccluded
    mv = np.zeros((h, w, 2))
    prev_uv = reproject_uv(mv, h, w)
    validity = disocclusion_validity(current_depth, history_depth, prev_uv)
    assert not validity.any()


def test_disocclusion_validity_accepts_small_depth_noise():
    h, w = 8, 8
    current_depth = np.full((h, w), 0.5)
    history_depth = np.full((h, w), 0.505)  # 1% diff, within default 2% threshold
    mv = np.zeros((h, w, 2))
    prev_uv = reproject_uv(mv, h, w)
    validity = disocclusion_validity(current_depth, history_depth, prev_uv)
    assert validity.all()


def test_prepare_pass_end_to_end_static_scene():
    h, w = 16, 16
    rng = np.random.default_rng(1)
    depth = rng.uniform(0.1, 1.0, size=(h, w))
    mv = np.zeros((h, w, 2))
    prev_uv, validity = prepare_pass(depth, depth, mv)
    assert prev_uv.shape == (h, w, 2)
    assert validity.shape == (h, w)
    assert validity.all()


def test_prepare_pass_mixed_occlusion():
    h, w = 8, 8
    current_depth = np.full((h, w), 0.5)
    history_depth = np.full((h, w), 0.5)
    # simulate an object that appeared this frame: right half occluded
    history_depth[:, w // 2:] = 0.9
    mv = np.zeros((h, w, 2))
    prev_uv, validity = prepare_pass(current_depth, history_depth, mv)
    assert validity[:, : w // 2].all()
    assert not validity[:, w // 2:].any()
