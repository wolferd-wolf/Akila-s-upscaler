import numpy as np
import pytest

from prepare import (
    pixel_uv_grid,
    reproject_uv,
    sample_bilinear,
    normalized_to_view_space_depth,
    depth_clip_factor,
    prepare_pass,
    FSR2_KSEP,
)


FOV_60 = np.radians(60.0)


def test_pixel_uv_grid_shape_and_range():
    grid = pixel_uv_grid(4, 8)
    assert grid.shape == (4, 8, 2)
    assert grid.min() > 0.0
    assert grid.max() < 1.0


def test_pixel_uv_grid_corners():
    h, w = 2, 2
    grid = pixel_uv_grid(h, w)
    assert np.allclose(grid[0, 0], [0.25, 0.25])
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
    w = 10
    buf = np.tile(np.arange(w, dtype=np.float64), (10, 1))
    uv = np.array([[0.55, 0.5]])
    result = sample_bilinear(buf, uv)
    assert 4.0 <= result[0] <= 6.0


def test_sample_bilinear_out_of_range_clamps_to_edge():
    buf = np.zeros((4, 4))
    buf[0, 0] = 9.0
    uv = np.array([[-0.5, -0.5]])
    result = sample_bilinear(buf, uv)
    assert np.isclose(result[0], 9.0)


def test_view_space_depth_near_and_far_bounds():
    near, far = 0.1, 1000.0
    z_near = normalized_to_view_space_depth(np.array([0.0]), near, far)
    z_far = normalized_to_view_space_depth(np.array([1.0]), near, far)
    assert np.isclose(z_near[0], near)
    assert np.isclose(z_far[0], far)


def test_view_space_depth_monotonic():
    near, far = 0.1, 1000.0
    d = np.linspace(0.0, 1.0, 20)
    z = normalized_to_view_space_depth(d, near, far)
    assert np.all(np.diff(z) > 0)


def test_depth_clip_factor_static_scene_not_rejected():
    # depth_clip_factor: 1.0 = reject, 0.0 = trust (matches AMD naming).
    h, w = 8, 8
    depth = np.full((h, w), 0.5)
    mv = np.zeros((h, w, 2))
    prev_uv = reproject_uv(mv, h, w)
    reject = depth_clip_factor(
        depth, depth, prev_uv, near=0.1, far=1000.0, fov_y_radians=FOV_60
    )
    assert np.all(reject < 0.01)


def test_depth_clip_factor_rejects_off_screen_reprojection():
    h, w = 8, 8
    depth = np.full((h, w), 0.5)
    mv = np.zeros((h, w, 2))
    mv[..., 0] = -2.0
    prev_uv = reproject_uv(mv, h, w)
    reject = depth_clip_factor(
        depth, depth, prev_uv, near=0.1, far=1000.0, fov_y_radians=FOV_60
    )
    assert np.all(reject == 1.0)


def test_depth_clip_factor_large_depth_jump_rejected():
    # Disocclusion = currently-visible surface is FARTHER than what
    # history held here (a nearer occluder from last frame is gone,
    # revealing background history never saw at this pixel).
    # depth_clip_factor should be HIGH (near 1) -- reject this history.
    h, w = 8, 8
    current_depth = np.full((h, w), 0.95)  # far now
    history_depth = np.full((h, w), 0.05)  # was near
    mv = np.zeros((h, w, 2))
    prev_uv = reproject_uv(mv, h, w)
    reject = depth_clip_factor(
        current_depth, history_depth, prev_uv,
        near=0.1, far=1000.0, fov_y_radians=FOV_60,
    )
    assert np.all(reject > 0.5)


def test_depth_clip_factor_tiny_depth_noise_not_rejected():
    h, w = 8, 8
    current_depth = np.full((h, w), 0.5)
    history_depth = current_depth + 1e-6
    mv = np.zeros((h, w, 2))
    prev_uv = reproject_uv(mv, h, w)
    reject = depth_clip_factor(
        current_depth, history_depth, prev_uv,
        near=0.1, far=1000.0, fov_y_radians=FOV_60,
    )
    assert np.all(reject < 0.1)


def test_depth_clip_factor_new_closer_geometry_not_penalized_by_this_term():
    # Current surface CLOSER than history (something moved toward camera,
    # or new geometry appeared in front) -- depth_diff <= 0, not penalized
    # by this depth-clip term specifically. Other terms (motion divergence)
    # are responsible for catching this case; not this function's job.
    # depth_clip_factor should stay LOW (trust) here.
    h, w = 8, 8
    current_depth = np.full((h, w), 0.1)  # near now
    history_depth = np.full((h, w), 0.9)  # was far
    mv = np.zeros((h, w, 2))
    prev_uv = reproject_uv(mv, h, w)
    reject = depth_clip_factor(
        current_depth, history_depth, prev_uv,
        near=0.1, far=1000.0, fov_y_radians=FOV_60,
    )
    assert np.all(reject < 0.01)


def test_ksep_constant_matches_fsr2_source():
    assert FSR2_KSEP == 1.37e-05


def test_prepare_pass_end_to_end_static_scene():
    h, w = 16, 16
    rng = np.random.default_rng(1)
    depth = rng.uniform(0.1, 1.0, size=(h, w))
    mv = np.zeros((h, w, 2))
    prev_uv, confidence = prepare_pass(depth, depth, mv)
    assert prev_uv.shape == (h, w, 2)
    assert confidence.shape == (h, w)
    assert np.all(confidence > 0.9)


def test_prepare_pass_mixed_occlusion():
    h, w = 8, 8
    current_depth = np.full((h, w), 0.05)
    history_depth = np.full((h, w), 0.05)
    # right half: something that was close (history) is gone, revealing
    # a much farther surface now -- history there is stale, should reject.
    current_depth[:, w // 2:] = 0.9
    history_depth[:, w // 2:] = 0.02
    mv = np.zeros((h, w, 2))
    prev_uv, confidence = prepare_pass(current_depth, history_depth, mv)
    assert np.all(confidence[:, : w // 2] > 0.9)
    assert np.all(confidence[:, w // 2:] < 0.5)
