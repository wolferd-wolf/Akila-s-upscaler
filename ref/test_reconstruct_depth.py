import numpy as np
import pytest

from reconstruct_depth import (
    rgb_to_luma,
    rgb_to_perceived_luma,
    compute_lock_input_luma,
    find_nearest_depth,
    bilinear_scatter_weights,
    reconstruct_prev_depth,
    reconstruct_and_dilate,
)


def test_rgb_to_luma_pure_green_weighted_highest():
    # Rec.709 luma weights green highest (0.7152) -- pure green should
    # register more luma than equal-magnitude pure red or blue.
    luma_r = rgb_to_luma(np.array([1.0, 0.0, 0.0]))
    luma_g = rgb_to_luma(np.array([0.0, 1.0, 0.0]))
    luma_b = rgb_to_luma(np.array([0.0, 0.0, 1.0]))
    assert luma_g > luma_r > luma_b


def test_rgb_to_perceived_luma_black_is_zero():
    assert np.isclose(rgb_to_perceived_luma(np.array([0.0, 0.0, 0.0])), 0.0)


def test_rgb_to_perceived_luma_increases_with_brightness():
    dim = rgb_to_perceived_luma(np.array([0.1, 0.1, 0.1]))
    bright = rgb_to_perceived_luma(np.array([0.8, 0.8, 0.8]))
    assert bright > dim


def test_compute_lock_input_luma_negative_rgb_clamped():
    # ffxMax(0,0,0) in source -- negative input shouldn't produce NaN
    # or negative luma.
    result = compute_lock_input_luma(
        np.array([-1.0, -1.0, -1.0]), exposure=1.0, pre_exposure=1.0
    )
    assert np.isfinite(result)
    assert result >= 0.0


def test_find_nearest_depth_center_is_nearest_returns_self():
    depth = np.full((5, 5), 0.5)
    depth[2, 2] = 0.1  # center pixel itself is nearest
    d, coord = find_nearest_depth(depth, (2, 2))
    assert np.isclose(d, 0.1)
    assert coord == (2, 2)


def test_find_nearest_depth_neighbor_closer_wins():
    depth = np.full((5, 5), 0.5)
    depth[2, 3] = 0.05  # neighbor to the right is much closer
    d, coord = find_nearest_depth(depth, (2, 2))
    assert np.isclose(d, 0.05)
    assert coord == (3, 2)


def test_find_nearest_depth_respects_screen_bounds():
    # Corner pixel -- some of the 3x3 neighborhood is off-screen and
    # must be skipped, not wrapped or crash.
    depth = np.full((3, 3), 0.5)
    depth[0, 0] = 0.2
    d, coord = find_nearest_depth(depth, (0, 0))
    assert np.isfinite(d)


def test_bilinear_scatter_weights_sum_to_one():
    (bx, by), taps = bilinear_scatter_weights((0.53, 0.47), (10, 10))
    total_weight = sum(w for _, w in taps)
    assert np.isclose(total_weight, 1.0)


def test_bilinear_scatter_weights_exact_texel_center_one_dominant_weight():
    # UV exactly at a texel center should give one tap ~1.0, others ~0.
    render_size = (10, 10)
    # texel (5,5) center in UV space:
    uv = ((5 + 0.5) / 10, (5 + 0.5) / 10)
    (bx, by), taps = bilinear_scatter_weights(uv, render_size)
    weights = sorted((w for _, w in taps), reverse=True)
    assert weights[0] > 0.99


def test_reconstruct_prev_depth_zero_motion_identity_scatter():
    # With zero motion everywhere, each pixel should scatter to (or very
    # near) its own position, reconstructing something close to the
    # original depth map.
    h, w = 6, 6
    depth = np.random.default_rng(0).uniform(0.1, 0.9, size=(h, w))
    mv = np.zeros((h, w, 2))
    result = reconstruct_prev_depth(depth, mv, (w, h), (w, h))
    finite_mask = np.isfinite(result)
    assert finite_mask.all()
    assert np.allclose(result[finite_mask], depth[finite_mask], atol=1e-6)


def test_reconstruct_prev_depth_keeps_nearest_on_collision():
    # Two source pixels scatter to overlapping destinations with
    # different depths -- the nearer (smaller) one should win, matching
    # the source's atomic-min semantics.
    h, w = 4, 4
    depth = np.full((h, w), 0.9)
    depth[1, 1] = 0.1  # one much-nearer pixel
    mv = np.zeros((h, w, 2))
    # push everything toward the same spot so depth[1,1]'s scatter
    # collides with neighbors' scatters
    mv[:, :, 0] = (1.5 - np.arange(w)) / w  # rough convergence toward x=1.5
    mv[:, :, 1] = (1.5 - np.arange(h))[:, None] / h
    result = reconstruct_prev_depth(depth, mv, (w, h), (w, h))
    # wherever a collision happened near the target, the min (0.1) should
    # appear somewhere in the output rather than being overwritten by 0.9
    assert np.any(np.isclose(result[np.isfinite(result)], 0.1, atol=1e-6))


def test_reconstruct_and_dilate_runs_end_to_end():
    h, w = 8, 8
    rng = np.random.default_rng(1)
    depth = rng.uniform(0.1, 0.9, size=(h, w))
    mv = np.zeros((h, w, 2))
    color = rng.uniform(0, 1, size=(h, w, 3))

    dilated_depth, dilated_mv, recon_prev_depth, lock_luma = reconstruct_and_dilate(
        current_depth=depth, input_motion_vectors=mv, current_color=color,
        render_size=(w, h), display_size=(w, h),
    )
    assert dilated_depth.shape == (h, w)
    assert dilated_mv.shape == (h, w, 2)
    assert recon_prev_depth.shape == (h, w)
    assert lock_luma.shape == (h, w)
    assert np.all(np.isfinite(lock_luma))
