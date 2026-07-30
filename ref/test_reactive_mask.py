import numpy as np
import pytest

from reactive_mask import (
    compute_motion_divergence,
    compute_depth_divergence,
    compute_temporal_motion_divergence,
    preprocess_reactive_masks,
)


def test_motion_divergence_zero_when_static():
    h, w = 8, 8
    mv = np.zeros((h, w, 2))
    result = compute_motion_divergence((4, 4), mv, (w, h))
    assert result == 0.0


def test_motion_divergence_zero_when_all_neighbors_move_same_direction():
    h, w = 8, 8
    mv = np.zeros((h, w, 2))
    mv[..., 0] = 0.05
    result = compute_motion_divergence((4, 4), mv, (w, h))
    assert result < 0.05


def test_motion_divergence_high_when_neighbors_move_opposite_directions():
    h, w = 8, 8
    mv = np.zeros((h, w, 2))
    mv[..., 0] = 0.05
    mv[3, 4, 0] = -0.05
    result = compute_motion_divergence((4, 4), mv, (w, h))
    assert result > 0.5


def test_depth_divergence_zero_for_flat_region():
    h, w = 8, 8
    depth = np.full((h, w), 0.5)
    result = compute_depth_divergence((4, 4), depth, far=1000.0)
    assert np.isclose(result, 0.0, atol=1e-6)


def test_depth_divergence_high_at_depth_edge():
    h, w = 8, 8
    depth = np.full((h, w), 0.1)
    depth[3:6, 5:8] = 0.9
    result = compute_depth_divergence((4, 4), depth, far=1000.0)
    assert result > 0.5


def test_depth_divergence_zero_when_max_distance_present():
    h, w = 8, 8
    depth = np.full((h, w), 0.5)
    depth[3, 5] = 1.0
    result = compute_depth_divergence((4, 4), depth, far=1000.0)
    assert result == 0.0


def test_temporal_motion_divergence_zero_below_pixel_threshold():
    h, w = 8, 8
    mv = np.zeros((h, w, 2))
    mv[4, 4] = [0.0001, 0.0]
    prev_mv = np.zeros((h, w, 2))
    result = compute_temporal_motion_divergence(
        (4, 4), mv, prev_mv, (w, h), (64, 64)
    )
    assert result == 0.0


def test_temporal_motion_divergence_zero_when_motion_matches_previous():
    h, w = 8, 8
    mv = np.zeros((h, w, 2))
    mv[4, 4] = [0.1, 0.0]
    prev_mv = np.zeros((h, w, 2))
    prev_mv[:, :] = [0.1, 0.0]
    result = compute_temporal_motion_divergence(
        (4, 4), mv, prev_mv, (w, h), (640, 640)
    )
    assert result < 0.1


def test_temporal_motion_divergence_high_when_motion_just_appeared():
    h, w = 8, 8
    mv = np.zeros((h, w, 2))
    mv[4, 4] = [0.1, 0.0]
    prev_mv = np.zeros((h, w, 2))
    result = compute_temporal_motion_divergence(
        (4, 4), mv, prev_mv, (w, h), (640, 640)
    )
    assert result > 0.5


def test_preprocess_reactive_masks_all_zero_masks_gives_motion_divergence_only():
    h, w = 8, 8
    color = np.random.default_rng(0).uniform(0, 1, size=(h, w, 3))
    reactive = np.zeros((h, w))
    transparency = np.zeros((h, w))
    reactive_factor, accumulation_mask = preprocess_reactive_masks(
        (4, 4), color, reactive, transparency,
        motion_divergence=0.3, render_size=(w, h),
    )
    assert reactive_factor == 0.0
    assert accumulation_mask == 0.3


def test_preprocess_reactive_masks_nonzero_mask_boosts_reactive_factor():
    h, w = 8, 8
    color = np.full((h, w, 3), 0.5)
    reactive = np.zeros((h, w))
    reactive[4, 4] = 0.8
    transparency = np.zeros((h, w))
    reactive_factor, accumulation_mask = preprocess_reactive_masks(
        (4, 4), color, reactive, transparency,
        motion_divergence=0.0, render_size=(w, h),
    )
    assert reactive_factor > 0.0


def test_preprocess_reactive_masks_similar_color_sample_weighted_higher():
    # 'Increase power for non-similar samples' means dissimilar-colored
    # neighbors get their reactive value suppressed MORE (pow with a
    # higher exponent shrinks a base < 1 further toward 0), not boosted.
    # A reactive hit on a color-matching neighbor is trusted more.
    h, w = 8, 8
    color_similar = np.full((h, w, 3), 0.5)
    reactive_similar = np.zeros((h, w))
    reactive_similar[3, 4] = 0.5
    transparency = np.zeros((h, w))

    color_dissimilar = np.full((h, w, 3), 0.5)
    color_dissimilar[3, 4] = [0.0, 0.0, 0.0]
    reactive_dissimilar = np.zeros((h, w))
    reactive_dissimilar[3, 4] = 0.5

    factor_similar, _ = preprocess_reactive_masks(
        (4, 4), color_similar, reactive_similar, transparency,
        motion_divergence=0.0, render_size=(w, h),
    )
    factor_dissimilar, _ = preprocess_reactive_masks(
        (4, 4), color_dissimilar, reactive_dissimilar, transparency,
        motion_divergence=0.0, render_size=(w, h),
    )
    assert factor_similar >= factor_dissimilar
