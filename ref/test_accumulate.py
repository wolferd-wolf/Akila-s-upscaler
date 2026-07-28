import numpy as np
import pytest

from accumulate import accumulate, rectify_history, FSR2_EPSILON


def test_accumulate_first_sample_takes_full_weight():
    history = np.array([0.0, 0.0, 0.0])
    upsampled = np.array([0.5, 0.2, -0.1])
    new_history, new_accum = accumulate(
        history_color=history, accumulation=0.0,
        upsampled_color=upsampled, upsampled_weight=1.0,
    )
    assert np.allclose(new_history, upsampled)
    assert np.isclose(new_accum, 1.0)


def test_accumulate_zero_new_weight_keeps_history():
    history = np.array([0.3, 0.1, 0.0])
    upsampled = np.array([0.9, 0.9, 0.9])
    new_history, new_accum = accumulate(
        history_color=history, accumulation=5.0,
        upsampled_color=upsampled, upsampled_weight=0.0,
    )
    assert np.allclose(new_history, history, atol=1e-5)


def test_accumulate_partial_weight_blends_between():
    history = np.array([0.0, 0.0, 0.0])
    upsampled = np.array([1.0, 1.0, 1.0])
    new_history, new_accum = accumulate(
        history_color=history, accumulation=1.0,
        upsampled_color=upsampled, upsampled_weight=1.0,
    )
    assert np.allclose(new_history, [0.5, 0.5, 0.5])


def test_accumulate_never_divides_by_true_zero():
    history = np.array([0.2, 0.2, 0.2])
    new_history, new_accum = accumulate(
        history_color=history, accumulation=0.0,
        upsampled_color=np.array([0.0, 0.0, 0.0]), upsampled_weight=0.0,
    )
    assert np.all(np.isfinite(new_history))
    assert new_accum >= FSR2_EPSILON


def test_rectify_history_no_change_when_inside_box():
    history = np.array([0.5, 0.5, 0.5])
    box_center = np.array([0.5, 0.5, 0.5])
    box_vec = np.array([0.1, 0.1, 0.1])
    aabb_min = np.array([0.0, 0.0, 0.0])
    aabb_max = np.array([1.0, 1.0, 1.0])
    accumulation = np.array([2.0, 2.0, 2.0])

    new_history, new_accum = rectify_history(
        history_color=history, accumulation=accumulation,
        box_center=box_center, box_vec=box_vec,
        aabb_min=aabb_min, aabb_max=aabb_max,
        downscale_factor=0.5, hr_velocity=0.0,
        depth_clip_factor=0.0, accumulation_mask=0.0,
        dilated_reactive_factor=0.0,
    )
    assert np.allclose(new_history, history)
    assert np.allclose(new_accum, accumulation)


def test_rectify_history_clamps_when_outside_box_and_no_lock():
    history = np.array([5.0, 5.0, 5.0])
    box_center = np.array([0.5, 0.5, 0.5])
    box_vec = np.array([0.05, 0.05, 0.05])
    aabb_min = np.array([0.0, 0.0, 0.0])
    aabb_max = np.array([1.0, 1.0, 1.0])
    accumulation = np.array([2.0, 2.0, 2.0])

    new_history, new_accum = rectify_history(
        history_color=history, accumulation=accumulation,
        box_center=box_center, box_vec=box_vec,
        aabb_min=aabb_min, aabb_max=aabb_max,
        downscale_factor=0.5, hr_velocity=0.0,
        depth_clip_factor=0.0, accumulation_mask=0.0,
        dilated_reactive_factor=0.0,
        lock_contribution_this_frame=0.0, luma_instability_factor=0.0,
    )
    assert np.all(new_history < 1.5)
    assert np.all(new_history >= aabb_min - 1e-6)


def test_rectify_history_full_lock_contribution_keeps_original():
    history = np.array([5.0, 5.0, 5.0])
    box_center = np.array([0.5, 0.5, 0.5])
    box_vec = np.array([0.05, 0.05, 0.05])
    aabb_min = np.array([0.0, 0.0, 0.0])
    aabb_max = np.array([1.0, 1.0, 1.0])
    accumulation = np.array([2.0, 2.0, 2.0])

    new_history, new_accum = rectify_history(
        history_color=history, accumulation=accumulation,
        box_center=box_center, box_vec=box_vec,
        aabb_min=aabb_min, aabb_max=aabb_max,
        downscale_factor=0.5, hr_velocity=0.0,
        depth_clip_factor=0.0, accumulation_mask=0.0,
        dilated_reactive_factor=0.0,
        lock_contribution_this_frame=1.0, luma_instability_factor=0.0,
    )
    assert np.allclose(new_history, history)


def test_rectify_history_runs_at_high_velocity_without_error():
    box_center = np.array([0.5, 0.5, 0.5])
    box_vec = np.array([0.05, 0.05, 0.05])
    aabb_min = np.array([0.0, 0.0, 0.0])
    aabb_max = np.array([1.0, 1.0, 1.0])
    history = np.array([0.65, 0.65, 0.65])
    accumulation = np.array([2.0, 2.0, 2.0])

    new_history, new_accum = rectify_history(
        history_color=history.copy(), accumulation=accumulation.copy(),
        box_center=box_center, box_vec=box_vec,
        aabb_min=aabb_min, aabb_max=aabb_max,
        downscale_factor=0.5, hr_velocity=100.0,
        depth_clip_factor=0.0, accumulation_mask=0.0,
        dilated_reactive_factor=0.0,
    )
    assert np.all(np.isfinite(new_history))
    assert np.all(np.isfinite(new_accum))
