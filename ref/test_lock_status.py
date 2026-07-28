import numpy as np
import pytest

from lock_status import (
    min_divided_by_max,
    kill_lock,
    initialize_new_lock_sample,
    update_lock_status,
    LOCK_LIFETIME_REMAINING,
    LOCK_TEMPORAL_LUMA,
)


def test_min_divided_by_max_basic():
    assert np.isclose(min_divided_by_max(2.0, 4.0), 0.5)
    assert np.isclose(min_divided_by_max(4.0, 2.0), 0.5)


def test_min_divided_by_max_both_zero_returns_zero():
    assert min_divided_by_max(0.0, 0.0) == 0.0


def test_min_divided_by_max_equal_values_returns_one():
    assert np.isclose(min_divided_by_max(3.0, 3.0), 1.0)


def test_kill_lock_zeros_lifetime_only():
    status = np.array([5.0, 0.8])  # [lifetime, temporal_luma]
    kill_lock(status)
    assert status[LOCK_LIFETIME_REMAINING] == 0.0
    assert status[LOCK_TEMPORAL_LUMA] == 0.8  # unaffected


def test_initialize_new_lock_sample_is_zeros():
    status = initialize_new_lock_sample()
    assert np.allclose(status, [0.0, 0.0])


def test_update_lock_status_first_sample_initializes_temporal_luma():
    status = initialize_new_lock_sample()
    _, new_status, _, _ = update_lock_status(
        shading_change_luma=0.7, reactive_factor=0.0, is_new_lock=False,
        lock_status=status, accumulation_mask=0.0, depth_clip_factor=0.0,
    )
    assert np.isclose(new_status[LOCK_TEMPORAL_LUMA], 0.7)


def test_update_lock_status_stable_luma_no_diff():
    # Same luma every frame -> luminance_diff should be ~0, no reactive
    # bump, lock should persist rather than being killed.
    status = np.array([2.0, 0.5])
    _, new_status, contribution, diff = update_lock_status(
        shading_change_luma=0.5, reactive_factor=0.0, is_new_lock=False,
        lock_status=status, accumulation_mask=0.0, depth_clip_factor=0.0,
    )
    assert np.isclose(diff, 0.0, atol=1e-6)


def test_update_lock_status_new_lock_resets_temporal_luma():
    # Use matching luma so luminance_diff stays ~0 and doesn't trigger
    # the reactive-factor decay path -- isolating just the lifetime-reset
    # arithmetic this test is actually about.
    status = np.array([0.0, 0.9])  # lifetime was 0, luma already matches
    _, new_status, _, _ = update_lock_status(
        shading_change_luma=0.9, reactive_factor=0.0, is_new_lock=True,
        lock_status=status, accumulation_mask=0.0, depth_clip_factor=0.0,
    )
    assert np.isclose(new_status[LOCK_TEMPORAL_LUMA], 0.9)
    assert np.isclose(new_status[LOCK_LIFETIME_REMAINING], 1.0)  # was 0 -> becomes 1


def test_update_lock_status_relock_when_already_had_lifetime():
    status = np.array([3.0, 0.9])  # nonzero lifetime, matching luma
    _, new_status, _, _ = update_lock_status(
        shading_change_luma=0.9, reactive_factor=0.0, is_new_lock=True,
        lock_status=status, accumulation_mask=0.0, depth_clip_factor=0.0,
    )
    assert np.isclose(new_status[LOCK_LIFETIME_REMAINING], 2.0)  # was nonzero -> becomes 2


def test_update_lock_status_large_luma_jump_on_new_lock_still_decays_lifetime():
    # A large luma jump raises reactive_factor toward 1.0, which then
    # decays lifetime *= (1 - reactive_factor) even on a fresh lock --
    # this coupling is intentional in the source, not a bug. Document it
    # as a real behavior, not paper over it with a convenient test.
    status = np.array([0.0, 0.1])
    _, new_status, _, diff = update_lock_status(
        shading_change_luma=0.9, reactive_factor=0.0, is_new_lock=True,
        lock_status=status, accumulation_mask=0.0, depth_clip_factor=0.0,
    )
    assert diff > 0.1
    assert new_status[LOCK_LIFETIME_REMAINING] < 1.0  # decayed despite new-lock branch


def test_update_lock_status_high_reactive_factor_decays_lifetime():
    status = np.array([2.0, 0.5])
    _, new_status, _, _ = update_lock_status(
        shading_change_luma=0.5, reactive_factor=1.0, is_new_lock=False,
        lock_status=status, accumulation_mask=0.0, depth_clip_factor=0.0,
    )
    # lifetime *= (1 - reactive_factor) = (1-1.0) = 0
    assert np.isclose(new_status[LOCK_LIFETIME_REMAINING], 0.0)


def test_update_lock_status_high_depth_clip_kills_lifetime():
    # depth_clip_factor >= 0.1 -> the (depth_clip_factor < 0.1) term is 0
    # -> lifetime forced to 0 regardless of everything else.
    status = np.array([5.0, 0.5])
    _, new_status, _, _ = update_lock_status(
        shading_change_luma=0.5, reactive_factor=0.0, is_new_lock=False,
        lock_status=status, accumulation_mask=0.0, depth_clip_factor=0.5,
    )
    assert np.isclose(new_status[LOCK_LIFETIME_REMAINING], 0.0)


def test_update_lock_status_large_luma_jump_raises_reactive_factor():
    status = np.array([5.0, 0.1])
    new_reactive, new_status, _, diff = update_lock_status(
        shading_change_luma=0.9, reactive_factor=0.0, is_new_lock=False,
        lock_status=status, accumulation_mask=0.0, depth_clip_factor=0.0,
    )
    assert diff > 0.1  # big luma jump
    assert new_reactive > 0.0  # should have been raised from 0


def test_update_lock_status_contribution_is_bounded():
    status = np.array([3.0, 0.5])
    _, _, contribution, _ = update_lock_status(
        shading_change_luma=0.5, reactive_factor=0.0, is_new_lock=False,
        lock_status=status, accumulation_mask=0.0, depth_clip_factor=0.0,
    )
    assert 0.0 <= contribution <= 1.0
