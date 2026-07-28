import numpy as np
import pytest

from upsample import (
    lanczos2_approx_sq,
    get_upsample_lanczos_weight,
    compute_max_kernel_weight,
    RectificationBox,
    compute_upsampled_color_and_weight,
    FSR2_UPSAMPLE_LANCZOS_WEIGHT_SCALE,
)


def test_lanczos2_approx_sq_at_zero_is_peak():
    # At distance 0, Lanczos should give its maximum weight (~1.0, the
    # kernel's central value before any falloff).
    val = lanczos2_approx_sq(0.0)
    assert val > 0.9


def test_lanczos2_approx_sq_clamped_beyond_support():
    # x2 > 4 (distance > 2, outside Lanczos-2's support) should clamp
    # to the same value as x2 == 4, not blow up or go negative unexpectedly.
    at_edge = lanczos2_approx_sq(4.0)
    beyond = lanczos2_approx_sq(100.0)
    assert np.isclose(at_edge, beyond)


def test_lanczos2_approx_sq_decreases_with_distance():
    near = lanczos2_approx_sq(0.1)
    far = lanczos2_approx_sq(2.0)
    assert near > far


def test_upsample_weight_scale_constant():
    # Regression guard -- don't silently drop this again.
    assert np.isclose(FSR2_UPSAMPLE_LANCZOS_WEIGHT_SCALE, 1.0 / 12.0)


def test_get_upsample_lanczos_weight_zero_offset():
    weight = get_upsample_lanczos_weight((0.0, 0.0), kernel_weight=1.0)
    assert weight > 0.9


def test_compute_max_kernel_weight_no_scaling_at_1to1():
    # downscale_factor == 1.0 (no actual upscale) -> kernel weight should
    # reduce to the base case (no extra bias needed).
    w = compute_max_kernel_weight(1.0)
    assert np.isclose(w, 1.0)


def test_compute_max_kernel_weight_capped():
    # Very small downscale_factor (aggressive upscale) should be capped,
    # not blow up unboundedly.
    w = compute_max_kernel_weight(0.1)
    assert w <= 1.99


def test_rectification_box_single_sample_zero_variance():
    box = RectificationBox()
    color = np.array([0.5, 0.1, -0.1])
    box.add_sample(True, color, 1.0)
    box.compute_variance_box_data()
    assert np.allclose(box.box_center, color)
    assert np.allclose(box.box_vec, 0.0)  # single sample -> zero stddev
    assert np.allclose(box.aabb_min, color)
    assert np.allclose(box.aabb_max, color)


def test_rectification_box_uniform_samples_zero_variance():
    box = RectificationBox()
    color = np.array([0.3, 0.0, 0.0])
    for i in range(5):
        box.add_sample(i == 0, color, 1.0)
    box.compute_variance_box_data()
    assert np.allclose(box.box_vec, 0.0, atol=1e-8)


def test_rectification_box_min_max_tracks_extremes():
    box = RectificationBox()
    samples = [
        np.array([0.1, 0.0, 0.0]),
        np.array([0.9, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
    ]
    for i, s in enumerate(samples):
        box.add_sample(i == 0, s, 1.0)
    assert np.isclose(box.aabb_min[0], 0.1)
    assert np.isclose(box.aabb_max[0], 0.9)


def test_rectification_box_nonzero_variance_for_varied_samples():
    box = RectificationBox()
    samples = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]
    for i, s in enumerate(samples):
        box.add_sample(i == 0, s, 1.0)
    box.compute_variance_box_data()
    assert box.box_vec[0] > 0.0


def test_compute_upsampled_color_and_weight_constant_field_returns_constant():
    # A perfectly uniform low-res color field should upsample to that same
    # constant color everywhere, regardless of jitter/position.
    h_lr, w_lr = 8, 8
    constant_color = np.array([0.4, 0.1, -0.05])
    prepared = np.tile(constant_color, (h_lr, w_lr, 1))

    color, weight, box = compute_upsampled_color_and_weight(
        prepared_color=prepared,
        px_hr_pos=(10, 10),
        render_size=(w_lr, h_lr),
        display_size=(w_lr * 2, h_lr * 2),
        jitter=(0.0, 0.0),
        reactive_factor=0.0,
        depth_clip_factor=0.0,
        is_new_sample=False,
        hr_velocity=0.0,
    )
    assert weight > 0.0
    assert np.allclose(color, constant_color, atol=1e-4)
    assert np.allclose(box.aabb_min, constant_color, atol=1e-6)
    assert np.allclose(box.aabb_max, constant_color, atol=1e-6)


def test_compute_upsampled_color_and_weight_new_sample_forces_full_reactive():
    # is_new_sample should behave like reactive_factor=1.0 for kernel bias
    # purposes (matches kernel_reactive_factor = max(reactive, is_new_sample))
    h_lr, w_lr = 8, 8
    prepared = np.random.default_rng(2).uniform(-0.5, 0.5, size=(h_lr, w_lr, 3))

    _, weight_new, _ = compute_upsampled_color_and_weight(
        prepared_color=prepared, px_hr_pos=(8, 8),
        render_size=(w_lr, h_lr), display_size=(w_lr * 2, h_lr * 2),
        jitter=(0.0, 0.0), reactive_factor=1.0, depth_clip_factor=0.0,
        is_new_sample=False, hr_velocity=0.0,
    )
    _, weight_forced, _ = compute_upsampled_color_and_weight(
        prepared_color=prepared, px_hr_pos=(8, 8),
        render_size=(w_lr, h_lr), display_size=(w_lr * 2, h_lr * 2),
        jitter=(0.0, 0.0), reactive_factor=0.0, depth_clip_factor=0.0,
        is_new_sample=True, hr_velocity=0.0,
    )
    assert np.isclose(weight_new, weight_forced, rtol=1e-6)


def test_compute_upsampled_color_and_weight_off_screen_samples_excluded():
    # Near the edge of the render target, some of the 3x3 taps fall
    # off-screen and should be excluded (on_screen factor = 0), not wrap
    # or crash.
    h_lr, w_lr = 8, 8
    prepared = np.random.default_rng(3).uniform(-0.5, 0.5, size=(h_lr, w_lr, 3))
    color, weight, box = compute_upsampled_color_and_weight(
        prepared_color=prepared, px_hr_pos=(0, 0),
        render_size=(w_lr, h_lr), display_size=(w_lr * 2, h_lr * 2),
        jitter=(0.0, 0.0), reactive_factor=0.0, depth_clip_factor=0.0,
        is_new_sample=False, hr_velocity=0.0,
    )
    assert weight > 0.0
    assert np.all(np.isfinite(color))
