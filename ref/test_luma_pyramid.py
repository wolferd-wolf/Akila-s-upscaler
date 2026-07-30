import numpy as np
import pytest

from luma_pyramid import (
    rgb_to_luma,
    build_log_luma_mip_pyramid,
    sample_mip_luma,
    get_shading_change_luma,
    FSR2_EPSILON,
)


def test_rgb_to_luma_matches_rec709_weights():
    result = rgb_to_luma(np.array([1.0, 0.0, 0.0]))
    assert np.isclose(result, 0.2126)


def test_build_pyramid_mip0_is_log_luma():
    h, w = 8, 8
    rgb = np.full((h, w, 3), 0.5)
    pyramid = build_log_luma_mip_pyramid(rgb, num_levels=4)
    expected = np.log(rgb_to_luma(np.array([0.5, 0.5, 0.5])))
    assert np.allclose(pyramid[0], expected, atol=1e-6)


def test_build_pyramid_black_pixels_clamped_by_epsilon():
    h, w = 8, 8
    rgb = np.zeros((h, w, 3))
    pyramid = build_log_luma_mip_pyramid(rgb, num_levels=2)
    assert np.all(np.isfinite(pyramid[0]))
    assert np.allclose(pyramid[0], np.log(FSR2_EPSILON))


def test_build_pyramid_each_level_halves_dimensions():
    h, w = 16, 16
    rgb = np.random.default_rng(0).uniform(0, 1, size=(h, w, 3))
    pyramid = build_log_luma_mip_pyramid(rgb, num_levels=5)
    assert pyramid[0].shape == (16, 16)
    assert pyramid[1].shape == (8, 8)
    assert pyramid[2].shape == (4, 4)
    assert pyramid[3].shape == (2, 2)
    assert pyramid[4].shape == (1, 1)


def test_build_pyramid_uniform_image_stays_uniform_at_every_level():
    h, w = 16, 16
    rgb = np.full((h, w, 3), 0.3)
    pyramid = build_log_luma_mip_pyramid(rgb, num_levels=4)
    for mip in pyramid:
        assert np.allclose(mip, mip.flat[0], atol=1e-6)


def test_build_pyramid_coarser_levels_average_out_local_variation():
    h, w = 16, 16
    rgb = np.zeros((h, w, 3))
    rgb[:, :, :] = 0.5
    rgb[::2, ::2, :] = 0.9
    pyramid = build_log_luma_mip_pyramid(rgb, num_levels=4)
    variances = [np.var(mip) for mip in pyramid]
    assert variances[0] >= variances[-1]


def test_sample_mip_luma_constant_level_returns_constant():
    h, w = 8, 8
    rgb = np.full((h, w, 3), 0.4)
    pyramid = build_log_luma_mip_pyramid(rgb, num_levels=3)
    val = sample_mip_luma(pyramid, level=0, uv=(0.5, 0.5))
    expected = np.log(rgb_to_luma(np.array([0.4, 0.4, 0.4])))
    assert np.isclose(val, expected, atol=1e-6)


def test_sample_mip_luma_clamps_level_to_available_range():
    h, w = 8, 8
    rgb = np.full((h, w, 3), 0.4)
    pyramid = build_log_luma_mip_pyramid(rgb, num_levels=3)
    val = sample_mip_luma(pyramid, level=99, uv=(0.5, 0.5))
    assert np.isfinite(val)


def test_get_shading_change_luma_roundtrips_through_log_exp():
    h, w = 8, 8
    rgb = np.full((h, w, 3), 0.5)
    pyramid = build_log_luma_mip_pyramid(rgb, num_levels=2)
    luma = rgb_to_luma(np.array([0.5, 0.5, 0.5]))
    result = get_shading_change_luma(pyramid, mip_level=0, hr_uv=(0.5, 0.5), exposure=1.0)
    expected = np.power(luma, 1.0 / 6.0)
    assert np.isclose(result, expected, atol=1e-4)


def test_get_shading_change_luma_scales_with_exposure():
    h, w = 8, 8
    rgb = np.full((h, w, 3), 0.5)
    pyramid = build_log_luma_mip_pyramid(rgb, num_levels=2)
    low_exposure = get_shading_change_luma(pyramid, 0, (0.5, 0.5), exposure=0.5)
    high_exposure = get_shading_change_luma(pyramid, 0, (0.5, 0.5), exposure=2.0)
    assert high_exposure > low_exposure
