"""
Prepare pass — reference implementation.

Given the current frame's depth buffer and per-pixel motion vectors,
compute where each current pixel came from in the previous frame
(reprojection), and whether that history sample is trustworthy
(disocclusion / rejection test).

This is ground truth. Every GLSL port must match this within
floating point tolerance before being trusted on-device.

Coordinate convention:
    - All buffers are (H, W) numpy arrays, row-major, origin top-left.
    - UV coordinates are normalized [0, 1], u = x/W, v = y/H.
    - Motion vectors are in UV space: mv = uv_current - uv_previous.
      (matches FSR2 convention: mv points from current pixel back
      to where it was in the previous frame)
"""

import numpy as np


def pixel_uv_grid(height: int, width: int) -> np.ndarray:
    """Return (H, W, 2) grid of UV coordinates at pixel centers."""
    ys, xs = np.meshgrid(
        (np.arange(height) + 0.5) / height,
        (np.arange(width) + 0.5) / width,
        indexing="ij",
    )
    return np.stack([xs, ys], axis=-1)  # (H, W, 2) -> [u, v]


def reproject_uv(motion_vectors: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Compute previous-frame UV for each current pixel.

    motion_vectors: (H, W, 2) array, mv = uv_current - uv_previous
    returns: (H, W, 2) array of previous-frame UVs (NOT clamped to [0,1])
    """
    assert motion_vectors.shape == (height, width, 2), (
        f"expected motion_vectors shape {(height, width, 2)}, "
        f"got {motion_vectors.shape}"
    )
    current_uv = pixel_uv_grid(height, width)
    prev_uv = current_uv - motion_vectors
    return prev_uv


def sample_bilinear(buffer: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """
    Bilinear sample a (H, W) or (H, W, C) buffer at normalized UV coords.
    uv: (..., 2) array of [u, v] in [0, 1] (out-of-range clamped to edge).
    Returns array of shape uv.shape[:-1] (+ (C,) if buffer has channels).
    """
    h, w = buffer.shape[0], buffer.shape[1]

    x = np.clip(uv[..., 0] * w - 0.5, 0, w - 1)
    y = np.clip(uv[..., 1] * h - 0.5, 0, h - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    fx = (x - x0)[..., None] if buffer.ndim == 3 else (x - x0)
    fy = (y - y0)[..., None] if buffer.ndim == 3 else (y - y0)

    top = buffer[y0, x0] * (1 - fx) + buffer[y0, x1] * fx
    bottom = buffer[y1, x0] * (1 - fx) + buffer[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


FSR2_KSEP = 1.37e-05  # empirically tuned constant, ported from AMD's source


def normalized_to_view_space_depth(normalized_depth, near, far):
    """Reconstruct linear view-space depth from normalized [0,1] depth."""
    return (near * far) / (far - normalized_depth * (far - near))


def depth_clip_factor(
    px_lr_pos,  # (x, y) integer low-res pixel position
    current_depth: np.ndarray,  # (H, W) normalized depth [0,1], 0=near
    history_depth: np.ndarray,  # (H, W) reconstructed previous depth
    reprojected_uv,  # (u, v) this pixel's reprojected UV into history_depth
    render_size,  # (w, h)
    near: float,
    far: float,
    fov_y_radians: float,
) -> float:
    """
    Ported from AMD FSR2's ComputeDepthClip (ffx_fsr2_depth_clip.h).

    CORRECTED from an earlier version of this function (found during a
    later session, while reading a DIFFERENT function in the same source
    file for an unrelated reason): the real algorithm evaluates the
    depth-diff/required-separation TERM SEPARATELY AT EACH OF 4 BILINEAR
    TAPS, then averages those terms with their bilinear weights. The
    previous version of this function instead bilinearly interpolated
    the history depth value FIRST, then computed one term from that
    single interpolated depth. Those two orders are not equivalent --
    interpolating depth before testing it smooths across exactly the
    kind of hard depth edge this disocclusion test exists to catch,
    which is the worst place for this function to be wrong. Averaging
    the terms afterward (as the source does) lets a genuine edge inside
    one 2x2 texel block still register as a strong reject signal even
    though the interpolated depth alone would look "fine".

    IMPORTANT -- semantics match AMD's naming, which is counter-intuitive
    on first read: this returns a REJECT signal, not a trust/confidence
    signal. 1.0 = disoccluded, reject history. 0.0 = no depth
    discontinuity detected, history is fine. Downstream (accumulate
    pass) uses it as `accumulation *= (1.0 - depth_clip_factor)`.

    returns: float in [0, 1]. 1.0 = reject, 0.0 = trust.
    """
    from reconstruct_depth import (
        bilinear_scatter_weights,
        RECONSTRUCTED_DEPTH_BILINEAR_WEIGHT_THRESHOLD,
    )

    w, h = render_size
    current_depth_sample = current_depth[px_lr_pos[1], px_lr_pos[0]]
    current_view_depth = normalized_to_view_space_depth(current_depth_sample, near, far)

    (base_x, base_y), taps = bilinear_scatter_weights(reprojected_uv, render_size)

    half_fov = fov_y_radians / 2.0
    kfov = np.sqrt(1.0 + np.tan(half_fov) ** 2)
    half_viewport_width = np.hypot(w, h)
    resolution_factor = np.clip(np.hypot(w, h) / np.hypot(1920.0, 1080.0), 0.0, 1.0)
    power = 1.0 + resolution_factor * 2.0  # lerp(1.0, 3.0, resolution_factor)

    accumulated_term = 0.0
    weight_sum = 0.0

    for (ox, oy), weight in taps:
        sx, sy = base_x + ox, base_y + oy
        if not (0 <= sx < w and 0 <= sy < h):
            continue
        if weight <= RECONSTRUCTED_DEPTH_BILINEAR_WEIGHT_THRESHOLD:
            continue

        prev_depth_sample = history_depth[sy, sx]
        if not np.isfinite(prev_depth_sample):
            continue  # texel never received a scattered depth -- no data
        prev_view_depth = normalized_to_view_space_depth(prev_depth_sample, near, far)

        depth_diff = current_view_depth - prev_view_depth
        if depth_diff <= 0.0:
            continue  # AMD: tap contributes nothing (not even weight) here

        plane_depth = max(prev_depth_sample, current_depth_sample)
        depth_threshold_view = max(current_view_depth, prev_view_depth)
        required_separation = FSR2_KSEP * kfov * half_viewport_width * depth_threshold_view

        term = np.power(
            np.clip(required_separation / max(depth_diff, 1e-8), 0.0, 1.0), power
        )
        accumulated_term += term * weight
        weight_sum += weight

    if weight_sum > 0.0:
        return float(np.clip(1.0 - accumulated_term / weight_sum, 0.0, 1.0))
    return 0.0


def prepare_pass(
    current_depth: np.ndarray,
    history_depth: np.ndarray,
    motion_vectors: np.ndarray,
    near: float = 0.1,
    far: float = 1000.0,
    fov_y_radians: float = np.radians(60.0),
):
    """
    Full prepare pass.

    near/far/fov_y_radians default to reasonable game camera values but
    should be passed explicitly from the actual camera in real use --
    the disocclusion threshold is directly sensitive to these.

    NOTE: EvaluateSurface (a small silhouette-edge heuristic AMD
    multiplies into the final depth-clip value) is not yet ported --
    stated gap, checked and confirmed unrelated to off-screen handling
    (see depth_clip_factor's docstring) while investigating a different
    question.

    Returns:
        prev_uv: (H, W, 2) reprojected UVs into previous frame (still
                 useful for callers that want it, e.g. visualization)
        trust: (H, W) float array in [0,1]. 1.0 = fully trusted history,
               0.0 = fully rejected/disoccluded.
    """
    h, w = current_depth.shape
    prev_uv = reproject_uv(motion_vectors, h, w)
    trust = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            reject = depth_clip_factor(
                px_lr_pos=(x, y),
                current_depth=current_depth,
                history_depth=history_depth,
                reprojected_uv=tuple(prev_uv[y, x]),
                render_size=(w, h),
                near=near, far=far, fov_y_radians=fov_y_radians,
            )
            trust[y, x] = 1.0 - reject
    return prev_uv, trust
