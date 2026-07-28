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


def disocclusion_validity(
    current_depth: np.ndarray,
    history_depth: np.ndarray,
    prev_uv: np.ndarray,
    depth_threshold: float = 0.02,
) -> np.ndarray:
    """
    Determine which reprojected history samples are trustworthy.

    A sample is INVALID (disoccluded) if:
      - prev_uv falls outside [0, 1] (reprojects off-screen), OR
      - the depth at the reprojected location in the previous frame
        differs from the current depth by more than depth_threshold
        (relative difference) -- this indicates the surface visible
        now was occluded/absent in the previous frame.

    current_depth: (H, W) array, normalized depth [0, 1] (0 = near)
    history_depth: (H, W) array, previous frame's depth buffer
    prev_uv: (H, W, 2) array from reproject_uv()
    returns: (H, W) boolean array, True = valid (usable) history
    """
    in_bounds = (
        (prev_uv[..., 0] >= 0.0)
        & (prev_uv[..., 0] <= 1.0)
        & (prev_uv[..., 1] >= 0.0)
        & (prev_uv[..., 1] <= 1.0)
    )

    sampled_history_depth = sample_bilinear(history_depth, prev_uv)

    # Relative depth difference test. Avoid div-by-zero on near-zero depth.
    denom = np.maximum(current_depth, 1e-5)
    relative_diff = np.abs(current_depth - sampled_history_depth) / denom

    depth_ok = relative_diff <= depth_threshold

    return in_bounds & depth_ok


def prepare_pass(
    current_depth: np.ndarray,
    history_depth: np.ndarray,
    motion_vectors: np.ndarray,
    depth_threshold: float = 0.02,
):
    """
    Full prepare pass.

    Returns:
        prev_uv: (H, W, 2) reprojected UVs into previous frame
        validity: (H, W) bool mask, True where history can be trusted
    """
    h, w = current_depth.shape
    prev_uv = reproject_uv(motion_vectors, h, w)
    validity = disocclusion_validity(
        current_depth, history_depth, prev_uv, depth_threshold
    )
    return prev_uv, validity
