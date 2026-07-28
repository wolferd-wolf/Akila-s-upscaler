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
    current_depth: np.ndarray,
    history_depth: np.ndarray,
    prev_uv: np.ndarray,
    near: float,
    far: float,
    fov_y_radians: float,
) -> np.ndarray:
    """
    Ported from AMD FSR2's ComputeDepthClip (ffx_fsr2_depth_clip.h).

    IMPORTANT -- semantics match AMD's naming exactly, which is
    counter-intuitive on first read: this returns a REJECT signal, not a
    trust/confidence signal. 1.0 = disoccluded, reject history. 0.0 = no
    depth discontinuity detected, history is fine. Downstream (accumulate
    pass) uses it as `accumulation *= (1.0 - depth_clip_factor)`.

    Deliberately named and structured to match the source 1:1 (per-tap term,
    then combined) rather than re-derived, because re-deriving this from
    "what should high/low mean" produced an inverted-sign bug during initial
    implementation. Match the source's data flow, not your own intuition
    about what the output "should" mean.

    current_depth, history_depth: (H, W) normalized depth [0,1], 0=near
    prev_uv: (H, W, 2) from reproject_uv()
    near, far: camera near/far planes (same units)
    fov_y_radians: vertical field of view

    returns: (H, W) float array in [0, 1]. 1.0 = reject, 0.0 = trust.
    """
    in_bounds = (
        (prev_uv[..., 0] >= 0.0)
        & (prev_uv[..., 0] <= 1.0)
        & (prev_uv[..., 1] >= 0.0)
        & (prev_uv[..., 1] <= 1.0)
    )

    current_view_depth = normalized_to_view_space_depth(current_depth, near, far)
    sampled_history_normalized = sample_bilinear(history_depth, prev_uv)
    history_view_depth = normalized_to_view_space_depth(
        sampled_history_normalized, near, far
    )

    # depth_diff > 0: current surface is FARTHER than history at this
    # location -- a nearer occluder from last frame is gone/moved, and a
    # farther surface (which history never saw here) is now visible.
    # That's the case this term detects and penalizes.
    depth_diff = current_view_depth - history_view_depth

    half_fov = fov_y_radians / 2.0
    kfov = np.sqrt(1.0 + np.tan(half_fov) ** 2)

    h, w = current_depth.shape
    half_viewport_width = np.hypot(w, h)

    depth_threshold_view = np.maximum(current_view_depth, history_view_depth)
    required_separation = (
        FSR2_KSEP * kfov * half_viewport_width * depth_threshold_view
    )

    resolution_factor = np.clip(np.hypot(w, h) / np.hypot(1920.0, 1080.0), 0.0, 1.0)
    power = 1.0 + resolution_factor * 2.0  # lerp(1.0, 3.0, resolution_factor)

    # Per-tap term from AMD source: pow(saturate(required/actual), power).
    # actual_gap >> required  -> term -> 0
    # actual_gap <= required  -> term -> 1
    term = np.where(
        depth_diff > 0.0,
        np.power(
            np.clip(required_separation / np.maximum(depth_diff, 1e-8), 0.0, 1.0),
            power,
        ),
        1.0,  # depth_diff <= 0: AMD skips this tap (weightSum stays 0 -> D=0
              # for single-tap case). Encode as term=1.0 here so that the
              # D = 1 - term step below correctly yields D=0.
    )

    # single-sample equivalent of AMD's `saturate(1 - fDepth/weightSum)`
    reject_signal = np.clip(1.0 - term, 0.0, 1.0)

    # off-screen reprojection: no history sample exists at all -> full reject
    return np.where(in_bounds, reject_signal, 1.0)


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

    Returns:
        prev_uv: (H, W, 2) reprojected UVs into previous frame
        trust: (H, W) float array in [0,1]. 1.0 = fully trusted history,
               0.0 = fully rejected/disoccluded. (Inverted from the raw
               depth_clip_factor -- this outer function hands callers the
               intuitive "how much do I trust this" value; depth_clip_factor
               itself keeps AMD's original reject-signal convention so it's
               directly comparable against the source during the shader port.)
    """
    h, w = current_depth.shape
    prev_uv = reproject_uv(motion_vectors, h, w)
    reject = depth_clip_factor(
        current_depth, history_depth, prev_uv, near, far, fov_y_radians
    )
    trust = 1.0 - reject
    return prev_uv, trust
