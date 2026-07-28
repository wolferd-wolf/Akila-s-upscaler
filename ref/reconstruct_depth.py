"""
Reconstruct dilated velocity and previous depth -- reference implementation.

Ported from AMD FSR2's ffx_fsr2_reconstruct_dilated_velocity_and_previous_depth.h.
This is the pass that RUNS BEFORE Prepare in the real pipeline and produces
the previous-frame depth buffer that every prior module in this package
(prepare.py's depth_clip_factor, pipeline.py's history_depth parameter)
has taken as a caller-supplied fake up to this point. Porting this closes
that gap directly.

Three pieces:
- rgb_to_perceived_luma: perceptual luma (feeds ComputeLockInputLuma)
- find_nearest_depth: 3x3 neighborhood dilation -- picks the nearest
  (smallest, assuming non-inverted depth) depth among a pixel and its
  8 neighbors, so thin/fast-moving objects don't get lost between
  low-res samples
- reconstruct_prev_depth: scatters current depth into a previous-depth
  buffer at the motion-vector-reprojected location. Multiple current
  pixels can scatter to the same previous location; the source uses an
  atomic MIN on the GPU (closest/nearest wins). This CPU reference
  mirrors that with an explicit min-reduction over all scatter writes.

SCOPE NOTE: this operates on a full low-res image at once (unlike most
of this package's single-pixel functions) because scatter operations
are awkward to express per-pixel without either a shared output buffer
threaded through every call or accepting the vectorized/whole-image
form here. The shader itself is also inherently a scatter (each
invocation writes to a *computed*, not fixed, output location), so
some deviation from strict per-invocation mirroring is unavoidable --
noted explicitly rather than silently changing structure without saying so.
"""

import numpy as np

FFX_FSR2_OPTION_INVERTED_DEPTH = False  # matches this project's convention so far
RECONSTRUCTED_DEPTH_BILINEAR_WEIGHT_THRESHOLD = 0.01  # ported constant


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb, np.array([0.2126, 0.7152, 0.0722]))


def rgb_to_perceived_luma(rgb: np.ndarray) -> np.ndarray:
    luminance = rgb_to_luma(rgb)
    low = luminance * (24389.0 / 27.0)
    high = np.power(np.maximum(luminance, 0.0), 1.0 / 3.0) * 116.0 - 16.0
    perceived = np.where(luminance <= 216.0 / 24389.0, low, high)
    return perceived * 0.01


def compute_lock_input_luma(rgb: np.ndarray, exposure: float, pre_exposure: float) -> np.ndarray:
    """rgb: (..., 3) linear RGB, single low-res pixel or batch."""
    rgb = np.maximum(rgb, 0.0)
    rgb = (rgb / pre_exposure) * exposure
    return np.power(rgb_to_perceived_luma(rgb), 1.0 / 6.0)


def find_nearest_depth(depth: np.ndarray, px_pos):
    """
    depth: (H, W) normalized depth, 0=near (non-inverted convention)
    px_pos: (x, y) integer position

    Returns: (nearest_depth, nearest_coord) -- nearest_coord is (x, y)
    """
    h, w = depth.shape
    x, y = px_pos
    offsets = [
        (0, 0), (1, 0), (0, 1), (0, -1),
        (-1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1),
    ]
    nearest_depth = depth[y, x]
    nearest_coord = (x, y)
    for dx, dy in offsets[1:]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            d = depth[ny, nx]
            if not FFX_FSR2_OPTION_INVERTED_DEPTH:
                if d < nearest_depth:
                    nearest_depth = d
                    nearest_coord = (nx, ny)
            else:
                if d > nearest_depth:
                    nearest_depth = d
                    nearest_coord = (nx, ny)
    return nearest_depth, nearest_coord


def bilinear_scatter_weights(reprojected_uv, render_size):
    """
    Matches AMD's GetBilinearSamplingData -- the 4 integer texel offsets
    and their bilinear weights for a given continuous UV coordinate.

    Returns: base_pos (x0, y0), list of ((ox, oy), weight) for the 4 taps.
    """
    w, h = render_size
    x = reprojected_uv[0] * w - 0.5
    y = reprojected_uv[1] * h - 0.5
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - x0, y - y0

    taps = [
        ((0, 0), (1 - fx) * (1 - fy)),
        ((1, 0), fx * (1 - fy)),
        ((0, 1), (1 - fx) * fy),
        ((1, 1), fx * fy),
    ]
    return (x0, y0), taps


def reconstruct_prev_depth(
    current_depth: np.ndarray,  # (H, W), already dilated (nearest-depth) values
    dilated_motion_vectors: np.ndarray,  # (H, W, 2), UV-space
    render_size,
    display_size,
) -> np.ndarray:
    """
    Scatters current_depth into a previous-frame depth buffer using
    reprojected (UV + motion_vector) positions with bilinear splat
    weights, keeping the MINIMUM (nearest) depth wherever multiple
    scatters land on the same output texel -- mirrors the source's
    atomic-min semantics.

    Returns: (H, W) reconstructed previous depth buffer. Texels no
    current pixel ever scattered into are left at +inf -- caller should
    treat +inf as "no valid history depth here" (equivalent to the
    source's implicit clear-to-far-plane behavior).
    """
    h, w = current_depth.shape
    output = np.full((h, w), np.inf)

    for y in range(h):
        for x in range(w):
            mv = dilated_motion_vectors[y, x].copy()
            hr_velocity_like = np.hypot(mv[0] * display_size[0], mv[1] * display_size[1])
            if hr_velocity_like <= 0.1:
                mv[:] = 0.0

            uv = ((x + 0.5) / w, (y + 0.5) / h)
            reprojected_uv = (uv[0] + mv[0], uv[1] + mv[1])

            (bx, by), taps = bilinear_scatter_weights(reprojected_uv, render_size)
            depth_val = current_depth[y, x]

            for (ox, oy), weight in taps:
                if weight <= RECONSTRUCTED_DEPTH_BILINEAR_WEIGHT_THRESHOLD:
                    continue
                sx, sy = bx + ox, by + oy
                if 0 <= sx < w and 0 <= sy < h:
                    if depth_val < output[sy, sx]:
                        output[sy, sx] = depth_val

    return output


def reconstruct_and_dilate(
    current_depth: np.ndarray,  # (H, W) raw (undilated) current depth
    input_motion_vectors: np.ndarray,  # (H, W, 2) raw (undilated) motion vectors
    current_color: np.ndarray,  # (H, W, 3) linear RGB, for lock input luma
    render_size,
    display_size,
    exposure: float = 1.0,
    pre_exposure: float = 1.0,
):
    """
    Full per-frame reconstruction: dilates depth+motion vectors via
    nearest-depth search, then scatters dilated depth into a
    reconstructed previous-depth buffer, and computes lock input luma.

    Returns:
        dilated_depth: (H, W)
        dilated_motion_vectors: (H, W, 2)
        reconstructed_prev_depth: (H, W) (+inf where nothing scattered)
        lock_input_luma: (H, W)
    """
    h, w = current_depth.shape
    dilated_depth = np.zeros((h, w))
    dilated_motion_vectors = np.zeros((h, w, 2))

    for y in range(h):
        for x in range(w):
            nearest_depth, (nx, ny) = find_nearest_depth(current_depth, (x, y))
            dilated_depth[y, x] = nearest_depth
            dilated_motion_vectors[y, x] = input_motion_vectors[ny, nx]

    reconstructed_prev_depth = reconstruct_prev_depth(
        dilated_depth, dilated_motion_vectors, render_size, display_size
    )

    lock_input_luma = compute_lock_input_luma(current_color, exposure, pre_exposure)

    return dilated_depth, dilated_motion_vectors, reconstructed_prev_depth, lock_input_luma
