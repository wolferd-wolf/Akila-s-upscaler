"""
Reactive mask preprocessing -- reference implementation.

Ported from AMD FSR2's ffx_fsr2_depth_clip.h: ComputeMotionDivergence,
ComputeDepthDivergence, ComputeTemporalMotionDivergence, and
PreProcessReactiveMasks. This is the pass that finally produces real
values for dilated_reactive_factor and accumulation_mask -- both of
which pipeline.py has taken as caller-supplied 0.0 defaults up to now.

Games typically supply their own reactive mask + transparency/composition
mask textures (marking particles, decals, etc. that don't reproject
cleanly). A game with no such content simply passes all-zero masks --
that's a legitimate, common case for testing, not a workaround.
"""

import numpy as np


def compute_motion_divergence(
    px_pos, input_motion_vectors: np.ndarray, render_size
) -> float:
    """
    High divergence = neighboring motion vectors point in very different
    directions from this pixel's own motion -- a sign of a moving edge
    silhouette, where reprojection is least trustworthy.
    """
    w, h = render_size
    x, y = px_pos
    mv_nucleus = input_motion_vectors[y, x]
    nucleus_velocity_lr = np.hypot(mv_nucleus[0] * w, mv_nucleus[1] * h)
    max_velocity_uv = np.hypot(mv_nucleus[0], mv_nucleus[1])

    motion_vector_velocity_epsilon = 1e-2
    if nucleus_velocity_lr <= motion_vector_velocity_epsilon:
        return 0.0

    min_convergence = 1.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sx = int(np.clip(x + dx, 0, w - 1))
            sy = int(np.clip(y + dy, 0, h - 1))
            mv = input_motion_vectors[sy, sx]
            velocity_uv = np.hypot(mv[0], mv[1])
            max_velocity_uv = max(velocity_uv, max_velocity_uv)
            velocity_uv = max(velocity_uv, max_velocity_uv)
            if velocity_uv > 0:
                cos_angle = (
                    mv[0] * mv_nucleus[0] + mv[1] * mv_nucleus[1]
                ) / (velocity_uv * velocity_uv)
                min_convergence = min(min_convergence, cos_angle)

    return float(
        np.clip(1.0 - min_convergence, 0.0, 1.0)
        * np.clip(max_velocity_uv / 0.01, 0.0, 1.0)
    )


def compute_depth_divergence(px_pos, dilated_depth: np.ndarray, far: float) -> float:
    """
    High divergence = a strong min/max depth spread in the 3x3
    neighborhood -- a depth edge/silhouette boundary.
    """
    w, h = dilated_depth.shape[1], dilated_depth.shape[0]
    x, y = px_pos
    depth_max = 0.0
    depth_min = far
    max_dist_found = False

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sx, sy = x + dx, y + dy
            on_screen = 0 <= sx < w and 0 <= sy < h
            depth = dilated_depth[sy, sx] * far if on_screen else 0.0
            if far == depth:
                max_dist_found = True
            depth_min = min(depth_min, depth)
            depth_max = max(depth_max, depth)

    if depth_max <= 0.0:
        return 0.0
    result = 1.0 - depth_min / depth_max
    return 0.0 if max_dist_found else result


def compute_temporal_motion_divergence(
    px_pos,
    dilated_motion_vectors: np.ndarray,
    previous_dilated_motion_vectors: np.ndarray,
    render_size,
    display_size,
) -> float:
    """
    High divergence = this pixel's motion vector has changed a lot since
    last frame -- suggests unstable/unreliable motion data here.
    """
    w, h = render_size
    x, y = px_pos
    uv = ((x + 0.5) / w, (y + 0.5) / h)
    mv = dilated_motion_vectors[y, x]
    reprojected_uv = (np.clip(uv[0] + mv[0], 0, 1), np.clip(uv[1] + mv[1], 0, 1))

    # nearest-sample previous dilated MV at reprojected location (bilinear
    # would be more faithful; nearest is a documented simplification)
    px = int(np.clip(reprojected_uv[0] * w - 0.5, 0, w - 1))
    py = int(np.clip(reprojected_uv[1] * h - 0.5, 0, h - 1))
    prev_mv = previous_dilated_motion_vectors[py, px]

    px_distance = np.hypot(mv[0] * display_size[0], mv[1] * display_size[1])
    if px_distance <= 1.0:
        return 0.0

    mv_len = np.hypot(mv[0], mv[1])
    prev_mv_len = np.hypot(prev_mv[0], prev_mv[1])
    ratio = np.clip(prev_mv_len / mv_len, 0.0, 1.0) if mv_len > 0 else 0.0
    t = np.clip(np.power(px_distance / 20.0, 3.0), 0.0, 1.0)
    return float((1.0 - ratio) * t)  # lerp(0, 1-ratio, t) == (1-ratio)*t


def preprocess_reactive_masks(
    px_lr_pos,
    input_color: np.ndarray,  # (H, W, 3) linear RGB, current frame
    reactive_mask: np.ndarray,  # (H, W) game-supplied, 0 if unused
    transparency_composition_mask: np.ndarray,  # (H, W) game-supplied, 0 if unused
    motion_divergence: float,
    render_size,
):
    """
    Returns (dilated_reactive_factor, accumulation_mask) for this pixel --
    both start with a motion_divergence-derived floor, then get boosted
    by a 3x3 neighborhood scan of the reactive/composition masks, biased
    toward whichever samples are LEAST similar in color to the center
    pixel (a mask value on a very differently-colored neighbor is
    considered more reliable evidence than one on a near-identical
    neighbor, which could just be dithering/noise).
    """
    w, h = render_size
    x, y = px_lr_pos
    reference_color = input_color[y, x].astype(np.float64)

    reactive_factor = np.array([0.0, motion_divergence])

    samples = []
    masks_sum = 0.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sx = int(np.clip(x + dx, 0, w - 1))
            sy = int(np.clip(y + dy, 0, h - 1))
            color_sample = input_color[sy, sx].astype(np.float64)
            reactive_sample = float(reactive_mask[sy, sx])
            transparency_sample = float(transparency_composition_mask[sy, sx])
            samples.append((color_sample, reactive_sample, transparency_sample))
            masks_sum += reactive_sample + transparency_sample

    if masks_sum > 0.0:
        for color_sample, reactive_sample, transparency_sample in samples:
            max_len_sq = max(
                np.dot(reference_color, reference_color),
                np.dot(color_sample, color_sample),
            )
            similarity = (
                np.dot(reference_color, color_sample) / max_len_sq
                if max_len_sq > 0
                else 1.0
            )
            power_bias_max = 6.0
            similarity_power = 1.0 + (power_bias_max - similarity * power_bias_max)
            weighted_reactive = np.power(reactive_sample, similarity_power)
            weighted_transparency = np.power(transparency_sample, similarity_power)
            reactive_factor = np.maximum(
                reactive_factor, [weighted_reactive, weighted_transparency]
            )

    return float(reactive_factor[0]), float(reactive_factor[1])
