"""
Per-segment feature extraction, used by the learned line scorer (`suggest.py`).

Why bother?
-----------
The geometry stage can only ever ask "does this segment point at a vanishing
point?". That question has a blind spot: a branch, a shadow edge, or a
reflection can point at a vanishing point purely by chance, and with enough of
them a spurious VP wins the RANSAC vote outright. Humans do not make that
mistake, because we also look at what the edge *looks like*: masonry edges are
high-contrast, locally isolated, and have a consistent gradient direction along
their length, whereas foliage edges sit in a dense thicket of other edges, are
low in contrast, and drift in colour.

So each segment gets a small appearance-and-geometry descriptor. `suggest.py`
learns, from the geometry stage's own verdicts plus the user's clicks, how to
map that descriptor to "is this a structural edge?".

All features are cheap (a few dozen samples per segment) and unitless, so the
descriptor transfers across image sizes.
"""

from __future__ import annotations

import cv2
import numpy as np

from .line_detection import LineSet

FEATURE_NAMES = [
    "length",            # normalised by image diagonal
    "orientation_cos",   # cos(2*theta): 180-degree-periodic orientation
    "orientation_sin",   # sin(2*theta)
    "pos_y",             # vertical position of the midpoint (sky vs ground)
    "grad_magnitude",    # mean edge strength along the segment
    "grad_consistency",  # how uniformly the gradient points across the edge
    "cross_contrast",    # brightness step from one side to the other
    "contrast_stability",# how constant that step is along the segment
    "greenness",         # vegetation cue
    "saturation",        # painted/natural surfaces vs grey masonry
    "edge_density",      # local edge clutter (high for foliage)
    "texture_std",       # local intensity variation
]

NUM_SAMPLES = 20   # sample points along each segment
NORMAL_OFFSET = 3  # pixels to each side when measuring the cross-edge step


def _sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sampling with clamping at the image border."""
    height, width = image.shape[:2]
    xi = np.clip(np.round(xs).astype(int), 0, width - 1)
    yi = np.clip(np.round(ys).astype(int), 0, height - 1)
    return image[yi, xi]


def compute_features(image_bgr: np.ndarray, lines: LineSet) -> np.ndarray:
    """Return an (N, len(FEATURE_NAMES)) float array, one row per segment."""
    height, width = image_bgr.shape[:2]
    diagonal = float(np.hypot(width, height))
    n = len(lines)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    blue, green, red = (image_bgr[:, :, i].astype(np.float32) for i in range(3))
    greenness = (green - 0.5 * (red + blue)) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_magnitude = np.hypot(gx, gy)
    grad_scale = float(np.percentile(grad_magnitude, 99)) + 1e-6

    # A blurred Canny map is a cheap proxy for "how cluttered is this
    # neighbourhood with other edges" - the single most useful foliage cue.
    edges = cv2.Canny(cv2.GaussianBlur(gray.astype(np.uint8), (3, 3), 0), 60, 160)
    edge_density_map = cv2.blur((edges > 0).astype(np.float32), (15, 15))

    mean_local = cv2.blur(gray, (15, 15))
    mean_local_sq = cv2.blur(gray * gray, (15, 15))
    texture_map = np.sqrt(np.maximum(mean_local_sq - mean_local**2, 0.0)) / 255.0

    t = np.linspace(0.15, 0.85, NUM_SAMPLES)  # skip the endpoints, they are noisy
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)

    for k in range(n):
        x1, y1, x2, y2 = lines.endpoints[k]
        dx, dy = lines.directions[k]
        nx, ny = -dy, dx  # unit normal

        xs = x1 + (x2 - x1) * t
        ys = y1 + (y2 - y1) * t

        gxs = _sample(gx, xs, ys)
        gys = _sample(gy, xs, ys)
        mags = np.hypot(gxs, gys)

        # Gradient direction consistency. Gradients on opposite sides of a
        # dark-to-light vs light-to-dark edge point in opposite directions, so
        # we compare *doubled* angles, which makes the measure sign-agnostic.
        angles = 2.0 * np.arctan2(gys, gxs)
        weights = mags / (mags.sum() + 1e-6)
        consistency = float(
            np.abs((weights * np.exp(1j * angles)).sum())
        )

        side_a = _sample(gray, xs + nx * NORMAL_OFFSET, ys + ny * NORMAL_OFFSET)
        side_b = _sample(gray, xs - nx * NORMAL_OFFSET, ys - ny * NORMAL_OFFSET)
        step = (side_a - side_b) / 255.0

        band_x = np.concatenate([xs, xs + nx * 6, xs - nx * 6])
        band_y = np.concatenate([ys, ys + ny * 6, ys - ny * 6])

        theta = np.arctan2(dy, dx)
        out[k] = [
            lines.lengths[k] / diagonal,
            np.cos(2 * theta),
            np.sin(2 * theta),
            lines.midpoints[k, 1] / height,
            float(mags.mean()) / grad_scale,
            consistency,
            float(np.abs(step).mean()),
            float(np.std(step)),
            float(_sample(greenness, band_x, band_y).mean()),
            float(_sample(saturation, band_x, band_y).mean()),
            float(_sample(edge_density_map, band_x, band_y).mean()),
            float(_sample(texture_map, band_x, band_y).mean()),
        ]

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
