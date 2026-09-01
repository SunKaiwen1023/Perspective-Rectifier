"""
Cropping the black wedges left behind by a perspective warp.

The problem
-----------
Rectifying a photograph is a projective map, so a rectangular input becomes a
quadrilateral output. Fitting a rectangular canvas around that quadrilateral
necessarily leaves triangular gaps at the corners, and the stronger the
correction the larger they get. Geometrically they are honest - they are the
parts of the frame the camera never saw from the corrected viewpoint - but they
make the result look broken rather than fixed.

The fix is the same one every perspective tool applies: find the largest
rectangle that fits entirely inside the region containing real pixels, and crop
to it. Content is lost. That is unavoidable and expected.

Finding that rectangle
----------------------
"Largest axis-aligned rectangle inside an arbitrary binary mask" has an exact
O(W·H) solution via the largest-rectangle-in-a-histogram algorithm, but it is a
per-pixel stack loop, which in Python costs a few hundred milliseconds - too
slow for something that re-runs on every click. Two things make it fast here:

1. **Search on a downsampled mask.** The answer is smooth in the mask, so a
   300-pixel-wide copy locates the rectangle to within a pixel or two of the
   full-resolution answer. The result is mapped back up and then verified
   against the full-resolution mask, shrinking slightly if the boundary was
   missed.

2. **Fix the aspect ratio, then binary search.** With the ratio fixed, the only
   unknown is scale, and "does a w x h rectangle fit anywhere?" is one
   vectorised sliding-window sum over an integral image - no Python loop at
   all. Binary searching the scale needs about eight of those.

The free-ratio mode then just sweeps a range of aspect ratios through the same
routine and keeps whichever gives the largest area. That is an approximation to
the exact algorithm, but it is a close one and it is fast.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Crop modes, in the order they appear in the interface.
CROP_ORIGINAL = "Keep original aspect ratio"
CROP_LARGEST = "Largest area (any ratio)"
CROP_OFF = "No crop (show the full warp)"
CROP_MODES = [CROP_ORIGINAL, CROP_LARGEST, CROP_OFF]

#: Longest side of the mask used for searching. Bigger is marginally more
#: accurate and linearly slower; 300 keeps a full recompute imperceptible.
SEARCH_SIDE = 300

#: Aspect ratios tried in free mode: 0.3 to 3.3, log-spaced so the sweep is
#: evenly spread in perceptual terms rather than clustered at wide ratios.
FREE_ASPECTS = np.exp(np.linspace(np.log(0.3), np.log(3.3), 23))


def valid_mask(image_shape, H: np.ndarray, source_shape) -> np.ndarray:
    """Boolean mask of output pixels that received real source pixels.

    Built by warping a solid white image with the same homography rather than
    by testing the warped photo for black, which would also flag genuinely
    dark pixels in the photograph itself. The result is eroded slightly to
    discard the antialiased sliver along the boundary, where interpolation has
    mixed real pixels with the background colour.
    """
    height, width = image_shape[:2]
    solid = np.full(source_shape[:2], 255, dtype=np.uint8)
    warped = cv2.warpPerspective(
        solid, H, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
    )
    mask = warped > 127
    return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0


def _fits(integral: np.ndarray, height: int, width: int):
    """Positions where a `height` x `width` rectangle contains no invalid pixel.

    `integral` is the summed-area table of the *invalid* mask, so a rectangle is
    entirely valid exactly when its sum is zero. Returns the (rows, cols) arrays
    of every top-left corner that works, or None if none does.
    """
    rows, cols = integral.shape[0] - 1, integral.shape[1] - 1
    if height < 1 or width < 1 or height > rows or width > cols:
        return None
    sums = (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )
    positions = np.where(sums == 0)
    return positions if len(positions[0]) else None


def _largest_at_aspect(integral, aspect: float, centre) -> tuple | None:
    """Largest rectangle of a given width/height ratio, by binary search.

    `centre` is the preferred centre (row, col); among equally large rectangles
    we take the one closest to it, which keeps the crop over the middle of the
    content instead of jammed into a corner.
    """
    rows, cols = integral.shape[0] - 1, integral.shape[1] - 1
    lo, hi = 2, int(min(rows, cols / aspect))
    best = None

    while lo <= hi:
        mid = (lo + hi) // 2
        width = max(2, int(round(mid * aspect)))
        found = _fits(integral, mid, width)
        if found is None:
            hi = mid - 1
            continue
        ys, xs = found
        distances = (ys + mid / 2 - centre[0]) ** 2 + (xs + width / 2 - centre[1]) ** 2
        pick = int(np.argmin(distances))
        best = (int(ys[pick]), int(xs[pick]), mid, width)
        lo = mid + 1

    return best


def largest_inscribed_rect(mask: np.ndarray, aspect: float | None = None):
    """Largest all-valid rectangle in `mask`, as `(y, x, height, width)`.

    `aspect` is width/height; pass None to sweep for the largest area at any
    ratio. Returns None when the mask has no usable region at all.
    """
    if mask is None or mask.size == 0 or not mask.any():
        return None

    full_h, full_w = mask.shape[:2]
    scale = min(1.0, SEARCH_SIDE / max(full_h, full_w))
    small = (
        mask
        if scale >= 1.0
        else cv2.resize(
            mask.astype(np.uint8),
            (max(8, int(full_w * scale)), max(8, int(full_h * scale))),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    )

    invalid = (~small).astype(np.int64)
    integral = np.pad(invalid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    # Prefer a crop centred on the centre of mass of the valid region.
    ys, xs = np.nonzero(small)
    centre = (float(ys.mean()), float(xs.mean()))

    candidates = [aspect] if aspect is not None else list(FREE_ASPECTS)
    best, best_area = None, 0
    for ratio in candidates:
        found = _largest_at_aspect(integral, float(ratio), centre)
        if found is not None and found[2] * found[3] > best_area:
            best, best_area = found, found[2] * found[3]
    if best is None:
        return None

    # Map back to full resolution, then pull in until it is genuinely valid:
    # the downsampled search can miss the boundary by a pixel or two.
    factor_y = full_h / small.shape[0]
    factor_x = full_w / small.shape[1]
    y, x, h, w = best
    y, h = int(round(y * factor_y)), int(round(h * factor_y))
    x, w = int(round(x * factor_x)), int(round(w * factor_x))

    for _ in range(12):
        y = max(0, min(y, full_h - 2))
        x = max(0, min(x, full_w - 2))
        h = max(2, min(h, full_h - y))
        w = max(2, min(w, full_w - x))
        if mask[y : y + h, x : x + w].all():
            return (y, x, h, w)
        # Shrink 1.5% about the centre and try again.
        shrink_y, shrink_x = max(1, int(h * 0.015)), max(1, int(w * 0.015))
        y, h = y + shrink_y, h - 2 * shrink_y
        x, w = x + shrink_x, w - 2 * shrink_x
        if h < 8 or w < 8:
            return None
    return None


def apply_crop(image: np.ndarray, mask: np.ndarray, mode: str, source_shape):
    """Crop `image` according to `mode`. Returns `(cropped, note)`.

    `source_shape` supplies the original photograph's proportions, which is
    what "keep original aspect ratio" means. On failure the image is returned
    untouched with an explanatory note rather than raising - a crop that cannot
    be found is a cosmetic disappointment, not an error worth breaking the
    interface over.
    """
    if mode == CROP_OFF or image is None or mask is None:
        return image, ""

    aspect = None
    if mode == CROP_ORIGINAL:
        aspect = float(source_shape[1]) / float(source_shape[0])

    rect = largest_inscribed_rect(mask, aspect=aspect)
    if rect is None:
        return image, "No usable crop found; showing the full warp."

    y, x, h, w = rect
    cropped = image[y : y + h, x : x + w]
    kept = 100.0 * (h * w) / float(image.shape[0] * image.shape[1])
    return cropped, f"Cropped to {w}x{h} px ({kept:.0f}% of the warped canvas)."
