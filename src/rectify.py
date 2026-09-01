"""
Stage 3 of the pipeline: turn vanishing points into a rectifying homography.

The construction
----------------
Take two vanishing points `v_a` and `v_b` belonging to two different line
families that lie in the *same* physical plane (e.g. the horizontal and
vertical edges of a building facade). The image line through them,
`l = v_a x v_b`, is that plane's vanishing line: every direction in the plane
vanishes somewhere on it.

Rectifying the plane then means sending `l` to the line at infinity, which the
matrix

    Hp = [[1, 0,  0],
          [0, 1,  0],
          [l1, l2, l3]]

does by construction, because `Hp^-T l = (0, 0, 1)`. That removes the
*projective* distortion: parallel lines in the plane become parallel again.
What survives is an affine distortion (shear plus anisotropic scale), which we
remove with a 2x2 matrix `A` chosen so the two families end up along the image
axes. The full correction is `H = A . Hp`, followed by a similarity that fits
the result inside the output canvas.

Two things this file is careful about
-------------------------------------
1. **Interpolation.** A "strength" of 0 must be the identity and 1 the full
   correction, with something sensible in between. Naively blending the two
   matrices entry-by-entry is not meaningful for homographies. Instead we
   blend the two *geometrically distinct* parts separately: scale the
   projective row toward zero, and blend the affine block toward the identity.
   Both endpoints come out exactly right and the path between them is smooth.

2. **Blow-ups.** If part of the image sits on the far side of the vanishing
   line, its points map behind the camera (`w <= 0`) and the warp explodes.
   Rather than emitting a garbage image, we sample the image, keep only the
   well-behaved region, and fit the canvas to that.
"""

from __future__ import annotations

import cv2
import numpy as np


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec


# --------------------------------------------------------------------------
# Calibration from vanishing points
# --------------------------------------------------------------------------


def focal_from_orthogonal_vps(v_a, v_b, image_shape):
    """Estimate the camera's focal length in pixels from two vanishing points.

    If two world directions are perpendicular, their vanishing points `v1`,
    `v2` satisfy `v1^T . w . v2 = 0`, where `w` is the image of the absolute
    conic. Assuming what almost every consumer camera satisfies closely -
    square pixels, no skew, principal point at the image centre - that
    constraint collapses to a single equation in one unknown:

        (u1 - cx)(u2 - cx) + (v1 - cy)(v2 - cy) + f^2 = 0

    so `f = sqrt(-(dot product))`. Returns None when the dot product is
    positive, which means the two families cannot be perpendicular under these
    assumptions - a sign the estimate is unreliable, not something to paper
    over - or when either vanishing point is at infinity, where the equation
    carries no information about `f`.
    """
    height, width = image_shape[:2]
    centre = np.array([width / 2.0, height / 2.0])

    if abs(v_a[2]) < 1e-8 or abs(v_b[2]) < 1e-8:
        return None
    p_a = v_a[:2] / v_a[2] - centre
    p_b = v_b[:2] / v_b[2] - centre

    dot = float(p_a @ p_b)
    if dot >= -1.0:  # perpendicularity implies a strictly negative dot product
        return None
    focal = float(np.sqrt(-dot))
    # Sanity gate: a plausible lens on this sensor. Outside this range the
    # estimate is being driven by vanishing-point error, not by the geometry.
    if not (0.2 * max(width, height) < focal < 20.0 * max(width, height)):
        return None
    return focal


def metric_homography(v_a, v_b, image_shape, focal: float):
    """Rectify by *rotating the camera*, which preserves the true aspect ratio.

    Once the focal length is known we can undo the projection properly rather
    than algebraically. Back-project each vanishing point to the 3D direction
    it represents, `d = K^-1 . v`; build the rotation `R` whose rows are those
    two directions and their cross product; then

        H = K . R . K^-1

    is exactly the image you would have taken had the camera been rotated to
    face the plane square-on. Because `R` is a rotation and not a general
    affine map, no shear or anisotropic scale is introduced - the rectified
    facade has the proportions of the real wall, which the purely algebraic
    construction below cannot guarantee.

    Returns None if the two directions are too close to parallel to define a
    plane reliably.
    """
    height, width = image_shape[:2]
    K = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )
    K_inv = np.linalg.inv(K)

    d1 = _unit(K_inv @ np.asarray(v_a, dtype=float))
    d2 = _unit(K_inv @ np.asarray(v_b, dtype=float))
    if abs(float(d1 @ d2)) > 0.98:
        return None

    # Point the axes the same way the image does, so the result is not
    # mirrored or upside down: first axis rightward, second axis downward.
    if d1[0] < 0:
        d1 = -d1
    if d2[1] < 0:
        d2 = -d2
    # The VP estimates are noisy, so d1 and d2 are only approximately
    # perpendicular. Gram-Schmidt makes the frame exactly orthonormal, which
    # a rotation matrix must be.
    d2 = _unit(d2 - float(d1 @ d2) * d1)
    d3 = np.cross(d1, d2)

    R = np.vstack([d1, d2, d3])
    return K @ R @ K_inv


def homography_from_two_vps(
    v_a: np.ndarray,
    v_b: np.ndarray,
    image_shape,
) -> np.ndarray | None:
    """Full rectification of the plane spanned by the directions `v_a`, `v_b`.

    `v_a` ends up along the output x-axis and `v_b` along the y-axis, so pass
    the horizontal family first and the vertical family second for a facade.
    Returns None if the two VPs are degenerate (identical or collinear with
    the image centre in a way that makes the plane ill-defined).
    """
    height, width = image_shape[:2]
    centre = np.array([width / 2.0, height / 2.0, 1.0])

    vanishing_line = np.cross(_unit(v_a), _unit(v_b))
    if np.linalg.norm(vanishing_line) < 1e-9:
        return None

    # Scale the vanishing line so the image centre keeps w = 1 under Hp. This
    # anchors the warp at the middle of the picture instead of the origin,
    # which keeps the output near the input's scale.
    denominator = float(vanishing_line @ centre)
    if abs(denominator) < 1e-9:
        return None
    vanishing_line = vanishing_line / denominator

    Hp = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], list(vanishing_line)],
        dtype=np.float64,
    )

    # After Hp both VPs are points at infinity; their (x, y) parts are pure
    # directions. Build the affine block that maps those two directions onto
    # the coordinate axes.
    a_inf = Hp @ v_a
    b_inf = Hp @ v_b
    basis = np.column_stack([a_inf[:2], b_inf[:2]])
    if abs(np.linalg.det(basis)) < 1e-9:
        return None
    affine2 = np.linalg.inv(basis)

    # Normalise to unit determinant: we only want the *shape* correction here,
    # not an arbitrary scale change (the canvas fit handles scale).
    det = abs(np.linalg.det(affine2))
    affine2 = affine2 / np.sqrt(det) if det > 1e-12 else affine2

    A = np.eye(3)
    A[:2, :2] = affine2
    return A @ Hp


def homography_vertical_only(v_vertical: np.ndarray, image_shape):
    """Keystone fix: make converging verticals vertical, touch nothing else.

    We pick the vanishing line to be the one through the vertical VP *and* the
    horizontal point at infinity `(1, 0, 0)`, i.e. `l = v x (1, 0, 0)`. Sending
    that to infinity straightens verticals while leaving the image's own
    horizontal direction untouched. The leftover distortion is then a pure
    horizontal shear, which one triangular matrix removes.
    """
    height, width = image_shape[:2]
    centre = np.array([width / 2.0, height / 2.0, 1.0])

    v = _unit(np.asarray(v_vertical, dtype=np.float64))
    if abs(v[2]) < 1e-9:
        # Already at infinity: verticals are parallel, nothing to correct.
        return np.eye(3)

    vanishing_line = np.cross(v, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(vanishing_line) < 1e-9:
        return np.eye(3)
    denominator = float(vanishing_line @ centre)
    if abs(denominator) < 1e-9:
        return np.eye(3)
    vanishing_line = vanishing_line / denominator

    Hp = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], list(vanishing_line)], dtype=np.float64
    )

    direction = (Hp @ v)[:2]
    if abs(direction[1]) < 1e-9:
        return Hp
    shear = np.eye(3)
    shear[0, 1] = -direction[0] / direction[1]  # send (dx, dy) -> (0, dy)
    return shear @ Hp


def scale_homography(H: np.ndarray, strength: float) -> np.ndarray:
    """Interpolate between the identity (0.0) and the full correction (1.0).

    Decompose `H = A . Hp`, damp each part toward the identity independently,
    then recombine. See the module docstring for why we do not blend `H` and
    `I` directly.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength >= 0.999:
        return H.copy()
    if strength <= 0.001:
        return np.eye(3)

    H = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

    # Recover the projective row and the affine block.
    projective = np.eye(3)
    projective[2, :] = H[2, :]
    affine = H @ np.linalg.inv(projective)

    damped_projective = np.eye(3)
    damped_projective[2, 0] = strength * H[2, 0]
    damped_projective[2, 1] = strength * H[2, 1]
    damped_projective[2, 2] = 1.0

    damped_affine = np.eye(3)
    damped_affine[:2, :2] = (1 - strength) * np.eye(2) + strength * affine[:2, :2]
    damped_affine[:2, 2] = strength * affine[:2, 2]

    return damped_affine @ damped_projective


def orientation_guard(H: np.ndarray, image_shape) -> np.ndarray:
    """Remove any mirroring the affine step introduced.

    `homography_from_two_vps` inverts a basis built from two vanishing-point
    directions. Depending on which way along each family the direction happens
    to point, that inverse can come out with a negative determinant, which
    silently produces a mirror-image result: geometrically a valid
    rectification, visually a wrong photograph with the text back-to-front.
    We detect it from the Jacobian at the image centre and flip back.

    Gross rotation is *not* corrected here, because it is not arbitrary: the
    affine step deliberately puts the first family along x and the second along
    y, and rotating after that would undo the alignment we just achieved.
    """
    height, width = image_shape[:2]
    centre = np.array([width / 2.0, height / 2.0, 1.0])
    mapped = H @ centre
    if abs(mapped[2]) < 1e-12:
        return H

    # Jacobian of the perspective division at the centre.
    jacobian = (
        H[:2, :2] * mapped[2] - np.outer(mapped[:2], H[2, :2])
    ) / (mapped[2] ** 2)

    if np.linalg.det(jacobian) < 0:
        flip = np.diag([1.0, -1.0, 1.0])  # mirror vertically to restore handedness
        return flip @ H
    return H


def fit_to_canvas(
    H: np.ndarray,
    image_shape,
    max_side: int = 900,
    anchors: np.ndarray | None = None,
    margin: float = 0.10,
    max_aspect: float = 3.0,
):
    """Post-multiply `H` with a similarity so the result fills the canvas well.

    Two framing strategies, in order of preference:

    * If `anchors` are supplied - the endpoints of the line segments that
      actually voted for the vanishing points in use - we frame those. This is
      almost always what the user wants: rectifying a facade tends to fling the
      sky and the road off toward infinity, and fitting the canvas to *those*
      leaves the building as a small patch in the corner.
    * Otherwise we sample a grid over the whole image and frame whatever lands
      in front of the camera.

    Either way, points that end up on the far side of the vanishing line are
    discarded (they map behind the camera and would blow the extent up), and
    percentiles rather than min/max absorb what remains of the stretching.

    Returns `(H_fitted, (out_width, out_height))`.
    """
    height, width = image_shape[:2]
    centre_w = (np.array([width / 2.0, height / 2.0, 1.0]) @ H.T)[2]
    sign = np.sign(centre_w) if abs(centre_w) > 1e-12 else 1.0

    def project(points, low_pct, high_pct):
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        warped = homogeneous @ H.T
        w = warped[:, 2]
        good = (np.sign(w) == sign) & (np.abs(w) > 1e-6)
        if good.sum() < 4:
            return None
        finite = warped[good, :2] / w[good, None]
        return (
            np.percentile(finite, low_pct, axis=0),
            np.percentile(finite, high_pct, axis=0),
            np.median(finite, axis=0),
        )

    bounds = None
    if anchors is not None and len(anchors) >= 4:
        bounds = project(np.asarray(anchors, dtype=float), 1.0, 99.0)
    if bounds is None:
        ys, xs = np.mgrid[0:31, 0:31]
        grid = np.stack(
            [xs.ravel() / 30.0 * (width - 1), ys.ravel() / 30.0 * (height - 1)], axis=1
        )
        bounds = project(grid, 0.5, 99.5)
    if bounds is None:
        return np.eye(3), (width, height)

    lo, hi, median = bounds
    extent = np.maximum(hi - lo, 1e-6)

    # Cap the aspect ratio. A top-down view of a ground plane stretches the far
    # distance without bound, and an unclamped fit turns that into a canvas
    # thousands of pixels tall and forty wide. Clamping around the *median* of
    # the warped points (not the midpoint of the extent, which sits out in the
    # stretched tail) keeps the bulk of the content in frame.
    clamped = False
    if extent[1] > max_aspect * extent[0]:
        extent[1], clamped = max_aspect * extent[0], True
    if extent[0] > max_aspect * extent[1]:
        extent[0], clamped = max_aspect * extent[1], True
    if clamped:
        # Only re-centre when the clamp actually fired; the facade modes are
        # never clamped and should keep their exact percentile framing.
        lo = median - extent / 2.0

    lo = lo - extent * margin
    extent = extent * (1 + 2 * margin)

    scale = float(np.clip(min(max_side / extent[0], max_side / extent[1]), 1e-6, 10.0))
    similarity = np.array(
        [[scale, 0.0, -lo[0] * scale], [0.0, scale, -lo[1] * scale], [0.0, 0.0, 1.0]]
    )
    out_w = int(np.clip(round(extent[0] * scale), 16, max_side))
    out_h = int(np.clip(round(extent[1] * scale), 16, max_side))
    return similarity @ H, (out_w, out_h)


def warp(
    image_bgr: np.ndarray,
    H: np.ndarray,
    strength: float = 1.0,
    max_side: int = 900,
    anchors: np.ndarray | None = None,
):
    """Apply a rectifying homography.

    Returns `(warped_image, valid_mask, H_final)`, where `valid_mask` marks the
    output pixels that actually received source pixels. Everything else is the
    background wedge left by the projective map, and `src/crop.py` uses the
    mask to crop it away.

    On any numerical failure this returns the input unchanged rather than
    raising, so a degenerate estimate degrades to "no correction" instead of
    breaking the interface mid-session.
    """
    from . import crop as crop_module

    try:
        H_scaled = scale_homography(H, strength)
        H_scaled = orientation_guard(H_scaled, image_bgr.shape)
        H_final, (out_w, out_h) = fit_to_canvas(
            H_scaled, image_bgr.shape, max_side, anchors=anchors
        )
        result = cv2.warpPerspective(
            image_bgr,
            H_final,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(24, 24, 24),
        )
        mask = crop_module.valid_mask((out_h, out_w), H_final, image_bgr.shape)
        return result, mask, H_final
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        copy = image_bgr.copy()
        return copy, np.ones(copy.shape[:2], dtype=bool), np.eye(3)
