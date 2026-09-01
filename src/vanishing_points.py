"""
Stage 2 of the pipeline: estimate vanishing points from a set of line segments.

The maths in one paragraph
--------------------------
Under a pinhole camera, a family of parallel 3D lines projects to a family of
image lines that all pass through a single point: the vanishing point (VP).
Written in homogeneous coordinates, a VP `v` satisfies `l . v = 0` for every
line `l` in the family, so two lines are enough to *propose* a VP (their cross
product) and many lines are needed to *verify* it. That is exactly the shape of
a RANSAC problem, and RANSAC is what we use: propose from minimal samples,
score by consensus, keep the best, then refine the winner with all its
inliers via a least-squares eigenproblem.

An architectural photo usually contains up to three mutually orthogonal
families (two horizontal directions plus vertical), so we run RANSAC
*sequentially*: find the strongest VP, remove its inliers, find the next,
remove those, find the third. This is simpler and far more stable than trying
to fit all three at once, and it degrades gracefully when a photo only
contains one or two usable families.

Why the consistency measure is angular
--------------------------------------
The naive residual `|l . v|` is an algebraic quantity with no units; it grows
with how far the VP happens to land from the origin, which makes a single
threshold meaningless. Instead we measure the *angle* between a segment's own
direction and the direction from its midpoint to the candidate VP. That is
scale-free, has an intuitive unit (degrees), and is what the user's threshold
slider actually controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .line_detection import LineSet


@dataclass
class VanishingPoint:
    """One estimated vanishing point and the evidence supporting it."""

    point: np.ndarray  # (3,) homogeneous, unit norm
    inliers: np.ndarray  # (K,) indices into the parent LineSet
    orientation: str  # "vertical" or "horizontal"
    mean_error_deg: float  # mean angular residual over its inliers

    @property
    def is_finite(self) -> bool:
        """False when the VP sits (numerically) at infinity - i.e. the family
        of lines is already parallel in the image and needs no correction."""
        return abs(self.point[2]) > 1e-8

    def pixel(self):
        """Cartesian location in pixels, or None if the VP is at infinity."""
        if not self.is_finite:
            return None
        return self.point[:2] / self.point[2]


# --------------------------------------------------------------------------
# Residuals
# --------------------------------------------------------------------------


def angular_residuals(lines: LineSet, vp: np.ndarray) -> np.ndarray:
    """Angle in degrees between each segment and the ray to `vp`.

    Vectorised over all segments. Note the algebraic trick: the direction from
    a midpoint `m` to a homogeneous point `v = (vx, vy, vw)` is proportional to
    `(vx, vy) - vw * m`, which avoids dividing by `vw` and therefore stays
    stable when the vanishing point is near or at infinity.
    """
    to_vp = vp[:2][None, :] - vp[2] * lines.midpoints  # (N, 2)
    norms = np.linalg.norm(to_vp, axis=1)
    norms = np.where(norms > 1e-12, norms, 1.0)
    to_vp = to_vp / norms[:, None]

    # |sin(angle)| via the 2D cross product; robust and avoids arccos of a
    # value that has drifted slightly outside [-1, 1].
    cross = np.abs(
        lines.directions[:, 0] * to_vp[:, 1] - lines.directions[:, 1] * to_vp[:, 0]
    )
    return np.degrees(np.arcsin(np.clip(cross, 0.0, 1.0)))


def _refine(lines: LineSet, idx: np.ndarray) -> np.ndarray:
    """Least-squares refinement of a VP from all of its inliers.

    We minimise `sum_i w_i (l_i . v)^2` subject to `||v|| = 1`. The solution is
    the eigenvector of `sum_i w_i l_i l_i^T` with the smallest eigenvalue.
    Segments are weighted by length because a long segment pins down its
    direction far more precisely than a short one.
    """
    if len(idx) < 2:
        return None
    L = lines.homog[idx]
    w = lines.lengths[idx]
    w = w / (w.sum() + 1e-12)
    scatter = (L * w[:, None]).T @ L  # (3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    v = eigenvectors[:, np.argmin(eigenvalues)]
    return v / (np.linalg.norm(v) + 1e-12)


# --------------------------------------------------------------------------
# Sequential RANSAC
# --------------------------------------------------------------------------


def _ransac_one(
    lines: LineSet,
    active: np.ndarray,
    threshold_deg: float,
    iterations: int,
    rng: np.random.Generator,
):
    """Find the single best-supported VP among the `active` segments.

    Returns (vp_vector, inlier_indices) or (None, empty) if there is not
    enough evidence. Candidate scoring is fully vectorised: we build all
    proposals at once and evaluate them against all active lines in one
    broadcasted pass, which keeps the interactive recompute well under a
    tenth of a second for a few hundred segments.
    """
    n = len(active)
    if n < 3:
        return None, np.array([], dtype=int)

    homog = lines.homog[active]
    mids = lines.midpoints[active]
    dirs = lines.directions[active]
    weights = lines.lengths[active]

    # --- propose: random pairs of distinct lines -> candidate VPs -----------
    m = min(iterations, max(200, n * n))
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    valid = i != j
    i, j = i[valid], j[valid]
    if len(i) == 0:
        return None, np.array([], dtype=int)

    candidates = np.cross(homog[i], homog[j])  # (M, 3)
    norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    keep = norms[:, 0] > 1e-9  # drop near-identical line pairs
    candidates = candidates[keep] / norms[keep]
    if len(candidates) == 0:
        return None, np.array([], dtype=int)

    # --- score: length-weighted count of segments within the angle gate ----
    # to_vp[c, k] = direction from segment k's midpoint to candidate c.
    to_vp = candidates[:, None, :2] - candidates[:, None, 2:3] * mids[None, :, :]
    tv_norm = np.linalg.norm(to_vp, axis=2)
    tv_norm = np.where(tv_norm > 1e-12, tv_norm, 1.0)
    cross = np.abs(
        dirs[None, :, 0] * to_vp[..., 1] - dirs[None, :, 1] * to_vp[..., 0]
    ) / tv_norm

    gate = np.sin(np.radians(threshold_deg))
    inlier_mask = cross < gate  # (M, N)
    scores = (inlier_mask * weights[None, :]).sum(axis=1)

    best = int(np.argmax(scores))
    best_mask = inlier_mask[best]
    if best_mask.sum() < 3:
        return None, np.array([], dtype=int)

    # --- refine: re-fit on the consensus set, then re-select inliers -------
    vp = candidates[best]
    for _ in range(3):  # a couple of IRLS-style passes is plenty
        idx_local = np.where(best_mask)[0]
        refined = _refine(lines.subset(active[idx_local]), np.arange(len(idx_local)))
        if refined is None:
            break
        vp = refined
        residuals = angular_residuals(lines.subset(active), vp)
        new_mask = residuals < threshold_deg
        if new_mask.sum() < 3 or np.array_equal(new_mask, best_mask):
            best_mask = new_mask if new_mask.sum() >= 3 else best_mask
            break
        best_mask = new_mask

    return vp, active[np.where(best_mask)[0]]


def estimate_vanishing_points(
    lines: LineSet,
    threshold_deg: float = 2.0,
    max_vps: int = 3,
    iterations: int = 3000,
    min_support: int = 3,
    seed: int = 0,
):
    """Sequentially extract up to `max_vps` vanishing points.

    Returns a list of `VanishingPoint`, ordered by support (strongest first),
    each tagged as "vertical" or "horizontal" based on the mean orientation of
    the segments that voted for it.
    """
    results = []
    if len(lines) < 3:
        return results

    rng = np.random.default_rng(seed)
    remaining = np.arange(len(lines))

    for _ in range(max_vps):
        if len(remaining) < min_support:
            break
        vp, inliers = _ransac_one(lines, remaining, threshold_deg, iterations, rng)
        if vp is None or len(inliers) < min_support:
            break

        residuals = angular_residuals(lines.subset(inliers), vp)
        # A segment votes "vertical" when its own direction is steeper than 45
        # degrees. The family inherits the majority vote of its members.
        steep = np.abs(lines.directions[inliers][:, 1]) > np.abs(
            lines.directions[inliers][:, 0]
        )
        orientation = "vertical" if steep.mean() > 0.5 else "horizontal"

        results.append(
            VanishingPoint(
                point=vp,
                inliers=inliers,
                orientation=orientation,
                mean_error_deg=float(residuals.mean()),
            )
        )
        remaining = np.setdiff1d(remaining, inliers)

    results.sort(key=lambda v: -len(v.inliers))
    return results


def split_by_orientation(vps):
    """Separate the estimated VPs into (vertical_vp_or_None, [horizontal_vps]).

    A scene has only one true vertical direction, so if several families claim
    to be vertical we keep the best-supported one (the list arrives sorted by
    support) and simply drop the others from the rectification candidates.
    They are still drawn in the overlay - the user should see that the
    estimator found them - but feeding a second "vertical" family into a
    horizontal slot would silently produce a nonsense homography.
    """
    vertical = None
    horizontal = []
    for vp in vps:
        if vp.orientation == "vertical":
            if vertical is None:
                vertical = vp
        else:
            horizontal.append(vp)
    return vertical, horizontal


def horizon_line(h_vps):
    """The horizon: the image line joining two horizontal vanishing points.

    Returns a unit-norm homogeneous line, or None if fewer than two horizontal
    VPs were found.
    """
    if len(h_vps) < 2:
        return None
    line = np.cross(h_vps[0].point, h_vps[1].point)
    norm = np.linalg.norm(line)
    if norm < 1e-12:
        return None
    return line / norm
