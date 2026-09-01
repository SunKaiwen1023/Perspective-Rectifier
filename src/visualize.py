"""
Rendering the overlay that makes the geometry visible - and clickable.

This module is the heart of the project's premise. Commercial perspective
tools apply a homography and show you the result; if the result is wrong you
get no explanation and no lever. Here every intermediate quantity is drawn:
which segments were detected, which vanishing point each one voted for, where
those vanishing points actually are, where the horizon runs, and which
segments the learned scorer distrusts. The user prunes by clicking, so the
overlay is simultaneously the explanation and the control surface.

Colour convention (kept consistent everywhere, including the legend):
    green    vertical family
    orange   first horizontal family
    blue     second horizontal family
    purple   any additional family
    grey     detected but unassigned to any vanishing point
    dark red deleted by the user (drawn dashed, still clickable to restore)
"""

from __future__ import annotations

import cv2
import numpy as np

# BGR, because everything internal stays in OpenCV's channel order until the
# very last step in `pipeline.py`.
COLOR_VERTICAL = (80, 220, 90)
COLOR_H1 = (40, 150, 255)
COLOR_H2 = (240, 180, 60)
COLOR_EXTRA = (220, 120, 220)
COLOR_UNASSIGNED = (150, 150, 150)
COLOR_DELETED = (70, 70, 200)
COLOR_HORIZON = (255, 255, 255)

GROUP_COLORS = [COLOR_H1, COLOR_H2, COLOR_EXTRA]


def group_color(group_id: int, orientation: str):
    """Stable colour for a vanishing-point family."""
    if orientation == "vertical":
        return COLOR_VERTICAL
    return GROUP_COLORS[group_id % len(GROUP_COLORS)]


def _dashed_line(canvas, p1, p2, color, thickness=1, dash=8):
    """OpenCV has no dashed-line primitive, so we step along the segment."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    length = float(np.linalg.norm(p2 - p1))
    if length < 1e-6:
        return
    steps = max(int(length / dash), 1)
    for i in range(steps):
        if i % 2:
            continue
        a = p1 + (p2 - p1) * (i / steps)
        b = p1 + (p2 - p1) * (min(i + 1, steps) / steps)
        cv2.line(canvas, tuple(np.int32(a)), tuple(np.int32(b)), color, thickness, cv2.LINE_AA)


def draw_overlay(
    image_bgr: np.ndarray,
    lines,
    assignment: np.ndarray,
    orientations: dict,
    vp_pixels: dict,
    deleted: set,
    suspicion: np.ndarray | None = None,
    show_rays: bool = True,
    show_vps: bool = True,
    horizon=None,
    suspicion_threshold: float = 0.35,
) -> np.ndarray:
    """Compose the annotated view.

    Parameters
    ----------
    assignment
        (N,) int array: index of the vanishing-point family each segment was
        assigned to, or -1 for unassigned.
    orientations
        {group_id: "vertical" | "horizontal"} - drives the colour choice.
    vp_pixels
        {group_id: (x, y)} for vanishing points that landed at a finite
        location; families whose VP is effectively at infinity are omitted.
    suspicion
        (N,) array of P(structural) from the learned scorer, or None. Segments
        below `suspicion_threshold` get a small warning tick so the user can
        see the model's opinion without it overriding their own.
    """
    canvas = image_bgr.copy()
    ray_layer = canvas.copy()

    # --- perspective rays: extend each segment toward its vanishing point ---
    if show_rays:
        for i in range(len(lines)):
            group = int(assignment[i])
            if group < 0 or i in deleted or group not in vp_pixels:
                continue
            colour = group_color(group, orientations.get(group, "horizontal"))
            vp = np.asarray(vp_pixels[group], dtype=float)
            for endpoint in (lines.endpoints[i][0:2], lines.endpoints[i][2:4]):
                cv2.line(
                    ray_layer,
                    tuple(np.int32(endpoint)),
                    tuple(np.int32(np.clip(vp, -1e5, 1e5))),
                    colour,
                    1,
                    cv2.LINE_AA,
                )
        # Rays are context, not content: keep them faint.
        canvas = cv2.addWeighted(ray_layer, 0.28, canvas, 0.72, 0)

    # --- horizon ------------------------------------------------------------
    if horizon is not None:
        height, width = canvas.shape[:2]
        a, b, c = horizon
        pts = []
        if abs(b) > 1e-9:
            pts = [(0, -c / b), (width - 1, -(a * (width - 1) + c) / b)]
        elif abs(a) > 1e-9:
            pts = [(-c / a, 0), (-(b * (height - 1) + c) / a, height - 1)]
        if pts and all(abs(p[1]) < 1e5 and abs(p[0]) < 1e5 for p in pts):
            _dashed_line(canvas, pts[0], pts[1], COLOR_HORIZON, 1, dash=14)
            cv2.putText(
                canvas, "horizon", (8, int(np.clip(pts[0][1] - 6, 12, canvas.shape[0] - 4))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_HORIZON, 1, cv2.LINE_AA,
            )

    # --- the segments themselves -------------------------------------------
    for i in range(len(lines)):
        x1, y1, x2, y2 = np.int32(lines.endpoints[i])
        if i in deleted:
            _dashed_line(canvas, (x1, y1), (x2, y2), COLOR_DELETED, 1, dash=6)
            continue

        group = int(assignment[i])
        if group < 0:
            colour, thickness = COLOR_UNASSIGNED, 1
        else:
            colour = group_color(group, orientations.get(group, "horizontal"))
            thickness = 2

        cv2.line(canvas, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)

        if suspicion is not None and i < len(suspicion) and suspicion[i] < suspicion_threshold:
            mid = np.int32(lines.midpoints[i])
            cv2.circle(canvas, tuple(mid), 4, (0, 0, 255), 1, cv2.LINE_AA)

    # --- vanishing point markers -------------------------------------------
    if show_vps:
        height, width = canvas.shape[:2]
        for group, (vx, vy) in vp_pixels.items():
            if not (-width * 0.6 < vx < width * 1.6 and -height * 0.6 < vy < height * 1.6):
                continue  # off-canvas VPs are reported numerically instead
            colour = group_color(group, orientations.get(group, "horizontal"))
            centre = (int(vx), int(vy))
            cv2.circle(canvas, centre, 9, colour, 2, cv2.LINE_AA)
            cv2.drawMarker(canvas, centre, colour, cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
            cv2.putText(
                canvas, f"VP{group + 1}", (centre[0] + 12, centre[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )

    return canvas


def nearest_line(lines, x: float, y: float, max_distance: float = 14.0):
    """Index of the segment nearest to a click, or None if the click missed.

    Distance is measured to the *segment*, not to its infinite extension, so
    clicking far off the end of a line does not select it. This is the only
    piece of hit-testing in the project and it needs to feel forgiving, hence
    the tolerance in pixels rather than an exact hit.
    """
    if len(lines) == 0:
        return None
    p = np.array([x, y], dtype=float)
    a = lines.endpoints[:, 0:2]
    b = lines.endpoints[:, 2:4]
    ab = b - a
    denominator = (ab * ab).sum(axis=1)
    denominator = np.where(denominator > 1e-9, denominator, 1.0)
    t = np.clip(((p - a) * ab).sum(axis=1) / denominator, 0.0, 1.0)
    projection = a + t[:, None] * ab
    distances = np.linalg.norm(projection - p, axis=1)
    best = int(np.argmin(distances))
    return best if distances[best] <= max_distance else None


def legend_markdown() -> str:
    """Colour key for the interface, kept next to the colours it describes."""
    return (
        "**Line colours** &nbsp; "
        "<span style='color:#5adc50'>&#9632;</span> vertical family &nbsp; "
        "<span style='color:#ff9628'>&#9632;</span> horizontal family 1 &nbsp; "
        "<span style='color:#3cb4f0'>&#9632;</span> horizontal family 2 &nbsp; "
        "<span style='color:#dc78dc'>&#9632;</span> extra family &nbsp; "
        "<span style='color:#969696'>&#9632;</span> unassigned &nbsp; "
        "<span style='color:#c84646'>&#9632;</span> deleted (dashed) &nbsp; "
        "<span style='color:#ff0000'>&#9675;</span> model flags as suspect"
    )
