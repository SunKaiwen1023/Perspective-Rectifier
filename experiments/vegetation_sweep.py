"""
Measure how vegetation degrades vanishing-point accuracy.

This produces the table in section 7 of the README. Run it with:

    python experiments/vegetation_sweep.py

It renders the same facade repeatedly with increasing amounts of foliage, and
for each render reports:

  * the share of detected line segments that sit on green pixels,
  * the angular error of the recovered vertical vanishing point with no
    intervention, and
  * the same error after every green line has been pruned - standing in for a
    user who patiently clicks away all the foliage.

Because the scenes come from the same pinhole camera as `examples/`, the true
vertical vanishing point is known in closed form, so "error" here is a real
measurement rather than an impression.

Read the results with the caveat stated in the README: these renders have far
crisper window frames than real masonry, so the structural lines survive more
occlusion than they would in a photograph. The experiment establishes the
*shape* of the degradation - automatic estimation becomes unreliable early and
erratically, manual pruning recovers it until too little structure remains -
not a threshold to quote for real images.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

import make_examples as mx  # noqa: E402

from src.features import FEATURE_NAMES  # noqa: E402
from src.pipeline import Session  # noqa: E402
from src.vanishing_points import split_by_orientation  # noqa: E402

# Same camera as the bundled facade scenes, so the ground truth carries over.
ROTATION = mx.rotation(24.0, 17.0)
CAMERA = np.array([0.0, 1.6, 0.0])
TRUE_VERTICAL_VP = mx.vanishing_point([0, 1, 0], ROTATION)

#: How green a segment's neighbourhood must be to count as vegetation. The same
#: threshold stands in for "a user can see this is a plant".
GREEN_THRESHOLD = 0.05

#: (number of trees, branch length). Chosen to sweep vegetation from none to
#: dominating the frame.
STEPS = [(0, 0), (3, 60), (5, 90), (7, 120), (9, 150), (12, 170), (16, 190),
         (22, 220), (32, 260)]


def render(n_trees: int, size: float, seed: int = 7) -> np.ndarray:
    """The standard facade scene with `n_trees` trees of the given scale."""
    mx.RNG = np.random.default_rng(seed)  # keep every render reproducible
    canvas = mx.sky_background()

    ground = np.array([[-14, 0, 4], [14, 0, 4], [14, 0, 30], [-14, 0, 30]], float)
    canvas = mx.paste_quad(
        canvas, mx.pavement_texture(), mx.project(ground[[3, 2, 1, 0]], ROTATION, CAMERA)
    )
    facade = np.array([[-19, 13, 17], [4, 13, 17], [4, 0, 17], [-19, 0, 17]], float)
    canvas = mx.paste_quad(
        canvas, mx.brick_facade(cols=8), mx.project(facade, ROTATION, CAMERA)
    )

    rng = np.random.default_rng(seed + 99)
    for x in np.linspace(40, 1100, max(n_trees, 1))[:n_trees]:
        for y in (760, 600, 440, 300):  # stack trunks to build up occlusion
            mx.draw_branch(
                canvas,
                (float(x), float(y)),
                -np.pi / 2 + float(rng.normal(0, 0.6)),
                float(rng.uniform(0.8, 1.2)) * size,
                9,
            )
    return canvas


def vertical_error_deg(session: Session, scale: float) -> float:
    """Angle between the recovered vertical VP and the rendered one.

    Compared as rays rather than pixel positions: two vanishing points far
    outside the frame can sit thousands of pixels apart while describing nearly
    the same 3D direction.
    """
    vertical, _ = split_by_orientation(session.vps)
    if vertical is None or vertical.pixel() is None:
        return float("nan")

    height, width = session.image_bgr.shape[:2]
    focal = mx.FOCAL * scale

    def ray(point):
        vector = np.array(
            [point[0] - width / 2.0, point[1] - height / 2.0, focal], dtype=float
        )
        return vector / np.linalg.norm(vector)

    expected = ray(np.array(TRUE_VERTICAL_VP) * scale)
    return float(np.degrees(np.arccos(np.clip(abs(expected @ ray(vertical.pixel())), 0, 1))))


def analyse(image_bgr: np.ndarray):
    """Return (green_share, raw_error, pruned_error) for one rendered scene."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    session = Session.from_rgb(rgb)
    session.recompute(threshold_deg=2.0)
    scale = session.image_bgr.shape[1] / image_bgr.shape[1]

    greenness = session.features[:, FEATURE_NAMES.index("greenness")]
    green_share = 100.0 * float((greenness > GREEN_THRESHOLD).mean())
    raw = vertical_error_deg(session, scale)

    # Simulate the patient user: delete every visibly green segment.
    for index in np.where(greenness > GREEN_THRESHOLD)[0]:
        session.toggle(int(index))
    session.recompute(threshold_deg=2.0)
    pruned = vertical_error_deg(session, scale)

    return green_share, raw, pruned, len(session.active_indices)


def main():
    print(f"{'trees':>6} {'green %':>8} {'raw err':>9} {'pruned':>8} {'lines left':>11}")
    print("-" * 46)
    for n_trees, size in STEPS:
        green, raw, pruned, remaining = analyse(render(n_trees, size))
        print(
            f"{n_trees:>6} {green:>7.0f}% {raw:>8.1f}° {pruned:>7.1f}° {remaining:>11}"
        )
    print(
        "\nRead these as the shape of the degradation, not as thresholds for real\n"
        "photographs - see the caveat in README section 7."
    )


if __name__ == "__main__":
    main()
