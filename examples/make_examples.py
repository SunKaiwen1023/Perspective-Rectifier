"""
Generate the bundled example photographs.

The repository ships synthetic images rather than downloaded photographs for
two reasons: there is no licensing question for a grader who clones the repo,
and - more importantly - the scenes are rendered through an explicit pinhole
camera, so the *true* vanishing points are known in closed form. That turns
"does the estimator look about right?" into a test with a number in it:
`tests/test_pipeline.py` checks the recovered vanishing points against the
ones written into `ground_truth.json` by this script.

Run `python examples/make_examples.py` to regenerate. Output is deterministic
(fixed RNG seed), so regenerating never produces a spurious diff.

Camera model
------------
World axes: X right, Y up, Z forward. A point `P` in world coordinates maps to
the image by `p ~ K . R . (P - C)`, where `R = FLIP_Y . Rx(pitch) . Ry(yaw)`
and `FLIP_Y` reconciles the y-up world with y-down pixel coordinates.

The vanishing point of any world direction `d` is then simply `K . R . d` -
note that it does not depend on the camera's position at all, only on its
orientation. That is precisely the property the estimator in
`src/vanishing_points.py` is trying to invert, which is why these renders make
a fair test of it.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(7)

WIDTH, HEIGHT = 1140, 760
FOCAL = 950.0
K = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1.0]])


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


#: Converts the y-up world convention into the y-down convention that image
#: pixel coordinates use. Without it, "up" in the world renders downward and
#: every building comes out inverted - which is exactly the bug this constant
#: was introduced to fix.
FLIP_Y = np.diag([1.0, -1.0, 1.0])


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """R = FLIP . Rx(pitch) . Ry(yaw). Positive pitch tilts the camera upward,
    which makes the scene's vertical lines converge toward a point above the
    frame - the classic keystone distortion the rectifier undoes."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
    )
    return FLIP_Y @ Rx @ Ry


def project(points_world, R, C):
    """Project world points to pixels. Returns (N, 2) float array."""
    cam = (np.asarray(points_world, dtype=float) - np.asarray(C, dtype=float)) @ R.T
    cam[:, 2] = np.where(np.abs(cam[:, 2]) < 1e-6, 1e-6, cam[:, 2])
    image = cam @ K.T
    return image[:, :2] / image[:, 2:3]


def vanishing_point(direction, R):
    """Pixel location of the vanishing point of a world direction."""
    v = K @ R @ np.asarray(direction, dtype=float)
    if abs(v[2]) < 1e-9:
        return None
    return (float(v[0] / v[2]), float(v[1] / v[2]))


# --------------------------------------------------------------------------
# Texture synthesis
# --------------------------------------------------------------------------


def brick_facade(width=900, height=700, base=(150, 158, 172), rows=8, cols=6):
    """A frontal building facade: masonry courses plus a grid of windows."""
    facade = np.full((height, width, 3), base, dtype=np.float32)
    facade += RNG.normal(0, 5, facade.shape)

    for y in range(0, height, 26):  # horizontal masonry courses
        facade[y : y + 2, :] *= 0.86
    for y in range(0, height, 26):  # staggered vertical joints
        offset = 0 if (y // 26) % 2 == 0 else 26
        for x in range(offset, width, 52):
            facade[y : y + 26, x : x + 2] *= 0.9
    facade = np.clip(facade, 0, 255).astype(np.uint8)

    margin_x, margin_y = width * 0.08, height * 0.12
    cell_w = (width - 2 * margin_x) / cols
    cell_h = (height - 2 * margin_y) / rows
    for r in range(rows):
        for c in range(cols):
            x0 = int(margin_x + c * cell_w + cell_w * 0.20)
            y0 = int(margin_y + r * cell_h + cell_h * 0.14)
            x1 = int(margin_x + c * cell_w + cell_w * 0.80)
            y1 = int(margin_y + r * cell_h + cell_h * 0.80)
            shade = int(RNG.integers(38, 78))
            cv2.rectangle(facade, (x0, y0), (x1, y1), (shade + 22, shade + 12, shade), -1)
            cv2.rectangle(facade, (x0, y0), (x1, y1), (215, 218, 222), 4)
            cv2.line(facade, ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1), (206, 210, 216), 3)

    cv2.rectangle(facade, (0, 0), (width, int(height * 0.05)), (196, 200, 208), -1)
    cv2.rectangle(
        facade, (0, int(height * 0.05)), (width, int(height * 0.068)), (118, 124, 134), -1
    )
    return facade


def pavement_texture(width=900, height=600, base=(126, 130, 136)):
    """Paving slabs: a grid, so the ground plane contributes real structure."""
    slab = np.full((height, width, 3), base, dtype=np.float32)
    slab += RNG.normal(0, 4, slab.shape)
    slab = np.clip(slab, 0, 255).astype(np.uint8)
    for y in range(0, height, 60):
        cv2.line(slab, (0, y), (width, y), (100, 104, 110), 3)
    for x in range(0, width, 90):
        cv2.line(slab, (x, 0), (x, height), (110, 114, 120), 3)
    return slab


def sky_background():
    """A soft vertical gradient - edge-free, so it adds no false segments."""
    top = np.array([210, 180, 145], dtype=np.float32)  # BGR
    bottom = np.array([240, 230, 216], dtype=np.float32)
    ramp = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
    canvas = top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp
    canvas = np.repeat(canvas, WIDTH, axis=1)
    canvas += RNG.normal(0, 2.0, canvas.shape)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def paste_quad(canvas, texture, quad_px):
    """Warp `texture` onto a projected quad (order: TL, TR, BR, BL)."""
    height, width = texture.shape[:2]
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    quad = np.float32(quad_px)
    if not np.all(np.isfinite(quad)) or np.any(np.abs(quad) > 1e5):
        return canvas
    H = cv2.getPerspectiveTransform(source, quad)
    warped = cv2.warpPerspective(texture, H, (WIDTH, HEIGHT))
    mask = cv2.warpPerspective(np.full((height, width), 255, np.uint8), H, (WIDTH, HEIGHT))
    canvas[mask > 0] = warped[mask > 0]
    return canvas


# --------------------------------------------------------------------------
# Clutter: the thing the user is meant to prune away
# --------------------------------------------------------------------------


def draw_branch(canvas, start, angle, length, thickness, depth=0):
    """Recursive tree branch, drawn with long straight strokes on purpose.

    Foliage that produces only tiny squiggles would be filtered out by the
    length threshold and prove nothing. These branches are long enough to
    survive detection and therefore genuinely compete with the architecture
    for RANSAC's vote - which is the failure mode the pruning interaction and
    the learned scorer exist to address.
    """
    if depth > 4 or length < 12:
        return
    end = (start[0] + length * np.cos(angle), start[1] + length * np.sin(angle))
    bark = (int(RNG.integers(28, 58)), int(RNG.integers(58, 105)), int(RNG.integers(24, 52)))
    cv2.line(
        canvas, tuple(np.int32(start)), tuple(np.int32(end)), bark,
        max(1, int(thickness)), cv2.LINE_AA,
    )
    for _ in range(int(RNG.integers(2, 4))):
        draw_branch(
            canvas, end, angle + RNG.normal(0, 0.55),
            length * float(RNG.uniform(0.55, 0.78)), thickness * 0.7, depth + 1,
        )
    if depth >= 2:
        for _ in range(int(RNG.integers(3, 7))):
            centre = (int(end[0] + RNG.normal(0, 14)), int(end[1] + RNG.normal(0, 14)))
            leaf = (
                int(RNG.integers(20, 55)), int(RNG.integers(85, 155)), int(RNG.integers(30, 70))
            )
            cv2.ellipse(
                canvas, centre,
                (int(RNG.integers(5, 11)), int(RNG.integers(3, 7))),
                float(RNG.uniform(0, 180)), 0, 360, leaf, -1, cv2.LINE_AA,
            )


# --------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------


def render_single_facade(with_foliage: bool):
    """One oblique facade plus pavement, shot from below with the camera
    tilted up - so both the horizontal and the vertical edges converge."""
    R = rotation(yaw_deg=24.0, pitch_deg=17.0)
    C = np.array([0.0, 1.6, 0.0])
    canvas = sky_background()

    # Pavement first, so the building overlaps it correctly.
    ground = np.array(
        [[-14, 0, 4], [14, 0, 4], [14, 0, 30], [-14, 0, 30]], dtype=float
    )
    canvas = paste_quad(canvas, pavement_texture(), project(ground[[3, 2, 1, 0]], R, C))

    # Facade: plane Z = 17, spanning X in [-19, 4], Y in [0, 13].
    facade = np.array([[-19, 13, 17], [4, 13, 17], [4, 0, 17], [-19, 0, 17]], dtype=float)
    canvas = paste_quad(canvas, brick_facade(cols=8), project(facade, R, C))

    if with_foliage:
        for base_x in (180, 520, 830):
            draw_branch(
                canvas, (base_x, 755),
                -np.pi / 2 + float(RNG.normal(0, 0.3)),
                float(RNG.uniform(120, 175)), 7,
            )

    truth = {
        "vp_horizontal": [vanishing_point([1, 0, 0], R)],
        "vp_vertical": vanishing_point([0, 1, 0], R),
        "camera": {"yaw_deg": 24.0, "pitch_deg": 17.0, "focal_px": FOCAL},
    }
    return canvas, truth


def render_street_corner():
    """A building corner: two facades at right angles plus the pavement, so
    all three orthogonal families are present and the horizon is well defined."""
    # The camera stands off to one side of the corner and aims at it, so both
    # walls are seen obliquely and each contributes its own horizontal family.
    R = rotation(yaw_deg=-31.0, pitch_deg=12.0)
    C = np.array([-12.0, 1.7, 0.0])
    canvas = sky_background()

    # Pavement first; the walls are drawn over it.
    ground = np.array([[-40, 0, 60], [40, 0, 60], [40, 0, 2], [-40, 0, 2]], dtype=float)
    canvas = paste_quad(canvas, pavement_texture(), project(ground, R, C))

    # Wall B: the plane X = 0, running away from the corner along +Z. Its
    # horizontal edges therefore vanish at the VP of the world direction Z.
    wall_z = np.array([[0, 13, 48], [0, 13, 20], [0, 0, 20], [0, 0, 48]], dtype=float)
    canvas = paste_quad(
        canvas, brick_facade(base=(138, 148, 166), cols=6), project(wall_z, R, C)
    )
    # Wall A: the plane Z = 20, running right from the corner along +X. Its
    # horizontal edges vanish at the VP of the world direction X.
    wall_x = np.array([[0, 13, 20], [25, 13, 20], [25, 0, 20], [0, 0, 20]], dtype=float)
    canvas = paste_quad(
        canvas, brick_facade(base=(118, 142, 172), cols=9, rows=7), project(wall_x, R, C)
    )

    truth = {
        "vp_horizontal": [vanishing_point([1, 0, 0], R), vanishing_point([0, 0, 1], R)],
        "vp_vertical": vanishing_point([0, 1, 0], R),
        "camera": {"yaw_deg": -31.0, "pitch_deg": 12.0, "focal_px": FOCAL},
    }
    return canvas, truth


def render_plaza():
    """A paved plaza shot from above eye level, looking down.

    This is the scene the ground-plane mode is for: most of the frame lies on
    a single physical plane, so sending the horizon to infinity produces a
    genuine top-down plan view of the paving rather than a smear.
    """
    R = rotation(yaw_deg=19.0, pitch_deg=-21.0)
    C = np.array([0.0, 7.0, 0.0])
    canvas = sky_background()

    ground = np.array([[-45, 0, 75], [45, 0, 75], [45, 0, 4], [-45, 0, 4]], dtype=float)
    canvas = paste_quad(
        canvas, pavement_texture(width=1100, height=900), project(ground, R, C)
    )

    # A low block at the back gives the scene a vertical family too, so the
    # focal-length estimate has a perpendicular pair to work with.
    block = np.array([[-26, 9, 46], [6, 9, 46], [6, 0, 46], [-26, 0, 46]], dtype=float)
    canvas = paste_quad(
        canvas, brick_facade(base=(146, 152, 166), rows=4, cols=7), project(block, R, C)
    )

    truth = {
        "vp_horizontal": [vanishing_point([1, 0, 0], R), vanishing_point([0, 0, 1], R)],
        "vp_vertical": vanishing_point([0, 1, 0], R),
        "camera": {"yaw_deg": 19.0, "pitch_deg": -21.0, "focal_px": FOCAL},
    }
    return canvas, truth


def main():
    scenes = {
        "01_facade_clean.jpg": lambda: render_single_facade(False),
        "02_facade_with_foliage.jpg": lambda: render_single_facade(True),
        "03_street_corner.jpg": render_street_corner,
        "04_plaza_ground.jpg": render_plaza,
    }
    ground_truth = {}
    for filename, build in scenes.items():
        image, truth = build()
        cv2.imwrite(os.path.join(HERE, filename), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ground_truth[filename] = truth
        print(f"wrote {filename}  {image.shape[1]}x{image.shape[0]}")

    with open(os.path.join(HERE, "ground_truth.json"), "w") as handle:
        json.dump(ground_truth, handle, indent=2)
    print("wrote ground_truth.json")


if __name__ == "__main__":
    main()
