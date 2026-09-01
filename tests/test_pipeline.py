"""
Tests for the rectification pipeline.

Run with:  pytest -q

These are not smoke tests. Because `examples/make_examples.py` renders the
sample images through an explicit pinhole camera, the true vanishing points and
the true focal length of every example are known in closed form, so the
estimator can be checked against a number rather than against a screenshot.
`examples/ground_truth.json` carries those values; the tests read it.

What each group checks:

    detection      the detector returns usable structure on every example
    estimation     recovered vanishing points match the rendered ones
    calibration    the focal length recovered from perpendicularity is right
    rectification  rectified line families really do end up axis-aligned
    interaction    deleting lines changes the estimate in the intended way
    robustness     degenerate inputs degrade instead of raising
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import crop, rectify  # noqa: E402
from src.features import FEATURE_NAMES  # noqa: E402
from src.pipeline import MODE_FACADE_A, Session  # noqa: E402
from src.vanishing_points import split_by_orientation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")

with open(os.path.join(EXAMPLES, "ground_truth.json")) as handle:
    GROUND_TRUTH = json.load(handle)

EXAMPLE_NAMES = sorted(GROUND_TRUTH)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def load_session(name: str):
    """Analyse one bundled example and return (session, scale_factor).

    `scale` is what the pipeline's internal downscaling applied, and every
    ground-truth pixel coordinate must be multiplied by it before comparison.
    """
    path = os.path.join(EXAMPLES, name)
    original = cv2.imread(path)
    assert original is not None, f"missing example image: {path}"
    session = Session.from_rgb(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    session.recompute(threshold_deg=2.0)
    return session, session.image_bgr.shape[1] / original.shape[1]


def ray(point_px, image_shape, focal):
    """Unit 3D ray through an image point, given the focal length.

    Vanishing points are compared as *rays*, not as pixel positions: a VP a
    thousand pixels outside the frame and one two thousand pixels outside can
    describe nearly the same 3D direction, so pixel distance would report a
    huge error where the geometry is nearly identical.
    """
    height, width = image_shape[:2]
    vector = np.array(
        [point_px[0] - width / 2.0, point_px[1] - height / 2.0, focal], dtype=float
    )
    return vector / np.linalg.norm(vector)


def ray_error_deg(a, b):
    """Angle between two rays, treating opposite directions as equivalent."""
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0.0, 1.0))))


def line_angle_after(session, index, H):
    """Orientation in degrees (mod 180) of one segment after applying H."""
    endpoints = np.hstack([session.lines.endpoints[index].reshape(2, 2), np.ones((2, 1))])
    warped = endpoints @ H.T
    warped = warped[:, :2] / warped[:, 2:3]
    delta = warped[1] - warped[0]
    return float(np.degrees(np.arctan2(delta[1], delta[0]))) % 180.0


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_detector_finds_enough_structure(name):
    session, _ = load_session(name)
    assert len(session.lines) >= 50, "too few segments to estimate anything"
    assert session.lines.detector != "none"
    # Every segment must yield a usable homogeneous line.
    assert np.all(np.isfinite(session.lines.homog))
    assert np.allclose(np.linalg.norm(session.lines.homog, axis=1), 1.0)


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_features_are_finite_and_shaped(name):
    session, _ = load_session(name)
    assert session.features.shape == (len(session.lines), len(FEATURE_NAMES))
    assert np.all(np.isfinite(session.features))


# --------------------------------------------------------------------------
# Vanishing-point estimation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_horizontal_vanishing_points_match_truth(name):
    session, scale = load_session(name)
    truth = GROUND_TRUTH[name]
    focal = truth["camera"]["focal_px"] * scale
    shape = session.image_bgr.shape

    _, horizontals = split_by_orientation(session.vps)
    estimated = [
        ray(vp.pixel(), shape, focal) for vp in horizontals if vp.pixel() is not None
    ]
    assert estimated, "no horizontal family recovered"

    for expected_px in truth["vp_horizontal"]:
        expected = ray(np.array(expected_px) * scale, shape, focal)
        best = min(ray_error_deg(expected, candidate) for candidate in estimated)
        assert best < 2.0, f"horizontal VP off by {best:.2f} degrees"


@pytest.mark.parametrize(
    "name", [n for n in EXAMPLE_NAMES if "foliage" not in n and "plaza" not in n]
)
def test_vertical_vanishing_point_matches_truth(name):
    """The clean scenes must recover the vertical family accurately.

    The foliage scene is excluded on purpose - getting the vertical family
    wrong there is the failure this project exists to let a user repair, and a
    separate test below checks that pruning repairs it. The plaza scene is
    excluded because its only vertical edges belong to one small block and are
    too short to survive the length filter.
    """
    session, scale = load_session(name)
    truth = GROUND_TRUTH[name]
    focal = truth["camera"]["focal_px"] * scale
    shape = session.image_bgr.shape

    vertical, _ = split_by_orientation(session.vps)
    assert vertical is not None and vertical.pixel() is not None

    expected = ray(np.array(truth["vp_vertical"]) * scale, shape, focal)
    error = ray_error_deg(expected, ray(vertical.pixel(), shape, focal))
    assert error < 2.0, f"vertical VP off by {error:.2f} degrees"


# --------------------------------------------------------------------------
# Self-calibration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_focal_length_recovered_from_perpendicularity(name):
    session, scale = load_session(name)
    expected = GROUND_TRUTH[name]["camera"]["focal_px"] * scale

    if session.focal is None:
        pytest.skip("no perpendicular vanishing-point pair available")
    relative_error = abs(session.focal - expected) / expected
    assert relative_error < 0.10, (
        f"focal length off by {relative_error * 100:.1f}% "
        f"({session.focal:.0f} px vs {expected:.0f} px)"
    )


# --------------------------------------------------------------------------
# Rectification
# --------------------------------------------------------------------------


def test_rectified_families_are_axis_aligned():
    """The defining property: after rectification, the horizontal family runs
    horizontally and the vertical family runs vertically."""
    session, _ = load_session("01_facade_clean.jpg")
    vertical, horizontals = split_by_orientation(session.vps)
    H = rectify.metric_homography(
        horizontals[0].point, vertical.point, session.image_bgr.shape, session.focal
    )
    assert H is not None

    for index in horizontals[0].inliers[:20]:
        angle = line_angle_after(session, index, H)
        assert min(angle, 180 - angle) < 2.0, f"horizontal line ended at {angle:.1f} deg"
    for index in vertical.inliers[:20]:
        angle = line_angle_after(session, index, H)
        assert abs(angle - 90.0) < 2.0, f"vertical line ended at {angle:.1f} deg"


def test_strength_zero_is_the_identity():
    """The strength slider must bottom out at 'do nothing', exactly."""
    session, _ = load_session("01_facade_clean.jpg")
    vertical, horizontals = split_by_orientation(session.vps)
    H = rectify.metric_homography(
        horizontals[0].point, vertical.point, session.image_bgr.shape, session.focal
    )
    damped = rectify.scale_homography(H, 0.0)
    assert np.allclose(damped, np.eye(3), atol=1e-9)
    assert np.allclose(rectify.scale_homography(H, 1.0), H, atol=1e-9)


def test_rectified_output_is_a_sane_image():
    session, _ = load_session("01_facade_clean.jpg")
    result, note = session.rectified_rgb(MODE_FACADE_A, 1.0)
    assert result.ndim == 3 and result.shape[2] == 3
    assert 16 <= result.shape[0] <= 900 and 16 <= result.shape[1] <= 900
    assert result.dtype == np.uint8
    assert "Metric rectification" in note


def test_orientation_guard_undoes_a_mirror():
    """A homography with a mirrored linear part must be flipped back."""
    mirrored = np.diag([1.0, -1.0, 1.0])  # reflects about the x axis
    guarded = rectify.orientation_guard(mirrored, (400, 600))
    jacobian = guarded[:2, :2]
    assert np.linalg.det(jacobian) > 0


# --------------------------------------------------------------------------
# Interaction
# --------------------------------------------------------------------------


def test_delete_and_restore_round_trip():
    session, _ = load_session("01_facade_clean.jpg")
    before = len(session.active_indices)

    session.toggle(0)
    assert len(session.active_indices) == before - 1
    session.toggle(0)
    assert len(session.active_indices) == before
    # A click that hits nothing must be a no-op with an explanation.
    message = session.toggle(None)
    assert "No line" in message and len(session.active_indices) == before


def test_pruning_foliage_repairs_the_vertical_estimate():
    """The project's central claim, as a test.

    In the foliage scene the tree trunks corrupt the vertical family. Here we
    simulate a user deleting the green lines - using the `greenness` feature
    only to *choose* which lines to click, exactly as a person would use their
    eyes - and assert that the vertical vanishing point moves substantially
    closer to the rendered truth.
    """
    name = "02_facade_with_foliage.jpg"
    session, scale = load_session(name)
    truth = GROUND_TRUTH[name]
    focal = truth["camera"]["focal_px"] * scale
    shape = session.image_bgr.shape
    expected = ray(np.array(truth["vp_vertical"]) * scale, shape, focal)

    def vertical_error():
        vertical, _ = split_by_orientation(session.vps)
        if vertical is None or vertical.pixel() is None:
            return 180.0
        return ray_error_deg(expected, ray(vertical.pixel(), shape, focal))

    before = vertical_error()

    greenness = session.features[:, FEATURE_NAMES.index("greenness")]
    for index in np.where(greenness > 0.05)[0]:
        session.toggle(int(index))
    session.recompute(threshold_deg=2.0)
    after = vertical_error()

    assert len(session.deleted) > 5, "the foliage cue selected almost nothing"
    assert after < before, f"pruning made it worse ({before:.2f} -> {after:.2f} deg)"
    assert after < 2.0, f"pruned estimate still off by {after:.2f} degrees"


def test_scorer_trains_and_produces_probabilities():
    session, _ = load_session("02_facade_with_foliage.jpg")
    assert session.score is not None
    assert len(session.score.probabilities) == len(session.lines)
    assert np.all((session.score.probabilities >= 0) & (session.score.probabilities <= 1))
    if session.score.trained:
        assert set(session.score.coefficients) == set(FEATURE_NAMES)


def test_auto_clean_only_removes_low_scoring_lines():
    session, _ = load_session("02_facade_with_foliage.jpg")
    if not session.score.trained:
        pytest.skip("scorer did not train on this image")
    threshold = 0.4
    scores = session.score.probabilities.copy()
    session.auto_clean(threshold)
    assert all(scores[i] < threshold for i in session.deleted)


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


def test_blank_image_does_not_raise():
    """A featureless image has no lines and therefore no vanishing points.
    The pipeline must say so rather than fall over."""
    blank = np.full((300, 400, 3), 200, dtype=np.uint8)
    session = Session.from_rgb(blank)
    session.recompute()
    assert session.vps == []
    result, note = session.rectified_rgb(MODE_FACADE_A, 1.0)
    assert result.shape[:2] == (300, 400)
    assert note  # an explanation, not silence


def test_deleting_every_line_does_not_raise():
    session, _ = load_session("01_facade_clean.jpg")
    session.deleted = set(range(len(session.lines)))
    session.recompute()
    assert session.vps == []
    overlay = session.overlay_rgb(True, True, 0.35)
    assert overlay.shape == session.image_bgr.shape
    assert isinstance(session.diagnostics_markdown(), str)


def test_reset_restores_everything():
    session, _ = load_session("01_facade_clean.jpg")
    session.deleted = {0, 1, 2}
    session.reset()
    session.recompute()
    assert len(session.active_indices) == len(session.lines)


# --------------------------------------------------------------------------
# Whole-family visibility
# --------------------------------------------------------------------------


def test_family_choices_describe_every_family():
    session, _ = load_session("03_street_corner.jpg")
    choices, checked = session.family_choices()
    assert len(choices) == len(session.vps)
    assert checked == [f"live-{i}" for i in range(len(session.vps))]
    # Labels carry the orientation and the support count, which is what makes
    # the control legible without cross-referencing the diagnostics table.
    for (label, _), vp in zip(choices, session.vps):
        assert vp.orientation in label
        assert str(len(vp.inliers)) in label


def test_hiding_a_family_removes_exactly_its_lines():
    session, _ = load_session("03_street_corner.jpg")
    _, checked = session.family_choices()
    target = session.vps[1]
    expected = {int(i) for i in target.inliers}

    session.apply_family_selection([k for k in checked if k != "live-1"])
    assert session.deleted == expected
    assert len(session.hidden_families) == 1


def test_hidden_family_survives_a_recompute_and_can_be_restored():
    """The group numbering changes when a family is removed, so the record has
    to remember line indices rather than a group id. This is that test."""
    session, _ = load_session("03_street_corner.jpg")
    _, checked = session.family_choices()
    before = len(session.active_indices)

    session.apply_family_selection([k for k in checked if k != "live-1"])
    session.recompute(threshold_deg=2.0)
    assert len(session.active_indices) < before

    choices, checked_now = session.family_choices()
    hidden_keys = [key for _, key in choices if key.startswith("hidden-")]
    assert len(hidden_keys) == 1, "the hidden family should still be offered back"

    session.apply_family_selection(checked_now + hidden_keys)
    session.recompute(threshold_deg=2.0)
    assert len(session.active_indices) == before
    assert session.hidden_families == []


def test_restoring_one_line_releases_it_from_its_family_record():
    session, _ = load_session("03_street_corner.jpg")
    _, checked = session.family_choices()
    session.apply_family_selection([k for k in checked if k != "live-1"])

    freed = next(iter(session.hidden_families[0]["indices"]))
    session.toggle(freed)
    assert all(freed not in r["indices"] for r in session.hidden_families)


def test_reset_clears_hidden_families():
    session, _ = load_session("03_street_corner.jpg")
    _, checked = session.family_choices()
    session.apply_family_selection([k for k in checked if k != "live-1"])
    session.reset()
    assert session.hidden_families == []


# --------------------------------------------------------------------------
# Cropping
# --------------------------------------------------------------------------


def test_largest_inscribed_rect_finds_a_known_rectangle():
    """A mask with one obvious rectangle of valid pixels: the search must find
    it, and must not stray outside it."""
    mask = np.zeros((300, 400), dtype=bool)
    mask[40:260, 60:340] = True

    y, x, h, w = crop.largest_inscribed_rect(mask)
    assert mask[y : y + h, x : x + w].all()
    # Within a few percent of the true 280 x 220 region.
    assert h * w > 0.90 * (220 * 280)


def test_crop_respects_the_requested_aspect_ratio():
    mask = np.zeros((300, 400), dtype=bool)
    mask[20:280, 30:370] = True

    for aspect in (1.0, 1.5, 0.75):
        y, x, h, w = crop.largest_inscribed_rect(mask, aspect=aspect)
        assert mask[y : y + h, x : x + w].all()
        assert abs((w / h) - aspect) < 0.05, f"got {w / h:.3f}, wanted {aspect}"


def test_crop_never_includes_an_invalid_pixel_on_a_real_warp():
    """The end-to-end guarantee: whatever the crop returns contains no border.

    We rebuild the validity mask the same way the pipeline does and assert the
    cropped region is entirely inside it.
    """
    session, _ = load_session("03_street_corner.jpg")
    vertical, horizontals = split_by_orientation(session.vps)
    H = rectify.metric_homography(
        horizontals[0].point, vertical.point, session.image_bgr.shape, session.focal
    )
    warped, mask, _ = rectify.warp(session.image_bgr, H, 1.0)
    assert warped.shape[:2] == mask.shape[:2]

    for mode in (crop.CROP_ORIGINAL, crop.CROP_LARGEST):
        rect = crop.largest_inscribed_rect(
            mask,
            aspect=(
                session.image_bgr.shape[1] / session.image_bgr.shape[0]
                if mode == crop.CROP_ORIGINAL
                else None
            ),
        )
        assert rect is not None, f"no crop found for {mode}"
        y, x, h, w = rect
        assert mask[y : y + h, x : x + w].all(), f"{mode} crop includes border pixels"


def test_crop_modes_produce_expected_shapes():
    session, _ = load_session("01_facade_clean.jpg")
    source_aspect = session.image_bgr.shape[1] / session.image_bgr.shape[0]

    uncropped, _ = session.rectified_rgb(MODE_FACADE_A, 1.0, crop.CROP_OFF)
    original, note = session.rectified_rgb(MODE_FACADE_A, 1.0, crop.CROP_ORIGINAL)
    largest, _ = session.rectified_rgb(MODE_FACADE_A, 1.0, crop.CROP_LARGEST)

    assert original.shape[0] <= uncropped.shape[0]
    assert original.shape[1] <= uncropped.shape[1]
    assert abs(original.shape[1] / original.shape[0] - source_aspect) < 0.05
    # Free-ratio cropping can only do better than a constrained one on area.
    assert largest.shape[0] * largest.shape[1] >= original.shape[0] * original.shape[1]
    assert "Cropped to" in note


# --------------------------------------------------------------------------
# Full-resolution export
# --------------------------------------------------------------------------


def _large_session(width=2600):
    """A session whose upload is much bigger than the analysis resolution.

    This is the case that matters: the preview pipeline downscales to 1000 px
    and then warps into a 900 px canvas, so without a separate export path a
    2600-pixel photograph comes back as a small, soft image.
    """
    original = cv2.imread(os.path.join(EXAMPLES, "01_facade_clean.jpg"))
    height = int(original.shape[0] * width / original.shape[1])
    original = cv2.resize(original, (width, height), interpolation=cv2.INTER_CUBIC)
    session = Session.from_rgb(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    session.recompute(threshold_deg=2.0)
    return session, original


def test_export_is_much_larger_than_the_preview():
    session, original = _large_session()
    assert session.original_bgr.shape == original.shape, "the upload must be retained"
    assert session.scale < 0.5, "this test needs a genuinely downscaled analysis image"

    preview, _ = session.rectified_rgb(MODE_FACADE_A, 1.0, crop.CROP_ORIGINAL)
    exported, note = session.export_full_resolution(MODE_FACADE_A, 1.0, crop.CROP_ORIGINAL)

    assert exported is not None
    assert exported.shape[1] > 2 * preview.shape[1], (
        f"export {exported.shape[1]} px wide vs preview {preview.shape[1]} px"
    )
    assert "Exported at" in note


def test_export_frames_the_same_view_as_the_preview():
    """The saved file must match what the user was looking at, not a re-search."""
    session, _ = _large_session()
    for crop_mode in (crop.CROP_ORIGINAL, crop.CROP_LARGEST, crop.CROP_OFF):
        preview, _ = session.rectified_rgb(MODE_FACADE_A, 1.0, crop_mode)
        exported, _ = session.export_full_resolution(MODE_FACADE_A, 1.0, crop_mode)
        preview_aspect = preview.shape[1] / preview.shape[0]
        export_aspect = exported.shape[1] / exported.shape[0]
        assert abs(preview_aspect - export_aspect) < 0.02, (
            f"{crop_mode}: preview {preview_aspect:.3f} vs export {export_aspect:.3f}"
        )


def test_export_keeps_the_original_aspect_ratio_when_asked():
    session, original = _large_session()
    exported, _ = session.export_full_resolution(MODE_FACADE_A, 1.0, crop.CROP_ORIGINAL)
    wanted = original.shape[1] / original.shape[0]
    got = exported.shape[1] / exported.shape[0]
    assert abs(got - wanted) < 0.02, f"got {got:.3f}, wanted {wanted:.3f}"


def test_export_is_sharper_than_upscaling_the_preview():
    """The point of the whole exercise, measured rather than asserted.

    Variance of the Laplacian is a standard focus measure: a soft image has
    little high-frequency energy. Upscaling the preview to the export's size
    adds pixels but no detail, so the real export should score far higher.
    """
    session, _ = _large_session()
    preview_rgb, _ = session.rectified_rgb(MODE_FACADE_A, 1.0, crop.CROP_ORIGINAL)
    exported, _ = session.export_full_resolution(MODE_FACADE_A, 1.0, crop.CROP_ORIGINAL)

    preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    upscaled = cv2.resize(
        preview_bgr, (exported.shape[1], exported.shape[0]), interpolation=cv2.INTER_CUBIC
    )

    def focus(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    assert focus(exported) > 2.0 * focus(upscaled)


def test_save_export_writes_both_formats(tmp_path):
    session, _ = _large_session(width=1600)
    for name in ("out.png", "out.jpg"):
        path = str(tmp_path / name)
        saved, note = session.save_export(path, MODE_FACADE_A, 1.0, crop.CROP_ORIGINAL)
        assert saved == path and os.path.getsize(path) > 1000
        # The written file must be readable back as an image, not just present.
        assert cv2.imread(path) is not None
        assert "Exported at" in note


def test_export_reports_rather_than_crashes_when_no_vps():
    blank = np.full((400, 600, 3), 210, dtype=np.uint8)
    session = Session.from_rgb(blank)
    session.recompute()
    image, note = session.export_full_resolution(MODE_FACADE_A, 1.0)
    assert image is None and note


def test_crop_handles_a_hopeless_mask():
    """Scattered noise contains no usable rectangle; say so, do not crash."""
    rng = np.random.default_rng(0)
    mask = rng.random((120, 160)) > 0.6
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    result, note = crop.apply_crop(image, mask, crop.CROP_ORIGINAL, (120, 160))
    assert result.shape == image.shape or result.size > 0
    assert isinstance(note, str)

    assert crop.largest_inscribed_rect(np.zeros((50, 50), dtype=bool)) is None
