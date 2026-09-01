"""
Orchestration: holds session state and wires the four stages together.

`app.py` deliberately contains no computer vision. It owns widgets; this file
owns the state machine. Keeping that boundary sharp means the whole pipeline
can be exercised headlessly (see `tests/test_pipeline.py`) without launching a
web server, and it makes the interactive loop easy to reason about:

    load image  ->  detect lines  ->  [ estimate VPs  ->  rectify ]
                                          ^                    |
                                          |                    v
                                     user clicks  <-  annotated overlay

Only the bracketed part re-runs when the user prunes a line, which is why the
interface responds immediately: detection and feature extraction, the two
expensive steps, happen exactly once per uploaded image.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import crop, rectify, suggest, visualize
from .features import compute_features
from .line_detection import detect_lines, resize_for_processing
from .vanishing_points import (
    estimate_vanishing_points,
    horizon_line,
    split_by_orientation,
)

# Names must match the radio choices in `app.py`.
MODE_FACADE_A = "Facade A - vertical x horizontal 1"
MODE_FACADE_B = "Facade B - vertical x horizontal 2"
MODE_GROUND = "Ground plane - horizontal 1 x horizontal 2"
MODE_VERTICAL = "Verticals only (keystone fix)"
RECTIFY_MODES = [MODE_FACADE_A, MODE_FACADE_B, MODE_GROUND, MODE_VERTICAL]


@dataclass
class Session:
    """Everything the interface needs to remember about one image."""

    image_bgr: np.ndarray
    lines: object
    features: np.ndarray

    # The upload at its native resolution, plus the factor that produced
    # `image_bgr` from it. Analysis runs on the small copy for speed; the
    # export re-warps this one so the saved file is not a blown-up preview.
    original_bgr: np.ndarray = None
    scale: float = 1.0
    deleted: set = field(default_factory=set)
    restored: set = field(default_factory=set)

    # Families the user has dismissed wholesale. Each record keeps the exact
    # set of line indices it hid, because the group numbering is *not* stable:
    # removing a family changes what RANSAC finds next time, and the remaining
    # families are renumbered. Remembering indices rather than a group id is
    # what lets a hidden family be brought back later.
    hidden_families: list = field(default_factory=list)
    _hidden_counter: int = 0

    # Filled in by `recompute()`.
    vps: list = field(default_factory=list)
    assignment: np.ndarray = None
    orientations: dict = field(default_factory=dict)
    vp_pixels: dict = field(default_factory=dict)
    horizon: np.ndarray = None
    focal: float = None  # estimated focal length in pixels, if recoverable
    score: object = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_rgb(
        cls,
        image_rgb: np.ndarray,
        min_length_frac: float = 0.025,
        max_lines: int = 300,
        max_side: int = 1000,
    ) -> "Session":
        """Build a session from a Gradio-supplied RGB array."""
        original_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
        image_bgr, scale = resize_for_processing(original_bgr, max_side=max_side)
        lines = detect_lines(image_bgr, min_length_frac=min_length_frac, max_lines=max_lines)
        features = compute_features(image_bgr, lines)
        session = cls(
            image_bgr=image_bgr,
            lines=lines,
            features=features,
            original_bgr=original_bgr,
            scale=scale,
        )
        session.assignment = np.full(len(lines), -1, dtype=int)
        return session

    # ------------------------------------------------------------------
    # Core recompute
    # ------------------------------------------------------------------

    @property
    def active_indices(self) -> np.ndarray:
        """Indices of segments the user has not deleted."""
        return np.array(
            [i for i in range(len(self.lines)) if i not in self.deleted], dtype=int
        )

    def recompute(self, threshold_deg: float = 2.0, max_vps: int = 3) -> None:
        """Re-estimate vanishing points from the surviving segments, then
        re-fit the learned scorer. Cheap enough to run on every click."""
        n = len(self.lines)
        self.assignment = np.full(n, -1, dtype=int)
        self.vps, self.vp_pixels, self.orientations = [], {}, {}
        self.horizon = None
        self.focal = None

        active = self.active_indices
        if len(active) >= 3:
            subset = self.lines.subset(active)
            vps = estimate_vanishing_points(
                subset, threshold_deg=threshold_deg, max_vps=max_vps
            )
            # Map inlier indices from the active subset back to global indices.
            for group_id, vp in enumerate(vps):
                vp.inliers = active[vp.inliers]
                self.assignment[vp.inliers] = group_id
                self.orientations[group_id] = vp.orientation
                pixel = vp.pixel()
                if pixel is not None and np.all(np.isfinite(pixel)):
                    self.vp_pixels[group_id] = (float(pixel[0]), float(pixel[1]))
            self.vps = vps

            _, horizontals = split_by_orientation(vps)
            self.horizon = horizon_line(horizontals)
            self._estimate_focal()

        self._refit_scorer()

    def _estimate_focal(self) -> None:
        """Recover the focal length from every pair of families that ought to
        be mutually perpendicular.

        In a rectangular building the vertical direction is perpendicular to
        both horizontal ones, and the two horizontals are perpendicular to each
        other, so up to three independent estimates are available. We take the
        median: it is robust to one bad vanishing point in a way that a mean is
        not, and with only two or three values that robustness matters.
        """
        vertical, horizontals = split_by_orientation(self.vps)
        pairs = []
        for horizontal in horizontals:
            if vertical is not None:
                pairs.append((vertical.point, horizontal.point))
        if len(horizontals) >= 2:
            pairs.append((horizontals[0].point, horizontals[1].point))

        estimates = [
            rectify.focal_from_orthogonal_vps(a, b, self.image_bgr.shape)
            for a, b in pairs
        ]
        estimates = [f for f in estimates if f is not None]
        self.focal = float(np.median(estimates)) if estimates else None

    def _refit_scorer(self) -> None:
        """Train the appearance model on geometry pseudo-labels + user clicks."""
        inlier_union = set()
        for vp in self.vps:
            inlier_union.update(int(i) for i in vp.inliers)
        labels, weights = suggest.build_pseudo_labels(
            len(self.lines), inlier_union, self.deleted, self.restored
        )
        self.score = suggest.score_lines(self.features, labels, weights)

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------

    def toggle(self, index: int) -> str:
        """Delete a segment, or restore it if it was already deleted."""
        if index is None or not (0 <= index < len(self.lines)):
            return "No line near that click - try clicking directly on a coloured line."
        if index in self.deleted:
            self.deleted.discard(index)
            self.restored.add(index)
            # If this line was hidden as part of a family, that family is no
            # longer intact; drop the line from its record so the family
            # checkbox never claims to control a line it no longer owns.
            self._forget_from_families(index)
            return f"Restored line #{index}."
        self.deleted.add(index)
        self.restored.discard(index)
        return f"Deleted line #{index}."

    def _forget_from_families(self, index: int) -> None:
        for record in self.hidden_families:
            record["indices"].discard(index)
        self.hidden_families = [r for r in self.hidden_families if r["indices"]]

    # ------------------------------------------------------------------
    # Whole-family visibility
    # ------------------------------------------------------------------

    def family_choices(self):
        """`(choices, checked)` for the interface's family checkbox group.

        `choices` is a list of `(label, key)` pairs: the keys are what the
        interface sends back, the labels are what the user reads. Live families
        are keyed by position and start checked; families the user has hidden
        keep their own stable key and stay unchecked until re-ticked.
        """
        choices, checked = [], []
        trained = self.score is not None and self.score.trained
        for group_id, vp in enumerate(self.vps):
            key = f"live-{group_id}"
            label = f"VP{group_id + 1} - {vp.orientation} - {len(vp.inliers)} lines"
            if trained and len(vp.inliers):
                # The scorer's average confidence in this family, shown on the
                # label so a family built out of foliage is visible as a number
                # before the user has to squint at the overlay.
                confidence = float(self.score.probabilities[vp.inliers].mean())
                label += f" - {confidence * 100:.0f}% structural"
            choices.append((label, key))
            checked.append(key)
        for record in self.hidden_families:
            choices.append((f"hidden: {record['label']}", record["key"]))
        return choices, checked

    def apply_family_selection(self, selected) -> str:
        """Hide or restore whole families to match the checkbox state.

        Unticking a live family deletes every line that voted for it in one
        action; re-ticking a hidden one brings exactly those lines back. This
        is the bulk counterpart to clicking individual lines, and it is the
        fast way to dismiss a family that is wrong in its entirety - a set of
        tree trunks that RANSAC has mistaken for the building's verticals, say.
        """
        selected = set(selected or [])
        messages = []

        for group_id, vp in enumerate(self.vps):
            if f"live-{group_id}" in selected:
                continue
            indices = {int(i) for i in vp.inliers}
            if not indices:
                continue
            self._hidden_counter += 1
            record = {
                "key": f"hidden-{self._hidden_counter}",
                "label": f"{vp.orientation} family ({len(indices)} lines)",
                "indices": indices,
            }
            self.hidden_families.append(record)
            self.deleted.update(indices)
            self.restored.difference_update(indices)
            messages.append(f"Hid the {record['label']}.")

        still_hidden = []
        for record in self.hidden_families:
            if record["key"] in selected:
                self.deleted.difference_update(record["indices"])
                messages.append(f"Restored the {record['label']}.")
            else:
                still_hidden.append(record)
        self.hidden_families = still_hidden

        return " ".join(messages)

    def auto_clean(self, threshold: float) -> str:
        """Delete every segment the scorer rates below `threshold`.

        This is the payoff of the learned scorer: a few manual corrections
        generalise to the rest of the image in one click.
        """
        if self.score is None or not self.score.trained:
            return "The scorer needs a few manual deletions before it can generalise."
        candidates = {
            i for i, p in enumerate(self.score.probabilities)
            if p < threshold and i not in self.deleted
        }
        self.deleted.update(candidates)
        return f"Auto-removed {len(candidates)} suspect line(s) at threshold {threshold:.2f}."

    def reset(self) -> str:
        self.deleted.clear()
        self.restored.clear()
        self.hidden_families.clear()
        return "Restored all detected lines."

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def overlay_rgb(self, show_rays: bool, show_vps: bool, suspicion_threshold: float):
        suspicion = self.score.probabilities if self.score is not None else None
        canvas = visualize.draw_overlay(
            self.image_bgr,
            self.lines,
            self.assignment,
            self.orientations,
            self.vp_pixels,
            self.deleted,
            suspicion=suspicion,
            show_rays=show_rays,
            show_vps=show_vps,
            horizon=self.horizon,
            suspicion_threshold=suspicion_threshold,
        )
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def _vps_for_mode(self, mode: str):
        """Resolve a mode name to the vanishing points it needs.

        Returns `(vp_a, vp_b, note)` as `VanishingPoint` objects; `vp_b` is
        None for the vertical-only mode. `note` explains any fallback, so the
        interface can say what actually happened rather than silently doing
        something different from what was asked.
        """
        vertical, horizontals = split_by_orientation(self.vps)

        if mode == MODE_VERTICAL:
            if vertical is None:
                return None, None, "No vertical family found - cannot fix keystoning."
            return vertical, None, ""

        if mode == MODE_GROUND:
            if len(horizontals) < 2:
                return None, None, "Need two horizontal families for a ground-plane view."
            return (
                horizontals[0],
                horizontals[1],
                "Top-down view: only surfaces genuinely on the ground plane come "
                "out flat. Walls stretch toward infinity, which is correct rather "
                "than a bug - this mode suits photos where paving or a floor "
                "fills most of the frame.",
            )

        want = 0 if mode == MODE_FACADE_A else 1
        if vertical is None:
            return None, None, "No vertical family found - try lowering the angle tolerance."
        if not horizontals:
            return None, None, "No horizontal family found for a facade."
        if len(horizontals) <= want:
            return (
                horizontals[0],
                vertical,
                "Only one horizontal family available - used it instead.",
            )
        return horizontals[want], vertical, ""

    def _anchor_points(self, *vps) -> np.ndarray:
        """Endpoints of the segments supporting the given families.

        These are handed to the canvas fit so the output frames the structure
        that was actually rectified, rather than the sky and road that the warp
        flings toward infinity. The ground mode needs a different rule and uses
        `_ground_anchors` instead.
        """
        indices = np.concatenate(
            [vp.inliers for vp in vps if vp is not None] or [np.array([], dtype=int)]
        )
        if len(indices) == 0:
            return np.zeros((0, 2))

        segments = self.lines.endpoints[indices]
        return np.vstack([segments[:, 0:2], segments[:, 2:4]])

    def _ground_anchors(self, horizon_margin: float = 0.10) -> np.ndarray:
        """Sample the image region that could plausibly *be* the ground plane.

        The ground mode cannot anchor on line segments the way the facade modes
        do, because walls share their horizontal families with the pavement,
        and a wall is not on the ground plane - under a top-down warp it
        stretches to infinity and hijacks the framing. Anchoring on a plain
        grid over the area below the horizon frames the visible ground instead.

        Points closer to the horizon than `horizon_margin` (as a fraction of
        image height) are dropped: they correspond to ground arbitrarily far
        away, and a handful of them would set the canvas extent on their own.
        """
        height, width = self.image_bgr.shape[:2]
        ys, xs = np.mgrid[0:25, 0:25]
        grid = np.stack(
            [xs.ravel() / 24.0 * (width - 1), ys.ravel() / 24.0 * (height - 1)], axis=1
        )
        if self.horizon is None:
            return grid[grid[:, 1] > 0.4 * height]

        a, b, c = self.horizon
        bottom = np.array([width / 2.0, height - 1.0, 1.0])
        sign = np.sign(self.horizon @ bottom)
        signed = a * grid[:, 0] + b * grid[:, 1] + c
        keep = (np.sign(signed) == sign) & (np.abs(signed) > horizon_margin * height)
        return grid[keep] if keep.sum() >= 8 else grid[grid[:, 1] > 0.4 * height]

    def _homography_for_mode(self, mode: str):
        """Resolve a mode to `(H, anchors, note)` in analysis-image coordinates.

        Shared by the interactive preview and the full-resolution export so the
        two cannot drift apart: whatever transform you are looking at is the
        transform that gets saved. `H` is None when the mode cannot be served,
        and `note` says why.
        """
        vp_a, vp_b, note = self._vps_for_mode(mode)
        if vp_a is None:
            return None, None, note

        if vp_b is None:
            # Keystone correction is meant to leave the framing alone, so the
            # canvas fits the whole picture rather than a cropped facade.
            H = rectify.homography_vertical_only(vp_a.point, self.image_bgr.shape)
            anchors = None
        else:
            # Prefer the calibrated route: it preserves the plane's true
            # proportions. Fall back to the uncalibrated algebraic
            # construction, which still removes the perspective distortion but
            # leaves an unknown aspect ratio, when the focal length could not
            # be recovered.
            H = None
            if self.focal is not None:
                H = rectify.metric_homography(
                    vp_a.point, vp_b.point, self.image_bgr.shape, self.focal
                )
                if H is not None:
                    note = (note + " Metric rectification using the recovered "
                            f"focal length ({self.focal:.0f} px).").strip()
            if H is None:
                H = rectify.homography_from_two_vps(
                    vp_a.point, vp_b.point, self.image_bgr.shape
                )
                if H is not None:
                    note = (note + " Focal length not recoverable - aspect ratio "
                            "of the result is not metric.").strip()
            anchors = (
                self._ground_anchors()
                if mode == MODE_GROUND
                else self._anchor_points(vp_a, vp_b)
            )
        if H is None:
            note = "Those vanishing points are degenerate - prune a few lines and retry."
        return H, anchors, note

    def rectified_rgb(
        self,
        mode: str,
        strength: float,
        crop_mode: str = crop.CROP_ORIGINAL,
        max_side: int = 900,
    ):
        """Warped, cropped preview image plus a short status note.

        Deliberately rendered small: this runs on every click, so it trades
        resolution for responsiveness. `export_full_resolution` produces the
        version worth saving.
        """
        H, anchors, note = self._homography_for_mode(mode)
        if H is None:
            return cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB), note

        warped, mask, _ = rectify.warp(
            self.image_bgr, H, strength=strength, max_side=max_side, anchors=anchors
        )
        warped, crop_note = crop.apply_crop(warped, mask, crop_mode, self.image_bgr.shape)
        note = " ".join(part for part in (note, crop_note) if part)
        return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), note

    # ------------------------------------------------------------------
    # Full-resolution export
    # ------------------------------------------------------------------

    def export_full_resolution(
        self,
        mode: str,
        strength: float,
        crop_mode: str = crop.CROP_ORIGINAL,
        preview_side: int = 900,
        max_side: int = 4000,
    ):
        """Re-warp the *original* upload at full resolution. Returns `(bgr, note)`.

        The preview pipeline works on a copy downscaled to at most 1000 px and
        then warps into a canvas of at most 900 px, so a 4000-pixel photograph
        loses most of its detail twice over before it is ever displayed. That
        is the right trade for something that recomputes on every click, and
        the wrong one for a file you intend to keep.

        The fix does not re-estimate anything. If `S` is the downscaling that
        produced the analysis image and `H` is the homography found in that
        frame, then a point of the original maps to the preview canvas by
        `H . S`, and scaling the canvas back up by `k` gives

            H_export = diag(k, k, 1) . H . S

        which warps the untouched original directly into a canvas `k` times the
        preview's size. Choosing `k = 1 / scale` restores the original pixel
        density. The crop rectangle is scaled by the same factor rather than
        searched again, so the exported file is framed exactly like the preview.
        """
        if self.original_bgr is None:
            return None, "No original image retained for this session."

        H, anchors, note = self._homography_for_mode(mode)
        if H is None:
            return None, note

        # The preview warp, purely to obtain the fitted canvas and its mask.
        _, preview_mask, H_preview = rectify.warp(
            self.image_bgr, H, strength=strength, max_side=preview_side, anchors=anchors
        )
        preview_h, preview_w = preview_mask.shape[:2]

        # Scale factor from the preview canvas to the export canvas, capped so
        # a very large upload cannot ask for an unreasonable allocation.
        k = 1.0 / max(self.scale, 1e-6)
        k = min(k, max_side / max(preview_w, preview_h))
        k = max(k, 1.0)

        S = np.diag([self.scale, self.scale, 1.0])
        K = np.diag([k, k, 1.0])
        H_export = K @ H_preview @ S

        out_w = int(round(preview_w * k))
        out_h = int(round(preview_h * k))

        try:
            warped = cv2.warpPerspective(
                self.original_bgr,
                H_export,
                (out_w, out_h),
                # Cubic rather than linear: this is the copy someone keeps, and
                # the extra cost is paid once instead of on every click.
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(24, 24, 24),
            )
        except (cv2.error, MemoryError) as exc:
            return None, f"Could not render at full resolution ({exc})."

        rect = crop.scale_rect(
            crop.crop_rect(preview_mask, crop_mode, self.image_bgr.shape),
            k,
            warped.shape,
        )
        if rect is not None:
            y, x, h, w = rect
            warped = warped[y : y + h, x : x + w]

        source = f"{self.original_bgr.shape[1]}x{self.original_bgr.shape[0]}"
        note = (
            f"{note} Exported at {warped.shape[1]}x{warped.shape[0]} px "
            f"from the {source} px original."
        ).strip()
        return warped, note

    def save_export(
        self,
        path: str,
        mode: str,
        strength: float,
        crop_mode: str = crop.CROP_ORIGINAL,
        jpeg_quality: int = 95,
    ):
        """Write the full-resolution result to `path`. Returns `(path, note)`.

        The file extension chooses the encoder: `.png` for a lossless copy,
        `.jpg` for a smaller one. Anything else is written as PNG.
        """
        image, note = self.export_full_resolution(mode, strength, crop_mode)
        if image is None:
            return None, note

        lower = path.lower()
        if lower.endswith((".jpg", ".jpeg")):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        else:
            params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
        if not cv2.imwrite(path, image, params):
            return None, "Could not write the exported file."
        return path, note

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics_markdown(self) -> str:
        """A plain-language report of every internal quantity worth seeing."""
        total = len(self.lines)
        kept = total - len(self.deleted)
        assigned = int((self.assignment >= 0).sum())

        rows = [
            f"**Detector:** `{self.lines.detector}` &nbsp;|&nbsp; "
            f"**Segments:** {total} detected, {kept} kept, {len(self.deleted)} deleted",
            f"**Assigned to a vanishing point:** {assigned} / {kept}",
            "",
            "| Family | Orientation | Support | Mean error | Location (px) |",
            "|---|---|---|---|---|",
        ]
        for group_id, vp in enumerate(self.vps):
            pixel = self.vp_pixels.get(group_id)
            location = (
                f"({pixel[0]:.0f}, {pixel[1]:.0f})" if pixel else "at infinity"
            )
            rows.append(
                f"| VP{group_id + 1} | {vp.orientation} | {len(vp.inliers)} lines "
                f"| {vp.mean_error_deg:.2f}&deg; | {location} |"
            )
        if not self.vps:
            rows.append("| _none found_ | | | | |")

        extras = []
        if self.horizon is not None:
            tilt = np.degrees(np.arctan2(-self.horizon[0], self.horizon[1]))
            extras.append(f"**Horizon tilt:** {tilt:+.2f}&deg; from level")
        if self.focal is not None:
            # Horizontal field of view implied by the recovered focal length -
            # a far more intuitive number than the focal length itself.
            fov = 2 * np.degrees(np.arctan(self.image_bgr.shape[1] / (2 * self.focal)))
            extras.append(
                f"**Recovered camera:** focal length {self.focal:.0f} px "
                f"({fov:.0f}&deg; horizontal field of view), inferred purely from "
                "the perpendicularity of the vanishing points"
            )
        if extras:
            rows += [""] + extras

        if self.score is not None:
            rows.append("")
            if self.score.trained:
                accuracy = (
                    f"{self.score.accuracy * 100:.0f}%"
                    if self.score.accuracy is not None
                    else "n/a"
                )
                rows.append(
                    f"**Line scorer:** trained on {self.score.n_positive} positive / "
                    f"{self.score.n_negative} negative examples &nbsp;|&nbsp; "
                    f"cross-validated agreement: {accuracy}"
                )
                influences = suggest.top_influences(self.score.coefficients)
                if influences:
                    pretty = ", ".join(
                        f"`{name}` ({value:+.2f})" for name, value in influences
                    )
                    rows.append(f"**Most influential cues:** {pretty}")
                rows.append(
                    "<sub>Agreement is measured against the geometry stage's own "
                    "verdicts plus your clicks, not against ground truth - read it "
                    "as 'appearance can predict this', not 'this is correct'.</sub>"
                )
            else:
                rows.append(f"**Line scorer:** {self.score.message}")

        return "\n".join(rows)
