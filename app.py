"""
Explainable Perspective Rectifier - interactive interface.

Run me:  python app.py       then open the printed http://127.0.0.1:7860 URL.

This file contains no computer vision. It builds the Gradio interface and
translates widget events into calls on `src.pipeline.Session`, which owns all
of the state and all of the maths. Keeping that split means the algorithm can
be tested without a browser (see `tests/`) and that reading this file tells you
what the user can *do*, not how any of it works.

Interaction model
-----------------
    upload / pick an example   ->  lines are detected and vanishing points
                                   estimated automatically
    click a line               ->  that line is deleted (click again to restore
                                   it); everything downstream recomputes
    untick a family            ->  every line that voted for one vanishing
                                   point is dismissed in a single action
    "Auto-remove suspect lines"->  applies the learned scorer's opinion to
                                   every line at once
    the sliders and radios     ->  re-run the relevant stage, or just redraw

Every handler returns the same six outputs, so they all share the `_render`
helper rather than each assembling the tuple by hand. The three tiers matter
for responsiveness: `analyze` re-detects lines (expensive), `recompute`
re-estimates vanishing points (cheap), and `redraw` only re-warps (cheapest).
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile

# Gradio phones home for usage analytics on startup. Disabling it before the
# import keeps a fresh clone from hanging on a network call when the machine is
# offline or behind a restrictive proxy - a reproducibility issue, not a
# privacy stance.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np

from src.crop import CROP_MODES, CROP_ORIGINAL
from src.pipeline import RECTIFY_MODES, Session
from src.visualize import legend_markdown, nearest_line

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(HERE, "examples")

# Defaults chosen by measuring vanishing-point error against the known ground
# truth of the bundled renders - see `tests/test_pipeline.py`. A longer minimum
# length loses the short vertical window edges, and without those the vertical
# family collapses.
DEFAULT_MIN_LENGTH_FRAC = 0.025
DEFAULT_MAX_LINES = 300
DEFAULT_ANGLE_TOL = 2.0

INTRO = """
# Explainable Perspective Rectifier

Automatic perspective correction usually works as a black box: it decides which
edges matter, and if it decides wrong you get a bent building and no
explanation. This tool shows its work instead.

It finds straight edges, groups them into families that converge on a shared
**vanishing point**, and uses those points to compute the transform that makes
the building's real right angles right again. Every step is drawn on the image,
and **any line you disagree with can be deleted with a click** - or a whole
family dismissed with one tick. The estimate recomputes instantly, and a small
model watches which lines you reject and learns to flag the rest.

*Start by picking one of the examples below, or upload your own photo.*
"""

HOW_TO = """
### How to use it

1. **Pick an example or upload a photo.** Buildings shot from below or at an
   angle work best. Analysis runs automatically.
2. **Read the overlay.** Coloured lines are edges that agreed on a vanishing
   point; grey lines agreed with nothing. Faint rays show where each family
   converges. Red circles mark lines the learned scorer distrusts.
3. **Click any line to delete it.** Tree branches, shadows and reflections are
   the usual culprits. Click a dashed (deleted) line to bring it back.
4. **Untick a family** under the overlay to dismiss every line that voted for
   one vanishing point at once - the quick fix when a whole family is wrong.
   Tick it again to bring the same lines back.
5. **Choose what to rectify**, how to crop the result, and drag the strength
   slider to taste.
6. **Press "Render full resolution"** to save the result. The preview redraws
   on every click, so it is deliberately small; this step re-warps your
   original upload at its native resolution and gives you a PNG or JPEG.

The panel under the result reports every internal quantity, including how far
each family's lines actually miss its vanishing point.
"""


# --------------------------------------------------------------------------
# Shared rendering
# --------------------------------------------------------------------------


def _empty():
    """The output tuple for "there is no session yet"."""
    return None, None, "", "Upload an image to begin.", gr.update(choices=[], value=[])


def _render(session, mode, strength, crop_mode, show_rays, show_vps, suspicion,
            status=""):
    """Produce the five non-state outputs every handler returns."""
    if session is None:
        return _empty()
    overlay = session.overlay_rgb(show_rays, show_vps, suspicion)
    result, note = session.rectified_rgb(mode, strength, crop_mode)
    message = " ".join(part for part in (status, note) if part)
    choices, checked = session.family_choices()
    return (
        overlay,
        result,
        session.diagnostics_markdown(),
        message,
        gr.update(choices=choices, value=checked),
    )


# --------------------------------------------------------------------------
# Handlers, in order of cost
# --------------------------------------------------------------------------


def analyze(image, min_length_frac, max_lines, angle_tol, *view):
    """Full analysis of a freshly supplied image. Re-detects everything."""
    if image is None:
        return (None, *_empty())
    session = Session.from_rgb(
        np.asarray(image),
        min_length_frac=float(min_length_frac),
        max_lines=int(max_lines),
    )
    session.recompute(threshold_deg=float(angle_tol))
    status = (
        f"Detected {len(session.lines)} line segments with "
        f"{session.lines.detector}. Click any line to remove it."
    )
    return (session, *_render(session, *view, status=status))


def recompute(session, angle_tol, *view):
    """Re-run vanishing-point estimation without re-detecting lines."""
    if session is None:
        return (None, *_empty())
    session.recompute(threshold_deg=float(angle_tol))
    return (session, *_render(session, *view))


def redraw(session, *view):
    """Cheapest path: nothing is re-estimated, only redrawn or re-warped."""
    if session is None:
        return (None, *_empty())
    return (session, *_render(session, *view))


def on_click(session, angle_tol, mode, strength, crop_mode, show_rays, show_vps,
             suspicion, event: gr.SelectData):
    """Delete (or restore) the line nearest the click, then re-estimate.

    This is the only handler that spells its view arguments out instead of
    collecting them with `*view`: Gradio appends the click's event data as the
    final positional argument, and a `*args` parameter would swallow it.
    """
    if session is None:
        return (None, *_empty())
    view = (mode, strength, crop_mode, show_rays, show_vps, suspicion)
    index = nearest_line(session.lines, event.index[0], event.index[1])
    status = session.toggle(index)
    session.recompute(threshold_deg=float(angle_tol))
    return (session, *_render(session, *view, status=status))


def on_families(session, selected, angle_tol, *view):
    """Hide or restore whole vanishing-point families from the checkbox group."""
    if session is None:
        return (None, *_empty())
    status = session.apply_family_selection(selected)
    session.recompute(threshold_deg=float(angle_tol))
    return (session, *_render(session, *view, status=status))


def on_auto_clean(session, threshold, angle_tol, *view):
    if session is None:
        return (None, *_empty())
    status = session.auto_clean(float(threshold))
    session.recompute(threshold_deg=float(angle_tol))
    return (session, *_render(session, *view, status=status))


def on_reset(session, angle_tol, *view):
    if session is None:
        return (None, *_empty())
    status = session.reset()
    session.recompute(threshold_deg=float(angle_tol))
    return (session, *_render(session, *view, status=status))


def on_export(session, mode, strength, crop_mode, file_format):
    """Render the full-resolution result and hand back a file to download.

    The preview panels are small on purpose - they re-render on every click -
    so this is a separate, deliberate step rather than something that happens
    automatically. It re-warps the original upload rather than upscaling what
    is on screen.
    """
    if session is None:
        return None, "Upload an image first."

    extension = "jpg" if file_format.lower().startswith("jpeg") else "png"
    slug = re.sub(r"[^a-z0-9]+", "-", mode.lower()).strip("-")[:40]
    # A fresh directory per export: Gradio serves files by path and would
    # happily hand back a cached copy if we kept overwriting one name.
    path = os.path.join(tempfile.mkdtemp(prefix="rectified-"), f"rectified-{slug}.{extension}")

    saved, note = session.save_export(path, mode, float(strength), crop_mode)
    if saved is None:
        return None, note
    size_mb = os.path.getsize(saved) / 1e6
    return saved, f"{note} Saved as {extension.upper()} ({size_mb:.1f} MB)."


def _on_release(component):
    """Slider drag-end event, falling back to `.change` on older Gradio.

    `Slider.release` fires once when the user lets go, which is what we want:
    `.change` fires on every pixel of the drag and would re-run RANSAC dozens
    of times per gesture.
    """
    return component.release if hasattr(component, "release") else component.change


def _on_input(component):
    """User-initiated change only.

    This matters for the family checkboxes: every handler rewrites their
    choices and values, and `.change` fires on programmatic updates too, so
    wiring the handler to `.change` would make the interface retrigger itself
    endlessly. `.input` fires only when a person clicks.
    """
    return component.input if hasattr(component, "input") else component.change


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    example_files = sorted(
        os.path.join(EXAMPLE_DIR, name)
        for name in os.listdir(EXAMPLE_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    with gr.Blocks(title="Explainable Perspective Rectifier") as demo:
        session = gr.State(None)
        gr.Markdown(INTRO)

        with gr.Row():
            # ---------------- controls ------------------------------------
            with gr.Column(scale=3, min_width=280):
                image_input = gr.Image(
                    label="Input photo", type="numpy", height=220, sources=["upload"]
                )
                if example_files:
                    gr.Examples(
                        examples=[[path] for path in example_files],
                        inputs=[image_input],
                        label="Examples (the second one is deliberately hard)",
                    )

                mode = gr.Radio(
                    RECTIFY_MODES,
                    value=RECTIFY_MODES[0],
                    label="What to rectify",
                    info="Which pair of vanishing points defines the plane to flatten.",
                )
                strength = gr.Slider(
                    0.0, 1.0, value=1.0, step=0.05,
                    label="Correction strength",
                    info="0 leaves the photo alone, 1 applies the full transform.",
                )
                crop_mode = gr.Radio(
                    CROP_MODES,
                    value=CROP_ORIGINAL,
                    label="Crop",
                    info="A perspective warp turns the rectangular frame into a "
                         "quadrilateral, leaving empty corners. Cropping takes the "
                         "largest rectangle of real pixels; content at the edges "
                         "is lost.",
                )
                angle_tol = gr.Slider(
                    0.5, 6.0, value=DEFAULT_ANGLE_TOL, step=0.1,
                    label="Vanishing-point tolerance (degrees)",
                    info="How far a line may point away from a vanishing point and "
                         "still count as a member of that family.",
                )

                with gr.Accordion("Line detection settings", open=False):
                    min_length = gr.Slider(
                        0.010, 0.080, value=DEFAULT_MIN_LENGTH_FRAC, step=0.005,
                        label="Minimum line length (fraction of image diagonal)",
                        info="Raise it for cleaner lines, lower it to catch short "
                             "window edges - the vertical family depends on those.",
                    )
                    max_lines = gr.Slider(
                        50, 400, value=DEFAULT_MAX_LINES, step=10,
                        label="Maximum number of lines",
                    )
                    redetect = gr.Button("Re-detect lines")

                with gr.Accordion("Learned line scorer", open=True):
                    gr.Markdown(
                        "A logistic-regression model trained live on the geometry "
                        "stage's own verdicts **plus every line you delete**. It "
                        "generalises your corrections to lines you have not "
                        "inspected."
                    )
                    suspicion = gr.Slider(
                        0.05, 0.95, value=0.35, step=0.05,
                        label="Suspicion threshold",
                        info="Lines scoring below this are circled in red.",
                    )
                    with gr.Row():
                        auto_button = gr.Button("Auto-remove suspect lines")
                        reset_button = gr.Button("Restore all lines")

                with gr.Row():
                    show_rays = gr.Checkbox(True, label="Perspective rays")
                    show_vps = gr.Checkbox(True, label="Vanishing points")

            # ---------------- interactive overlay --------------------------
            with gr.Column(scale=5, min_width=380):
                overlay = gr.Image(
                    label="Detected structure - click a line to delete it",
                    type="numpy",
                    interactive=False,
                    height=460,
                    format="png",
                )
                families = gr.CheckboxGroup(
                    choices=[], value=[],
                    label="Line families - untick one to dismiss all of its lines",
                    info="Each family is the set of lines that agreed on one "
                         "vanishing point, with the scorer's average confidence "
                         "in it. Unticking hides them all at once and ticking a "
                         "hidden family brings exactly those lines back - but it "
                         "is a blunt instrument: if a family mixes real edges "
                         "with clutter, click the bad lines individually instead.",
                )
                gr.Markdown(legend_markdown())
                status = gr.Markdown("Upload an image to begin.")

            # ---------------- result ---------------------------------------
            with gr.Column(scale=4, min_width=320):
                # format="png" matters: Gradio's own download button defaults to
                # webp, which many image editors will not open.
                result = gr.Image(
                    label="Rectified result (preview)",
                    type="numpy",
                    height=460,
                    format="png",
                )
                with gr.Accordion("Save the result", open=True):
                    gr.Markdown(
                        "The preview above is rendered small so it can redraw on "
                        "every click. This re-warps your **original upload** at "
                        "full resolution instead of enlarging the preview."
                    )
                    with gr.Row():
                        file_format = gr.Radio(
                            ["PNG (lossless)", "JPEG (smaller)"],
                            value="PNG (lossless)",
                            label="File format",
                            scale=2,
                        )
                        export_button = gr.Button(
                            "Render full resolution", variant="primary", scale=1
                        )
                    download = gr.File(label="Download", interactive=False)
                with gr.Accordion("What the system computed", open=True):
                    diagnostics = gr.Markdown()

        with gr.Accordion("How to use this", open=False):
            gr.Markdown(HOW_TO)

        # ------------------ wiring ----------------------------------------
        # Every handler produces this same tuple, in this order.
        outputs = [session, overlay, result, diagnostics, status, families]
        # …and takes these as its trailing arguments, in this order.
        view_controls = [mode, strength, crop_mode, show_rays, show_vps, suspicion]

        analyze_inputs = [image_input, min_length, max_lines, angle_tol, *view_controls]
        image_input.change(analyze, analyze_inputs, outputs)
        redetect.click(analyze, analyze_inputs, outputs)

        # Changing the tolerance re-runs RANSAC; changing a view control only
        # redraws. Separating them keeps the interface feeling instant.
        _on_release(angle_tol)(recompute, [session, angle_tol, *view_controls], outputs)
        for control in (mode, crop_mode, show_rays, show_vps):
            control.change(redraw, [session, *view_controls], outputs)
        for control in (strength, suspicion):
            _on_release(control)(redraw, [session, *view_controls], outputs)

        overlay.select(on_click, [session, angle_tol, *view_controls], outputs)
        _on_input(families)(
            on_families, [session, families, angle_tol, *view_controls], outputs
        )
        auto_button.click(
            on_auto_clean, [session, suspicion, angle_tol, *view_controls], outputs
        )
        reset_button.click(on_reset, [session, angle_tol, *view_controls], outputs)

        # The export has its own, smaller output set: it produces a file and a
        # status line, and changes no session state.
        export_button.click(
            on_export,
            [session, mode, strength, crop_mode, file_format],
            [download, status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Explainable Perspective Rectifier")
    parser.add_argument("--port", type=int, default=7860, help="port to serve on")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    parser.add_argument(
        "--share", action="store_true", help="create a temporary public Gradio link"
    )
    args = parser.parse_args()

    # Only the widely-supported launch arguments are passed, so the same
    # command works across Gradio 4, 5 and 6.
    build_interface().launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
