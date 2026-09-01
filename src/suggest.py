"""
A line scorer that learns, during the session, which edges are structural.

The idea
--------
We never ask the user to label a training set. Instead the model bootstraps
itself from two sources that are free:

1. **Self-supervision from geometry.** After the RANSAC stage runs, segments
   that agree tightly with an estimated vanishing point are treated as weak
   positives, and segments that agree with nothing are treated as weak
   negatives. These labels are noisy - that is the whole point, a lucky branch
   gets labelled positive - but there are hundreds of them.

2. **The user's clicks.** Every segment the user deletes is a confident
   negative and every segment they restore is a confident positive. There are
   few of these, so they get a much larger sample weight.

A logistic regression on the descriptors from `features.py` then predicts
P(structural) for *every* segment, including ones the user has not looked at.
That is what makes the interaction feel like more than manual labour: correct
three branches and the model offers to remove the other twelve.

Why logistic regression and not something bigger? With tens to hundreds of
samples and twelve features it is the right capacity, it trains in
milliseconds inside the click loop, and - because the project is about making
an opaque process legible - its coefficients can be read straight off and
shown to the user as "here is what the model thinks matters".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES

# Weight given to a click relative to a geometry-derived pseudo-label. The
# user is far more reliable than RANSAC, but not infinitely so, and a huge
# weight on a handful of points makes the fit unstable.
USER_LABEL_WEIGHT = 8.0
PSEUDO_LABEL_WEIGHT = 1.0


@dataclass
class ScoreResult:
    """Predicted structural probability per segment, plus a model report."""

    probabilities: np.ndarray
    trained: bool = False
    accuracy: float | None = None
    n_positive: int = 0
    n_negative: int = 0
    coefficients: dict = field(default_factory=dict)
    message: str = ""


def build_pseudo_labels(
    n_lines: int,
    vp_inlier_indices,
    deleted: set,
    restored: set,
):
    """Assemble (labels, weights, mask_of_labelled) for the training set.

    `vp_inlier_indices` is the union of every vanishing point's inlier set.
    Segments in it become weak positives; the rest become weak negatives.
    User decisions override both, at a higher weight.
    """
    labels = np.zeros(n_lines, dtype=np.float64)
    weights = np.full(n_lines, PSEUDO_LABEL_WEIGHT, dtype=np.float64)

    inlier_mask = np.zeros(n_lines, dtype=bool)
    if len(vp_inlier_indices) > 0:
        inlier_mask[np.asarray(list(vp_inlier_indices), dtype=int)] = True
    labels[inlier_mask] = 1.0

    for i in deleted:
        if 0 <= i < n_lines:
            labels[i] = 0.0
            weights[i] = USER_LABEL_WEIGHT
    for i in restored:
        if 0 <= i < n_lines:
            labels[i] = 1.0
            weights[i] = USER_LABEL_WEIGHT

    return labels, weights


def score_lines(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    seed: int = 0,
) -> ScoreResult:
    """Fit the scorer and predict P(structural) for every segment.

    Falls back to a neutral score of 0.5 everywhere when the data cannot
    support a fit (too few samples, or only one class present) rather than
    letting an exception reach the interface.
    """
    n = len(features)
    if n == 0:
        return ScoreResult(np.zeros(0), message="No lines to score.")

    n_pos = int((labels > 0.5).sum())
    n_neg = int((labels <= 0.5).sum())
    if n_pos < 4 or n_neg < 4:
        return ScoreResult(
            np.full(n, 0.5),
            n_positive=n_pos,
            n_negative=n_neg,
            message=(
                "Not enough contrast between structural and non-structural "
                "examples yet - delete a few bad lines to teach the model."
            ),
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
    )

    try:
        model.fit(features, (labels > 0.5).astype(int), logisticregression__sample_weight=weights)
        probabilities = model.predict_proba(features)[:, 1]
    except Exception as exc:  # pragma: no cover - defensive
        return ScoreResult(np.full(n, 0.5), message=f"Scorer unavailable: {exc}")

    # Honest generalisation estimate. Cross-validation on self-supervised
    # labels measures "can appearance predict what geometry decided", not
    # ground truth - the UI wording says so explicitly.
    accuracy = None
    if min(n_pos, n_neg) >= 5:
        try:
            folds = int(min(5, n_pos, n_neg))
            scores = cross_val_score(
                model, features, (labels > 0.5).astype(int), cv=folds
            )
            accuracy = float(scores.mean())
        except Exception:
            accuracy = None

    coefficients = {}
    try:
        raw = model.named_steps["logisticregression"].coef_[0]
        coefficients = {
            name: float(value) for name, value in zip(FEATURE_NAMES, raw)
        }
    except Exception:
        pass

    return ScoreResult(
        probabilities=probabilities,
        trained=True,
        accuracy=accuracy,
        n_positive=n_pos,
        n_negative=n_neg,
        coefficients=coefficients,
        message="Scorer trained.",
    )


def top_influences(coefficients: dict, k: int = 4):
    """The k features the model leans on hardest, for the diagnostics panel."""
    if not coefficients:
        return []
    ordered = sorted(coefficients.items(), key=lambda kv: -abs(kv[1]))
    return ordered[:k]
