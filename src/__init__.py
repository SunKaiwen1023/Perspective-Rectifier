"""Explainable perspective rectification - core library.

Four stages, one per module:

    line_detection   find candidate structural segments
    vanishing_points RANSAC estimation of where parallel families converge
    rectify          build and apply the rectifying homography
    suggest          a scorer that learns which segments are structural

`features` supplies the descriptors the scorer consumes, `visualize` draws the
clickable overlay, and `pipeline` holds the session state that ties it all
together. `app.py` at the repository root is the entry point.
"""

__all__ = [
    "line_detection",
    "vanishing_points",
    "rectify",
    "features",
    "suggest",
    "visualize",
    "pipeline",
]
