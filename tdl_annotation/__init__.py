"""
Top Down Listening Annotation Tools

A collection of tools for annotating audio clips using spectrograms and audio playback.
Designed for processing data collected using Autonomous Recording Units (ARUs).
"""

from .annotation import (
    plot_clip,
    user_input,
    save_annotations_file,
    load_scores_df,
    annotate,
)
from .annotation_mixit import (
    plot_clip_mixit,
    user_input_mixit,
    save_annotations_file as save_mixit_annotations_file,
    load_scores_df as load_mixit_scores_df,
    annotate_mixit,
)

__version__ = "0.1.0"
__author__ = "Leonardo Viotti"

__all__ = [
    "plot_clip",
    "user_input", 
    "save_annotations_file",
    "load_scores_df",
    "annotate",
    # mixit variants
    "plot_clip_mixit",
    "user_input_mixit",
    "save_mixit_annotations_file",
    "load_mixit_scores_df",
    "annotate_mixit",
] 