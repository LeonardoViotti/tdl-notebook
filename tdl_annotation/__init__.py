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

__version__ = "0.1.0"
__author__ = "Leonardo Viotti"

__all__ = [
    "plot_clip",
    "user_input", 
    "save_annotations_file",
    "load_scores_df",
    "annotate",
] 