"""VoxTraceAI — Contrastive Speech Analytics & Temporal Flaw Grounding."""
from .pipeline import analyze_files, analyze_arrays
from . import features, alignment, grounding, scoring, dataset, utils

__version__ = "0.1.0"
__all__ = ["analyze_files", "analyze_arrays", "features", "alignment", "grounding", "scoring", "dataset", "utils"]
