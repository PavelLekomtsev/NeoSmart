"""Evaluation utilities for the YOLO car detector.

``compare``          — TP/FP/FN + precision/recall/F1 on a labeled split.
``threshold_sweep``  — F1 vs. confidence threshold scan.
``latency``          — per-stage latency benchmark (YOLO + SORT).

Pure libraries: no CLI, no ClearML side effects. The CLI that
orchestrates them is ``Validation/evaluate.py``.
"""

from neosmart.eval.compare import (
    EvaluationResult,
    evaluate_model,
    iou,
    load_gt,
    match_detections,
)
from neosmart.eval.latency import LatencyReport, benchmark
from neosmart.eval.threshold_sweep import SweepPoint, best_conf, sweep

__all__ = [
    "EvaluationResult",
    "LatencyReport",
    "SweepPoint",
    "benchmark",
    "best_conf",
    "evaluate_model",
    "iou",
    "load_gt",
    "match_detections",
    "sweep",
]
