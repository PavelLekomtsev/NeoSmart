"""Confidence-threshold sweep for a YOLO detector.

Ultralytics ``model.val()`` gives the classic mAP curve over conf at a
range of IoUs. This module does a complementary job: at a fixed IoU,
it reports precision / recall / F1 at several explicit conf points so
we can pick a *deployment* threshold and show the trade-off on one
plot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neosmart.eval.compare import evaluate_model


@dataclass
class SweepPoint:
    conf: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULT_CONF_GRID = (0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80)


def sweep(
    model_path: str | Path,
    data_dir: str | Path,
    *,
    conf_values: list[float] | None = None,
    iou_thresh: float = 0.5,
) -> list[SweepPoint]:
    """Evaluate the model at each conf value, return the points."""
    values = list(conf_values) if conf_values else list(_DEFAULT_CONF_GRID)
    points: list[SweepPoint] = []
    for c in values:
        r = evaluate_model(
            model_path,
            data_dir,
            conf_thresh=c,
            iou_thresh=iou_thresh,
        )
        points.append(
            SweepPoint(
                conf=float(c),
                precision=r.precision,
                recall=r.recall,
                f1=r.f1,
                tp=r.tp,
                fp=r.fp,
                fn=r.fn,
            )
        )
    return points


def best_conf(points: list[SweepPoint]) -> SweepPoint:
    """Return the point with the highest F1."""
    if not points:
        raise ValueError("no sweep points")
    return max(points, key=lambda p: p.f1)
