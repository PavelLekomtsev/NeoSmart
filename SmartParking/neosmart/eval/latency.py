"""Per-stage latency benchmark for the detector pipeline.

Times YOLO inference on its own and optionally adds the SORT update so
the defense can quote a live-pipeline figure separately from the raw
model forward. Warms up for ``warmup`` frames (discarded) and then
measures ``runs`` frames, reporting mean / p50 / p95 in milliseconds.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class LatencyReport:
    stage: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    n_runs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct * (len(s) - 1)))))
    return s[k]


def _summarize(stage: str, times: list[float]) -> LatencyReport:
    return LatencyReport(
        stage=stage,
        mean_ms=statistics.fmean(times) * 1000 if times else 0.0,
        p50_ms=_percentile(times, 0.5) * 1000,
        p95_ms=_percentile(times, 0.95) * 1000,
        n_runs=len(times),
    )


def benchmark(
    model_path: str | Path,
    image_paths: Sequence[Path],
    *,
    warmup: int = 10,
    runs: int = 100,
    conf: float = 0.5,
    include_sort: bool = False,
) -> list[LatencyReport]:
    """Run the benchmark. Frames are cycled modulo the input list."""
    if not image_paths:
        raise ValueError("image_paths must be non-empty")
    model = YOLO(str(model_path))

    frames = [cv2.imread(str(p)) for p in image_paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise ValueError("no readable frames in image_paths")

    for i in range(warmup):
        model(frames[i % len(frames)], conf=conf, verbose=False)

    yolo_times: list[float] = []
    pipeline_times: list[float] = []
    sort_times: list[float] = []

    sort_tracker = None
    if include_sort:
        from neosmart.tracking.sort import Sort
        sort_tracker = Sort()

    for i in range(runs):
        frame = frames[i % len(frames)]

        t0 = time.perf_counter()
        results = model(frame, conf=conf, verbose=False)
        t_inf = time.perf_counter() - t0
        yolo_times.append(t_inf)

        t1 = time.perf_counter()
        if results and results[0].boxes is not None and len(results[0].boxes):
            pred = np.hstack([
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.conf.cpu().numpy().reshape(-1, 1),
            ])
        else:
            pred = np.zeros((0, 5), dtype=np.float32)
        t_post = time.perf_counter() - t1
        pipeline_times.append(t_inf + t_post)

        if sort_tracker is not None:
            t2 = time.perf_counter()
            sort_tracker.update(pred)
            sort_times.append(time.perf_counter() - t2)

    reports = [
        _summarize("yolo_inference", yolo_times),
        _summarize("yolo_plus_bbox_extract", pipeline_times),
    ]
    if sort_times:
        reports.append(_summarize("sort_update", sort_times))
    return reports
