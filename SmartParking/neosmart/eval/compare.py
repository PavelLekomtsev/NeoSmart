"""YOLO detector evaluation on a labeled split.

TP/FP/FN counting at a fixed IoU threshold, per-image detail, and
derived precision/recall/F1. No ClearML, no plotting — orchestration
lives in ``Validation/evaluate.py``.

Matching algorithm (classic greedy confidence-sorted):
  1. For each image build an IoU matrix between predictions and GT.
  2. Sort predictions by confidence (descending).
  3. Each prediction grabs the highest-IoU unmatched GT.
  4. IoU >= ``iou_thresh`` becomes TP, otherwise FP.
  5. GT left unmatched becomes FN.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class EvaluationResult:
    model_name: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    confidences: list[float] = field(default_factory=list)
    per_image: list[dict[str, Any]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def total_gt(self) -> int:
        return self.tp + self.fn

    @property
    def total_pred(self) -> int:
        return self.tp + self.fp

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "total_gt": self.total_gt,
            "total_pred": self.total_pred,
        }


def iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Axis-aligned bbox IoU. Boxes are (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_file: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load a YOLO-format label file → Nx4 array of pixel-space xyxy."""
    if not label_file.exists():
        return np.zeros((0, 4), dtype=np.float32)
    boxes: list[list[float]] = []
    with label_file.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = (float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]))
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def match_detections(
    pred_boxes: np.ndarray,
    pred_conf: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresh: float,
) -> tuple[int, int, int]:
    """Greedy confidence-sorted IoU matching → (tp, fp, fn)."""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    if len(gt_boxes) == 0:
        return 0, int(len(pred_boxes)), 0
    if len(pred_boxes) == 0:
        return 0, 0, int(len(gt_boxes))

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            iou_matrix[i, j] = iou(p, g)

    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = fp = 0
    for pi in np.argsort(pred_conf)[::-1]:
        best_j = -1
        best_iou = 0.0
        for j in range(len(gt_boxes)):
            if not gt_matched[j] and iou_matrix[pi, j] > best_iou:
                best_iou = float(iou_matrix[pi, j])
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            gt_matched[best_j] = True
        else:
            fp += 1
    fn = int(len(gt_boxes) - gt_matched.sum())
    return tp, fp, fn


def evaluate_model(
    model_path: str | Path,
    data_dir: str | Path,
    *,
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    model_name: str | None = None,
    image_paths: Iterable[Path] | None = None,
) -> EvaluationResult:
    """Run YOLO inference over a labeled split and aggregate metrics.

    ``data_dir`` is expected to contain ``images/`` and ``labels/``.
    """
    model = YOLO(str(model_path))
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if image_paths is None:
        image_paths = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        )
    result = EvaluationResult(
        model_name=model_name or Path(model_path).stem,
    )

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        preds = model(str(img_path), conf=conf_thresh, verbose=False)
        if preds and preds[0].boxes is not None and len(preds[0].boxes):
            pred_boxes = preds[0].boxes.xyxy.cpu().numpy()
            pred_conf = preds[0].boxes.conf.cpu().numpy()
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_conf = np.zeros(0, dtype=np.float32)

        gt = load_gt(labels_dir / (img_path.stem + ".txt"), w, h)
        tp, fp, fn = match_detections(pred_boxes, pred_conf, gt, iou_thresh)

        result.tp += tp
        result.fp += fp
        result.fn += fn
        result.confidences.extend(float(c) for c in pred_conf)
        result.per_image.append({
            "image": img_path.name,
            "tp": tp, "fp": fp, "fn": fn,
            "n_gt": int(len(gt)),
            "n_pred": int(len(pred_boxes)),
        })
    return result
