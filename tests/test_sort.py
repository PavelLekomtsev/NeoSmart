"""SORT tracker — track ID stability, IoU matching, lifecycle.

SORT is the core of the paid-zone "suspicious parking" detection: the
controller decides a car has been stationary too long only because SORT
gives it a persistent track ID across frames. Regressions in the Kalman
update or the Hungarian association silently break that downstream
decision, hence these tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from neosmart.tracking.sort import (
    KalmanBoxTracker,
    Sort,
    associate_detections_to_trackers,
    iou_batch,
)


@pytest.fixture(autouse=True)
def _reset_tracker_id_counter() -> None:
    """KalmanBoxTracker.count is a class-level counter.

    Left alone, it leaks across tests and makes IDs non-deterministic — a
    real gotcha reproducing the production experience where track IDs
    depend on process start time.
    """
    KalmanBoxTracker.count = 0


def _det(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> np.ndarray:
    return np.array([[x1, y1, x2, y2, score]], dtype=float)


def test_iou_identical_bboxes_is_one() -> None:
    a = np.array([[10.0, 10.0, 50.0, 50.0]])
    assert iou_batch(a, a)[0, 0] == pytest.approx(1.0)


def test_iou_disjoint_bboxes_is_zero() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert iou_batch(a, b)[0, 0] == 0.0


def test_iou_half_overlap_matches_expected() -> None:
    # a and b overlap on their right half / left half → 1/3 of the union.
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    iou = iou_batch(a, b)[0, 0]
    # intersection = 5*10 = 50, union = 100 + 100 - 50 = 150
    assert iou == pytest.approx(50.0 / 150.0)


def test_associate_matches_overlapping_pair_leaves_the_outlier_unmatched() -> None:
    detections = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [100.0, 100.0, 110.0, 110.0],
    ])
    trackers = np.array([
        [0.0, 0.0, 10.0, 10.0],
    ])
    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        detections, trackers, iou_threshold=0.3,
    )
    assert matches.shape == (1, 2)
    assert tuple(matches[0]) == (0, 0)
    assert set(unmatched_dets.tolist()) == {1}
    assert unmatched_trks.size == 0


def test_single_stationary_box_keeps_stable_id_across_frames() -> None:
    """A motionless car must not change its track ID between frames — the
    suspicious-parking logic aggregates dwell time per ID and would reset
    on every ID flip."""
    sort = Sort(max_age=1, min_hits=1, iou_threshold=0.3)
    bbox = _det(100.0, 100.0, 200.0, 200.0)

    ids = []
    for _ in range(5):
        out = sort.update(bbox)
        assert out.shape == (1, 5)
        ids.append(int(out[0, 4]))

    assert len(set(ids)) == 1, f"track id flipped across frames: {ids}"


def test_two_disjoint_boxes_get_two_distinct_ids() -> None:
    sort = Sort(max_age=1, min_hits=1, iou_threshold=0.3)
    dets = np.array([
        [100.0, 100.0, 200.0, 200.0, 0.9],
        [400.0, 400.0, 500.0, 500.0, 0.9],
    ])
    # Warm-up frame — trackers are created but not all are emitted yet
    # depending on initialization order. Run once to stabilize, then check.
    sort.update(dets)
    out = sort.update(dets)

    assert out.shape[0] == 2
    ids = set(out[:, 4].astype(int).tolist())
    assert len(ids) == 2, f"expected two distinct track IDs, got {ids}"


def test_tracker_is_pruned_after_max_age_of_missed_frames() -> None:
    """When a detection disappears for longer than ``max_age`` frames, the
    tracker must be dropped; otherwise a re-appearing car would inherit the
    old ID and the dwell timer would not reset."""
    sort = Sort(max_age=1, min_hits=1, iou_threshold=0.3)
    bbox = _det(100.0, 100.0, 200.0, 200.0)

    sort.update(bbox)
    assert len(sort.trackers) == 1

    # Two empty frames — tracker's time_since_update climbs past max_age
    sort.update(np.empty((0, 5)))
    sort.update(np.empty((0, 5)))

    assert len(sort.trackers) == 0, "stale tracker was not pruned"
