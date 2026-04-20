"""Wrong-parking detection — bbox-vs-polygon overlap geometry.

The "wrong parking" verdict for a car is a pure geometric question:
what fraction of the car's bbox falls outside its assigned parking
polygon? The threshold that turns that fraction into a boolean verdict
is resolved from config (covered in ``test_config.py``); this module
pins the raster geometry that feeds the threshold.

We exercise ``ParkingDetector._compute_outside_percentage`` as an
unbound function — it does not touch ``self``, so we can dodge the
full YOLO-loading ``__init__`` and test the primitive in isolation.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import pytest

# detector.py lives under SmartParking/web_app/ and isn't installable;
# mirror what the live app does.
WEB_APP = Path(__file__).resolve().parent.parent / "SmartParking" / "web_app"
if str(WEB_APP) not in sys.path:
    sys.path.insert(0, str(WEB_APP))

from detector import ParkingDetector  # noqa: E402


def _pct(polygon: list, bbox: tuple, shape: tuple = (480, 640, 3)) -> float:
    """Invoke _compute_outside_percentage without instantiating ParkingDetector.

    The method never reads self — binding to None avoids YOLO load.
    """
    return ParkingDetector._compute_outside_percentage(  # type: ignore[call-arg]
        None, polygon, bbox, shape
    )


# --- Geometry primitives --------------------------------------------------


def test_bbox_fully_inside_polygon_is_zero_percent_outside() -> None:
    # Polygon: big square (0..200). Bbox (50,50,50,50) sits strictly inside.
    polygon = [(0, 0), (200, 0), (200, 200), (0, 200)]
    bbox = (50, 50, 50, 50)  # x, y, w, h -> covers (50..100, 50..100)
    assert _pct(polygon, bbox) == pytest.approx(0.0)


def test_bbox_fully_outside_polygon_is_one_hundred_percent() -> None:
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    bbox = (300, 300, 50, 50)  # completely disjoint
    assert _pct(polygon, bbox) == pytest.approx(100.0)


def test_bbox_half_inside_polygon_is_fifty_percent() -> None:
    """Polygon covers x in [0, 100]. Bbox spans x in [50, 150], same y band.
    Exactly half the bbox area is outside the polygon."""
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    bbox = (50, 0, 100, 100)  # 100x100 bbox, right half outside
    pct = _pct(polygon, bbox)
    # Raster-based so off-by-one on the edge column is expected; allow ~2%.
    assert pct == pytest.approx(50.0, abs=2.0)


def test_zero_area_bbox_short_circuits_to_zero() -> None:
    """Guard in the method: a degenerate (0-area) bbox must not divide by
    zero — it reports 0% outside and leaves downstream logic alone."""
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert _pct(polygon, (50, 50, 0, 0)) == pytest.approx(0.0)


def test_bbox_just_peeking_over_edge_is_small_percentage() -> None:
    """A mostly-inside bbox whose top 10% spills above the polygon top
    should report ~10% outside — the typical 'car parked a bit forward'
    case that the per-camera threshold is tuned against."""
    polygon = [(0, 50), (200, 50), (200, 250), (0, 250)]
    # Bbox 200x100 at (0, 40) -> top 10px outside, bottom 90px inside.
    bbox = (0, 40, 200, 100)
    pct = _pct(polygon, bbox)
    assert pct == pytest.approx(10.0, abs=2.0)


# --- Ordering property ---------------------------------------------------


def test_sliding_bbox_monotonically_increases_outside_percentage() -> None:
    """Slide a bbox rightward across a polygon boundary — the outside
    percentage should be non-decreasing. If it ever drops, the raster
    mask bookkeeping is broken."""
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    percentages = [_pct(polygon, (x, 0, 50, 50)) for x in range(0, 151, 25)]
    # x=0 fully inside (0%); x=150 fully outside (100%); monotonic in between.
    for prev, cur in pairwise(percentages):
        assert cur >= prev - 1e-6, f"non-monotonic slide: {percentages}"
    assert percentages[0] == pytest.approx(0.0)
    assert percentages[-1] == pytest.approx(100.0)
