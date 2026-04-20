"""Settings loading, derived views, threshold resolver, env overrides."""

from __future__ import annotations

import pytest
from neosmart.config import (
    AppSettings,
    CameraSettings,
    EdgeRule,
    IndexOverride,
    WrongParkingSettings,
    get_settings,
    reload_settings,
)
from pydantic import ValidationError


def test_default_yaml_loads() -> None:
    s = get_settings()
    assert set(s.camera_ids) == {f"camera{i}" for i in range(1, 7)}
    assert s.detector.confidence == 0.65


def test_derived_camera_views() -> None:
    s = get_settings()
    assert s.parking_camera_ids == ["camera1", "camera2", "camera3"]
    assert s.road_camera_ids == ["camera4"]
    assert set(s.barrier_camera_ids) == {"camera5", "camera6"}
    assert s.tracking_camera_ids == ["camera3"]


def test_env_override_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEOSMART_DETECTOR__CONFIDENCE", "0.42")
    s = reload_settings()
    assert s.detector.confidence == pytest.approx(0.42)


def test_env_override_barrier_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEOSMART_BARRIER__PLATE_MIN_AGREEING_READS", "5")
    s = reload_settings()
    assert s.barrier.plate_min_agreeing_reads == 5


def test_invalid_confidence_raises() -> None:
    with pytest.raises(ValidationError):
        AppSettings(detector={"confidence": 1.5})  # type: ignore[arg-type]


def test_tracking_only_on_parking_cameras_raises() -> None:
    with pytest.raises(ValidationError):
        AppSettings(cameras={"camera9": CameraSettings(role="road", tracking=True)})


def test_frame_path_is_absolute() -> None:
    s = get_settings()
    p = s.frame_path("camera1")
    assert p.is_absolute()
    assert p.name == "camera1.png"


# --- Threshold resolver ----------------------------------------------


@pytest.fixture
def wp() -> WrongParkingSettings:
    return WrongParkingSettings(
        outside_threshold_default=25,
        outside_threshold_edge=45,
        per_camera_default={"camera2": 31, "camera3": 36},
        edge_polygon_rules={
            "camera1": EdgeRule(first_n=2, last_n=2),
            "camera2": EdgeRule(last_n=2),
        },
        index_overrides={
            "camera1": [IndexOverride(indices=[0, 1], threshold=52)],
            "camera3": [IndexOverride(indices=[0, 1], threshold=46)],
        },
    )


class TestResolveThreshold:
    def test_explicit_index_override_wins(self, wp: WrongParkingSettings) -> None:
        # camera1 index 0: override (52) beats edge rule (45).
        assert wp.resolve_threshold("camera1", 0, total_polygons=12) == 52
        assert wp.resolve_threshold("camera1", 1, total_polygons=12) == 52

    def test_edge_rule_applies_when_no_override(self, wp: WrongParkingSettings) -> None:
        # camera1 last 2 polygons are edge — no index override, gets edge threshold.
        assert wp.resolve_threshold("camera1", 10, total_polygons=12) == 45
        assert wp.resolve_threshold("camera1", 11, total_polygons=12) == 45

    def test_middle_polygon_falls_through_to_default(self, wp: WrongParkingSettings) -> None:
        # camera1 middle polygons have no rule → global default.
        assert wp.resolve_threshold("camera1", 5, total_polygons=12) == 25

    def test_per_camera_default_without_rule(self, wp: WrongParkingSettings) -> None:
        # camera3 has per-camera default 36, no edge rule. Index 5 (not overridden)
        # should get 36, not 25.
        assert wp.resolve_threshold("camera3", 5, total_polygons=12) == 36

    def test_per_camera_override_beats_edge(self, wp: WrongParkingSettings) -> None:
        # camera3 index 0 has override 46; camera3 has no edge rule, so without
        # the override it would be 36 (per-camera). Override wins.
        assert wp.resolve_threshold("camera3", 0, total_polygons=12) == 46

    def test_camera2_far_edge(self, wp: WrongParkingSettings) -> None:
        # camera2 has last_n=2 and per-camera default 31.
        # Last 2 polygons → edge threshold (45).
        assert wp.resolve_threshold("camera2", 10, total_polygons=12) == 45
        assert wp.resolve_threshold("camera2", 11, total_polygons=12) == 45
        # Middle polygon → per-camera default 31.
        assert wp.resolve_threshold("camera2", 5, total_polygons=12) == 31

    def test_unknown_camera_uses_global_default(self, wp: WrongParkingSettings) -> None:
        assert wp.resolve_threshold("camera9", 0, total_polygons=12) == 25
