"""Typed application settings.

Settings are assembled from, in order of decreasing precedence:

1. Explicit init kwargs (used mainly by tests).
2. Environment variables prefixed with ``NEOSMART_`` (``__`` is the
   nesting delimiter, e.g. ``NEOSMART_DETECTOR__CONFIDENCE=0.5``).
3. An optional ``.env`` file at the repo root.
4. ``config/default.yaml`` plus an optional ``config/<profile>.yaml``
   where ``profile`` comes from ``NEOSMART_CONFIG_PROFILE``.
5. Field defaults on the pydantic models themselves.

Use ``get_settings()`` to obtain the cached singleton; FastAPI routes
can declare ``settings: AppSettings = Depends(get_settings)``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# SmartParking/neosmart/config.py → parents[2] is repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = REPO_ROOT / "config"


# --- Leaf models ------------------------------------------------------


class PathsSettings(BaseModel):
    """Filesystem layout. Relative paths are resolved against the repo root."""

    frames_dir: Path = Path("SmartParking/frames")
    models_dir: Path = Path("Models")
    car_detector: Path = Path("Models/Car_Detector.pt")
    plate_scanner_dir: Path = Path("Models/plate_scanner")
    parking_polygons_dir: Path = Path("SmartParking/CarParkingSpace")
    barrier_zones_dir: Path = Path("SmartParking/BarrierSystem")
    traffic_crossing_dir: Path = Path("SmartParking/TrafficCounting")
    wrong_parking_calib_dir: Path = Path("SmartParking/WrongParking")
    db_path: Path = Path("SmartParking/web_app/data/barrier.db")
    seed_plates: Path = Path("SmartParking/web_app/data/seed_plates.json")
    logs_dir: Path = Path("logs")

    def resolve(self, p: Path) -> Path:
        """Resolve a configured path against the repo root if relative."""
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()


CameraRole = Literal["parking", "road", "barrier_plate", "barrier_safety"]


class CameraSettings(BaseModel):
    role: CameraRole
    tracking: bool = False


class DetectorSettings(BaseModel):
    confidence: float = Field(0.65, ge=0.0, le=1.0)
    plate_confidence: float = Field(0.3, ge=0.0, le=1.0)


class TrackingSettings(BaseModel):
    sort_max_age: int = Field(20, ge=1)
    sort_min_hits: int = Field(5, ge=0)
    sort_iou_threshold: float = Field(0.3, ge=0.0, le=1.0)
    suspicious_time_threshold: float = Field(30.0, ge=0.0)
    traffic_sort_max_age: int = Field(50, ge=1)
    traffic_sort_min_hits: int = Field(0, ge=0)
    traffic_sort_iou_threshold: float = Field(0.15, ge=0.0, le=1.0)


class BarrierSettings(BaseModel):
    plate_min_agreeing_reads: int = Field(3, ge=1)
    plate_confidence_threshold: float = Field(0.3, ge=0.0, le=1.0)
    access_denied_timeout: float = Field(5.0, ge=0.0)
    safety_zone_clear_delay: float = Field(1.0, ge=0.0)
    barrier_open_timeout: float = Field(30.0, ge=0.0)
    safety_grace_period: float = Field(3.0, ge=0.0)
    barrier_opening_timeout: float = Field(30.0, ge=0.0)
    barrier_closing_timeout: float = Field(30.0, ge=0.0)
    max_recent_events: int = Field(20, ge=1)
    entry_plate_camera: str = "camera5"
    entry_safety_camera: str = "camera6"


class EdgeRule(BaseModel):
    """Selects polygon indices near an image edge (fisheye distortion)."""

    first_n: int = Field(0, ge=0)
    last_n: int = Field(0, ge=0)

    def indices(self, total: int) -> set[int]:
        result: set[int] = set()
        if self.first_n:
            result.update(range(min(self.first_n, total)))
        if self.last_n:
            result.update(range(max(0, total - self.last_n), total))
        return result


class IndexOverride(BaseModel):
    indices: list[int]
    threshold: int = Field(ge=0, le=100)


class WrongParkingSettings(BaseModel):
    outside_threshold_default: int = Field(25, ge=0, le=100)
    outside_threshold_edge: int = Field(45, ge=0, le=100)
    per_camera_default: dict[str, int] = Field(default_factory=dict)
    edge_polygon_rules: dict[str, EdgeRule] = Field(default_factory=dict)
    index_overrides: dict[str, list[IndexOverride]] = Field(default_factory=dict)

    def resolve_threshold(
        self, camera_id: str, polygon_index: int, total_polygons: int
    ) -> int:
        """Resolve the 'bbox outside polygon' threshold for a spot.

        Precedence:
          1. Explicit per-index override.
          2. Edge-polygon rule → outside_threshold_edge.
          3. Per-camera default.
          4. Global default.
        """
        for override in self.index_overrides.get(camera_id, []):
            if polygon_index in override.indices:
                return override.threshold

        rule = self.edge_polygon_rules.get(camera_id)
        if rule and polygon_index in rule.indices(total_polygons):
            return self.outside_threshold_edge

        if camera_id in self.per_camera_default:
            return self.per_camera_default[camera_id]

        return self.outside_threshold_default


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_to_file: bool = True


class CrossingRegion(BaseModel):
    incoming: list[list[int]] = Field(default_factory=list)
    outgoing: list[list[int]] = Field(default_factory=list)


# --- Root settings ----------------------------------------------------


def _yaml_files() -> list[Path]:
    """YAML layers to load, in read order (later layers override earlier)."""
    files = [CONFIG_DIR / "default.yaml"]
    profile = os.getenv("NEOSMART_CONFIG_PROFILE", "").strip()
    if profile:
        profile_path = CONFIG_DIR / f"{profile}.yaml"
        if profile_path.exists():
            files.append(profile_path)
    return files


class AppSettings(BaseSettings):
    """Top-level settings, cached via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="NEOSMART_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    paths: PathsSettings = Field(default_factory=PathsSettings)
    cameras: dict[str, CameraSettings] = Field(default_factory=dict)
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    wrong_parking: WrongParkingSettings = Field(default_factory=WrongParkingSettings)
    barrier: BarrierSettings = Field(default_factory=BarrierSettings)
    crossing_regions_fallback: dict[str, CrossingRegion] = Field(default_factory=dict)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @field_validator("cameras")
    @classmethod
    def _validate_cameras(
        cls, v: dict[str, CameraSettings]
    ) -> dict[str, CameraSettings]:
        if not v:
            return v
        tracking_roles = {"parking"}
        for cam_id, spec in v.items():
            if spec.tracking and spec.role not in tracking_roles:
                raise ValueError(
                    f"Camera {cam_id!r}: tracking=true only allowed for parking cameras"
                )
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=_yaml_files())
        # Precedence (first wins in pydantic-settings):
        #   init > env > dotenv > yaml layers > file secrets.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_source,
            file_secret_settings,
        )

    # --- Derived views over `cameras` --------------------------------

    @property
    def camera_ids(self) -> list[str]:
        return list(self.cameras.keys())

    @property
    def parking_camera_ids(self) -> list[str]:
        return [c for c, s in self.cameras.items() if s.role == "parking"]

    @property
    def road_camera_ids(self) -> list[str]:
        return [c for c, s in self.cameras.items() if s.role == "road"]

    @property
    def barrier_camera_ids(self) -> list[str]:
        return [
            c
            for c, s in self.cameras.items()
            if s.role in ("barrier_plate", "barrier_safety")
        ]

    @property
    def tracking_camera_ids(self) -> list[str]:
        return [c for c, s in self.cameras.items() if s.tracking]

    # --- Path helpers ------------------------------------------------

    def frame_path(self, camera_id: str) -> Path:
        return self.paths.resolve(self.paths.frames_dir) / f"{camera_id}.png"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the process-wide settings singleton."""
    return AppSettings()


def reload_settings() -> AppSettings:
    """Drop the cache and re-read sources. Test-only helper."""
    get_settings.cache_clear()
    return get_settings()
