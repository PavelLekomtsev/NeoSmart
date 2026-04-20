"""Entry-barrier state machine — full happy path + critical branches.

BarrierController is the biggest piece of business logic in the web app:
nine states, two cameras (plate + safety), sticky UE5 commands, an
asymmetric OCR exit policy (instant grant on whitelist hit, consensus
required for denial), and two grace/clear timers. A regression here
either locks out legitimate plates or leaves the barrier open — both
are the kind of failure mode the defense committee will probe.

These tests exercise the state machine directly: real ``BarrierDatabase``
on a tmp file, a stub plate recognizer fed pre-queued readings, a fake
clock driven by the test. The image parameter is a zero-filled array
since ``_advance_state`` never touches its pixels outside of drawing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WEB_APP = Path(__file__).resolve().parent.parent / "SmartParking" / "web_app"
if str(WEB_APP) not in sys.path:
    sys.path.insert(0, str(WEB_APP))

import barrier_controller  # noqa: E402
from barrier_controller import (  # noqa: E402
    STATE_ACCESS_DENIED,
    STATE_BARRIER_CLOSING,
    STATE_BARRIER_OPENING,
    STATE_CAR_APPROACHING,
    STATE_CAR_PASSING,
    STATE_IDLE,
    STATE_READING_PLATE,
    BarrierController,
)
from barrier_db import BarrierDatabase  # noqa: E402

# --- Helpers --------------------------------------------------------------


class FakeClock:
    """Monotonic clock the test can advance by hand — replaces ``time.time``
    inside the controller via monkeypatch."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def time(self) -> float:
        return self.now


class StubPlateRecognizer:
    """Returns pre-queued OCR readings, one per call.

    Matches PlateRecognizer.detect_plate's contract: returns a dict with
    keys ``plate_text``, ``confidence``, and optionally ``plate_image``,
    or None when the queue is exhausted (simulates "no plate detected").
    """

    def __init__(self, readings: list[dict] | None = None) -> None:
        self._queue: list[dict | None] = list(readings or [])
        self.calls = 0

    def detect_plate(self, img, car_bbox):
        self.calls += 1
        if not self._queue:
            return None
        return self._queue.pop(0)

    def queue(self, reading: dict | None) -> None:
        self._queue.append(reading)


def _reading(text: str, conf: float = 0.9) -> dict:
    return {"plate_text": text, "confidence": conf, "plate_image": None}


# Zones are axis-aligned rectangles the state machine converts to polygons
# via ``_to_polygon``. Coordinates were picked so each zone is disjoint
# — a car "in approach" is unambiguously out of the reading zone.
APPROACH = [(0, 0), (100, 0), (100, 100), (0, 100)]
READING = [(150, 0), (250, 0), (250, 100), (150, 100)]
SAFETY = [(300, 0), (400, 0), (400, 100), (300, 100)]


def _car(cx: int, cy: int, w: int = 40, h: int = 40) -> dict:
    """Detection dict matching what ParkingDetector emits."""
    return {
        "bbox": (cx - w // 2, cy - h // 2, w, h),
        "center": (cx, cy),
        "confidence": 0.9,
    }


CAR_IN_APPROACH = _car(50, 50)
CAR_IN_READING = _car(200, 50)
CAR_IN_SAFETY = _car(350, 50)


# --- Fixtures -------------------------------------------------------------


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    c = FakeClock()
    # Patch the `time` module that barrier_controller imported at module
    # load — barrier_controller uses `time.time()` for `state_enter_time`
    # and for every timeout comparison in `_advance_state`.
    monkeypatch.setattr(barrier_controller.time, "time", c.time)
    return c


@pytest.fixture
def db(tmp_path: Path) -> BarrierDatabase:
    return BarrierDatabase(db_path=str(tmp_path / "barrier.db"))


@pytest.fixture
def recognizer() -> StubPlateRecognizer:
    return StubPlateRecognizer()


@pytest.fixture
def controller(
    clock: FakeClock,
    db: BarrierDatabase,
    recognizer: StubPlateRecognizer,
) -> BarrierController:
    return BarrierController(
        barrier_id="entry",
        camera_id="camera5",
        plate_recognizer=recognizer,
        database=db,
        approach_zone=APPROACH,
        reading_zone=READING,
        safety_zone=SAFETY,
        safety_camera_id="camera6",
    )


@pytest.fixture
def frame() -> np.ndarray:
    # ``_advance_state`` never reads pixels; ``draw_overlay`` does but only
    # writes back onto the array. 300x500 is enough to clear all zone polys.
    return np.zeros((300, 500, 3), dtype=np.uint8)


# --- Tests ----------------------------------------------------------------


def test_idle_to_approaching_when_car_enters_approach_zone(
    controller: BarrierController, frame: np.ndarray,
) -> None:
    assert controller.state == STATE_IDLE
    controller.process_frame(frame, [CAR_IN_APPROACH])
    assert controller.state == STATE_CAR_APPROACHING


def test_approaching_to_reading_when_car_reaches_reading_zone(
    controller: BarrierController, frame: np.ndarray,
) -> None:
    controller.process_frame(frame, [CAR_IN_APPROACH])
    controller.process_frame(frame, [CAR_IN_READING])
    assert controller.state == STATE_READING_PLATE


def test_approaching_returns_to_idle_after_timeout_when_car_leaves(
    controller: BarrierController, frame: np.ndarray, clock: FakeClock,
) -> None:
    """If a car enters the approach zone but leaves without progressing to
    the reading zone, the controller waits 3 s of empty-approach before
    returning to IDLE — prevents flapping when detections jitter."""
    controller.process_frame(frame, [CAR_IN_APPROACH])
    assert controller.state == STATE_CAR_APPROACHING

    # Empty frame right away → still APPROACHING because elapsed < 3 s
    controller.process_frame(frame, [])
    assert controller.state == STATE_CAR_APPROACHING

    # After 3.1 s of empty approach, idle reset kicks in
    clock.advance(3.1)
    controller.process_frame(frame, [])
    assert controller.state == STATE_IDLE


def test_full_happy_path_whitelisted_plate(
    controller: BarrierController,
    frame: np.ndarray,
    clock: FakeClock,
    db: BarrierDatabase,
    recognizer: StubPlateRecognizer,
) -> None:
    """End-to-end: IDLE → APPROACHING → READING → GRANTED (instant) →
    OPENING → (ack) → CAR_PASSING → (safety clear + delay) → CLOSING
    → (ack) → IDLE. This is THE demo path — if it regresses, nothing
    works in the live system."""
    db.add_plate("AAA111")
    recognizer.queue(_reading("AAA111"))

    # 1. Approach
    controller.process_frame(frame, [CAR_IN_APPROACH])
    assert controller.state == STATE_CAR_APPROACHING

    # 2. Reaches reading zone
    controller.process_frame(frame, [CAR_IN_READING])
    assert controller.state == STATE_READING_PLATE

    # 3. OCR returns whitelisted plate → instant grant
    controller.process_frame(frame, [CAR_IN_READING])
    # Instant grant flows: READING_PLATE → ACCESS_GRANTED during the OCR
    # accumulation, then GRANTED → BARRIER_OPENING on the very next frame.
    controller.process_frame(frame, [CAR_IN_READING])
    assert controller.state == STATE_BARRIER_OPENING
    assert controller.barrier_command == "open"

    # 4. UE5 acks open_complete → CAR_PASSING
    controller.ack_from_ue5("open_complete")
    assert controller.state == STATE_CAR_PASSING
    assert controller.barrier_position == "open"
    assert controller.barrier_command == "idle"

    # 5. Grace period: safety zone not even checked for SAFETY_GRACE_PERIOD
    controller.update_safety([])
    controller.process_frame(frame, [])
    assert controller.state == STATE_CAR_PASSING

    # 6. After grace, still empty safety zone for >= SAFETY_ZONE_CLEAR_DELAY
    clock.advance(3.1)                                   # past grace
    controller.update_safety([])
    controller.process_frame(frame, [])                 # safety_clear_time set
    clock.advance(1.1)                                  # past clear delay
    controller.update_safety([])
    controller.process_frame(frame, [])
    assert controller.state == STATE_BARRIER_CLOSING
    assert controller.barrier_command == "close"

    # 7. UE5 acks close_complete → IDLE
    controller.ack_from_ue5("close_complete")
    assert controller.state == STATE_IDLE
    assert controller.barrier_position == "closed"

    # Side effects persisted: granted log + active session
    assert db.get_today_stats()["entries"] == 1
    assert len(db.get_active_sessions()) == 1


def test_reading_decides_with_best_reading_when_car_leaves_before_consensus(
    controller: BarrierController,
    frame: np.ndarray,
    recognizer: StubPlateRecognizer,
) -> None:
    """Asymmetric-exit branch: car ducks out of the reading zone before we
    collected PLATE_MIN_AGREEING_READS matching OCR runs. Controller must
    still decide — using the single best reading it has — otherwise
    flickering detections leave the state machine hung in READING_PLATE
    forever."""
    # Queue one reading for a plate that is NOT whitelisted → DENIED.
    recognizer.queue(_reading("UNKNOWN1"))

    controller.process_frame(frame, [CAR_IN_READING])  # idle → reading (skips approach; idle jumps straight to reading if car is already there)
    assert controller.state == STATE_READING_PLATE
    controller.process_frame(frame, [CAR_IN_READING])  # accumulate one reading
    assert len(controller.ocr_buffer) == 1

    # Car leaves the reading zone (and also approach) entirely
    controller.process_frame(frame, [])

    assert controller.state == STATE_ACCESS_DENIED
    assert controller.last_plate == "UNKNOWN1"
    assert controller.last_access_result == "denied"


def test_access_denied_returns_to_idle_after_timeout(
    controller: BarrierController,
    frame: np.ndarray,
    clock: FakeClock,
    recognizer: StubPlateRecognizer,
) -> None:
    recognizer.queue(_reading("UNKNOWN1"))
    controller.process_frame(frame, [CAR_IN_READING])
    controller.process_frame(frame, [CAR_IN_READING])
    controller.process_frame(frame, [])  # decide → DENIED
    assert controller.state == STATE_ACCESS_DENIED

    clock.advance(5.1)
    controller.process_frame(frame, [])
    assert controller.state == STATE_IDLE


def test_manual_open_goes_directly_to_barrier_opening(
    controller: BarrierController, frame: np.ndarray,
) -> None:
    """Operator override from the dashboard bypasses plate recognition and
    the approach/reading flow entirely — used for guest entries and when
    plate OCR fails but the operator can see the car."""
    controller.manual_open()
    controller.process_frame(frame, [])
    assert controller.state == STATE_BARRIER_OPENING
    assert controller.barrier_command == "open"
    assert controller.last_access_result == "manual_override"


def test_ack_without_matching_state_is_ignored(
    controller: BarrierController,
) -> None:
    """Late acks (state already advanced past OPENING/CLOSING) must not
    retrigger transitions — otherwise a stale UE5 animation ack could
    reopen a just-closed barrier."""
    assert controller.state == STATE_IDLE
    controller.ack_from_ue5("open_complete")
    assert controller.state == STATE_IDLE
    controller.ack_from_ue5("close_complete")
    assert controller.state == STATE_IDLE


def test_car_passing_stays_open_while_car_in_safety_zone(
    controller: BarrierController,
    frame: np.ndarray,
    clock: FakeClock,
    recognizer: StubPlateRecognizer,
    db: BarrierDatabase,
) -> None:
    """Core safety invariant: while a car is detected in the safety zone
    (camera6 on the far side of the barrier), the barrier MUST stay open
    — even if the clear-delay timer has expired. Otherwise the barrier
    closes on top of a passing car."""
    db.add_plate("AAA111")
    recognizer.queue(_reading("AAA111"))

    # Drive through to CAR_PASSING
    controller.process_frame(frame, [CAR_IN_APPROACH])
    controller.process_frame(frame, [CAR_IN_READING])
    controller.process_frame(frame, [CAR_IN_READING])  # instant-grant
    controller.process_frame(frame, [CAR_IN_READING])  # GRANTED → OPENING
    controller.ack_from_ue5("open_complete")
    assert controller.state == STATE_CAR_PASSING

    # Past grace period, with a car sitting in the safety zone
    clock.advance(5.0)
    controller.update_safety([CAR_IN_SAFETY])
    controller.process_frame(frame, [])
    assert controller.state == STATE_CAR_PASSING
    assert controller.safety_clear_time is None  # never "cleared"
