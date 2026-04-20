"""
Barrier Controller Module
State machine for managing parking barrier entry/exit with plate recognition.
"""

import time
import logging
import cv2
import numpy as np
from collections import deque
from pathlib import Path

from plate_scanner import PlateRecognizer
from barrier_db import BarrierDatabase

logger = logging.getLogger(__name__)


# --- State Machine Constants ---

STATE_IDLE = "idle"
STATE_CAR_APPROACHING = "car_approaching"
STATE_READING_PLATE = "reading_plate"
STATE_ACCESS_GRANTED = "access_granted"
STATE_ACCESS_DENIED = "access_denied"
STATE_BARRIER_OPENING = "barrier_opening"
STATE_CAR_PASSING = "car_passing"
STATE_BARRIER_CLOSING = "barrier_closing"

# Minimum number of agreeing OCR readings to auto-decide access while car is still in reading zone
PLATE_MIN_AGREEING_READS = 3
# Minimum PlateScanner confidence to accept a reading
PLATE_CONFIDENCE_THRESHOLD = 0.3
# How long to show "Access Denied" before returning to idle
ACCESS_DENIED_TIMEOUT = 5.0
# How long to wait after safety zone clears before closing barrier
SAFETY_ZONE_CLEAR_DELAY = 1.0
# Maximum time barrier stays open without a car (safety timeout)
BARRIER_OPEN_TIMEOUT = 30.0
# Grace period after barrier opens before checking safety zone (seconds)
SAFETY_GRACE_PERIOD = 3.0
# Max wait for UE5 "open_complete" ack before we force-advance (failsafe only).
# Long enough that slow animations never trip it and lose the handshake.
BARRIER_OPENING_TIMEOUT = 30.0
# Max wait for UE5 "close_complete" ack before we force-advance.
BARRIER_CLOSING_TIMEOUT = 30.0
# Maximum events to keep in memory for dashboard
MAX_RECENT_EVENTS = 20

# Zone overlay colors (BGR)
COLOR_APPROACH = (200, 200, 0)     # Cyan-ish
COLOR_READING = (0, 200, 200)      # Yellow-ish
COLOR_SAFETY_CLEAR = (0, 200, 0)   # Green
COLOR_SAFETY_OCCUPIED = (0, 200, 255)  # Yellow/Orange
COLOR_SAFETY_MOVING = (0, 0, 255)  # Red


class BarrierController:
    """
    Manages the barrier state machine for a single entry/exit point.
    One instance per barrier camera (entry or exit).
    """

    def __init__(self, barrier_id: str, camera_id: str,
                 plate_recognizer: PlateRecognizer,
                 database: BarrierDatabase,
                 approach_zone: list = None,
                 reading_zone: list = None,
                 safety_zone: list = None,
                 safety_camera_id: str = None):
        """
        Args:
            barrier_id: "entry"
            camera_id: e.g. "camera5" (plate camera)
            plate_recognizer: PlateRecognizer instance
            database: BarrierDatabase instance
            approach_zone: List of (x, y) points defining approach polygon
            reading_zone: List of (x, y) points defining plate reading trigger polygon
            safety_zone: List of (x, y) points defining barrier arm safety polygon
            safety_camera_id: If set, safety zone is monitored from a different camera
                              (e.g. "camera6"). Call update_safety() with that camera's detections.
        """
        self.barrier_id = barrier_id
        self.camera_id = camera_id
        self.plate_recognizer = plate_recognizer
        self.db = database

        # Zone polygons (numpy arrays for cv2.pointPolygonTest)
        self.approach_zone = self._to_polygon(approach_zone) if approach_zone else None
        self.reading_zone = self._to_polygon(reading_zone) if reading_zone else None
        self.safety_zone = self._to_polygon(safety_zone) if safety_zone else None

        # Safety camera support (two-camera entry barrier)
        self._safety_camera_id = safety_camera_id
        self._safety_detections = []  # Updated by safety camera via update_safety()

        # State machine
        self.state = STATE_IDLE
        self.state_enter_time = time.time()

        # OCR accumulation — every frame while car sits in reading zone
        self.ocr_buffer = []          # List of plate reading dicts

        # Current plate info
        self.last_plate = None         # Last recognized plate text
        self.last_plate_confidence = 0.0
        self.last_plate_image = None   # Cropped plate image (numpy)
        self.last_access_result = None  # "granted", "denied", or None

        # Barrier command for UE5
        self.barrier_command = "idle"   # "open", "close", "idle"
        self.barrier_position = "closed"  # "open", "closed", "opening", "closing"

        # Safety monitoring
        self.safety_clear_time = None   # When safety zone last became clear

        # Event log for dashboard
        self.recent_events = deque(maxlen=MAX_RECENT_EVENTS)

        # Manual override flag
        self._manual_override = None    # "open" or "close" or None

        safety_label = f", safety from {safety_camera_id}" if safety_camera_id else ""
        logger.info("[%s] Controller initialized for %s%s",
                    barrier_id, camera_id, safety_label)

    @staticmethod
    def _to_polygon(points: list) -> np.ndarray | None:
        if not points:
            return None
        return np.array(points, np.int32).reshape(-1, 1, 2)

    def _set_state(self, new_state: str):
        """Transition to a new state."""
        if new_state != self.state:
            old = self.state
            self.state = new_state
            self.state_enter_time = time.time()
            # Reset state-specific data on transition
            if new_state == STATE_READING_PLATE:
                self.ocr_buffer = []
            elif new_state == STATE_CAR_PASSING:
                self.safety_clear_time = None
            elif new_state == STATE_IDLE:
                self.barrier_command = "idle"
                self.last_access_result = None

    def _car_in_zone(self, detections: list, zone: np.ndarray) -> dict | None:
        """
        Check if any car's center is inside a zone polygon.
        Returns the detection dict of the car, or None.
        """
        if zone is None:
            return None
        for det in detections:
            if cv2.pointPolygonTest(zone, det["center"], False) >= 0:
                return det
        return None

    def _any_car_in_zone(self, detections: list, zone: np.ndarray) -> bool:
        """Check if any car overlaps the zone (bbox intersection, not just center)."""
        if zone is None:
            return False
        # Check center point first (fast)
        for det in detections:
            if cv2.pointPolygonTest(zone, det["center"], False) >= 0:
                return True
        # Also check if any bbox corner is inside zone
        for det in detections:
            x, y, w, h = det["bbox"]
            corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
            for corner in corners:
                if cv2.pointPolygonTest(zone, corner, False) >= 0:
                    return True
        return False

    def update_safety(self, detections: list):
        """
        Update safety zone detections from the safety camera (e.g. camera6).
        Called each frame for the safety camera, separate from process_frame().
        """
        self._safety_detections = detections

    def process_frame(self, img: np.ndarray, car_detections: list) -> dict:
        """
        Called every frame for this barrier's camera.
        Advances the state machine based on car positions and plate readings.

        Args:
            img: Current frame (BGR)
            car_detections: List of car detection dicts from ParkingDetector

        Returns:
            dict with barrier state info for WebSocket/API
        """
        # Handle manual override
        if self._manual_override:
            self._handle_manual_override()
            self.draw_overlay(img)
            return self._build_result()

        # Advance state machine
        self._advance_state(img, car_detections)

        # Draw overlay
        self.draw_overlay(img)

        return self._build_result()

    def _advance_state(self, img: np.ndarray, detections: list):
        """Core state machine logic."""
        now = time.time()
        elapsed_in_state = now - self.state_enter_time

        if self.state == STATE_IDLE:
            # Pick up a car that's already in the reading zone first — handles
            # the "second car arrived mid barrier cycle" case, where it drove
            # past the approach zone while we were stuck in BARRIER_CLOSING and
            # is now sitting in reading zone when we finally return to IDLE.
            if self._car_in_zone(detections, self.reading_zone) is not None:
                self._set_state(STATE_READING_PLATE)
            elif self._car_in_zone(detections, self.approach_zone) is not None:
                self._set_state(STATE_CAR_APPROACHING)

        elif self.state == STATE_CAR_APPROACHING:
            # Check if car moved into reading zone
            car = self._car_in_zone(detections, self.reading_zone)
            if car is not None:
                self._set_state(STATE_READING_PLATE)
            # If car left approach zone entirely, go back to idle
            elif not self._any_car_in_zone(detections, self.approach_zone):
                if elapsed_in_state > 3.0:
                    self._set_state(STATE_IDLE)

        elif self.state == STATE_READING_PLATE:
            # Asymmetric exit policy:
            #   - GRANT: a single OCR reading that hits the whitelist exits
            #     immediately (handled inside _accumulate_ocr). Low risk because
            #     the plate text must match a known whitelisted plate exactly.
            #   - DENY:  still requires consensus or car leaving — one bad OCR
            #     should never lock out a valid car.
            car = self._car_in_zone(detections, self.reading_zone)

            if car is not None:
                self._accumulate_ocr(img, car)
                # _accumulate_ocr may have transitioned us to ACCESS_GRANTED
                # already; only fall through to consensus check if we're still
                # in READING_PLATE.
                if self.state == STATE_READING_PLATE and self._has_plate_consensus():
                    self._decide_access()
            else:
                if self.ocr_buffer:
                    self._decide_access()
                elif self._any_car_in_zone(detections, self.approach_zone):
                    self._set_state(STATE_CAR_APPROACHING)
                else:
                    self._set_state(STATE_IDLE)

        elif self.state == STATE_ACCESS_GRANTED:
            # Wait briefly then command barrier to open
            self.barrier_command = "open"
            self.barrier_position = "opening"
            self._set_state(STATE_BARRIER_OPENING)

        elif self.state == STATE_ACCESS_DENIED:
            # Show denied for a few seconds, then reset
            if elapsed_in_state >= ACCESS_DENIED_TIMEOUT:
                self._set_state(STATE_IDLE)

        elif self.state == STATE_BARRIER_OPENING:
            # Keep "open" command sticky until UE5 acks open_complete.
            # This guarantees the command is delivered regardless of poll timing
            # and prevents sending "close" mid-opening-animation — which was
            # causing UE5 to drop the close command. ack_from_ue5() advances.
            self.barrier_command = "open"
            if elapsed_in_state > BARRIER_OPENING_TIMEOUT:
                logger.warning(
                    "[%s] BARRIER_OPENING timed out after %.0fs with no "
                    "open_complete ack — forcing advance",
                    self.barrier_id, BARRIER_OPENING_TIMEOUT,
                )
                self.barrier_position = "open"
                self.barrier_command = "idle"
                self._set_state(STATE_CAR_PASSING)

        elif self.state == STATE_CAR_PASSING:
            # Grace period: don't check safety zone immediately after opening.
            # The car needs time to drive from the plate camera to the safety zone.
            if elapsed_in_state < SAFETY_GRACE_PERIOD:
                return

            # Use safety camera detections if a separate safety camera is configured
            safety_dets = self._safety_detections if self._safety_camera_id else detections

            # Monitor safety zone — barrier stays open while car is present
            car_in_safety = self._any_car_in_zone(safety_dets, self.safety_zone)

            if car_in_safety:
                self.safety_clear_time = None
            else:
                if self.safety_clear_time is None:
                    self.safety_clear_time = now
                elif now - self.safety_clear_time >= SAFETY_ZONE_CLEAR_DELAY:
                    # Safety zone has been clear long enough — close barrier
                    self.barrier_command = "close"
                    self.barrier_position = "closing"
                    self._set_state(STATE_BARRIER_CLOSING)

            # Safety timeout: auto-close after long time with no car
            if elapsed_in_state > BARRIER_OPEN_TIMEOUT and not car_in_safety:
                self.barrier_command = "close"
                self.barrier_position = "closing"
                self._set_state(STATE_BARRIER_CLOSING)

        elif self.state == STATE_BARRIER_CLOSING:
            # Keep "close" command sticky until UE5 acks close_complete.
            # Long timeout gives a slow animation plenty of time to finish and
            # the UE5 poll loop multiple chances to pick up the command.
            self.barrier_command = "close"
            if elapsed_in_state > BARRIER_CLOSING_TIMEOUT:
                logger.warning(
                    "[%s] BARRIER_CLOSING timed out after %.0fs with no "
                    "close_complete ack — forcing advance",
                    self.barrier_id, BARRIER_CLOSING_TIMEOUT,
                )
                self.barrier_position = "closed"
                self.barrier_command = "idle"
                self._set_state(STATE_IDLE)

    def _accumulate_ocr(self, img: np.ndarray, car: dict):
        """Run plate detection/OCR once on this frame, accumulate the result,
        and immediately grant access if the plate is whitelisted.

        Asymmetric-grant rationale: a single positive DB hit is enough because
        the OCR text has to match an exact whitelisted plate; the cost of a
        false grant requires misreading some other plate AS a whitelisted one,
        which is rare. Denials still go through consensus — see _advance_state."""
        x, y, w, h = car["bbox"]
        car_bbox = (x, y, x + w, y + h)

        result = self.plate_recognizer.detect_plate(img, car_bbox)
        if not result or not result["plate_text"]:
            return

        self.ocr_buffer.append(result)

        plate_text = result["plate_text"]
        if self.db.is_plate_allowed(plate_text):
            self._grant_access(result)

    def _grant_access(self, reading: dict):
        """Single-reading access grant — bypasses consensus and opens immediately."""
        plate_text = reading["plate_text"]
        confidence = reading["confidence"]

        self.last_plate = plate_text
        self.last_plate_confidence = confidence
        self.last_plate_image = reading.get("plate_image")
        self.last_access_result = "granted"

        self.db.log_access(
            barrier_id=self.barrier_id,
            plate=plate_text,
            confidence=confidence,
            result="granted",
        )
        self.db.start_session(plate_text)

        info = self.db.get_plate_info(plate_text) or {}
        self.recent_events.appendleft({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "plate": plate_text,
            "owner": info.get("owner_name", ""),
            "confidence": round(confidence, 2),
            "result": "granted",
            "barrier_id": self.barrier_id,
        })

        logger.info(
            "[%s] INSTANT GRANT plate=%r conf=%.2f (after %d reading(s))",
            self.barrier_id, plate_text, confidence, len(self.ocr_buffer),
        )

        self._set_state(STATE_ACCESS_GRANTED)

    def _has_plate_consensus(self) -> bool:
        """True if enough OCR readings agree on the same plate text to decide."""
        if len(self.ocr_buffer) < PLATE_MIN_AGREEING_READS:
            return False
        from collections import Counter
        texts = [r["plate_text"] for r in self.ocr_buffer if r["plate_text"]]
        if not texts:
            return False
        _, count = Counter(texts).most_common(1)[0]
        return count >= PLATE_MIN_AGREEING_READS

    def _decide_access(self):
        """Make access decision based on accumulated OCR readings."""
        if not self.ocr_buffer:
            self._set_state(STATE_IDLE)
            return

        # Find most common plate text (majority vote)
        from collections import Counter
        texts = [r["plate_text"] for r in self.ocr_buffer if r["plate_text"]]
        if not texts:
            self._set_state(STATE_IDLE)
            return

        counter = Counter(texts)
        best_text, count = counter.most_common(1)[0]

        # Get best confidence reading for this plate
        matching = [r for r in self.ocr_buffer if r["plate_text"] == best_text]
        best_reading = max(matching, key=lambda r: r["confidence"])

        self.last_plate = best_text
        self.last_plate_confidence = best_reading["confidence"]
        self.last_plate_image = best_reading.get("plate_image")

        # Check database
        allowed = self.db.is_plate_allowed(best_text)
        result = "granted" if allowed else "denied"
        self.last_access_result = result

        # Log to database
        self.db.log_access(
            barrier_id=self.barrier_id,
            plate=best_text,
            confidence=best_reading["confidence"],
            result=result
        )

        # Start parking session on successful entry
        if allowed:
            self.db.start_session(best_text)

        # Add to recent events
        info = self.db.get_plate_info(best_text) or {}
        self.recent_events.appendleft({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "plate": best_text,
            "owner": info.get("owner_name", ""),
            "confidence": round(best_reading["confidence"], 2),
            "result": result,
            "barrier_id": self.barrier_id,
        })

        # Transition state
        if allowed:
            self._set_state(STATE_ACCESS_GRANTED)
        else:
            self._set_state(STATE_ACCESS_DENIED)

    def _handle_manual_override(self):
        """Handle manual barrier control from operator."""
        if self._manual_override == "open":
            self.barrier_command = "open"
            self.barrier_position = "opening"
            self._set_state(STATE_BARRIER_OPENING)
            self.last_access_result = "manual_override"

            self.recent_events.appendleft({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "plate": "MANUAL",
                "owner": "Operator",
                "confidence": 1.0,
                "result": "manual_override",
                "barrier_id": self.barrier_id,
            })
            self.db.log_access(self.barrier_id, "MANUAL", 1.0, "manual_override")

        elif self._manual_override == "close":
            self.barrier_command = "close"
            self.barrier_position = "closing"
            self._set_state(STATE_BARRIER_CLOSING)

        self._manual_override = None

    def manual_open(self):
        """Queue a manual open command."""
        self._manual_override = "open"

    def manual_close(self):
        """Queue a manual close command."""
        self._manual_override = "close"

    # --- UE5 Communication ---

    def get_barrier_api_state(self) -> dict:
        """Return state dict for GET /api/barrier/{id} (UE5 polling)."""
        return {
            "state": self.state,
            "command": self.barrier_command,
            "barrier_position": self.barrier_position,
            "last_plate": self.last_plate or "",
            "access": self.last_access_result or "none",
        }

    def ack_from_ue5(self, event: str):
        """
        Handle acknowledgment from UE5 (animation complete).

        Args:
            event: "open_complete" or "close_complete"
        """
        if event == "open_complete":
            if self.state == STATE_BARRIER_OPENING:
                self.barrier_position = "open"
                self.barrier_command = "idle"
                self._set_state(STATE_CAR_PASSING)
            else:
                logger.warning(
                    "[%s] got open_complete ack in state %r — ignoring "
                    "(state machine already advanced)",
                    self.barrier_id, self.state,
                )
        elif event == "close_complete":
            if self.state == STATE_BARRIER_CLOSING:
                self.barrier_position = "closed"
                self.barrier_command = "idle"
                self._set_state(STATE_IDLE)
            else:
                logger.warning(
                    "[%s] got close_complete ack in state %r — ignoring",
                    self.barrier_id, self.state,
                )

    # --- Overlay Drawing ---

    def draw_overlay(self, img: np.ndarray):
        """Draw approach + reading zones, state info, and recognition overlay
        on the plate camera frame (camera5). Safety zone is NOT drawn here —
        it belongs to the safety camera and is drawn by draw_safety_overlay()."""
        # Draw approach zone
        if self.approach_zone is not None:
            color = COLOR_APPROACH
            if self.state == STATE_CAR_APPROACHING:
                color = (0, 255, 255)
                alpha_zone = 0.35
            else:
                alpha_zone = 0.15
            zone_overlay = img.copy()
            cv2.fillPoly(zone_overlay, [self.approach_zone], color)
            cv2.addWeighted(zone_overlay, alpha_zone, img, 1 - alpha_zone, 0, img)
            cv2.polylines(img, [self.approach_zone], True, color, 2)
            self._draw_zone_label(img, self.approach_zone, "APPROACH", color)

        # Draw reading zone — red + SCANNING label while actively reading a plate
        if self.reading_zone is not None:
            scanning = (self.state == STATE_READING_PLATE)
            if scanning:
                color = (0, 0, 255)       # bright red
                alpha_zone = 0.45
            else:
                color = COLOR_READING
                alpha_zone = 0.2
            zone_overlay = img.copy()
            cv2.fillPoly(zone_overlay, [self.reading_zone], color)
            cv2.addWeighted(zone_overlay, alpha_zone, img, 1 - alpha_zone, 0, img)
            cv2.polylines(img, [self.reading_zone], True, color, 3 if scanning else 2)
            label = f"READING - SCANNING ({len(self.ocr_buffer)})" if scanning else "READING"
            self._draw_zone_label(img, self.reading_zone, label, color)

        # Draw state info panel (top-left)
        self._draw_state_panel(img)

        # Draw plate recognition result
        if self.last_plate and self.last_access_result:
            self._draw_plate_result(img)

    def draw_safety_overlay(self, img: np.ndarray):
        """Draw only safety zone + state panel on the safety camera frame (camera6)."""
        if self.safety_zone is not None:
            if self.state in (STATE_BARRIER_OPENING, STATE_BARRIER_CLOSING):
                color = COLOR_SAFETY_MOVING
            elif self.state == STATE_CAR_PASSING and self.safety_clear_time is None:
                color = COLOR_SAFETY_OCCUPIED
            else:
                color = COLOR_SAFETY_CLEAR
            zone_overlay = img.copy()
            cv2.fillPoly(zone_overlay, [self.safety_zone], color)
            cv2.addWeighted(zone_overlay, 0.2, img, 0.8, 0, img)
            cv2.polylines(img, [self.safety_zone], True, color, 2)
            self._draw_zone_label(img, self.safety_zone, "SAFETY", color)

        # Draw state panel
        self._draw_state_panel(img)

    def _draw_zone_label(self, img: np.ndarray, polygon: np.ndarray,
                         label: str, color: tuple):
        """Draw a label at the centroid of a zone polygon."""
        centroid = polygon.mean(axis=0).astype(int)[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cx, cy = centroid[0] - tw // 2, centroid[1]
        cv2.rectangle(img, (cx - 3, cy - th - 3), (cx + tw + 3, cy + 3), (0, 0, 0), -1)
        cv2.putText(img, label, (cx, cy), font, 0.5, color, 1)

    def _draw_state_panel(self, img: np.ndarray):
        """Draw barrier state info panel at top-left."""
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # State name
        state_display = self.state.replace("_", " ").upper()
        barrier_label = f"BARRIER: {self.barrier_id.upper()}"

        # Background panel
        panel_h = 70
        panel_w = 280
        cv2.rectangle(img, (10, 10), (10 + panel_w, 10 + panel_h), (30, 30, 30), -1)
        cv2.rectangle(img, (10, 10), (10 + panel_w, 10 + panel_h), (100, 100, 100), 1)

        # State indicator color
        state_colors = {
            STATE_IDLE: (128, 128, 128),
            STATE_CAR_APPROACHING: (0, 255, 255),
            STATE_READING_PLATE: (0, 165, 255),
            STATE_ACCESS_GRANTED: (0, 255, 0),
            STATE_ACCESS_DENIED: (0, 0, 255),
            STATE_BARRIER_OPENING: (0, 255, 0),
            STATE_CAR_PASSING: (255, 200, 0),
            STATE_BARRIER_CLOSING: (0, 128, 255),
        }
        indicator_color = state_colors.get(self.state, (128, 128, 128))

        # Indicator dot
        cv2.circle(img, (28, 30), 8, indicator_color, -1)

        # Labels
        cv2.putText(img, barrier_label, (45, 35), font, 0.5, (255, 255, 255), 1)
        cv2.putText(img, state_display, (20, 58), font, 0.5, indicator_color, 1)

        # Barrier position icon
        pos_text = f"[{self.barrier_position.upper()}]"
        cv2.putText(img, pos_text, (180, 58), font, 0.4, (200, 200, 200), 1)

    def _draw_plate_result(self, img: np.ndarray):
        """Draw the recognized plate and access result."""
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Position: bottom-center
        plate_text = self.last_plate
        result = self.last_access_result

        # Colors
        if result == "granted" or result == "manual_override":
            result_color = (0, 255, 0)
            result_text = "ACCESS GRANTED"
        elif result == "denied":
            result_color = (0, 0, 255)
            result_text = "ACCESS DENIED"
        else:
            result_color = (200, 200, 200)
            result_text = "PENDING"

        # Panel background
        panel_w = 320
        panel_h = 65
        px = (w - panel_w) // 2
        py = h - panel_h - 15

        cv2.rectangle(img, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
        cv2.rectangle(img, (px, py), (px + panel_w, py + panel_h), result_color, 2)

        # Plate text (large, monospace-style)
        (tw, th), _ = cv2.getTextSize(plate_text, font, 1.0, 2)
        text_x = px + (panel_w - tw) // 2
        cv2.putText(img, plate_text, (text_x, py + 28), font, 1.0, (255, 255, 255), 2)

        # Result text
        (tw2, th2), _ = cv2.getTextSize(result_text, font, 0.6, 1)
        text_x2 = px + (panel_w - tw2) // 2
        cv2.putText(img, result_text, (text_x2, py + 52), font, 0.6, result_color, 1)

    # --- Result Building ---

    def _build_result(self) -> dict:
        """Build the result dict for WebSocket/API."""
        return {
            "state": self.state,
            "barrier_id": self.barrier_id,
            "last_plate": self.last_plate or "",
            "last_plate_confidence": round(self.last_plate_confidence, 2),
            "access_result": self.last_access_result or "none",
            "barrier_position": self.barrier_position,
            "barrier_command": self.barrier_command,
            "recent_events": list(self.recent_events),
            "today_stats": self.db.get_today_stats(),
        }
