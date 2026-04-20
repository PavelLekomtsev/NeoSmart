"""
Parking Detector Module
Combines car counting and parking space detection functionality.
"""

import math
import time
import cv2
import cvzone
import numpy as np
import pickle
import mss
import win32gui
from pathlib import Path
from ultralytics import YOLO
from sort import Sort


class ParkingDetector:
    """
    A class that handles parking detection with two modes:
    1. Parking spaces mode - detects occupied/free parking spots using polygons
    2. Car counter mode - simple car counting without polygons
    """

    # Detection modes
    MODE_PARKING_SPACES = "parking_spaces"
    MODE_CAR_COUNTER = "car_counter"

    # Wrong parking: % of bbox outside polygon to classify as wrong
    OUTSIDE_THRESHOLD_DEFAULT = 25
    OUTSIDE_THRESHOLD_EDGE = 45

    # Per-camera default threshold overrides
    OUTSIDE_THRESHOLD_PER_CAMERA = {
        "camera2": 31,
        "camera3": 36,
    }

    # Per-camera edge polygon indices (fish-eye distortion → higher threshold)
    # camera1: first 2 (left edge) and last 2 (right edge)
    # camera2: last 2 (far edge)
    EDGE_INDICES = {
        "camera1": lambda n: {0, 1, n - 2, n - 1},
        "camera2": lambda n: {n - 2, n - 1},
    }

    # Per-camera per-index threshold overrides (takes priority over edge/default)
    OUTSIDE_THRESHOLD_OVERRIDES = {
        "camera1": lambda n: {0: 52, 1: 52},
        "camera3": lambda n: {0: 46, 1: 46},
    }

    CAMERA_IDS = ["camera1", "camera2", "camera3", "camera4", "camera5", "camera6"]

    # Road cameras (traffic counting, no parking spaces)
    ROAD_CAMERA_IDS = ["camera4"]

    # Cameras that use SORT tracking (paid zone only)
    TRACKING_CAMERA_IDS = ["camera3"]

    # Barrier cameras (plate recognition + barrier control, no parking logic)
    BARRIER_CAMERA_IDS = ["camera5", "camera6"]

    # SORT tracker parameters
    SORT_MAX_AGE = 20       # frames to keep a track alive without detections
    SORT_MIN_HITS = 5       # min detections before track is confirmed
    SORT_IOU_THRESHOLD = 0.3

    # Suspicious parking: time threshold in seconds
    SUSPICIOUS_TIME_THRESHOLD = 30

    # Traffic counting: SORT parameters for road cameras
    TRAFFIC_SORT_MAX_AGE = 50      # keep tracks alive longer (cars may be briefly occluded)
    TRAFFIC_SORT_MIN_HITS = 0      # show track on creation frame (hit_streak>=0 is always true)
    TRAFFIC_SORT_IOU_THRESHOLD = 0.15  # lenient matching to survive detection gaps

    # Fallback crossing regions (used only if no calibration pickle exists)
    CROSSING_REGIONS_FALLBACK = {
        "camera4": {
            "incoming": [[400, 350], [600, 350], [600, 380], [400, 380]],
            "outgoing": [[400, 450], [600, 450], [600, 480], [400, 480]],
        }
    }

    def __init__(self, model_path: str = None):
        """
        Initialize the parking detector with per-camera polygon support.

        Args:
            model_path: Path to YOLO model file
        """
        base_dir = Path(__file__).parent
        parking_dir = base_dir.parent / "CarParkingSpace"

        if model_path is None:
            model_path = str(base_dir.parent.parent / "Models" / "Car_Detector.pt")

        self.model = YOLO(model_path)
        self.class_names = ["car"]
        self.confidence = 0.65

        # Per-camera polygons and stats
        self.camera_polygons = {}
        self.camera_total_spaces = {}
        self.camera_stats = {}

        for cam_id in self.CAMERA_IDS:
            # Barrier cameras don't have parking spaces (handled by BarrierController)
            if cam_id in self.BARRIER_CAMERA_IDS:
                self.camera_polygons[cam_id] = []
                self.camera_total_spaces[cam_id] = 0
                self.camera_stats[cam_id] = {
                    "total_spaces": 0,
                    "occupied": 0,
                    "available": 0,
                    "cars_detected": 0,
                    "wrong_count": 0
                }
                print(f"Barrier camera {cam_id} initialized (plate recognition mode)")
                continue

            # Road cameras don't have parking spaces
            if cam_id in self.ROAD_CAMERA_IDS:
                self.camera_polygons[cam_id] = []
                self.camera_total_spaces[cam_id] = 0
                self.camera_stats[cam_id] = {
                    "total_spaces": 0,
                    "occupied": 0,
                    "available": 0,
                    "cars_detected": 0,
                    "wrong_count": 0
                }
                print(f"Road camera {cam_id} initialized (traffic counting only)")
                continue

            polygons = []
            cam_file = parking_dir / f"{cam_id}_parkings.p"
            fallback_file = parking_dir / "polygons.p"

            if cam_file.exists():
                try:
                    with open(cam_file, 'rb') as f:
                        polygons = pickle.load(f)
                    print(f"Loaded {len(polygons)} polygons for {cam_id}")
                except Exception as e:
                    print(f"Warning: Could not load {cam_file}: {e}")
            elif fallback_file.exists():
                try:
                    with open(fallback_file, 'rb') as f:
                        polygons = pickle.load(f)
                    print(f"Loaded {len(polygons)} polygons from fallback for {cam_id}")
                except Exception as e:
                    print(f"Warning: Could not load {fallback_file}: {e}")

            total = len(polygons) if polygons else 12
            self.camera_polygons[cam_id] = polygons
            self.camera_total_spaces[cam_id] = total
            self.camera_stats[cam_id] = {
                "total_spaces": total,
                "occupied": 0,
                "available": total,
                "cars_detected": 0,
                "wrong_count": 0
            }

        # Legacy single-camera compat
        self.polygons = self.camera_polygons.get("camera1", [])
        self.total_spaces = self.camera_total_spaces.get("camera1", 12)

        self.mode = self.MODE_PARKING_SPACES

        self._last_found_window_title = ""
        self._window_found_before = False

        self.last_stats = self.camera_stats.get("camera1", {
            "total_spaces": 12,
            "occupied": 0,
            "available": 12,
            "cars_detected": 0,
            "wrong_count": 0
        })

        # SORT trackers only for paid zone cameras
        self.camera_trackers = {}
        self.camera_track_times = {}  # {cam_id: {track_id: first_seen_timestamp}}
        self.camera_display_ids = {}  # {cam_id: {sort_id: display_id}}
        self.camera_next_display_id = {}  # {cam_id: next_id}
        for cam_id in self.TRACKING_CAMERA_IDS:
            self.camera_trackers[cam_id] = Sort(
                max_age=self.SORT_MAX_AGE,
                min_hits=self.SORT_MIN_HITS,
                iou_threshold=self.SORT_IOU_THRESHOLD
            )
            self.camera_track_times[cam_id] = {}
            self.camera_display_ids[cam_id] = {}
            self.camera_next_display_id[cam_id] = 1
        print(f"SORT trackers initialized for {self.TRACKING_CAMERA_IDS} (suspicious threshold: {self.SUSPICIOUS_TIME_THRESHOLD}s)")

        # SORT trackers for traffic counting (road cameras)
        self.traffic_trackers = {}
        self.traffic_display_ids = {}
        self.traffic_next_display_id = {}
        self.traffic_crossed = {}   # {cam_id: {"incoming": set(), "outgoing": set()}}
        self.traffic_counts = {}    # {cam_id: {"incoming": int, "outgoing": int}}
        self.traffic_flash_times = {}  # {cam_id: {"incoming": timestamp, "outgoing": timestamp}}
        for cam_id in self.ROAD_CAMERA_IDS:
            self.traffic_trackers[cam_id] = Sort(
                max_age=self.TRAFFIC_SORT_MAX_AGE,
                min_hits=self.TRAFFIC_SORT_MIN_HITS,
                iou_threshold=self.TRAFFIC_SORT_IOU_THRESHOLD
            )
            self.traffic_display_ids[cam_id] = {}
            self.traffic_next_display_id[cam_id] = 1
            self.traffic_crossed[cam_id] = {"incoming": set(), "outgoing": set()}
            self.traffic_counts[cam_id] = {"incoming": 0, "outgoing": 0}
            self.traffic_flash_times[cam_id] = {"incoming": 0.0, "outgoing": 0.0}
        # Load crossing regions from pickle or use fallback
        traffic_dir = base_dir.parent / "TrafficCounting"
        self.crossing_regions = {}
        for cam_id in self.ROAD_CAMERA_IDS:
            crossing_file = traffic_dir / f"{cam_id}_crossing.p"
            if crossing_file.exists():
                try:
                    with open(crossing_file, "rb") as f:
                        data = pickle.load(f)
                    self.crossing_regions[cam_id] = {
                        "incoming": data["incoming"],
                        "outgoing": data["outgoing"],
                    }
                    print(f"Loaded crossing regions for {cam_id} from {crossing_file}")
                except Exception as e:
                    print(f"Warning: Could not load {crossing_file}: {e}")
                    self.crossing_regions[cam_id] = self.CROSSING_REGIONS_FALLBACK.get(cam_id, {})
            else:
                self.crossing_regions[cam_id] = self.CROSSING_REGIONS_FALLBACK.get(cam_id, {})
                print(f"No calibration for {cam_id}, using fallback regions. "
                      f"Run TrafficCounting/mark_crossing_regions.py to calibrate.")

        if self.ROAD_CAMERA_IDS:
            print(f"Traffic trackers initialized for {self.ROAD_CAMERA_IDS}")

    def set_mode(self, mode: str):
        """Set detection mode."""
        if mode in [self.MODE_PARKING_SPACES, self.MODE_CAR_COUNTER]:
            self.mode = mode

    def find_unreal_window(self) -> dict | None:
        """
        Search for an Unreal Engine window and return its coordinates.

        Returns:
            dict with top, left, width, height or None if not found
        """
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)

                if any(x in window_title for x in ["Detection", "Parking", "Smart Parking"]):
                    return True

                if ("Unreal Editor" in window_title or
                    "UE5" in window_title or
                    "UnrealEditor" in window_title or
                    window_title.endswith(" - Unreal Editor")):
                    windows.append((hwnd, window_title))
            return True

        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)

        if windows:
            hwnd, title = windows[0]

            if not self._window_found_before or self._last_found_window_title != title:
                print(f"Found Unreal Engine window: {title}")
                self._last_found_window_title = title
                self._window_found_before = True

            rect = win32gui.GetWindowRect(hwnd)

            border_width = 8
            title_height = 30

            return {
                "top": rect[1] + title_height,
                "left": rect[0] + border_width,
                "width": rect[2] - rect[0] - (border_width * 2),
                "height": rect[3] - rect[1] - title_height - border_width
            }
        else:
            if self._window_found_before:
                print("Unreal Engine window lost")
                self._window_found_before = False
                self._last_found_window_title = ""

        return None

    def capture_frame(self) -> np.ndarray | None:
        """
        Capture a screenshot of the Unreal Engine window.

        Returns:
            numpy.ndarray in BGR format or None if capture fails
        """
        window_region = self.find_unreal_window()

        if window_region is None:
            return None

        try:
            with mss.mss() as sct:
                screenshot = sct.grab(window_region)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
        except Exception as e:
            print(f"Error capturing window: {e}")
            return None

    def detect_cars(self, img: np.ndarray, draw: bool = True) -> list:
        """
        Detect cars in the image using YOLO.

        Args:
            img: Input image in BGR format
            draw: Whether to draw bounding boxes

        Returns:
            List of detected objects with bbox, center, confidence, class
        """
        results = self.model(img, stream=False, verbose=False)
        object_list = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                if conf > self.confidence:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    center = (x1 + w // 2, y1 + h // 2)
                    class_name = self.class_names[int(box.cls[0])]

                    object_list.append({
                        "bbox": (x1, y1, w, h),
                        "center": center,
                        "conf": conf,
                        "class": class_name
                    })

                    if draw:
                        cvzone.cornerRect(img, (x1, y1, w, h))
                        cvzone.putTextRect(img, f'{class_name} {conf}',
                                          (max(0, x1), max(35, y1)),
                                          scale=1, thickness=1)

        return object_list

    def _get_threshold(self, camera_id: str, index: int, total: int) -> float:
        """Get wrong-parking threshold for a polygon by camera and index."""
        override_fn = self.OUTSIDE_THRESHOLD_OVERRIDES.get(camera_id)
        if override_fn:
            overrides = override_fn(total)
            if index in overrides:
                return overrides[index]
        edge_fn = self.EDGE_INDICES.get(camera_id)
        if edge_fn and index in edge_fn(total):
            return self.OUTSIDE_THRESHOLD_EDGE
        return self.OUTSIDE_THRESHOLD_PER_CAMERA.get(camera_id, self.OUTSIDE_THRESHOLD_DEFAULT)

    def overlay_parking_spaces(self, img: np.ndarray, object_list: list,
                               polygons: list = None,
                               camera_id: str = "camera1") -> tuple[int, int]:
        """
        Overlay parking space polygons and detect wrong parking.

        Free spots are green, occupied spots are red.
        If a car's bbox extends beyond the threshold outside its polygon,
        it gets a red bbox and a "Wrong" label.

        Returns:
            (occupied_count, wrong_count)
        """
        if polygons is None:
            polygons = self.polygons
        if not polygons:
            return 0, 0

        overlay = img.copy()
        occupied_count = 0
        wrong_count = 0
        wrong_cars = []
        total = len(polygons)

        for i, polygon in enumerate(polygons):
            polygon_array = np.array(polygon, np.int32).reshape((-1, 1, 2))

            car_in_spot = None
            for obj in object_list:
                if cv2.pointPolygonTest(polygon_array, obj["center"], False) >= 0:
                    car_in_spot = obj
                    break

            if car_in_spot is None:
                cv2.fillPoly(overlay, [polygon_array], (0, 255, 0))  # Green - free
            else:
                cv2.fillPoly(overlay, [polygon_array], (0, 0, 255))  # Red - occupied
                occupied_count += 1

                threshold = self._get_threshold(camera_id, i, total)
                pct = self._compute_outside_percentage(
                    polygon, car_in_spot["bbox"], img.shape)
                if pct > threshold:
                    wrong_count += 1
                    wrong_cars.append((car_in_spot["bbox"], pct))

        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        # Draw wrong parking indicators on top of everything
        for bbox, pct in wrong_cars:
            x, y, w, h = bbox
            cvzone.cornerRect(img, (x, y, w, h),
                              colorR=(0, 0, 255), colorC=(0, 0, 255))
            label_y = y if y >= 200 else y + h + 25
            cvzone.putTextRect(img, f"Wrong ({pct:.0f}%)", (x, label_y),
                               scale=1.2, colorR=(0, 0, 255), thickness=2)

        return occupied_count, wrong_count

    def _compute_outside_percentage(self, polygon_pts: list, bbox: tuple,
                                       img_shape: tuple) -> float:
        """Calculate what % of a car's bbox is outside its parking polygon."""
        height, width = img_shape[:2]

        poly_mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(polygon_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(poly_mask, [pts], 255)

        car_mask = np.zeros((height, width), dtype=np.uint8)
        x, y, w, h = bbox
        cv2.rectangle(car_mask, (x, y), (x + w, y + h), 255, -1)

        total_car_area = cv2.countNonZero(car_mask)
        if total_car_area == 0:
            return 0.0

        intersection = cv2.bitwise_and(poly_mask, car_mask)
        intersection_area = cv2.countNonZero(intersection)
        return ((total_car_area - intersection_area) / total_car_area) * 100

    def _convert_detections_for_sort(self, object_list: list) -> np.ndarray:
        """Convert detector object_list to SORT format [[x1,y1,x2,y2,conf], ...]."""
        if not object_list:
            return np.empty((0, 5))
        detections = np.empty((0, 5))
        for obj in object_list:
            x1, y1, w, h = obj["bbox"]
            x2, y2 = x1 + w, y1 + h
            detections = np.vstack([detections, [x1, y1, x2, y2, obj["conf"]]])
        return detections

    def _get_parking_spot_index(self, center: tuple, camera_id: str) -> int:
        """Return the index of the parking polygon the point is in, or -1 if none."""
        polygons = self.camera_polygons.get(camera_id, [])
        for i, polygon in enumerate(polygons):
            polygon_array = np.array(polygon, np.int32).reshape((-1, 1, 2))
            if cv2.pointPolygonTest(polygon_array, center, False) >= 0:
                return i
        return -1

    def _update_tracking(self, img: np.ndarray, object_list: list,
                         camera_id: str) -> tuple[int, int]:
        """
        Run SORT tracker.
        - Track ID is always shown for every tracked car.
        - Timer starts only when a car enters a parking spot.
        - Timer resets if the car changes to a different spot.
        - Blue frame while parked, red frame + SUSPICIOUS after threshold.

        Returns:
            (tracked_count, suspicious_count)
        """
        tracker = self.camera_trackers[camera_id]
        track_times = self.camera_track_times[camera_id]
        display_ids = self.camera_display_ids[camera_id]
        # track_times stores {display_id: (start_time, spot_index)}
        now = time.time()

        dets = self._convert_detections_for_sort(object_list)
        tracks = tracker.update(dets)

        active_display_ids = set()
        suspicious_count = 0

        for track in tracks:
            x1, y1, x2, y2, sort_id = track
            x1, y1, x2, y2, sort_id = int(x1), int(y1), int(x2), int(y2), int(sort_id)

            # Map internal SORT ID to sequential display ID
            if sort_id not in display_ids:
                display_ids[sort_id] = self.camera_next_display_id[camera_id]
                self.camera_next_display_id[camera_id] += 1
            display_id = display_ids[sort_id]
            active_display_ids.add(display_id)

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            spot_index = self._get_parking_spot_index(center, camera_id)

            if spot_index >= 0:
                # Car is in a parking spot
                if display_id in track_times:
                    prev_time, prev_spot = track_times[display_id]
                    if prev_spot != spot_index:
                        # Changed spot — reset timer
                        track_times[display_id] = (now, spot_index)
                else:
                    # First time entering a spot
                    track_times[display_id] = (now, spot_index)

                start_time, _ = track_times[display_id]
                elapsed = now - start_time
                is_suspicious = elapsed >= self.SUSPICIOUS_TIME_THRESHOLD

                if is_suspicious:
                    suspicious_count += 1

                self._draw_track_info(img, x1, y1, x2, y2, display_id, elapsed, is_suspicious)
            else:
                # Car is not in any spot — show ID only, reset timer
                track_times.pop(display_id, None)
                self._draw_track_id(img, x1, y1, x2, y2, display_id)

        # Clean up tracks that are no longer active
        stale_ids = [tid for tid in track_times if tid not in active_display_ids]
        for tid in stale_ids:
            del track_times[tid]

        # Clean up stale display ID mappings
        active_sort_ids = {int(t[4]) for t in tracks}
        stale_sort_ids = [sid for sid in display_ids if sid not in active_sort_ids]
        for sid in stale_sort_ids:
            del display_ids[sid]

        return len(tracks), suspicious_count

    def _draw_track_id(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       track_id: int):
        """Draw only the track ID (car not in a parking spot yet)."""
        label_y = y2 + 20 if y2 + 20 < img.shape[0] - 10 else y1 - 10
        cvzone.putTextRect(img, f"ID:{track_id}", (max(0, x1), label_y),
                           scale=0.8, thickness=1,
                           colorR=(255, 200, 0), colorT=(255, 255, 255))

    def _draw_track_info(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                         track_id: int, elapsed: float, is_suspicious: bool):
        """Draw tracking ID + parked duration. Suspicious info in a box to the right of bbox."""
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        time_str = f"{minutes}:{seconds:02d}" if minutes > 0 else f"{seconds}s"

        if is_suspicious:
            color = (0, 0, 255)  # Red
        else:
            color = (255, 0, 0)  # Blue

        # ID label below bbox
        label_y = y2 + 20 if y2 + 20 < img.shape[0] - 10 else y1 - 10
        cvzone.putTextRect(img, f"ID:{track_id} {time_str}", (max(0, x1), label_y),
                           scale=0.8, thickness=1,
                           colorR=color, colorT=(255, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Suspicious box to the right of bbox
        if is_suspicious:
            box_x = x2 + 5
            box_y = y1
            overtime = elapsed - self.SUSPICIOUS_TIME_THRESHOLD
            ot_min = int(overtime) // 60
            ot_sec = int(overtime) % 60
            ot_str = f"{ot_min}:{ot_sec:02d}" if ot_min > 0 else f"{ot_sec}s"
            line1 = "SUSPICIOUS"
            line2 = f"+{ot_str}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
            (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
            box_w = max(w1, w2) + 12
            box_h = h1 + h2 + 18
            # Clamp to image bounds
            img_h, img_w = img.shape[:2]
            if box_x + box_w > img_w:
                box_x = x1 - box_w - 5
            if box_y + box_h > img_h:
                box_y = img_h - box_h
            if box_x < 0:
                box_x = 0
            if box_y < 0:
                box_y = 0
            # Background
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 180), -1)
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 255), 2)
            # Text
            cv2.putText(img, line1, (box_x + 6, box_y + h1 + 5), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(img, line2, (box_x + 6, box_y + h1 + h2 + 13), font, font_scale, (255, 255, 255), thickness)

    def _update_traffic_tracking(self, img: np.ndarray, object_list: list,
                                  camera_id: str) -> dict:
        """
        Run SORT tracker for traffic counting.
        Counts cars crossing two directional regions.

        Returns:
            {"incoming": int, "outgoing": int}
        """
        tracker = self.traffic_trackers[camera_id]
        display_ids = self.traffic_display_ids[camera_id]
        crossed = self.traffic_crossed[camera_id]
        flash_times = self.traffic_flash_times[camera_id]
        regions = self.crossing_regions.get(camera_id, {})
        now = time.time()

        dets = self._convert_detections_for_sort(object_list)
        tracks = tracker.update(dets)

        # Prepare region polygons
        region_arrays = {}
        for direction, pts in regions.items():
            region_arrays[direction] = np.array(pts, np.int32).reshape(-1, 1, 2)

        active_sort_ids = set()

        for track in tracks:
            x1, y1, x2, y2, sort_id = track
            x1, y1, x2, y2, sort_id = int(x1), int(y1), int(x2), int(y2), int(sort_id)
            active_sort_ids.add(sort_id)

            # Map to sequential display ID
            if sort_id not in display_ids:
                display_ids[sort_id] = self.traffic_next_display_id[camera_id]
                self.traffic_next_display_id[camera_id] += 1
            display_id = display_ids[sort_id]

            w, h = x2 - x1, y2 - y1
            center = (x1 + w // 2, y1 + h // 2)

            # Draw center point (used for crossing detection)
            cv2.circle(img, center, 4, (0, 255, 255), -1)

            # Check crossing for each direction
            for direction, poly_np in region_arrays.items():
                if cv2.pointPolygonTest(poly_np, center, False) >= 0:
                    if display_id not in crossed[direction]:
                        crossed[direction].add(display_id)
                        self.traffic_counts[camera_id][direction] += 1
                        flash_times[direction] = now

            # Small ID badge inside bbox (bottom-right corner)
            id_text = str(display_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(id_text, font, 0.45, 1)
            pad = 3
            bx = x2 - tw - pad * 2
            by = y2 - th - pad * 2
            cv2.rectangle(img, (bx, by), (x2, y2), (40, 40, 40), -1)
            cv2.putText(img, id_text, (bx + pad, y2 - pad), font, 0.45, (255, 255, 255), 1)

        # Clean up stale display ID mappings
        stale_sort_ids = [sid for sid in display_ids if sid not in active_sort_ids]
        for sid in stale_sort_ids:
            del display_ids[sid]

        # Draw crossing regions (flash red for 1s on new crossing)
        flash_duration = 1.0
        for direction, poly_np in region_arrays.items():
            count = self.traffic_counts[camera_id][direction]
            is_flashing = (now - flash_times.get(direction, 0)) < flash_duration

            if is_flashing:
                region_color = (0, 0, 255)  # Red flash
                alpha = 0.5
            elif direction == "incoming":
                region_color = (0, 200, 0)  # Green
                alpha = 0.25
            else:
                region_color = (0, 100, 255)  # Orange
                alpha = 0.25

            label = f"{'Incoming' if direction == 'incoming' else 'Outgoing'}: {count}"

            overlay = img.copy()
            cv2.fillPoly(overlay, [poly_np], region_color)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            cv2.polylines(img, [poly_np], True, region_color, 2)

            # Label near the region
            centroid = poly_np.mean(axis=0).astype(int)[0]
            cvzone.putTextRect(img, label, (centroid[0] - 60, centroid[1] - 20),
                               scale=0.7, thickness=1,
                               colorR=region_color, colorT=(255, 255, 255))

        # Draw total count
        total = self.traffic_counts[camera_id]["incoming"] + self.traffic_counts[camera_id]["outgoing"]
        cvzone.putTextRect(img, f"Total Traffic: {total}", (20, 40),
                           scale=1, thickness=2,
                           colorR=(50, 50, 50), colorT=(255, 255, 255))

        return self.traffic_counts[camera_id]

    def process_frame(self, img: np.ndarray, camera_id: str = "camera1") -> tuple[np.ndarray, dict]:
        """
        Process a frame with current detection mode using per-camera polygons.

        Args:
            img: Input image in BGR format
            camera_id: Which camera this frame belongs to

        Returns:
            Tuple of (processed_image, stats_dict)
        """
        # Barrier cameras: detection only, no parking/tracking logic
        # (BarrierController handles the rest in main.py)
        if camera_id in self.BARRIER_CAMERA_IDS:
            object_list = self.detect_cars(img, draw=True)
            stats = {
                "total_spaces": 0,
                "occupied": 0,
                "available": 0,
                "cars_detected": len(object_list),
                "wrong_count": 0,
                "tracked_count": 0,
                "suspicious_count": 0,
                "incoming_count": 0,
                "outgoing_count": 0,
                "mode": "barrier",
            }
            self.camera_stats[camera_id] = stats
            return img, stats, object_list

        object_list = self.detect_cars(img, draw=True)
        cars_detected = len(object_list)

        total_spaces = self.camera_total_spaces.get(camera_id, 12)
        polygons = self.camera_polygons.get(camera_id, [])

        wrong_count = 0

        if self.mode == self.MODE_PARKING_SPACES and polygons:
            occupied, wrong_count = self.overlay_parking_spaces(img, object_list, polygons, camera_id)
            available = total_spaces - occupied
        else:
            occupied = 0
            available = total_spaces

        # Run SORT tracking (paid zone only)
        tracked_count, suspicious_count = 0, 0
        if camera_id in self.TRACKING_CAMERA_IDS:
            tracked_count, suspicious_count = self._update_tracking(img, object_list, camera_id)

        # Run traffic counting (road cameras)
        incoming_count, outgoing_count = 0, 0
        if camera_id in self.ROAD_CAMERA_IDS:
            traffic = self._update_traffic_tracking(img, object_list, camera_id)
            incoming_count = traffic["incoming"]
            outgoing_count = traffic["outgoing"]

        stats = {
            "total_spaces": total_spaces,
            "occupied": occupied,
            "available": available,
            "cars_detected": cars_detected,
            "wrong_count": wrong_count,
            "tracked_count": tracked_count,
            "suspicious_count": suspicious_count,
            "incoming_count": incoming_count,
            "outgoing_count": outgoing_count,
            "mode": self.mode
        }
        self.camera_stats[camera_id] = stats
        self.last_stats = stats

        return img, stats

    def get_stats(self, camera_id: str = None) -> dict:
        """Get last computed statistics for a camera."""
        if camera_id and camera_id in self.camera_stats:
            return self.camera_stats[camera_id]
        return self.last_stats


def create_placeholder_image(message: str = "Waiting for Unreal Engine...") -> np.ndarray:
    """Create a placeholder image when UE5 is not available."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(480):
        img[i, :] = [int(20 + i * 0.05), int(20 + i * 0.03), int(30 + i * 0.02)]

    cv2.putText(img, message, (50, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(img, "Make sure UE5 is running", (50, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return img
