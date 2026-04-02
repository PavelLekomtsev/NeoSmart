"""
Parking Detector Module
Combines car counting and parking space detection functionality.
"""

import math
import cv2
import cvzone
import numpy as np
import pickle
import mss
import win32gui
from pathlib import Path
from ultralytics import YOLO


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
    OUTSIDE_THRESHOLD_DEFAULT = 35
    OUTSIDE_THRESHOLD_EDGE = 45

    # Per-camera edge polygon indices (fish-eye distortion → higher threshold)
    # camera1: first 2 (left edge) and last 2 (right edge)
    # camera2: last 2 (far edge)
    EDGE_INDICES = {
        "camera1": lambda n: {0, 1, n - 2, n - 1},
        "camera2": lambda n: {n - 2, n - 1},
    }

    CAMERA_IDS = ["camera1", "camera2"]

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
        self.confidence = 0.8

        # Per-camera polygons and stats
        self.camera_polygons = {}
        self.camera_total_spaces = {}
        self.camera_stats = {}

        for cam_id in self.CAMERA_IDS:
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
        edge_fn = self.EDGE_INDICES.get(camera_id)
        if edge_fn and index in edge_fn(total):
            return self.OUTSIDE_THRESHOLD_EDGE
        return self.OUTSIDE_THRESHOLD_DEFAULT

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

    def process_frame(self, img: np.ndarray, camera_id: str = "camera1") -> tuple[np.ndarray, dict]:
        """
        Process a frame with current detection mode using per-camera polygons.

        Args:
            img: Input image in BGR format
            camera_id: Which camera this frame belongs to

        Returns:
            Tuple of (processed_image, stats_dict)
        """
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

        stats = {
            "total_spaces": total_spaces,
            "occupied": occupied,
            "available": available,
            "cars_detected": cars_detected,
            "wrong_count": wrong_count,
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
