"""
Wrong Parking Detection - Advanced (ROI-based).

Detects wrong parking by checking how much of a car's bounding box
extends outside its assigned parking space polygon.

Captures frames in real-time from the Unreal Engine 5 window.
Loads parking polygons from CarParkingSpace/ (camera1_parkings.p / camera2_parkings.p).

Controls:
  Q - Quit
"""

import math
import cv2
import cvzone
import mss
import numpy as np
import pickle
import win32gui
import time
from pathlib import Path
from ultralytics import YOLO

# ------------ Paths --------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = str(BASE_DIR.parent.parent / "Models" / "Car_Detector.pt")
PARKING_DIR = BASE_DIR.parent / "CarParkingSpace"

# ------------ Settings --------------
confidence = 0.8
class_names = ["car"]

# If more than this % of the car bbox is outside its parking polygon → wrong parking
OUTSIDE_THRESHOLD = 20

# ------------ Camera selection --------------
print("=" * 50)
print("  Wrong Parking Detection (Advanced - ROI)")
print("=" * 50)
print()

while True:
    cam_num = input("Camera number (1 or 2): ").strip()
    if cam_num in ("1", "2"):
        break
    print("Please enter 1 or 2.")

camera_id = f"camera{cam_num}"

# ------------ Load model and data --------------
print(f"\nLoading YOLO model...")
model = YOLO(MODEL_PATH)

# Load parking polygons
polygon_path = PARKING_DIR / f"{camera_id}_parkings.p"
rois = []

if polygon_path.exists():
    try:
        with open(polygon_path, "rb") as f:
            rois = pickle.load(f)
        print(f"Loaded {len(rois)} parking polygons for {camera_id}")
    except Exception as e:
        print(f"Warning: Could not load polygons: {e}")
else:
    print(f"WARNING: No polygon file found: {polygon_path}")
    print(f"  Run mark_parking_spaces.py first for {camera_id}.")

# Global for OpenCV window HWND (to exclude from UE5 search)
opencv_window_hwnd = None
last_found_window_title = ""
window_found_before = False


def find_unreal_window(exclude_hwnd=None):
    """Search for an Unreal Engine window and return its screen coordinates."""
    global last_found_window_title, window_found_before

    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if exclude_hwnd and hwnd == exclude_hwnd:
                return True
            if "Wrong Parking" in title or "Detection" in title:
                return True
            if ("Unreal Editor" in title or
                    "UE5" in title or
                    "UnrealEditor" in title or
                    title.endswith(" - Unreal Editor")):
                windows.append((hwnd, title))
        return True

    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)

    if windows:
        hwnd, title = windows[0]
        if not window_found_before or last_found_window_title != title:
            print(f"Found Unreal Engine window: {title}")
            last_found_window_title = title
            window_found_before = True

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
        if window_found_before:
            print("Unreal Engine window lost")
            window_found_before = False
            last_found_window_title = ""
    return None


def capture_unreal_window(exclude_hwnd=None):
    """Capture a screenshot of the Unreal Engine window."""
    region = find_unreal_window(exclude_hwnd)
    if region is None:
        return None
    try:
        with mss.mss() as sct:
            screenshot = sct.grab(region)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        print(f"Capture error: {e}")
        return None


def get_object_list_yolo(_model, _img, _class_names, _confidence=0.5, draw=True):
    """Detect objects using YOLO model and return list of detections."""
    _results = _model(_img, stream=False, verbose=False)
    _object_list = []

    for r in _results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > _confidence:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                center = (x1 + w // 2, y1 + h // 2)
                class_name = _class_names[int(box.cls[0])]

                _object_list.append({
                    "bbox": (x1, y1, w, h),
                    "center": center,
                    "conf": conf,
                    "class": class_name
                })

                if draw:
                    cvzone.cornerRect(_img, (x1, y1, w, h))
                    cvzone.putTextRect(_img, f'{class_name} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1)
    return _object_list


def find_polygon_index(point, polygons):
    """Find which polygon contains the given point. Returns index or -1."""
    for idx, polygon_pts in enumerate(polygons):
        pts = np.array(polygon_pts, np.int32).reshape((-1, 1, 2))
        result = cv2.pointPolygonTest(pts, point, False)
        if result >= 0:
            return idx
    return -1


def compute_outside_percentage(polygon_pts, bbox, img_shape):
    """
    Calculate what percentage of a car's bbox is outside its parking polygon.

    Uses mask intersection: creates a mask for the polygon and a mask for the
    car bbox, then computes how much of the car area falls outside the polygon.

    Returns:
        float: percentage of bbox area outside the polygon (0-100)
    """
    height, width = img_shape[:2]

    # Mask for parking polygon
    poly_mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon_pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(poly_mask, [pts], 255)

    # Mask for car bbox
    car_mask = np.zeros((height, width), dtype=np.uint8)
    x, y, w, h = bbox
    cv2.rectangle(car_mask, (x, y), (x + w, y + h), 255, -1)

    total_car_area = cv2.countNonZero(car_mask)
    if total_car_area == 0:
        return 0.0

    # Area of car inside the polygon
    intersection = cv2.bitwise_and(poly_mask, car_mask)
    intersection_area = cv2.countNonZero(intersection)

    # Area of car outside the polygon
    outside_area = total_car_area - intersection_area
    return (outside_area / total_car_area) * 100


def detect_wrong_parking(object_list, img, polygons, threshold=OUTSIDE_THRESHOLD):
    """
    Check each detected car against its parking polygon.

    A car is "wrong" if too much of its bounding box extends outside
    the assigned parking space polygon.

    Returns:
        int: number of wrongly parked cars
    """
    wrong_count = 0

    for obj in object_list:
        center = obj["center"]
        bbox = obj["bbox"]
        x, y, w, h = bbox

        # Find which parking spot this car belongs to
        idx = find_polygon_index(center, polygons)
        if idx == -1:
            # Car center is not inside any parking spot
            label_y = y if y >= 200 else y + h + 25
            cvzone.putTextRect(img, "No spot", (x, label_y),
                               scale=1.2, colorR=(128, 128, 128), thickness=2)
            continue

        pct_outside = compute_outside_percentage(polygons[idx], bbox, img.shape)

        if pct_outside > threshold:
            color = (0, 0, 255)
            text = f"Wrong ({pct_outside:.0f}%)"
            wrong_count += 1
        else:
            color = (0, 255, 0)
            text = f"OK ({pct_outside:.0f}%)"

        label_y = y if y >= 200 else y + h + 25
        cvzone.putTextRect(img, text, (x, label_y),
                           scale=1.2, colorR=color, thickness=2)

    return wrong_count


def draw_parking_overlay(img, polygons):
    """Draw semi-transparent parking space polygons with indices."""
    overlay = img.copy()
    for i, poly in enumerate(polygons):
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (50, 50, 50))
        cv2.polylines(img, [pts], True, (255, 255, 0), 1)
        # Polygon number at centroid
        cx = sum(p[0] for p in poly) // len(poly)
        cy = sum(p[1] for p in poly) // len(poly)
        cv2.putText(img, str(i + 1), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)


# ------------ Window setup --------------
window_name = f"Wrong Parking Advanced - {camera_id}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Find OpenCV window HWND to exclude it from UE5 search
try:
    time.sleep(0.1)

    def _find_opencv_hwnd():
        global opencv_window_hwnd

        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                if win32gui.GetWindowText(hwnd) == window_name:
                    opencv_window_hwnd = hwnd
                    return False
            return True

        win32gui.EnumWindows(enum_callback, None)

    _find_opencv_hwnd()
except Exception:
    pass

print()
print(f"Starting wrong parking detection (advanced) for {camera_id}...")
print(f"Parking spots: {len(rois)}")
print(f"Threshold: {OUTSIDE_THRESHOLD}% outside")
print()
print("Controls:")
print("  Q - Quit")
print()

# ------------ Main loop --------------
while True:
    img = capture_unreal_window(opencv_window_hwnd)

    if img is not None:
        # Detect cars
        object_list = get_object_list_yolo(model, img, class_names, confidence, draw=True)

        # Draw parking spot overlays
        if rois:
            draw_parking_overlay(img, rois)

        # Check for wrong parking
        wrong_count = detect_wrong_parking(object_list, img, rois)
        total = len(object_list)

        info_text = f"{camera_id} | Cars: {total} | Wrong: {wrong_count} | Spots: {len(rois)}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, img)
    else:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Searching for Unreal Engine 5 window...", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(blank, "Make sure UE5 is running and visible", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(blank, "Press 'q' to quit", (50, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.imshow(window_name, blank)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
print("Wrong parking detection (advanced) stopped.")
