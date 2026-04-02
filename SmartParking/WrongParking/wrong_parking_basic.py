# Import necessary libraries
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

# ------------ Variables --------------
model_path = "../../Models/Car_Detector.pt"
confidence = 0.8
class_names = ["car"]

# If actual_ratio < expected_correct_ratio * THRESHOLD_FACTOR -> wrong parking
# 0.87 means: if ratio is less than 87% of what a correctly-parked car would have
# at that position, it's considered wrong parking
THRESHOLD_FACTOR = 0.87

# Path to calibration files
CALIBRATION_DIR = Path(__file__).parent

# Camera selection
print("=" * 50)
print("  Wrong Parking Detection")
print("=" * 50)
print()

while True:
    cam_num = input("Camera number (1 or 2): ").strip()
    if cam_num in ("1", "2"):
        break
    print("Please enter 1 or 2.")

camera_id = f"camera{cam_num}"
CALIBRATION_PATH = CALIBRATION_DIR / f"{camera_id}_calibration.pkl"

# Load the YOLO model
model = YOLO(model_path)

# Load unified calibration file
homography_matrix = None
adaptive_calib = None

if CALIBRATION_PATH.exists():
    try:
        with open(CALIBRATION_PATH, "rb") as f:
            calibration = pickle.load(f)

        homography_matrix = calibration.get("homography")
        if homography_matrix is not None:
            print(f"Loaded perspective calibration (homography)")

        if "near_y" in calibration and "far_y" in calibration:
            adaptive_calib = {
                "near_y": calibration["near_y"],
                "near_ratio": calibration["near_ratio"],
                "far_y": calibration["far_y"],
                "far_ratio": calibration["far_ratio"],
            }
            print(f"Loaded adaptive calibration:")
            print(f"  Near: Y={adaptive_calib['near_y']}, ratio={adaptive_calib['near_ratio']:.2f}")
            print(f"  Far:  Y={adaptive_calib['far_y']}, ratio={adaptive_calib['far_ratio']:.2f}")

    except Exception as e:
        print(f"Warning: Could not load calibration: {e}")

if adaptive_calib is None:
    print("WARNING: No calibration found!")
    print(f"  Run calibrate_perspective.py first for {camera_id}.")

opencv_window_hwnd = None

last_found_window_title = ""
window_found_before = False


def find_unreal_window(opencv_window_hwnd=None):
    """Search for an Unreal Engine window and return its adjusted coordinates."""
    global last_found_window_title, window_found_before

    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)

            if opencv_window_hwnd and hwnd == opencv_window_hwnd:
                return True

            if "Wrong Parking" in window_title or "Detection" in window_title:
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


def capture_unreal_window(opencv_window_hwnd=None):
    """Capture a screenshot of the Unreal Engine window."""
    window_region = find_unreal_window(opencv_window_hwnd)

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


def get_object_list_yolo(_model, _img, _class_names, _confidence=0.5, draw=True):
    """Detect objects in the image using YOLO and draw bounding boxes."""
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
                center = x1 + (w // 2), y1 + (h // 2)
                class_name = _class_names[int(box.cls[0])]

                _object_list.append({"bbox": (x1, y1, w, h),
                                     "center": center,
                                     "conf": conf,
                                     "class": class_name})

                if draw:
                    cvzone.cornerRect(_img, (x1, y1, w, h))
                    cvzone.putTextRect(_img, f'{class_name} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1)
    return _object_list


def compute_ratio(x, y, w, h, H=None):
    """
    Compute aspect ratio, optionally with perspective correction.

    Args:
        x, y, w, h: Bounding box
        H: Homography matrix (optional)

    Returns:
        float: max/min aspect ratio
    """
    if H is not None:
        corners = np.array([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

        t_w = (np.linalg.norm(transformed[1] - transformed[0]) +
               np.linalg.norm(transformed[2] - transformed[3])) / 2
        t_h = (np.linalg.norm(transformed[3] - transformed[0]) +
               np.linalg.norm(transformed[2] - transformed[1])) / 2

        long_side = max(t_w, t_h)
        short_side = min(t_w, t_h)
    else:
        long_side = max(h, w)
        short_side = min(h, w)

    return long_side / short_side if short_side > 0 else 0


def get_adaptive_threshold(y_center, calib):
    """
    Compute the adaptive threshold for a car at a given Y position.

    Interpolates between the calibrated near and far "correct" ratios,
    then applies THRESHOLD_FACTOR to get the actual wrong/correct boundary.

    Args:
        y_center: Y coordinate of the car's center
        calib: Calibration dict with near_y, near_ratio, far_y, far_ratio

    Returns:
        float: Threshold ratio below which parking is "wrong"
    """
    y_near = calib["near_y"]
    y_far = calib["far_y"]
    r_near = calib["near_ratio"]
    r_far = calib["far_ratio"]

    # Clamp t to [0, 1] with some extrapolation margin
    if y_near == y_far:
        t = 0.5
    else:
        t = (y_center - y_near) / (y_far - y_near)
        t = max(-0.2, min(1.2, t))  # Allow slight extrapolation

    # Interpolate expected "correct" ratio at this Y position
    expected_correct_ratio = r_near + t * (r_far - r_near)

    # The threshold = fraction of the expected correct ratio
    return expected_correct_ratio * THRESHOLD_FACTOR


def find_wrong_parking(_object_list, _img, _H=None, _calib=None):
    """
    Identify and label wrong parking with adaptive threshold.

    If adaptive calibration is available, the threshold varies by Y position.
    Otherwise falls back to a fixed THRESHOLD_FACTOR-based threshold.

    Parameters:
    - _object_list: List of detected objects.
    - _img: Input image.
    - _H: Homography matrix (optional).
    - _calib: Adaptive calibration dict (optional).
    """
    wrong_count = 0
    for obj in _object_list:
        x, y, w, h = obj["bbox"]
        cy = y + h // 2

        ratio = compute_ratio(x, y, w, h, _H)

        if _calib is not None:
            threshold = get_adaptive_threshold(cy, _calib)
        else:
            threshold = 1.4

        if ratio < threshold:
            color = (0, 0, 255)
            text = "Wrong"
            wrong_count += 1
        else:
            color = (0, 255, 0)
            text = "Correct"

        label_y = y
        if y < 200:
            label_y = y + h + 25

        cvzone.putTextRect(_img, f"{text} ({ratio:.2f}/{threshold:.2f})", (x, label_y),
                           scale=1.2, colorR=color, thickness=2)

    return wrong_count


window_name = f"Wrong Parking Detection - {camera_id}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

try:
    time.sleep(0.1)

    def find_opencv_window():
        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title == window_name:
                    global opencv_window_hwnd
                    opencv_window_hwnd = hwnd
                    return False
            return True

        win32gui.EnumWindows(enum_callback, None)

    find_opencv_window()
except:
    pass

print()
print(f"Starting wrong parking detection for {camera_id}...")
print(f"Threshold factor: {THRESHOLD_FACTOR}")
if adaptive_calib:
    print("Using adaptive calibration (position-dependent threshold)")
else:
    print("Using fixed threshold (run calibrate_perspective.py to calibrate)")
print()
print("Controls:")
print("  Q - Quit")
print()

while True:
    img = capture_unreal_window(opencv_window_hwnd)

    if img is not None:
        object_list = get_object_list_yolo(model, img, class_names, confidence, draw=True)

        wrong_count = find_wrong_parking(object_list, img, homography_matrix, adaptive_calib)
        total = len(object_list)
        mode_label = "adaptive" if adaptive_calib else "fixed"
        info_text = f"{camera_id} | Cars: {total} | Wrong: {wrong_count} | {mode_label}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, img)
    else:
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_img, "Searching for Unreal Engine 5 window...", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(blank_img, "Make sure UE5 is running and visible", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(blank_img, "Press 'q' to quit", (50, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.imshow(window_name, blank_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
print("Wrong parking detection stopped.")
