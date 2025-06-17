import math
import cv2
import cvzone
from ultralytics import YOLO
import mss
import numpy as np
import win32gui

# Model and configuration
model_path = "Car_Detector.pt"
confidence = 0.7
# class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#                "traffic light", "fire hydrant", "parking meter", "bench", "bird", "cat",
#                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#                "teddy bear", "hair drier", "toothbrush"]
class_names = ["car"]

model = YOLO(model_path)

# Global variable to store the HWND of OpenCV window
opencv_window_hwnd = None


def find_unreal_window(opencv_window_hwnd=None):
    """
    Search for an Unreal Engine window and return its adjusted coordinates.

    Args:
        opencv_window_hwnd (int or None): Handle to an OpenCV window to exclude from the search.

    Returns:
        dict or None: A dictionary with the position and size of the found Unreal Engine window:
            {
                "top": int,      # Top Y-coordinate (adjusted for title bar)
                "left": int,     # Left X-coordinate (adjusted for border)
                "width": int,    # Width excluding window borders
                "height": int    # Height excluding title bar and bottom border
            }
            Returns None if no matching window is found.
    """

    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)

            # Skip the OpenCV window if provided
            if opencv_window_hwnd and hwnd == opencv_window_hwnd:
                return True

            # Skip windows related to our application
            if "Object Detection" in window_title or "Detection" in window_title:
                return True

            # Include only Unreal Engine windows
            if ("Unreal Editor" in window_title or
                "UE5" in window_title or
                "UnrealEditor" in window_title or
                window_title.endswith(" - Unreal Editor")):
                windows.append((hwnd, window_title))
        return True

    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)

    if windows:
        # Use the first matching Unreal Engine window
        hwnd, title = windows[0]
        print(f"Found Unreal Engine window: {title}")
        rect = win32gui.GetWindowRect(hwnd)

        # Estimate border and title bar size
        border_width = 8
        title_height = 30

        return {
            "top": rect[1] + title_height,
            "left": rect[0] + border_width,
            "width": rect[2] - rect[0] - (border_width * 2),
            "height": rect[3] - rect[1] - title_height - border_width
        }

    return None


def capture_unreal_window(opencv_window_hwnd=None):
    """
    Capture a screenshot of the Unreal Engine window.

    Args:
        opencv_window_hwnd (int or None): Handle to an OpenCV window to exclude from detection.

    Returns:
        numpy.ndarray or None: Captured image as a NumPy array in BGR format.
            Returns None if the Unreal Engine window is not found or if an error occurs.
    """
    window_region = find_unreal_window(opencv_window_hwnd)

    if window_region is None:
        print("Unreal Engine window not found!")
        return None

    try:
        with mss.mss() as sct:
            # Capture the specified region of the screen
            screenshot = sct.grab(window_region)

            # Convert the raw image to a NumPy array and then to BGR format
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
    except Exception as e:
        print(f"Error capturing window: {e}")
        return None


def get_object_list_yolo(_model, _img, _class_names, _confidence=0.5, draw=True):
    """Detect objects in the image using YOLO and draw bounding boxes"""
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


# Create the unique-name-window
window_name = "AI Object Detection - UE5"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Receive HWND OpenCV HWND window to delete it from search
try:
    import time

    time.sleep(0.1)  # Delay to create a window

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

print("Starting object detection for Unreal Engine 5...")
print("Press 'q' to quit or close the window")

# Main loop
frame_count = 0
while True:
    img = capture_unreal_window()

    if img is not None:
        object_list = get_object_list_yolo(model, img, class_names, confidence, draw=True)

        # Help information about number of found objects
        info_text = f"Objects found: {len(object_list)} | Frame: {frame_count}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, img)
        frame_count += 1
    else:
        # Display a message if capture fails
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_img, "Searching for Unreal Engine 5 window...", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(blank_img, "Make sure UE5 is running and visible", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(blank_img, "Press 'q' to quit", (50, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.imshow(window_name, blank_img)

    # Handle window close or 'q' key to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
print("Object detection stopped.")