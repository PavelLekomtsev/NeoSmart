import math
import cv2
import cvzone
from ultralytics import YOLO
import pickle
import numpy as np
import mss
import win32gui

# --------------- Variables -------------------
model_path = "../../Models/Car_Detector.pt"
confidence = 0.8
class_names = ["car"]
polygon_file_path = 'polygons.p'

# Load previously defined Regions of Interest (ROIs) polygons from a file
file_obj = open(polygon_file_path, 'rb')
rois = pickle.load(file_obj)
file_obj.close()

# Load the YOLO model
model = YOLO(model_path)

# Global variable to store the HWND of OpenCV window
opencv_window_hwnd = None

last_found_window_title = ""
window_found_before = False

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
    global last_found_window_title, window_found_before

    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)

            # Skip the OpenCV window if provided
            if opencv_window_hwnd and hwnd == opencv_window_hwnd:
                return True

            # Skip windows related to our application
            if "Parking Detection" in window_title or "Detection" in window_title:
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

        if not window_found_before or last_found_window_title != title:
            print(f"Found Unreal Engine window: {title}")
            last_found_window_title = title
            window_found_before = True

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
    else:
        if window_found_before:
            print("Unreal Engine window lost")
            window_found_before = False
            last_found_window_title = ""

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

# Function to get a list of objects detected by YOLO in an image
def get_object_list_yolo(_model, _img, _class_names, _confidence=0.5, draw=True):
    """
        Detect objects using YOLO model.

        Parameters:
        - _model: YOLO model for object detection.
        - _img: Input image for object detection.
        - _class_names: List of class names to detect.
        - _confidence: Confidence threshold for object detection.
        - draw: Whether to draw bounding boxes on the image.

        Returns:
        - _object_list: List of dictionaries containing information about detected objects.
        """
    # Run YOLO on the input image
    _results = _model(_img, stream=False, verbose=False)
    _object_list = []

    # Iterate through the detected results
    for r in _results:
        boxes = r.boxes
        for box in boxes:
            # Extract information about the detected object
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > _confidence:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                center = x1 + (w // 2), y1 + (h // 2)
                class_name = _class_names[int(box.cls[0])]

                # Append information to the object list
                _object_list.append({"bbox": (x1, y1, w, h),
                                     "center": center,
                                     "conf": conf,
                                     "class": class_name})

                # Draw bounding box and class label on the image if specified
                if draw:
                    cvzone.cornerRect(_img, (x1, y1, w, h))
                    cvzone.putTextRect(_img, f'{class_name} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1)
    return _object_list

# Function to overlay polygons on the image based on the occupancy status
def overlay_polygons(_image, _object_list, _parking_spaces, _draw_occupied=False):
    """
    Overlay polygons on the image and return the count of occupied spaces.
    
    Returns:
        int: Number of occupied parking spaces
    """
    overlay = _image.copy()
    occupied_count = 0

    for parking_space in _parking_spaces:
        is_empty = True

        # Convert polygon to numpy array and reshape
        parking_space_array = np.array(parking_space, np.int32).reshape((-1, 1, 2))

        # Check if any car is present in this polygon
        for obj in _object_list:
            car_center = obj["center"]
            result = cv2.pointPolygonTest(parking_space_array, car_center, False)
            if result >= 0:
                is_empty = False
                occupied_count += 1
                break

        if is_empty:
            cv2.fillPoly(overlay, [parking_space_array], (0, 255, 0))  # Green for empty space
        if not is_empty and _draw_occupied:
            cv2.fillPoly(overlay, [parking_space_array], (0, 0, 255))  # Red for occupied space

    cv2.addWeighted(overlay, 0.35, _image, 0.65, 0, _image)
    return occupied_count

# Create the unique-name-window
window_name = "AI Parking Detection - UE5"
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

print("Starting parking detection for Unreal Engine 5...")
print("Press 'q' to quit or close the window")

# Main loop
while True:
    img = capture_unreal_window(opencv_window_hwnd)

    if img is not None:
        object_list = get_object_list_yolo(model, img, class_names, confidence)

        # Overlay polygons and get the count of occupied spaces
        occupied_spaces = overlay_polygons(img, object_list, rois, _draw_occupied=True)

        # Calculate available parking spaces correctly
        total_spaces = 12
        available_spaces = total_spaces - occupied_spaces
        
        if available_spaces == 0:
            color = (0, 0, 255)  # Red for no available spaces
        else:
            color = (0, 200, 0)  # Green for available spaces

        cvzone.putTextRect(img, f"Available: {available_spaces}/{total_spaces}", (20, 50), colorR=color)

        cv2.imshow(window_name, img)
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
print("Parking detection stopped.")