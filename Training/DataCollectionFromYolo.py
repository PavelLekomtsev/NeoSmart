import math
import os
import time
import cv2
import cvzone
from ultralytics import YOLO
import mss
import numpy as np
import win32gui

# Initialize variables
totalObjects = 5  # Total objects to detect before saving the image
model_path = "../Models/yolov8x.pt"  # Path to YOLO model
model = YOLO(model_path)
output_image_folder = "output_images"  # Folder to save output images and labels
classToDetect = ['truck', 'car']  # Classes to detect
imageFrequency = 2  # Capture every x image/frame
confidence = 0.1  # Confidence threshold for object detection
minArea, maxArea = 0, 10000000000  # Minimum and maximum area of detected objects
roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, 2560, 1440  # Define Region of Interest (ROI) coordinates

# Create output folder if it doesn't exist
os.makedirs(output_image_folder, exist_ok=True)

# List of class names for YOLO model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Global variable for OpenCV window HWND
opencv_window_hwnd = None


def get_window_client_area(hwnd):
    """Get the client area coordinates of a window (content area without borders/title)"""
    try:
        # Get window rectangle (includes borders and title bar)
        window_rect = win32gui.GetWindowRect(hwnd)

        # Get client rectangle (content area only)
        client_rect = win32gui.GetClientRect(hwnd)

        # Convert client coordinates to screen coordinates
        client_point = win32gui.ClientToScreen(hwnd, (0, 0))

        return {
            "top": client_point[1],
            "left": client_point[0],
            "width": client_rect[2],
            "height": client_rect[3]
        }
    except Exception as e:
        print(f"Error getting client area: {e}")
        return None


def find_unreal_window():
    """Find Unreal Engine window and return its client area coordinates"""

    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)

            # Exclude our OpenCV window
            if opencv_window_hwnd and hwnd == opencv_window_hwnd:
                return True

            # Exclude windows with detection-related names
            if "Detection" in window_title or "Image" in window_title:
                return True

            # Look for genuine Unreal Engine windows
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
        print(f"Found Unreal Engine window: {title}")

        # Get precise client area instead of estimating borders
        client_area = get_window_client_area(hwnd)

        if client_area:
            print(
                f"Client area: {client_area['width']}x{client_area['height']} at ({client_area['left']}, {client_area['top']})")
            return client_area
        else:
            # Fallback to old method if client area detection fails
            rect = win32gui.GetWindowRect(hwnd)
            return {
                "top": rect[1] + 30,
                "left": rect[0] + 8,
                "width": rect[2] - rect[0] - 16,
                "height": rect[3] - rect[1] - 38
            }
    return None


def capture_unreal_window():
    """Capture only the Unreal Engine window"""
    window_region = find_unreal_window()

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


# Create window with unique name
window_name = "Unreal Engine Object Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Get HWND of our OpenCV window to exclude from search
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

# Initialize variables
frame_count = 0
saved_images_count = 0
prev_frame_time = 0
new_frame_time = 0

print(f"Starting Unreal Engine object detection...")
print(f"Looking for classes: {classToDetect}")
print(f"Will save images when {totalObjects} objects are detected")
print(f"Output folder: {output_image_folder}")
print("Press 'q' to quit")

while True:
    frame_count += 1
    new_frame_time = time.time()

    # Capture image from Unreal Engine window
    img = capture_unreal_window()

    if img is None:
        # Display message if capture fails
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_img, "Searching for Unreal Engine window...", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(blank_img, "Make sure UE5 is running and visible", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow(window_name, blank_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Create display copy
    imgDisplay = img.copy()

    # Adjust ROI to image dimensions if necessary
    img_height, img_width = img.shape[:2]
    roi_x2 = min(roi_x2, img_width)
    roi_y2 = min(roi_y2, img_height)

    # Crop the Region of Interest (ROI) from the image
    img_roi = img[roi_y1:roi_y2, roi_x1:roi_x2]

    # Only process every (imageFrequency) frames
    if frame_count % imageFrequency != 0:
        # Calculate and display FPS
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time

        # Display info on image
        cv2.putText(imgDisplay, f"FPS: {int(fps)} | Frame: {frame_count} | Saved: {saved_images_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(imgDisplay, f"Processing every {imageFrequency} frames | Size: {img_width}x{img_height}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw ROI rectangle
        cv2.rectangle(imgDisplay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        cv2.putText(imgDisplay, "ROI", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow(window_name, imgDisplay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Apply the YOLO model only on the ROI
    results = model(img_roi, stream=True, verbose=False)
    annotation_data = []  # Store annotation data for detected objects
    detected_objects = 0

    # Process YOLO results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            area = (x2 - x1) * (y2 - y1)

            # Convert ROI coordinates back to full image coordinates
            x1, y1, x2, y2 = x1 + roi_x1, y1 + roi_y1, x2 + roi_x1, y2 + roi_y1
            name = classNames[cls]

            # Check if the detected object is in the classes to detect and meets criteria
            if name in classToDetect and conf > confidence and minArea < area < maxArea:
                detected_objects += 1

                # Highlight the detected object on the image
                cvzone.cornerRect(imgDisplay, (x1, y1, x2 - x1, y2 - y1))
                cvzone.putTextRect(imgDisplay, f'{name} {conf:.2f} A:{area}',
                                   (max(0, x1), max(35, y1)), scale=0.8, thickness=1)

                # Calculate YOLO format coordinates (normalized)
                w = x2 - x1
                h = y2 - y1
                x_center = (x1 + (w / 2)) / img_width
                y_center = (y1 + (h / 2)) / img_height
                w_norm = w / img_width
                h_norm = h / img_height

                # Check
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))

                # Append annotation data in YOLO format
                annotation_data.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time

    # Display information on image
    cv2.putText(imgDisplay,
                f"FPS: {int(fps)} | Objects: {detected_objects}/{totalObjects} | Saved: {saved_images_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw ROI rectangle
    cv2.rectangle(imgDisplay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    cv2.putText(imgDisplay, "ROI", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # If the specified number of objects is detected, save the image and corresponding label
    if len(annotation_data) == totalObjects:
        timeNow = time.time()
        image_name = f"unreal_{int(timeNow)}_{saved_images_count:04d}.jpg"
        label_name = f"unreal_{int(timeNow)}_{saved_images_count:04d}.txt"

        # Save original image (without annotations)
        cv2.imwrite(os.path.join(output_image_folder, image_name), img)

        # Save YOLO format labels
        with open(os.path.join(output_image_folder, label_name), "w") as f:
            for line in annotation_data:
                f.write(line + "\n")

        saved_images_count += 1
        print(f"Saved image {saved_images_count}: {image_name} with {len(annotation_data)} objects")
        print(f"Image dimensions: {img_width}x{img_height}")

        # Visual feedback
        cv2.putText(imgDisplay, f"SAVED! Image #{saved_images_count}",
                    (10, img_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show the result
    cv2.imshow(window_name, imgDisplay)

    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
print(f"Detection stopped. Total images saved: {saved_images_count}")
print(f"Check the '{output_image_folder}' folder for saved images and labels.")