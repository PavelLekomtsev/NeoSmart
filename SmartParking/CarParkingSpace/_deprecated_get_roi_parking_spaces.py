# Import necessary libraries
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy library for numerical operations
import pickle  # Pickle library for serializing Python objects
import win32gui
import mss

# ------------ Variables --------------
totalSpaces = 12
polygons = []
current_polygon = []
counter = 0

last_found_window_title = ""
window_found_before = False
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
    global last_found_window_title, window_found_before

    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)

            # Skip the OpenCV window if provided
            if opencv_window_hwnd and hwnd == opencv_window_hwnd:
                return True

            # Skip windows related to our application
            if "Mark Parking Spaces" in window_title or "Detection" in window_title:
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

# Function to handle mouse events (used to mark points for polygons)
def mousePoints(event, x, y, flags, params):
    global counter, current_polygon

    # If left mouse button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the clicked point (x, y) to the current_polygon list
        current_polygon.append((x, y))

        # If we have collected four points for one polygon
        if len(current_polygon) == 4:
            polygons.append(current_polygon)  # Add the polygon to the list
            current_polygon = []  # Reset for the next polygon
            counter += 1  # Increment the counter
            print(polygons)  # Print the collected polygons

# Create the unique-name-window
window_name = "Mark Parking Spaces - UE5"
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

print("Starting parking space marking for Unreal Engine 5...")
print("Click to mark 4 points for each parking space. Press 'q' to quit early.")

# Main loop for capturing window and marking parking spaces
while True:
    img = capture_unreal_window(opencv_window_hwnd)

    if img is not None:
        # Draw the collected polygons on the image
        for polygon in polygons:
            cv2.polylines(img, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)

        # If we have collected all polygons, then save and exit the loop
        if counter == totalSpaces:
            with open('polygons.p', 'wb') as fileObj:
                pickle.dump(polygons, fileObj)  # Save the polygons to a file
            print("Saved all polygon points.")
            break

        # Display the image with marked polygons
        cv2.imshow(window_name, img)

        # Set the mouse callback function for marking points
        cv2.setMouseCallback(window_name, mousePoints)
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
        if counter > 0:
            with open('polygons.p', 'wb') as fileObj:
                pickle.dump(polygons, fileObj)
            print(f"Saved {counter} polygons early.")
        break

cv2.destroyAllWindows()
print("Parking space marking stopped.")