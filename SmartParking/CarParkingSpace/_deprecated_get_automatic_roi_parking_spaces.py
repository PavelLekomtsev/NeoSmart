import cv2
import numpy as np
import pickle
import mss
import win32gui

# --------------- Variables -------------------
polygon_file_path = 'polygons.p'

# Global variables
opencv_window_hwnd = None
last_found_window_title = ""
window_found_before = False
detected_parking_spaces = []

def find_unreal_window(opencv_window_hwnd=None):
    """Search for an Unreal Engine window and return its adjusted coordinates."""
    global last_found_window_title, window_found_before

    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)

            if opencv_window_hwnd and hwnd == opencv_window_hwnd:
                return True

            if "Parking Detection" in window_title or "Detection" in window_title:
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
        print("Unreal Engine window not found!")
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

def detect_white_lines(img):
    """Detect white lines in the image using color thresholding and edge detection."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    return white_mask

def find_parking_lines(white_mask, img_shape):
    """Find vertical and horizontal lines that could be parking space boundaries."""
    lines = cv2.HoughLinesP(white_mask, 1, np.pi / 180, threshold=50, 
                            minLineLength=30, maxLineGap=20)
    
    vertical_lines = []
    horizontal_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle > 70 and angle < 110:
                vertical_lines.append((x1, y1, x2, y2))
            elif angle < 20 or angle > 160:
                horizontal_lines.append((x1, y1, x2, y2))
    
    return vertical_lines, horizontal_lines

def merge_similar_lines(lines, threshold=30, is_vertical=True):
    """Merge lines that are close to each other."""
    if not lines:
        return []
    
    merged = []
    lines = sorted(lines, key=lambda l: l[0] if is_vertical else l[1])
    
    current_group = [lines[0]]
    
    for i in range(1, len(lines)):
        if is_vertical:
            if abs(lines[i][0] - current_group[-1][0]) < threshold:
                current_group.append(lines[i])
            else:
                merged.append(average_line(current_group, is_vertical))
                current_group = [lines[i]]
        else:
            if abs(lines[i][1] - current_group[-1][1]) < threshold:
                current_group.append(lines[i])
            else:
                merged.append(average_line(current_group, is_vertical))
                current_group = [lines[i]]
    
    merged.append(average_line(current_group, is_vertical))
    return merged

def average_line(lines, is_vertical):
    """Calculate average line from a group of similar lines."""
    x1_avg = int(np.mean([l[0] for l in lines]))
    y1_avg = int(np.mean([l[1] for l in lines]))
    x2_avg = int(np.mean([l[2] for l in lines]))
    y2_avg = int(np.mean([l[3] for l in lines]))
    return (x1_avg, y1_avg, x2_avg, y2_avg)

def extend_line(x1, y1, x2, y2, img_height):
    """Extend a line to cover more of the image."""
    if y2 == y1:
        return x1, y1, x2, y2
    
    slope = (x2 - x1) / (y2 - y1) if y2 != y1 else 0
    
    y_start = 0
    y_end = img_height
    x_start = int(x1 + slope * (y_start - y1))
    x_end = int(x1 + slope * (y_end - y1))
    
    return x_start, y_start, x_end, y_end

def find_parking_spaces(vertical_lines, horizontal_lines, img_shape):
    """Find parking spaces by matching vertical lines with horizontal boundaries."""
    parking_spaces = []
    height, width = img_shape[:2]
    
    vertical_lines = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
    
    for i in range(len(vertical_lines) - 1):
        left_line = vertical_lines[i]
        right_line = vertical_lines[i + 1]
        
        left_x = (left_line[0] + left_line[2]) / 2
        right_x = (right_line[0] + right_line[2]) / 2
        width_space = right_x - left_x
        
        if 40 < width_space < 250:
            top_y = min(left_line[1], left_line[3], right_line[1], right_line[3])
            bottom_y = max(left_line[1], left_line[3], right_line[1], right_line[3])
            
            top_boundary = top_y
            bottom_boundary = bottom_y
            
            for h_line in horizontal_lines:
                h_y = (h_line[1] + h_line[3]) / 2
                h_x1 = min(h_line[0], h_line[2])
                h_x2 = max(h_line[0], h_line[2])
                
                if h_x1 <= right_x and h_x2 >= left_x:
                    if abs(h_y - top_y) < 50 and h_y < (top_y + bottom_y) / 2:
                        top_boundary = min(top_boundary, h_y)
                    elif abs(h_y - bottom_y) < 50 and h_y > (top_y + bottom_y) / 2:
                        bottom_boundary = max(bottom_boundary, h_y)
            
            parking_space = [
                [int(left_x), int(top_boundary)],
                [int(right_x), int(top_boundary)],
                [int(right_x), int(bottom_boundary)],
                [int(left_x), int(bottom_boundary)]
            ]
            
            parking_spaces.append(parking_space)
    
    return parking_spaces

def draw_detected_spaces(img, parking_spaces):
    """Draw detected parking spaces on the image."""
    overlay = img.copy()
    
    for i, space in enumerate(parking_spaces):
        space_array = np.array(space, np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(img, [space_array], True, (0, 255, 0), 2)
        
        cv2.fillPoly(overlay, [space_array], (0, 255, 0))
        
        center_x = int(np.mean([p[0] for p in space]))
        center_y = int(np.mean([p[1] for p in space]))
        cv2.putText(img, str(i + 1), (center_x - 10, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    return img

window_name = "Automatic Parking Space Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

try:
    import time
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

print("=" * 60)
print("Automatic Parking Space Detection")
print("=" * 60)
print("Instructions:")
print("  - Press SPACE to detect parking spaces")
print("  - Press 's' to save detected spaces to file")
print("  - Press 'r' to reset detection")
print("  - Press 'q' to quit")
print("=" * 60)

detection_done = False

while True:
    img = capture_unreal_window(opencv_window_hwnd)

    if img is not None:
        display_img = img.copy()
        
        if detection_done and detected_parking_spaces:
            display_img = draw_detected_spaces(display_img, detected_parking_spaces)
            cv2.putText(display_img, f"Detected: {len(detected_parking_spaces)} spaces", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_img, "Press 's' to save, 'r' to reset", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(display_img, "Press SPACE to detect parking spaces", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_img)
    else:
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_img, "Searching for Unreal Engine 5 window...", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(blank_img, "Make sure UE5 is running and visible", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow(window_name, blank_img)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        if img is not None:
            print("\nDetecting parking spaces...")
            
            white_mask = detect_white_lines(img)
            
            vertical_lines, horizontal_lines = find_parking_lines(white_mask, img.shape)
            
            print(f"Found {len(vertical_lines)} vertical lines")
            print(f"Found {len(horizontal_lines)} horizontal lines")
            
            vertical_lines = merge_similar_lines(vertical_lines, threshold=30, is_vertical=True)
            horizontal_lines = merge_similar_lines(horizontal_lines, threshold=30, is_vertical=False)
            
            print(f"After merging: {len(vertical_lines)} vertical, {len(horizontal_lines)} horizontal")
            
            detected_parking_spaces = find_parking_spaces(vertical_lines, horizontal_lines, img.shape)
            
            print(f"Detected {len(detected_parking_spaces)} parking spaces!")
            detection_done = True
    
    elif key == ord('s'):
        if detected_parking_spaces:
            with open(polygon_file_path, 'wb') as f:
                pickle.dump(detected_parking_spaces, f)
            print(f"\nSaved {len(detected_parking_spaces)} parking spaces to '{polygon_file_path}'")
            print("You can now use these with the main parking detection program!")
        else:
            print("\nNo parking spaces detected yet! Press SPACE first.")
    
    elif key == ord('r'):
        detected_parking_spaces = []
        detection_done = False
        print("\nDetection reset. Press SPACE to detect again.")
    
    elif key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
print("\nProgram stopped.")