"""
Wrong Parking Calibration Tool (unified).

Captures frames from UE5, lets you calibrate everything in one run:
  Step 1: Mark 4 perspective points (rectangle in real world)
  Step 2: Click on a correctly-parked car NEAR the camera
  Step 3: Click on a correctly-parked car FAR from the camera

Saves a single .pkl file per camera (e.g. camera1_calibration.pkl)
containing homography + adaptive threshold data.

Controls:
  Left click  - Place point / select car
  Right click - Remove last point
  R           - Reload image from UE5
  Z           - Reset current step
  S / Enter   - Confirm current step and proceed
  Q           - Quit
"""

import math
import cv2
import cvzone
import numpy as np
import pickle
import mss
import win32gui
import time
from pathlib import Path
from ultralytics import YOLO


last_found_window_title = ""
window_found_before = False


def find_unreal_window():
    global last_found_window_title, window_found_before

    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if any(x in window_title for x in ["Calibration", "Detection"]):
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
    return None


def capture_frame():
    region = find_unreal_window()
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


def detect_cars(model, img):
    """Detect cars and return list with bbox info."""
    results = model(img, stream=False, verbose=False)
    object_list = []
    for r in results:
        for box in r.boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > 0.8:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                center = (x1 + w // 2, y1 + h // 2)
                object_list.append({"bbox": (x1, y1, w, h), "center": center, "conf": conf})
    return object_list


def compute_ratio(x, y, w, h, H=None):
    """Compute aspect ratio, optionally with perspective correction."""
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


def find_closest_car(click_x, click_y, object_list):
    """Find the car whose center is closest to the click point."""
    best = None
    best_dist = float("inf")
    for obj in object_list:
        cx, cy = obj["center"]
        dist = math.hypot(click_x - cx, click_y - cy)
        if dist < best_dist:
            best_dist = dist
            best = obj
    return best


def main():
    print("=" * 55)
    print("  Wrong Parking Calibration Tool")
    print("=" * 55)
    print()

    # Choose camera
    while True:
        cam_num = input("Camera number (1 or 2): ").strip()
        if cam_num in ("1", "2"):
            break
        print("Please enter 1 or 2.")

    camera_id = f"camera{cam_num}"
    save_path = Path(__file__).parent / f"{camera_id}_calibration.pkl"

    print(f"\nCalibrating for {camera_id}")
    print(f"Will save to: {save_path.name}")

    # Load YOLO model (needed for steps 2-3)
    print("\nLoading YOLO model...")
    model = YOLO(str(Path(__file__).parent.parent.parent / "Models" / "Car_Detector.pt"))

    # Capture initial frame
    print("Waiting for UE5 window...")
    base_image = None
    while base_image is None:
        base_image = capture_frame()
        if base_image is None:
            time.sleep(1)

    img_h, img_w = base_image.shape[:2]
    print(f"Captured frame: {img_w}x{img_h}")

    # State
    STEP_PERSPECTIVE = 0   # Mark 4 points for homography
    STEP_NEAR_CAR = 1      # Click on correctly-parked car near camera
    STEP_FAR_CAR = 2       # Click on correctly-parked car far from camera
    STEP_DONE = 3

    step = STEP_PERSPECTIVE
    perspective_points = []
    point_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    homography = None
    near_data = None
    click_point = None

    window_name = f"Calibration - {camera_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(img_w, 1280), min(img_h, 720))

    def mouse_callback(event, x, y, flags, param):
        nonlocal click_point
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if step == STEP_PERSPECTIVE and perspective_points:
                removed = perspective_points.pop()
                print(f"  Removed point ({removed[0]}, {removed[1]})")

    cv2.setMouseCallback(window_name, mouse_callback)

    print()
    print("=" * 55)
    print("  STEP 1: Mark 4 perspective points")
    print("=" * 55)
    print("Click 4 corners of a real-world RECTANGLE on the parking area:")
    print("  1=Top-Left  2=Top-Right  3=Bottom-Right  4=Bottom-Left")
    print("Then press S to confirm.")
    print()

    while step != STEP_DONE:
        # Reload live frame for car detection steps
        if step in (STEP_NEAR_CAR, STEP_FAR_CAR):
            new_img = capture_frame()
            if new_img is not None:
                base_image = new_img

        img = base_image.copy()

        # ---- STEP 1: Perspective points ----
        if step == STEP_PERSPECTIVE:
            # Handle click
            if click_point is not None:
                if len(perspective_points) < 4:
                    perspective_points.append(click_point)
                    label = point_labels[len(perspective_points) - 1]
                    print(f"  Point {len(perspective_points)}: {label} = ({click_point[0]}, {click_point[1]})")
                click_point = None

            # Draw
            status = f"STEP 1: Perspective | Points: {len(perspective_points)}/4"
            if len(perspective_points) < 4:
                status += f" | Next: {point_labels[len(perspective_points)]}"
            else:
                status += " | Press S to confirm"
            cv2.rectangle(img, (0, 0), (len(status) * 10 + 20, 30), (0, 0, 0), -1)
            cv2.putText(img, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            colors = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0)]
            for i, pt in enumerate(perspective_points):
                cv2.circle(img, pt, 8, colors[i], -1)
                cv2.circle(img, pt, 10, (255, 255, 255), 2)
                cv2.putText(img, f"{i+1}: {point_labels[i]}", (pt[0]+15, pt[1]+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

            if len(perspective_points) >= 2:
                for i in range(len(perspective_points) - 1):
                    cv2.line(img, perspective_points[i], perspective_points[i+1], (0, 255, 255), 2)
                if len(perspective_points) == 4:
                    cv2.line(img, perspective_points[3], perspective_points[0], (0, 255, 255), 2)
                    overlay = img.copy()
                    pts_arr = np.array(perspective_points, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts_arr], (0, 255, 255))
                    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        # ---- STEP 2: Near car ----
        elif step == STEP_NEAR_CAR:
            object_list = detect_cars(model, img)

            # Draw cars with ratios
            for obj in object_list:
                x, y, w, h = obj["bbox"]
                cvzone.cornerRect(img, (x, y, w, h))
                ratio = compute_ratio(x, y, w, h, homography)
                cy = y + h // 2
                label_y = y if y >= 200 else y + h + 25
                cvzone.putTextRect(img, f"R={ratio:.2f} Y={cy}", (x, label_y),
                                   scale=1.2, colorR=(255, 165, 0), thickness=2)

            status = "STEP 2: Click on a CORRECTLY parked car NEAR the camera"
            cv2.rectangle(img, (0, 0), (len(status) * 10 + 20, 30), (0, 0, 0), -1)
            cv2.putText(img, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if click_point is not None:
                car = find_closest_car(click_point[0], click_point[1], object_list)
                click_point = None
                if car is not None:
                    x, y, w, h = car["bbox"]
                    ratio = compute_ratio(x, y, w, h, homography)
                    car_cy = y + h // 2
                    near_data = {"y": car_cy, "ratio": ratio}
                    print(f"  Near reference: Y={car_cy}, ratio={ratio:.2f}")
                    step = STEP_FAR_CAR
                    print()
                    print("  Now click on a CORRECTLY parked car FAR from the camera")

        # ---- STEP 3: Far car ----
        elif step == STEP_FAR_CAR:
            object_list = detect_cars(model, img)

            for obj in object_list:
                x, y, w, h = obj["bbox"]
                cvzone.cornerRect(img, (x, y, w, h))
                ratio = compute_ratio(x, y, w, h, homography)
                cy = y + h // 2
                label_y = y if y >= 200 else y + h + 25
                cvzone.putTextRect(img, f"R={ratio:.2f} Y={cy}", (x, label_y),
                                   scale=1.2, colorR=(255, 165, 0), thickness=2)

            status = "STEP 3: Click on a CORRECTLY parked car FAR from the camera"
            cv2.rectangle(img, (0, 0), (len(status) * 10 + 20, 30), (0, 0, 0), -1)
            cv2.putText(img, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if click_point is not None:
                car = find_closest_car(click_point[0], click_point[1], object_list)
                click_point = None
                if car is not None:
                    x, y, w, h = car["bbox"]
                    ratio = compute_ratio(x, y, w, h, homography)
                    car_cy = y + h // 2
                    far_data = {"y": car_cy, "ratio": ratio}
                    print(f"  Far reference: Y={car_cy}, ratio={ratio:.2f}")

                    near = near_data
                    far = far_data

                    # Ensure near_y > far_y (near camera = lower in image = higher Y)
                    if near["y"] < far["y"]:
                        near, far = far, near
                        print("  (Swapped near/far based on Y position)")

                    # Save everything in one file
                    calibration = {
                        "camera_id": camera_id,
                        "homography": homography,
                        "perspective_points": [list(p) for p in perspective_points],
                        "near_y": near["y"],
                        "near_ratio": near["ratio"],
                        "far_y": far["y"],
                        "far_ratio": far["ratio"],
                        "image_size": (img_w, img_h),
                    }

                    with open(save_path, "wb") as f:
                        pickle.dump(calibration, f)

                    print()
                    print("=" * 55)
                    print(f"  Calibration saved to {save_path.name}!")
                    print("=" * 55)
                    print(f"  Camera: {camera_id}")
                    print(f"  Homography: {'yes' if homography is not None else 'skipped'}")
                    print(f"  Near: Y={near['y']}, correct_ratio={near['ratio']:.2f}")
                    print(f"  Far:  Y={far['y']}, correct_ratio={far['ratio']:.2f}")
                    print()

                    step = STEP_DONE

        cv2.imshow(window_name, img)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        elif key == ord("r"):
            new_img = capture_frame()
            if new_img is not None:
                base_image = new_img
                print("  Image reloaded from UE5")

        elif key == ord("z"):
            if step == STEP_PERSPECTIVE:
                perspective_points.clear()
                print("  All points cleared")
            elif step == STEP_NEAR_CAR:
                print("  Reset to step 1")
                step = STEP_PERSPECTIVE
                perspective_points.clear()
                homography = None
            elif step == STEP_FAR_CAR:
                print("  Back to near car selection")
                step = STEP_NEAR_CAR
                near_data = None

        elif key in (ord("s"), 13):  # S or Enter
            if step == STEP_PERSPECTIVE:
                if len(perspective_points) == 4:
                    # Ask for real-world dimensions
                    print()
                    print("Enter real-world dimensions of the marked rectangle.")
                    print("(Any unit - meters, parking-space-widths, etc.)")

                    real_w = None
                    while real_w is None:
                        try:
                            val = float(input("  Real-world width (left-to-right): ").strip())
                            if val > 0:
                                real_w = val
                        except ValueError:
                            pass
                        if real_w is None:
                            print("  Enter a positive number.")

                    real_h = None
                    while real_h is None:
                        try:
                            val = float(input("  Real-world height (top-to-bottom): ").strip())
                            if val > 0:
                                real_h = val
                        except ValueError:
                            pass
                        if real_h is None:
                            print("  Enter a positive number.")

                    src_pts = np.array(perspective_points, dtype=np.float32)
                    scale = 200
                    dst_pts = np.array([
                        [0, 0], [real_w * scale, 0],
                        [real_w * scale, real_h * scale], [0, real_h * scale]
                    ], dtype=np.float32)

                    H, _ = cv2.findHomography(src_pts, dst_pts)

                    if H is not None:
                        homography = H
                        print(f"\n  Homography computed!")

                        # Show bird's-eye preview
                        bev_w = int(real_w * scale)
                        bev_h = int(real_h * scale)
                        bev = cv2.warpPerspective(base_image, H, (bev_w, bev_h))
                        preview_name = "Bird's Eye View Preview"
                        cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(preview_name, min(bev_w, 800), min(bev_h, 600))
                        cv2.imshow(preview_name, bev)
                        print("  Preview window opened. Press any key to continue.")
                        cv2.waitKey(0)
                        cv2.destroyWindow(preview_name)

                        step = STEP_NEAR_CAR
                        print()
                        print("=" * 55)
                        print("  STEP 2: Select NEAR car")
                        print("=" * 55)
                        print("Place a CORRECTLY parked car at the NEAREST parking spot.")
                        print("Click on it.")
                        print()
                    else:
                        print("  ERROR: Could not compute homography. Try different points.")

                elif len(perspective_points) == 0:
                    # Skip perspective calibration
                    print("  Skipping perspective (no points). Proceeding to car selection.")
                    step = STEP_NEAR_CAR
                    print()
                    print("=" * 55)
                    print("  STEP 2: Select NEAR car")
                    print("=" * 55)
                    print("Place a CORRECTLY parked car at the NEAREST parking spot.")
                    print("Click on it.")
                    print()
                else:
                    print(f"  Need exactly 4 points (or 0 to skip), have {len(perspective_points)}")

    cv2.destroyAllWindows()
    print("Calibration tool closed.")


if __name__ == "__main__":
    main()
