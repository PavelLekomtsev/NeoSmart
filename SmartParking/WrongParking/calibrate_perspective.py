"""
Wrong Parking Calibration Tool

Reads frames from the frames/ directory (camera1.png / camera2.png).
Three-step calibration:
  Step 1: Mark 4 perspective points (rectangle in real world)
  Step 2: Mark 4 corners of a correctly-parked car NEAR the camera
  Step 3: Mark 4 corners of a correctly-parked car FAR from the camera

Saves camera{N}_calibration.pkl compatible with wrong_parking_basic.py.

Camera 1: horizontal parking (cars face toward/away from camera, parked in a row)
Camera 2: perpendicular parking (cars in a column going away from camera)

Controls:
  Left click  - Place point
  Right click - Remove last point
  Z           - Reset current step (if no points — go back one step)
  S / Enter   - Confirm current step and proceed
  Q           - Quit
"""

import cv2
import numpy as np
import pickle
from pathlib import Path


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


def bbox_from_points(points):
    """Derive axis-aligned bounding box from 4 points: (x, y, w, h)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    return x, y, w, h


def draw_quad(img, points, color, closed=False, label_names=None):
    """Draw points and connecting lines for a quadrilateral."""
    for i, pt in enumerate(points):
        cv2.circle(img, pt, 6, color, -1)
        cv2.circle(img, pt, 8, (255, 255, 255), 2)
        if label_names and i < len(label_names):
            cv2.putText(img, label_names[i], (pt[0] + 10, pt[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, 2)
        if closed and len(points) == 4:
            cv2.line(img, points[3], points[0], color, 2)


def draw_status_bar(img, text):
    """Draw black status bar at the top of the image."""
    cv2.rectangle(img, (0, 0), (len(text) * 10 + 20, 30), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


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
    frames_dir = Path(__file__).parent.parent / "frames"
    frame_path = frames_dir / f"{camera_id}.png"

    print(f"\nCalibrating for {camera_id}")
    print(f"Will save to: {save_path.name}")

    if cam_num == "1":
        print("  Orientation: horizontal parking (cars face toward/away from camera)")
    else:
        print("  Orientation: perpendicular parking (cars in column, going away)")

    # Load frame from frames/ directory
    if not frame_path.exists():
        print(f"\nERROR: Frame not found: {frame_path}")
        print("Place a frame image there before running calibration.")
        return

    base_image = cv2.imread(str(frame_path))
    if base_image is None:
        print(f"\nERROR: Could not read image: {frame_path}")
        return

    img_h, img_w = base_image.shape[:2]
    print(f"Loaded frame: {frame_path.name} ({img_w}x{img_h})")

    # State
    STEP_PERSPECTIVE = 0
    STEP_NEAR_CAR = 1
    STEP_FAR_CAR = 2
    STEP_DONE = 3

    step = STEP_PERSPECTIVE
    perspective_points = []
    perspective_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    corner_labels = ["TL", "TR", "BR", "BL"]
    near_car_points = []
    far_car_points = []
    homography = None
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
            elif step == STEP_NEAR_CAR and near_car_points:
                removed = near_car_points.pop()
                print(f"  Removed point ({removed[0]}, {removed[1]})")
            elif step == STEP_FAR_CAR and far_car_points:
                removed = far_car_points.pop()
                print(f"  Removed point ({removed[0]}, {removed[1]})")

    cv2.setMouseCallback(window_name, mouse_callback)

    print()
    print("=" * 55)
    print("  STEP 1: Mark 4 perspective points")
    print("=" * 55)
    print("Click 4 corners of a real-world RECTANGLE on the parking area:")
    print("  1=Top-Left  2=Top-Right  3=Bottom-Right  4=Bottom-Left")
    print("Press S to confirm. (S with 0 points to skip perspective.)")
    print()

    while step != STEP_DONE:
        img = base_image.copy()

        # ---- STEP 1: Perspective points ----
        if step == STEP_PERSPECTIVE:
            if click_point is not None:
                if len(perspective_points) < 4:
                    perspective_points.append(click_point)
                    label = perspective_labels[len(perspective_points) - 1]
                    print(f"  Point {len(perspective_points)}: {label} = ({click_point[0]}, {click_point[1]})")
                click_point = None

            status = f"STEP 1: Perspective | Points: {len(perspective_points)}/4"
            if len(perspective_points) < 4:
                status += f" | Next: {perspective_labels[len(perspective_points)]}"
            else:
                status += " | Press S to confirm"
            draw_status_bar(img, status)

            colors = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0)]
            for i, pt in enumerate(perspective_points):
                cv2.circle(img, pt, 8, colors[i], -1)
                cv2.circle(img, pt, 10, (255, 255, 255), 2)
                cv2.putText(img, f"{i+1}: {perspective_labels[i]}", (pt[0] + 15, pt[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

            if len(perspective_points) >= 2:
                for i in range(len(perspective_points) - 1):
                    cv2.line(img, perspective_points[i], perspective_points[i + 1], (0, 255, 255), 2)
                if len(perspective_points) == 4:
                    cv2.line(img, perspective_points[3], perspective_points[0], (0, 255, 255), 2)
                    overlay = img.copy()
                    pts_arr = np.array(perspective_points, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts_arr], (0, 255, 255))
                    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        # ---- STEP 2: Near car (4 corners) ----
        elif step == STEP_NEAR_CAR:
            if click_point is not None:
                if len(near_car_points) < 4:
                    near_car_points.append(click_point)
                    lbl = corner_labels[len(near_car_points) - 1]
                    print(f"  Near car point {len(near_car_points)}: {lbl} = ({click_point[0]}, {click_point[1]})")
                click_point = None

            status = f"STEP 2: NEAR car corners | Points: {len(near_car_points)}/4"
            if len(near_car_points) < 4:
                status += f" | Next: {corner_labels[len(near_car_points)]}"
            else:
                status += " | Press S to confirm"
            draw_status_bar(img, status)

            draw_quad(img, near_car_points, (0, 255, 0),
                      closed=(len(near_car_points) == 4), label_names=corner_labels)

        # ---- STEP 3: Far car (4 corners) ----
        elif step == STEP_FAR_CAR:
            if click_point is not None:
                if len(far_car_points) < 4:
                    far_car_points.append(click_point)
                    lbl = corner_labels[len(far_car_points) - 1]
                    print(f"  Far car point {len(far_car_points)}: {lbl} = ({click_point[0]}, {click_point[1]})")
                click_point = None

            status = f"STEP 3: FAR car corners | Points: {len(far_car_points)}/4"
            if len(far_car_points) < 4:
                status += f" | Next: {corner_labels[len(far_car_points)]}"
            else:
                status += " | Press S to confirm"
            draw_status_bar(img, status)

            # Show confirmed near car outline (dimmer)
            if near_car_points:
                pts = np.array(near_car_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, (0, 180, 0), 1)

            draw_quad(img, far_car_points, (255, 165, 0),
                      closed=(len(far_car_points) == 4), label_names=corner_labels)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        elif key == ord("z"):
            if step == STEP_PERSPECTIVE:
                perspective_points.clear()
                print("  All perspective points cleared")
            elif step == STEP_NEAR_CAR:
                if near_car_points:
                    near_car_points.clear()
                    print("  Near car points cleared")
                else:
                    step = STEP_PERSPECTIVE
                    perspective_points.clear()
                    homography = None
                    print("  Back to step 1")
            elif step == STEP_FAR_CAR:
                if far_car_points:
                    far_car_points.clear()
                    print("  Far car points cleared")
                else:
                    step = STEP_NEAR_CAR
                    near_car_points.clear()
                    print("  Back to step 2")

        elif key in (ord("s"), 13):  # S or Enter
            if step == STEP_PERSPECTIVE:
                if len(perspective_points) == 4:
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
                        print("  STEP 2: Mark NEAR car (4 corners)")
                        print("=" * 55)
                        print("Click 4 corners of a CORRECTLY parked car NEAR the camera:")
                        print("  TL -> TR -> BR -> BL")
                        print()
                    else:
                        print("  ERROR: Could not compute homography. Try different points.")

                elif len(perspective_points) == 0:
                    print("  Skipping perspective (no points).")
                    step = STEP_NEAR_CAR
                    print()
                    print("=" * 55)
                    print("  STEP 2: Mark NEAR car (4 corners)")
                    print("=" * 55)
                    print("Click 4 corners of a CORRECTLY parked car NEAR the camera:")
                    print("  TL -> TR -> BR -> BL")
                    print()
                else:
                    print(f"  Need exactly 4 points (or 0 to skip), have {len(perspective_points)}")

            elif step == STEP_NEAR_CAR:
                if len(near_car_points) == 4:
                    x, y, w, h = bbox_from_points(near_car_points)
                    ratio = compute_ratio(x, y, w, h, homography)
                    cy = y + h // 2
                    print(f"  Near car: bbox=({x}, {y}, {w}, {h}), Y_center={cy}, ratio={ratio:.2f}")

                    step = STEP_FAR_CAR
                    print()
                    print("=" * 55)
                    print("  STEP 3: Mark FAR car (4 corners)")
                    print("=" * 55)
                    print("Click 4 corners of a CORRECTLY parked car FAR from the camera:")
                    print("  TL -> TR -> BR -> BL")
                    print()
                else:
                    print(f"  Need 4 points, have {len(near_car_points)}")

            elif step == STEP_FAR_CAR:
                if len(far_car_points) == 4:
                    # Compute near car data
                    nx, ny, nw, nh = bbox_from_points(near_car_points)
                    near_ratio = compute_ratio(nx, ny, nw, nh, homography)
                    near_cy = ny + nh // 2

                    # Compute far car data
                    fx, fy, fw, fh = bbox_from_points(far_car_points)
                    far_ratio = compute_ratio(fx, fy, fw, fh, homography)
                    far_cy = fy + fh // 2

                    print(f"  Far car: bbox=({fx}, {fy}, {fw}, {fh}), Y_center={far_cy}, ratio={far_ratio:.2f}")

                    # Ensure near_y > far_y (near camera = lower in image = higher Y)
                    near = {"y": near_cy, "ratio": near_ratio}
                    far = {"y": far_cy, "ratio": far_ratio}
                    if near["y"] < far["y"]:
                        near, far = far, near
                        print("  (Swapped near/far based on Y position)")

                    calibration = {
                        "camera_id": camera_id,
                        "homography": homography,
                        "perspective_points": [list(p) for p in perspective_points],
                        "near_y": near["y"],
                        "near_ratio": near["ratio"],
                        "far_y": far["y"],
                        "far_ratio": far["ratio"],
                        "near_car_points": [list(p) for p in near_car_points],
                        "far_car_points": [list(p) for p in far_car_points],
                        "image_size": (img_w, img_h),
                    }

                    with open(save_path, "wb") as f:
                        pickle.dump(calibration, f)

                    print()
                    print("=" * 55)
                    print(f"  Calibration saved to {save_path.name}!")
                    print("=" * 55)
                    print(f"  Camera:     {camera_id}")
                    print(f"  Homography: {'yes' if homography is not None else 'skipped'}")
                    print(f"  Near: Y={near['y']}, correct_ratio={near['ratio']:.2f}")
                    print(f"  Far:  Y={far['y']}, correct_ratio={far['ratio']:.2f}")
                    print()

                    step = STEP_DONE
                else:
                    print(f"  Need 4 points, have {len(far_car_points)}")

    cv2.destroyAllWindows()
    print("Calibration tool closed.")


if __name__ == "__main__":
    main()
