"""
Parking Space Marking Tool for Multi-Camera Setup.

Loads a frame image from the frames/ directory (same images UE5 writes),
lets you mark parking space polygons (4 points each) with mouse clicks,
and saves per-camera polygon files (e.g. camera1_parkings.p).

Controls:
  Left click  - Add a point to the current polygon
  Right click - Remove the last point
  Z           - Undo last completed polygon
  R           - Reload image from disk
  S           - Save current polygons
  Q / close   - Save and quit
"""

import cv2
import numpy as np
import pickle
from pathlib import Path


def main():
    base_dir = Path(__file__).parent
    frames_dir = base_dir.parent / "frames"

    # --- User input ---
    print("=" * 50)
    print("  Parking Space Marking Tool")
    print("=" * 50)
    print()

    while True:
        cam_num = input("Camera number (1, 2 or 3): ").strip()
        if cam_num in ("1", "2", "3"):
            break
        print("Please enter 1, 2 or 3.")

    camera_id = f"camera{cam_num}"
    frame_path = frames_dir / f"{camera_id}.png"

    if not frame_path.exists():
        print(f"ERROR: Frame not found at {frame_path}")
        print("Make sure UE5 is running and has exported at least one frame.")
        return

    while True:
        try:
            total_spaces = int(input("Total parking spaces to mark: ").strip())
            if total_spaces > 0:
                break
            print("Must be > 0.")
        except ValueError:
            print("Enter a number.")

    save_path = base_dir / f"{camera_id}_parkings.p"

    # --- Load existing polygons if file exists ---
    polygons = []
    if save_path.exists():
        try:
            with open(save_path, "rb") as f:
                polygons = pickle.load(f)
            print(f"Loaded {len(polygons)} existing polygons from {save_path.name}")
        except Exception:
            polygons = []

    current_polygon = []
    base_image = cv2.imread(str(frame_path))
    if base_image is None:
        print(f"ERROR: Could not read image at {frame_path}")
        return

    h, w = base_image.shape[:2]
    print(f"\nImage size: {w}x{h}")
    print(f"Marking {total_spaces} spaces for {camera_id}")
    print(f"Already marked: {len(polygons)}/{total_spaces}")
    print()
    print("Controls:")
    print("  Left click  - Add point")
    print("  Right click - Remove last point")
    print("  Z - Undo last polygon")
    print("  R - Reload image from disk")
    print("  S - Save")
    print("  Q - Save and quit")
    print()

    window_name = f"Mark Parking Spaces - {camera_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(w, 1280), min(h, 720))

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(polygons) >= total_spaces and len(current_polygon) == 0:
                return  # all spaces already marked
            current_polygon.append((x, y))

            if len(current_polygon) == 4:
                polygons.append(list(current_polygon))
                current_polygon = []
                print(f"  Polygon {len(polygons)}/{total_spaces} completed")

                if len(polygons) == total_spaces:
                    save_polygons()
                    print(f"All {total_spaces} spaces marked! Auto-saved.")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if current_polygon:
                current_polygon.pop()

    cv2.setMouseCallback(window_name, mouse_callback)

    def save_polygons():
        with open(save_path, "wb") as f:
            pickle.dump(polygons, f)
        print(f"Saved {len(polygons)} polygons to {save_path.name}")

    def draw_frame():
        img = base_image.copy()
        overlay = img.copy()

        # Draw completed polygons
        for i, poly in enumerate(polygons):
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

            # Draw polygon number at centroid
            cx = sum(p[0] for p in poly) // 4
            cy = sum(p[1] for p in poly) // 4
            cv2.putText(img, str(i + 1), (cx - 8, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        # Draw current in-progress polygon points
        for pt in current_polygon:
            cv2.circle(img, pt, 5, (0, 0, 255), -1)

        if len(current_polygon) >= 2:
            for i in range(len(current_polygon) - 1):
                cv2.line(img, current_polygon[i], current_polygon[i + 1], (0, 0, 255), 2)

        # Status bar at top
        status = f"{camera_id} | Spaces: {len(polygons)}/{total_spaces}"
        if current_polygon:
            status += f" | Points: {len(current_polygon)}/4"
        cv2.rectangle(img, (0, 0), (len(status) * 12 + 20, 30), (0, 0, 0), -1)
        cv2.putText(img, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        return img

    while True:
        display = draw_frame()
        cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            if polygons:
                save_polygons()
            break

        elif key == ord("s"):
            save_polygons()

        elif key == ord("z"):
            if current_polygon:
                current_polygon = []
                print("  Cleared current points")
            elif polygons:
                polygons.pop()
                print(f"  Undid last polygon. Now {len(polygons)}/{total_spaces}")

        elif key == ord("r"):
            new_img = cv2.imread(str(frame_path))
            if new_img is not None:
                base_image = new_img
                print("  Image reloaded from disk")
            else:
                print("  Could not reload image")

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
