"""
Traffic Crossing Region Marking Tool
Interactive GUI for defining two crossing regions (incoming / outgoing)
on a camera frame. Each region is a 4-point polygon.

Usage:
    python mark_crossing_regions.py

Controls:
    Left click  - Add point (4 points per region)
    Right click - Remove last point
    S / Enter   - Confirm current region and move to next step
    Z           - Clear current region points
    Q           - Quit (saves if both regions are defined)
"""

import cv2
import numpy as np
import pickle
from pathlib import Path


FRAMES_DIR = Path(__file__).parent.parent / "frames"
SAVE_DIR = Path(__file__).parent
WINDOW_NAME = "Mark Crossing Regions"

# Colors (BGR)
COLOR_INCOMING = (0, 200, 0)      # Green
COLOR_OUTGOING = (0, 100, 255)    # Orange
COLOR_POINT = (0, 255, 255)       # Yellow
COLOR_ACTIVE_POINT = (255, 255, 0)  # Cyan
COLOR_STATUS_BG = (0, 0, 0)
COLOR_STATUS_TEXT = (0, 255, 255)

STEP_INCOMING = 0
STEP_OUTGOING = 1
STEP_DONE = 2

STEP_LABELS = {
    STEP_INCOMING: "INCOMING region (green)",
    STEP_OUTGOING: "OUTGOING region (orange)",
}


def get_camera_id():
    """Ask user which camera to calibrate."""
    print("\n=== Traffic Crossing Region Marking Tool ===\n")

    available = sorted(FRAMES_DIR.glob("camera*.png"))
    if not available:
        print(f"No camera frames found in {FRAMES_DIR}")
        print("Make sure UE5 has exported at least one frame.")
        return None

    print("Available cameras:")
    for f in available:
        cam_id = f.stem
        print(f"  - {cam_id}")

    while True:
        cam_input = input("\nEnter camera ID (e.g. camera4): ").strip()
        if not cam_input:
            continue
        frame_path = FRAMES_DIR / f"{cam_input}.png"
        if frame_path.exists():
            return cam_input
        print(f"  Frame not found: {frame_path}")


def load_existing(camera_id):
    """Load existing crossing regions if available."""
    save_path = SAVE_DIR / f"{camera_id}_crossing.p"
    if save_path.exists():
        try:
            with open(save_path, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded existing regions from {save_path}")
            return data
        except Exception as e:
            print(f"Warning: Could not load {save_path}: {e}")
    return None


def save_regions(camera_id, incoming_pts, outgoing_pts):
    """Save crossing regions to pickle file."""
    save_path = SAVE_DIR / f"{camera_id}_crossing.p"
    data = {
        "camera_id": camera_id,
        "incoming": [list(p) for p in incoming_pts],
        "outgoing": [list(p) for p in outgoing_pts],
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved crossing regions to {save_path}")
    return save_path


def draw_status_bar(img, text):
    """Draw status bar at the top of the image."""
    h, w = img.shape[:2]
    bar_height = 35
    cv2.rectangle(img, (0, 0), (w, bar_height), COLOR_STATUS_BG, -1)
    cv2.putText(img, text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_STATUS_TEXT, 1)


def draw_region(img, points, color, label, filled=False):
    """Draw a polygon region on the image."""
    if not points:
        return

    pts_array = np.array(points, np.int32).reshape(-1, 1, 2)

    if filled and len(points) == 4:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts_array], color)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        cv2.polylines(img, [pts_array], True, color, 2)
    else:
        # Draw lines between consecutive points
        for i in range(len(points)):
            cv2.circle(img, tuple(points[i]), 6, COLOR_ACTIVE_POINT, -1)
            cv2.circle(img, tuple(points[i]), 6, color, 2)
            if i > 0:
                cv2.line(img, tuple(points[i - 1]), tuple(points[i]), color, 2)
        # Close polygon if 4 points
        if len(points) == 4:
            cv2.line(img, tuple(points[3]), tuple(points[0]), color, 2)

    # Draw label at centroid
    if len(points) >= 3:
        centroid = np.mean(points, axis=0).astype(int)
        cv2.putText(img, label, (centroid[0] - 30, centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, label, (centroid[0] - 30, centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def main():
    camera_id = get_camera_id()
    if camera_id is None:
        return

    frame_path = FRAMES_DIR / f"{camera_id}.png"
    base_img = cv2.imread(str(frame_path))
    if base_img is None:
        print(f"Error: Could not read {frame_path}")
        return

    # Try loading existing regions
    existing = load_existing(camera_id)

    incoming_pts = []
    outgoing_pts = []
    step = STEP_INCOMING

    if existing:
        choice = input("Existing regions found. Load them? (y/n): ").strip().lower()
        if choice == "y":
            incoming_pts = [tuple(p) for p in existing.get("incoming", [])]
            outgoing_pts = [tuple(p) for p in existing.get("outgoing", [])]
            if len(incoming_pts) == 4 and len(outgoing_pts) == 4:
                step = STEP_DONE
                print("Both regions loaded. Press S to re-save or Q to quit.")
                print("Press Z to clear and start over.")
            elif len(incoming_pts) == 4:
                step = STEP_OUTGOING
                print("Incoming region loaded. Mark outgoing region next.")
            else:
                incoming_pts = []
                outgoing_pts = []
                print("Incomplete data. Starting fresh.")

    click_point = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal click_point
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point = ("add", x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            click_point = ("remove", x, y)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(base_img.shape[1], 1400), min(base_img.shape[0], 900))
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("\nControls:")
    print("  Left click  - Add point (4 per region)")
    print("  Right click - Remove last point")
    print("  S / Enter   - Confirm region")
    print("  Z           - Clear current region")
    print("  Q           - Quit\n")

    while True:
        display = base_img.copy()

        # Process pending click
        if click_point is not None:
            action, cx, cy = click_point
            click_point = None

            if step == STEP_INCOMING:
                if action == "add" and len(incoming_pts) < 4:
                    incoming_pts.append((cx, cy))
                elif action == "remove" and incoming_pts:
                    incoming_pts.pop()
            elif step == STEP_OUTGOING:
                if action == "add" and len(outgoing_pts) < 4:
                    outgoing_pts.append((cx, cy))
                elif action == "remove" and outgoing_pts:
                    outgoing_pts.pop()

        # Draw confirmed regions
        if step > STEP_INCOMING or (step == STEP_DONE):
            draw_region(display, incoming_pts, COLOR_INCOMING, "INCOMING", filled=True)
        if step > STEP_OUTGOING or (step == STEP_DONE):
            draw_region(display, outgoing_pts, COLOR_OUTGOING, "OUTGOING", filled=True)

        # Draw active region being edited
        if step == STEP_INCOMING:
            draw_region(display, incoming_pts, COLOR_INCOMING, "INCOMING",
                        filled=(len(incoming_pts) == 4))
        elif step == STEP_OUTGOING:
            draw_region(display, outgoing_pts, COLOR_OUTGOING, "OUTGOING",
                        filled=(len(outgoing_pts) == 4))

        # Status bar
        if step == STEP_DONE:
            status = f"{camera_id} | DONE | Press S to save, Z to redo, Q to quit"
        else:
            label = STEP_LABELS[step]
            current_pts = incoming_pts if step == STEP_INCOMING else outgoing_pts
            status = f"{camera_id} | Mark {label} | Points: {len(current_pts)}/4 | S=confirm, Z=clear"
        draw_status_bar(display, status)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF

        # Window closed
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == ord("q"):
            break
        elif key in (ord("s"), 13):  # S or Enter
            if step == STEP_INCOMING and len(incoming_pts) == 4:
                step = STEP_OUTGOING
                print("Incoming region confirmed. Now mark OUTGOING region.")
            elif step == STEP_OUTGOING and len(outgoing_pts) == 4:
                step = STEP_DONE
                save_path = save_regions(camera_id, incoming_pts, outgoing_pts)
                print(f"Both regions saved to {save_path}")
            elif step == STEP_DONE:
                save_path = save_regions(camera_id, incoming_pts, outgoing_pts)
                print(f"Re-saved to {save_path}")
        elif key == ord("z"):
            if step == STEP_INCOMING:
                incoming_pts.clear()
                print("Incoming points cleared.")
            elif step == STEP_OUTGOING:
                outgoing_pts.clear()
                print("Outgoing points cleared.")
            elif step == STEP_DONE:
                incoming_pts.clear()
                outgoing_pts.clear()
                step = STEP_INCOMING
                print("All cleared. Start over with INCOMING region.")

    cv2.destroyAllWindows()

    # Auto-save on quit if both regions are complete
    if len(incoming_pts) == 4 and len(outgoing_pts) == 4:
        save_regions(camera_id, incoming_pts, outgoing_pts)
        print("Auto-saved on exit.")
    else:
        print("Exited without complete regions (need 4 points each).")


if __name__ == "__main__":
    main()
