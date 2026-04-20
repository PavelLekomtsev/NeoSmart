"""
Barrier Zone Marking Tool
Interactive GUI for defining zones for barrier cameras.

Modes:
  1) Plate camera  - approach + reading zones (for camera5)
  2) Safety camera - safety zone only (for camera6)
  3) Exit camera   - approach + safety zones (for camera7)
  4) All zones     - approach + reading + safety

Usage:
    python mark_barrier_zones.py

Controls:
    Left click  - Add point (min 3 per zone, no limit)
    Right click - Remove last point
    S / Enter   - Confirm current zone (min 3 points) and move to next step
    Z           - Clear current zone points
    Q           - Quit (saves if all required zones are defined)
"""

import cv2
import numpy as np
import pickle
from pathlib import Path


FRAMES_DIR = Path(__file__).parent.parent / "frames"
SAVE_DIR = Path(__file__).parent
WINDOW_NAME = "Mark Barrier Zones"

# Colors (BGR)
COLOR_APPROACH = (200, 200, 0)      # Cyan
COLOR_READING = (0, 200, 200)       # Yellow
COLOR_SAFETY = (0, 200, 0)          # Green
COLOR_ACTIVE_POINT = (255, 255, 0)  # Cyan points
COLOR_STATUS_BG = (0, 0, 0)
COLOR_STATUS_TEXT = (0, 255, 255)

STEP_APPROACH = 0
STEP_READING = 1
STEP_SAFETY = 2

STEP_CONFIG = {
    STEP_APPROACH: {"label": "APPROACH zone (cyan)", "color": COLOR_APPROACH, "key": "approach"},
    STEP_READING:  {"label": "READING zone (yellow)", "color": COLOR_READING, "key": "reading"},
    STEP_SAFETY:   {"label": "SAFETY zone (green)", "color": COLOR_SAFETY, "key": "safety"},
}

# Mode definitions: which steps to include
MODES = {
    "1": {"name": "Plate camera", "steps": [STEP_APPROACH, STEP_READING],
           "desc": "approach + reading zones (e.g. camera5)"},
    "2": {"name": "Safety camera", "steps": [STEP_SAFETY],
           "desc": "safety zone only (e.g. camera6)"},
    "3": {"name": "Exit camera", "steps": [STEP_APPROACH, STEP_SAFETY],
           "desc": "approach + safety zones (e.g. camera7)"},
    "4": {"name": "All zones", "steps": [STEP_APPROACH, STEP_READING, STEP_SAFETY],
           "desc": "approach + reading + safety"},
}


def get_camera_id():
    """Ask user which barrier camera to calibrate."""
    print("\n=== Barrier Zone Marking Tool ===\n")

    available = sorted(FRAMES_DIR.glob("camera*.png"))
    if not available:
        print(f"No camera frames found in {FRAMES_DIR}")
        print("Make sure UE5 has exported at least one frame.")
        return None

    print("Available cameras:")
    for f in available:
        print(f"  - {f.stem}")

    while True:
        cam_input = input("\nEnter barrier camera ID (e.g. camera5): ").strip()
        if not cam_input:
            continue
        frame_path = FRAMES_DIR / f"{cam_input}.png"
        if frame_path.exists():
            return cam_input
        print(f"  Frame not found: {frame_path}")


def get_mode():
    """Ask user which calibration mode to use."""
    print("\nSelect calibration mode:")
    for key, mode in MODES.items():
        print(f"  {key}) {mode['name']} — {mode['desc']}")

    while True:
        choice = input("\nMode (1-4): ").strip()
        if choice in MODES:
            mode = MODES[choice]
            print(f"  Selected: {mode['name']}")
            return mode["steps"]
        print("  Invalid choice. Enter 1, 2, 3, or 4.")


def load_existing(camera_id):
    """Load existing barrier zones if available."""
    save_path = SAVE_DIR / f"{camera_id}_barrier.p"
    if save_path.exists():
        try:
            with open(save_path, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded existing zones from {save_path}")
            return data
        except Exception as e:
            print(f"Warning: Could not load {save_path}: {e}")
    return None


def save_zones(camera_id, zones):
    """Save barrier zones to pickle file."""
    save_path = SAVE_DIR / f"{camera_id}_barrier.p"
    data = {
        "camera_id": camera_id,
        "approach": [list(p) for p in zones["approach"]],
        "reading": [list(p) for p in zones["reading"]],
        "safety": [list(p) for p in zones["safety"]],
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved barrier zones to {save_path}")
    return save_path


def draw_status_bar(img, text):
    """Draw status bar at the top of the image."""
    h, w = img.shape[:2]
    bar_height = 35
    cv2.rectangle(img, (0, 0), (w, bar_height), COLOR_STATUS_BG, -1)
    cv2.putText(img, text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_STATUS_TEXT, 1)


def draw_zone(img, points, color, label, filled=False):
    """Draw a polygon zone on the image."""
    if not points:
        return

    pts_array = np.array(points, np.int32).reshape(-1, 1, 2)

    if filled and len(points) >= 3:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts_array], color)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.polylines(img, [pts_array], True, color, 2)
    else:
        for i in range(len(points)):
            cv2.circle(img, tuple(points[i]), 6, COLOR_ACTIVE_POINT, -1)
            cv2.circle(img, tuple(points[i]), 6, color, 2)
            if i > 0:
                cv2.line(img, tuple(points[i - 1]), tuple(points[i]), color, 2)
        # Close the polygon if 3+ points
        if len(points) >= 3:
            cv2.line(img, tuple(points[-1]), tuple(points[0]), color, 2)

    if len(points) >= 3:
        centroid = np.mean(points, axis=0).astype(int)
        cv2.putText(img, label, (centroid[0] - 40, centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, label, (centroid[0] - 40, centroid[1]),
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

    active_steps = get_mode()
    active_keys = [STEP_CONFIG[s]["key"] for s in active_steps]

    existing = load_existing(camera_id)

    zones = {"approach": [], "reading": [], "safety": []}
    step_idx = 0  # Index into active_steps
    DONE = len(active_steps)

    if existing:
        choice = input("Existing zones found. Load them? (y/n): ").strip().lower()
        if choice == "y":
            for key in zones:
                pts = existing.get(key, [])
                zones[key] = [tuple(p) for p in pts]

            # Check if all active zones are complete
            all_complete = all(len(zones[k]) >= 3 for k in active_keys)
            if all_complete:
                step_idx = DONE
                print("All required zones loaded. Press S to re-save, Z to redo, Q to quit.")
            else:
                # Find first incomplete active step
                for i, s in enumerate(active_steps):
                    key = STEP_CONFIG[s]["key"]
                    if len(zones[key]) < 3:
                        step_idx = i
                        zones[key] = []
                        break
                print(f"Starting from step: {STEP_CONFIG[active_steps[step_idx]]['label']}")

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
    print("  Left click  - Add point (min 3, no limit)")
    print("  Right click - Remove last point")
    print("  S / Enter   - Confirm zone (need 3+ points)")
    print("  Z           - Clear current zone")
    print("  Q           - Quit\n")
    print("Zones to mark:")
    for i, s in enumerate(active_steps):
        print(f"  {i + 1}) {STEP_CONFIG[s]['label']}")
    print()

    while True:
        display = base_img.copy()

        # Current step (actual STEP_* constant)
        current_step = active_steps[step_idx] if step_idx < DONE else None

        # Process pending click
        if click_point is not None and step_idx < DONE:
            action, cx, cy = click_point
            click_point = None
            current_key = STEP_CONFIG[current_step]["key"]
            if action == "add":
                zones[current_key].append((cx, cy))
            elif action == "remove" and zones[current_key]:
                zones[current_key].pop()
        else:
            click_point = None

        # Draw all zones (only active ones)
        for i, s in enumerate(active_steps):
            cfg = STEP_CONFIG[s]
            key = cfg["key"]
            pts = zones[key]
            is_active = (step_idx < DONE and s == current_step)
            is_complete = len(pts) >= 3
            is_confirmed = (step_idx < DONE and i < step_idx) or (step_idx == DONE)

            if is_confirmed and is_complete:
                draw_zone(display, pts, cfg["color"], key.upper(), filled=True)
            elif is_active:
                draw_zone(display, pts, cfg["color"], key.upper(),
                          filled=is_complete)

        # Status bar
        if step_idx == DONE:
            status = f"{camera_id} | ALL ZONES DONE | S=save, Z=redo, Q=quit"
        else:
            cfg = STEP_CONFIG[current_step]
            current_pts = zones[cfg["key"]]
            step_num = step_idx + 1
            total = len(active_steps)
            status = f"{camera_id} | [{step_num}/{total}] Mark {cfg['label']} | Points: {len(current_pts)} (min 3) | S=confirm, Z=clear"
        draw_status_bar(display, status)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == ord("q"):
            break
        elif key in (ord("s"), 13):
            if step_idx < DONE:
                current_key = STEP_CONFIG[current_step]["key"]
                if len(zones[current_key]) >= 3:
                    step_idx += 1
                    if step_idx == DONE:
                        save_zones(camera_id, zones)
                        print("All required zones saved!")
                    else:
                        next_step = active_steps[step_idx]
                        print(f"Confirmed. Now mark {STEP_CONFIG[next_step]['label']}.")
            elif step_idx == DONE:
                save_zones(camera_id, zones)
                print("Re-saved.")
        elif key == ord("z"):
            if step_idx < DONE:
                current_key = STEP_CONFIG[current_step]["key"]
                zones[current_key] = []
                print(f"{current_key.upper()} zone cleared.")
            elif step_idx == DONE:
                for k in active_keys:
                    zones[k] = []
                step_idx = 0
                print("All zones cleared. Starting over.")

    cv2.destroyAllWindows()

    all_complete = all(len(zones[k]) == 4 for k in active_keys)
    if all_complete:
        save_zones(camera_id, zones)
        print("Auto-saved on exit.")
    else:
        print("Exited without complete zones (need at least 3 points each for required zones).")


if __name__ == "__main__":
    main()
