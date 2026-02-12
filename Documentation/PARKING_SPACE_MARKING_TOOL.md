# Multi-Camera Parking Space Marking Tool

## Context

The system now supports two cameras (`camera1`, `camera2`). UE5 sends frames to `frames/camera1.png` and `frames/camera2.png`.

The old `get_roi_parking_spaces.py` marks polygons by capturing the UE5 window directly via `mss` — but now frames arrive as PNG files. We need a new marking tool that loads these PNGs (guaranteeing resolution match with the web app) and saves per-camera polygon files. Then `detector.py` must use per-camera polygons.

---

## Changes

### 1. New file: `SmartParking/CarParkingSpace/mark_parking_spaces.py`

Simple OpenCV UI tool:

- On launch: console prompts for camera number (`1` or `2`) and total parking spaces count
- Loads image from `../frames/cameraX.png` (same files UE5 writes — resolution matches exactly)
- Shows image in OpenCV window, user clicks 4 points per polygon (same as old tool)
- Real-time drawing: green polygon outlines + semi-transparent fill for completed polygons, red dots for in-progress points

**Keyboard & mouse controls:**

| Key / Action    | Description                                        |
| --------------- | -------------------------------------------------- |
| Left click      | Add a polygon point                                |
| Right click     | Undo last point                                    |
| `Z`             | Remove last completed polygon                      |
| `R`             | Reload image from disk (if UE5 updates it)         |
| `S`             | Save current polygons                              |
| `Q` / Close     | Save partial progress and exit                     |

Auto-saves when all spaces are marked. Saves to `CarParkingSpace/camera1_parkings.p` / `camera2_parkings.p`.

### 2. Modify: `SmartParking/web_app/detector.py`

- `__init__` loads per-camera polygon files (`camera1_parkings.p`, `camera2_parkings.p`) from `CarParkingSpace/` directory
- Falls back to `polygons.p` if per-camera files not found (backward compatibility)
- Stores `self.camera_polygons = {"camera1": [...], "camera2": [...]}` and `self.camera_total_spaces = {...}`
- Stores `self.camera_stats = {"camera1": {...}, "camera2": {...}}`
- `process_frame(img, camera_id)` uses correct polygons for that camera
- `get_stats(camera_id)` returns camera-specific stats

### 3. Modify: `SmartParking/web_app/main.py`

- Pass `camera_id` to `det.process_frame(processed_frame, cam_id)`
- Use `det.get_stats(cam_id)` for per-camera stats
- Update placeholder stats to use `det.camera_total_spaces.get(cam_id, 12)`
- Update `/api/stats` to use per-camera stats

---

## Files Summary

| Action   | Path                                              |
| -------- | ------------------------------------------------- |
| **New**  | `SmartParking/CarParkingSpace/mark_parking_spaces.py` |
| Modify   | `SmartParking/web_app/detector.py`                |
| Modify   | `SmartParking/web_app/main.py`                    |

---

## Verification

1. Run UE5 to have `frames/camera1.png` on disk
2. Run `python mark_parking_spaces.py`, select camera 1, mark spaces — `camera1_parkings.p` created
3. Restart web app — parking space overlay uses camera1-specific polygons
4. Repeat for camera 2
