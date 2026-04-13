# Traffic Counting System

## Overview

The Traffic Counting system monitors a road adjacent to the parking lot using a dedicated road camera (`camera4`). It detects vehicles in real time using YOLO and tracks them across frames with the SORT algorithm. Two directional crossing zones — **incoming** and **outgoing** — are defined on the road. When a tracked car's center enters a zone, the corresponding counter increments.

Unlike parking cameras (camera1–camera3) which monitor static occupancy, the road camera is concerned with **flow**: how many vehicles pass in each direction. The system provides:

- Per-direction counts (incoming / outgoing)
- Total traffic count
- Real-time visual feedback: ID badges, center points, zone highlighting, flash-on-crossing
- A calibration tool to mark custom crossing regions on any camera frame

---

## Technologies

### YOLO v8x (Object Detection)

Each frame from UE5 passes through a custom-trained YOLOv8x model (`Car_Detector.pt`) with a confidence threshold of `0.65`. The model detects cars and returns bounding boxes with confidence scores. Detections are drawn on the frame with `cvzone.cornerRect`.

### SORT (Simple Online and Realtime Tracking)

SORT assigns persistent track IDs to detected cars across frames using two classical algorithms:

1. **Kalman Filter** — predicts where each tracked car will appear in the next frame, based on its position and velocity history. This allows the tracker to maintain a car's identity even when YOLO misses a detection for several frames.

2. **Hungarian Algorithm** — optimally matches predicted track positions to new detections using **IoU (Intersection over Union)** as the cost metric. Each detection is assigned to the closest existing track, or spawns a new track if no match is found.

```
Frame N (from UE5)
        │
        ▼
┌───────────────────┐
│   YOLO Detection   │
│   → bounding boxes │
│   → confidence     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Kalman Filter    │
│   Predict where    │
│   existing tracks  │
│   should be now    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Hungarian Match   │
│  Predictions ↔     │
│  New detections    │
│  (IoU cost matrix) │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Update tracks     │
│  Create new tracks │
│  Remove dead ones  │
└───────────────────┘
```

### Why SORT Over DeepSORT?

DeepSORT adds a deep appearance model for re-identification, which increases latency. For traffic counting on a fixed camera, SORT is sufficient because:

- Cars move in predictable straight lines along the road
- The camera viewpoint is stable (no camera motion)
- Vehicles rarely occlude each other when passing through
- We only need to count crossings, not re-identify cars across cameras

### Dependencies

| Library | Purpose |
|---------|---------|
| `filterpy` | Kalman filter implementation (`KalmanBoxTracker` in `sort.py`) |
| `scipy` | Hungarian algorithm via `linear_sum_assignment` |
| `numpy` | Matrix operations, IoU computation, polygon testing |
| `opencv-python` | Frame processing, `cv2.pointPolygonTest` for zone crossing |
| `cvzone` | Bounding box and label rendering |

---

## Architecture

### Data Flow

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  UE5 Engine      │     │  FastAPI Server   │     │  ParkingDetector │
│  Renders scene   │────▶│  POST /api/frame  │────▶│  process_frame() │
│  Exports         │     │  Stores frame in  │     │                  │
│  camera4.png     │     │  frame_storages   │     │                  │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                            │
                         ┌──────────────────────────────────┘
                         ▼
              ┌─────────────────────┐
              │  detect_cars()      │
              │  YOLO → bbox list   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────────────┐
              │  _update_traffic_tracking()  │
              │  1. Convert dets → SORT fmt  │
              │  2. SORT.update() → tracks[] │
              │  3. Map sort_id → display_id │
              │  4. Compute bbox center      │
              │  5. pointPolygonTest vs      │
              │     incoming & outgoing zones │
              │  6. Increment counter on     │
              │     first crossing           │
              │  7. Draw overlays            │
              └──────────┬──────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  WebSocket stream   │
              │  Frame + stats →    │
              │  Browser dashboard  │
              └─────────────────────┘
```

### Zone Crossing Detection

Each crossing region is a 4-point polygon defined via the calibration tool. The system determines crossings by testing whether a tracked car's bounding box **center point** falls inside a region polygon:

```python
center = (x1 + w // 2, y1 + h // 2)
for direction, poly_np in region_arrays.items():
    if cv2.pointPolygonTest(poly_np, center, False) >= 0:
        if display_id not in crossed[direction]:
            crossed[direction].add(display_id)
            traffic_counts[camera_id][direction] += 1
```

**Key design decisions:**

- **Center point, not full bbox** — the center is a single unambiguous point. Using the full bbox would cause premature counting when the car's edge enters the zone.
- **`crossed` set tracks IDs, not positions** — once a car's display ID is recorded in the set for a direction, it is never counted again, even if it re-enters the zone.
- **Display IDs, not SORT IDs** — SORT's internal IDs are non-sequential (the `KalmanBoxTracker.count` class variable is shared across all SORT instances). A display ID mapping provides clean sequential IDs per camera.

---

## Implementation

### Display ID Mapping

SORT uses a class-level counter (`KalmanBoxTracker.count`) shared across all `Sort` instances. This means that if the parking tracker (camera3) creates tracks 1–9, the traffic tracker (camera4) might start at track 10. Additionally, dead tracks consume IDs that are never shown. The display ID mapping solves this:

```python
# When a new SORT track appears
if sort_id not in display_ids:
    display_ids[sort_id] = self.traffic_next_display_id[camera_id]
    self.traffic_next_display_id[camera_id] += 1
display_id = display_ids[sort_id]
```

Each camera maintains its own counter starting at 1, so the user sees sequential IDs (1, 2, 3, ...) regardless of SORT internals.

### Zone Flash Feedback

When a car crosses a zone for the first time, the zone flashes red for 1 second to provide visual feedback:

```python
if display_id not in crossed[direction]:
    crossed[direction].add(display_id)
    self.traffic_counts[camera_id][direction] += 1
    flash_times[direction] = now  # timestamp for flash animation
```

During rendering, if less than 1 second has passed since the last crossing, the zone is drawn in red with 50% opacity. Otherwise, it uses its default color (green for incoming, orange for outgoing) at 25% opacity.

### Visual Elements

| Element | Color | Description |
|---------|-------|-------------|
| YOLO bounding box | White corners | Standard detection box from `cvzone.cornerRect` |
| Center point | Yellow `(0, 255, 255)` | 4px filled circle at bbox center |
| ID badge | White on dark gray | Small box in bottom-right corner of bbox |
| Incoming zone | Green `(0, 200, 0)` | Semi-transparent polygon overlay |
| Outgoing zone | Orange `(0, 100, 255)` | Semi-transparent polygon overlay |
| Zone flash | Red `(0, 0, 255)` | 1-second flash when a car crosses |
| Zone label | White on zone color | "Incoming: N" / "Outgoing: N" at zone centroid |
| Total counter | White on dark gray | "Total Traffic: N" in top-left corner |

### Stale Track Cleanup

When a SORT track disappears (car leaves the frame), its display ID mapping is cleaned up:

```python
stale_sort_ids = [sid for sid in display_ids if sid not in active_sort_ids]
for sid in stale_sort_ids:
    del display_ids[sid]
```

However, the `crossed` set retains the display ID permanently — this ensures that if a car somehow re-appears with a new track ID and crosses the same zone, it would get a new display ID (and be counted), while the original crossing record remains.

---

## Configuration

All parameters are defined as class constants in `ParkingDetector` ([detector.py](../SmartParking/web_app/detector.py)):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ROAD_CAMERA_IDS` | `["camera4"]` | Cameras that use traffic counting instead of parking detection |
| `TRAFFIC_SORT_MAX_AGE` | `50` | Frames to keep a track alive without new detections |
| `TRAFFIC_SORT_MIN_HITS` | `0` | Minimum detections before a track is shown |
| `TRAFFIC_SORT_IOU_THRESHOLD` | `0.15` | Minimum IoU overlap to match a detection to an existing track |
| `confidence` | `0.65` | YOLO confidence threshold for car detection |

### Why These SORT Parameters?

The traffic tracker uses significantly different parameters from the parking tracker. Each was tuned to solve specific real-world problems:

**`TRAFFIC_SORT_MAX_AGE = 50`** (parking uses 20)

Cars on a road move faster and YOLO may lose detection for several frames due to motion blur or angle changes. A high max_age keeps the track alive through these gaps, preventing a new ID from being assigned when the car reappears.

**`TRAFFIC_SORT_MIN_HITS = 0`** (parking uses 5)

This is the most critical parameter. SORT only returns a track when `hit_streak >= min_hits` OR `frame_count <= min_hits`. With `min_hits = 0`, the condition `hit_streak >= 0` is always true, so every track is visible from its very first frame. This is essential for traffic counting because:

- A car may pass through the crossing zone in just a few frames
- If the track requires 5 confirmations (parking default), the car could cross the zone before its track is confirmed, and the crossing would never be counted
- The first car to enter the frame was consistently missed with `min_hits > 0` because `frame_count` was already high

**`TRAFFIC_SORT_IOU_THRESHOLD = 0.15`** (parking uses 0.3)

A lower IoU threshold allows more lenient matching between predictions and detections. When YOLO detection drops for a frame and the Kalman prediction drifts slightly, a strict threshold (0.3) would fail to match the detection when it reappears, creating a new track ID. The value of 0.15 provides enough tolerance for road driving speeds while still preventing cross-track contamination.

### Comparison Table

| Parameter | Parking (camera3) | Traffic (camera4) | Reason for Difference |
|-----------|-------------------|-------------------|-----------------------|
| `max_age` | 20 | 50 | Cars on road have longer detection gaps |
| `min_hits` | 5 | 0 | Must count zone crossing on first frame |
| `iou_threshold` | 0.3 | 0.15 | Moving cars need lenient re-matching |

---

## Calibration Tool

### `mark_crossing_regions.py`

An interactive OpenCV GUI for defining two 4-point crossing zone polygons on a camera frame. Located at `SmartParking/TrafficCounting/mark_crossing_regions.py`.

```bash
cd SmartParking/TrafficCounting
python mark_crossing_regions.py
```

### Workflow

1. **Select camera** — the tool scans `SmartParking/frames/` for available camera frames and prompts for a camera ID (e.g., `camera4`).
2. **Mark incoming region** — click 4 points to define the incoming (green) polygon. Press `S` or `Enter` to confirm.
3. **Mark outgoing region** — click 4 points to define the outgoing (orange) polygon. Press `S` or `Enter` to confirm and save.

### Controls

| Key | Action |
|-----|--------|
| Left click | Add a point (max 4 per region) |
| Right click | Remove the last point |
| `S` / `Enter` | Confirm current region and advance |
| `Z` | Clear current region's points |
| `Q` | Quit (auto-saves if both regions are complete) |

### Output Format

The tool saves a pickle file at `SmartParking/TrafficCounting/{camera_id}_crossing.p` containing:

```python
{
    "camera_id": "camera4",
    "incoming": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "outgoing": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}
```

If no calibration file exists for a camera, `ParkingDetector` falls back to hardcoded regions defined in `CROSSING_REGIONS_FALLBACK`.

### Tips for Good Zone Placement

- Place zones perpendicular to the road direction
- Leave a gap between the two zones so that cars are counted only once per direction
- Test with the live dashboard to verify counting accuracy
- Zones should be wide enough that no car can pass through without its center entering the polygon

---

## Integration with the Web Dashboard

### Backend Stats

`process_frame()` returns a stats dictionary for each camera. Road cameras include `incoming_count` and `outgoing_count`:

```python
stats = {
    "incoming_count": incoming_count,
    "outgoing_count": outgoing_count,
    "cars_detected": cars_detected,
    # ... other fields
}
```

These stats are sent to the browser via WebSocket at ~10 FPS alongside the processed frame.

### Frontend Display

The dashboard includes a **Road Traffic** zone with:

- **Navigation tab** — "Road Traffic" with a road icon, badge showing total count
- **Overview card** — summary with Incoming / Outgoing / Total mini-stats
- **Camera view** — original and processed frames side by side
- **Stat cards** — Incoming (green), Outgoing (orange), Total (blue)

The road zone is independent from parking zones. Camera4's car detections are excluded from the aggregate parking totals to avoid inflating occupancy numbers.

### Tab Persistence

The active tab (overview, free, paid, or road) is saved to `localStorage` and restored on page reload, so users don't lose their view when the page refreshes.

---

## Files

| File | Purpose |
|------|---------|
| `web_app/detector.py` | `_update_traffic_tracking()` — tracking, zone crossing, drawing. Traffic SORT parameters and crossing region loading |
| `web_app/sort.py` | SORT algorithm: `Sort`, `KalmanBoxTracker`, `associate_detections_to_trackers` |
| `TrafficCounting/mark_crossing_regions.py` | Interactive calibration tool for crossing zone polygons |
| `TrafficCounting/{camera_id}_crossing.p` | Pickle file with calibrated crossing region coordinates |
| `web_app/main.py` | Server: camera4 in `CAMERA_IDS`, `FRAME_PATHS`, `frame_storages` |
| `web_app/static/js/app.js` | Frontend: road zone stats, camera rendering, tab persistence |
| `web_app/templates/index.html` | HTML: Road Traffic tab, overview card, stat cards, camera canvases |
| `web_app/static/css/styles.css` | CSS: road zone color variables, zone indicator styling |

---

## Adding a New Road Camera

To add another road camera (e.g., `camera5`):

1. Add `"camera5"` to `CAMERA_IDS` in both `main.py` and `detector.py`
2. Add `"camera5"` to `ROAD_CAMERA_IDS` in `detector.py`
3. Add the frame path to `FRAME_PATHS` in `main.py`
4. Run `mark_crossing_regions.py` and select `camera5` to calibrate its crossing zones
5. Assign it to a zone in `app.js` (or create a new zone)
6. Add corresponding HTML elements in `index.html`

---

## Troubleshooting

**First car doesn't get counted when crossing a zone:**
- Verify that `TRAFFIC_SORT_MIN_HITS` is `0`. Any value > 0 can cause the track to be invisible during the first frames when the car enters the zone.

**Track IDs keep incrementing (1, 2, 3 ... 15, 16 ...) even for the same car:**
- This indicates the track is being lost and recreated. Increase `TRAFFIC_SORT_MAX_AGE` to keep tracks alive through detection gaps, or lower `TRAFFIC_SORT_IOU_THRESHOLD` to allow more lenient matching.

**Cars are counted twice (once for each direction):**
- The crossing zones may overlap. Use `mark_crossing_regions.py` to re-position them with a gap between them. The `crossed` set prevents the same display ID from being counted twice in the same direction, but a car passing through both zones will be counted in both.

**Zone never flashes red:**
- Ensure the calibration pickle file exists at `TrafficCounting/{camera_id}_crossing.p`. If no file is found, fallback regions are used, which may not align with the actual road position.

**"No calibration for camera4, using fallback regions" message on startup:**
- Run the calibration tool: `python SmartParking/TrafficCounting/mark_crossing_regions.py`
- Select `camera4` and mark both crossing regions.

**`ModuleNotFoundError: No module named 'filterpy'`:**
- Install dependencies: `pip install -r SmartParking/requirements.txt`
