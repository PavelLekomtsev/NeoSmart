# Wrong Parking Detection

## Overview

The Wrong Parking Detection system identifies cars that are parked at an angle (not properly aligned with the parking space). It uses YOLO-based car detection combined with **bounding box aspect ratio analysis** and an **adaptive threshold** that accounts for perspective distortion.

---

## How It Works

### Core Idea: Aspect Ratio

When a car is parked correctly (aligned with the space), its bounding box is a tall, narrow rectangle. When a car is parked at an angle, its bounding box becomes more square-shaped.

![Correct vs Wrong parking bounding boxes](extras/WBP_Logic1.jpg)

*Left: Correctly parked car — tall bounding box (high aspect ratio). Right: Wrongly parked car — more square bounding box (low aspect ratio).*

The **aspect ratio** is computed as:

```
ratio = max(height, width) / min(height, width)
```

Using `max/min` instead of just `height/width` makes it independent of camera rotation — a car viewed from the side or from the front both produce high ratios when correctly parked.

### The Problem: Perspective Distortion

A single fixed threshold doesn't work because perspective distortion causes the aspect ratio to vary with position:

![Perspective distortion problem](extras/WBP_PerspectiveProblem.png)

Cars **near** the camera have taller bounding boxes (ratio ~1.86), while cars **far** from the camera have squarer bounding boxes (ratio ~1.31) — even though both are correctly parked. A fixed threshold would either miss wrong parking near the camera or produce false positives far from it.

### Solution: Two-Part Calibration

The system uses two techniques to handle this:

#### 1. Homography (Perspective Correction)

A homography matrix transforms bounding box corners from the distorted camera view into a bird's-eye view, partially correcting for perspective:

![Homography transform](extras/WBP_HomographyTransform.png)

This reduces the ratio variation but doesn't eliminate it completely (because the bounding box is a 2D projection of a 3D car, not a ground-plane feature).

#### 2. Adaptive Threshold (Position-Dependent)

Two reference points are calibrated — a correctly-parked car **near** the camera and one **far** from the camera. For any car at an intermediate Y position, the expected "correct" ratio is linearly interpolated between the two reference values. The threshold is then set at a configurable percentage (default: 87%) of this expected ratio:

![Adaptive threshold graph](extras/WBP_AdaptiveThreshold.png)

```
threshold(Y) = interpolated_correct_ratio(Y) * THRESHOLD_FACTOR
```

If a car's actual ratio falls below this threshold, it is classified as **wrongly parked**.

---

## Files

| File | Purpose |
|------|---------|
| `WrongParking/calibrate_perspective.py` | One-time calibration tool (3-step wizard) |
| `WrongParking/wrong_parking_basic.py` | Real-time wrong parking detection |
| `WrongParking/camera1_calibration.pkl` | Calibration data for camera 1 |
| `WrongParking/camera2_calibration.pkl` | Calibration data for camera 2 |

---

## Calibration (calibrate_perspective.py)

The calibration tool is a unified 3-step wizard that creates a single `.pkl` file per camera.

![Calibration workflow](extras/WBP_CalibrationWorkflow.png)

### Running the Calibration

```bash
cd SmartParking/WrongParking
python calibrate_perspective.py
```

On launch, you'll be asked which camera to calibrate (1 or 2).

### Step 1: Perspective Points (Homography)

Mark 4 corners of a rectangle that you know is rectangular in the real world (e.g., the outline of a parking space or a group of spaces):

1. Click 4 points: **Top-Left**, **Top-Right**, **Bottom-Right**, **Bottom-Left**
2. Press **S** to confirm
3. Enter the real-world width and height of that rectangle (any unit — meters, parking widths, etc.)
4. A bird's-eye preview will appear to verify the transform

If you want to **skip** homography, press **S** with 0 points marked.

| Control | Action |
|---------|--------|
| Left click | Place a point |
| Right click | Remove last point |
| R | Reload image from UE5 |
| Z | Clear all points |
| S / Enter | Confirm and proceed |

### Step 2: Near Car Reference

YOLO detection activates. Each detected car shows its aspect ratio and Y position.

**Click on a CORRECTLY parked car that is NEAR the camera** (close to the bottom of the frame). The system records its Y position and aspect ratio.

### Step 3: Far Car Reference

**Click on a CORRECTLY parked car that is FAR from the camera** (close to the top of the frame). The system records its Y position and aspect ratio.

The tool automatically swaps near/far if needed based on Y coordinates.

### Output

The calibration is saved to `camera1_calibration.pkl` or `camera2_calibration.pkl` containing:

```python
{
    "camera_id": "camera1",          # Which camera
    "homography": np.array(...),     # 3x3 matrix (or None if skipped)
    "perspective_points": [...],     # The 4 clicked points
    "near_y": 450,                   # Y position of near reference car
    "near_ratio": 1.86,             # Aspect ratio of near reference car
    "far_y": 180,                    # Y position of far reference car
    "far_ratio": 1.31,             # Aspect ratio of far reference car
    "image_size": (1280, 720),      # Frame dimensions
}
```

---

## Detection (wrong_parking_basic.py)

The detection script captures the UE5 window in real time, detects cars with YOLO, and classifies each as correctly or wrongly parked.

### Running Detection

```bash
cd SmartParking/WrongParking
python wrong_parking_basic.py
```

On launch, you'll be asked which camera to use (1 or 2). The script loads the corresponding `camera{N}_calibration.pkl` file.

### What It Shows

Each detected car displays:
- **"Correct"** (green) or **"Wrong"** (red) label
- The actual ratio and threshold in parentheses: `Correct (1.82/1.56)` means ratio=1.82, threshold=1.56

The top bar shows: `camera1 | Cars: 5 | Wrong: 1 | adaptive`

### Wrong Parking Examples

![Wrong parking detection examples](extras/WBP_Logic2.jpg)

*Red bounding boxes and labels indicate wrongly parked cars. Cars parked at an angle, diagonally, or sideways all produce lower aspect ratios.*

| Control | Action |
|---------|--------|
| Q / Close window | Quit |

### Configuration

The `THRESHOLD_FACTOR` constant (default `0.87`) controls sensitivity:

- **Higher value** (e.g. 0.92): More strict — more cars flagged as wrong
- **Lower value** (e.g. 0.80): More lenient — only severely angled cars flagged

If no calibration file exists, a fixed threshold of 1.4 is used (less accurate).

---

## Algorithm Details

### Ratio Computation

```python
def compute_ratio(x, y, w, h, H=None):
    if H is not None:
        # Transform bbox corners through homography
        corners = [[x,y], [x+w,y], [x+w,y+h], [x,y+h]]
        transformed = cv2.perspectiveTransform(corners, H)
        # Compute width and height in transformed space
        t_w = average of top and bottom edge lengths
        t_h = average of left and right edge lengths
        ratio = max(t_w, t_h) / min(t_w, t_h)
    else:
        ratio = max(h, w) / min(h, w)
    return ratio
```

### Adaptive Threshold

```python
def get_adaptive_threshold(y_center, calib):
    # Linear interpolation between near and far reference points
    t = (y_center - near_y) / (far_y - near_y)
    t = clamp(t, -0.2, 1.2)  # slight extrapolation allowed

    expected_ratio = near_ratio + t * (far_ratio - near_ratio)
    threshold = expected_ratio * THRESHOLD_FACTOR

    return threshold
```

### Decision

```
if car_ratio < threshold:
    → WRONG parking
else:
    → CORRECT parking
```

---

## Troubleshooting

**All cars show "Wrong":**
- Re-run calibration. Make sure you click on CORRECTLY parked cars in steps 2-3.
- Try increasing `THRESHOLD_FACTOR` (make it more lenient).

**No cars are flagged as "Wrong":**
- Decrease `THRESHOLD_FACTOR` to be more strict.
- Verify the calibration file matches the current camera angle.

**Calibration tool hangs after pressing S:**
- After marking 4 points and pressing S, the script waits for console input (real-world dimensions). Switch to the terminal window and enter the values.

**"No calibration found" warning:**
- Run `calibrate_perspective.py` first for the selected camera.

**Detection window doesn't capture UE5:**
- Make sure UE5 is running and the viewport is visible (not minimized).
- The script looks for windows with "Unreal Editor", "UE5", or "UnrealEditor" in the title.
