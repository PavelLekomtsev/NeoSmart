# Legacy Tools

`SmartParking/legacy/` contains the standalone prototype scripts that
preceded the current web application. They are **not** part of the
live pipeline and are **not** covered by tests or CI. They are kept in
the repository on purpose — as an honest record of how the system was
built iteratively.

## Why keep them

Each of the current web-app components grew out of one of these
scripts. The path from prototype → productionised module is part of
the work and worth preserving:

| Prototype (legacy/)                                | Production module (web_app / neosmart)                                          |
| -------------------------------------------------- | ------------------------------------------------------------------------------- |
| `car_counter.py`                                   | `ParkingDetector` in *car_counter* mode + traffic counter                       |
| `car_parking_space.py`                             | `ParkingDetector` in *parking_spaces* mode (`overlay_parking_spaces`)           |
| `wrong_parking_basic.py`                           | (retired) — homography approach replaced by bbox-overlap                        |
| `wrong_parking_advanced.py`                        | `ParkingDetector.overlay_parking_spaces` bbox-vs-polygon overlap                |
| `suspicious_car_detector.py`                       | Paid-zone tracking in `ParkingDetector._update_tracking` (YOLO + SORT)          |
| `_deprecated_get_roi_parking_spaces.py`            | `CarParkingSpace/mark_parking_spaces.py`                                        |
| `_deprecated_get_automatic_roi_parking_spaces.py`  | `CarParkingSpace/mark_parking_spaces.py`                                        |

## What they have in common

All scripts except `suspicious_car_detector.py` capture the Unreal
Engine 5 viewport via `win32gui` + `mss` and display results in a
standalone OpenCV window. This was the original data path before the
UE5 Blueprint started exporting PNG frames to
`SmartParking/frames/` and the FastAPI server took over rendering.

That coupling is why they are Windows-only. The production web app has
no `win32gui` / `mss` dependency in its core path — it reads PNG frames
from disk, so it runs on Linux and in Docker as well.

## Running them (only if you want to)

1. Windows + Python 3.11.
2. `pip install -r SmartParking/requirements.txt`
3. `pip install -r SmartParking/requirements-win.txt`
4. Launch UE5 with the parking scene so the viewport is visible.
5. From the repo root, for example:
   ```powershell
   python SmartParking/legacy/wrong_parking_advanced.py
   python SmartParking/legacy/car_counter.py
   ```

Calibration pickles still live next to their original interactive
tools (`SmartParking/CarParkingSpace/*.p`,
`SmartParking/WrongParking/*.pkl`) and the legacy scripts reach back
into those directories — they are not duplicated.

## Deliberately not maintained

- These scripts do not use the typed `neosmart` settings layer; they
  have their own hardcoded paths and thresholds. The values inside the
  web app and `config/default.yaml` are authoritative; divergence in a
  legacy script is not a bug.
- They log via `print()`, not the structured logger.
- The already-prefixed `_deprecated_*` files were marked deprecated in
  the original project and kept only so the ROI-picker history is
  visible.

## If this directory ever goes away

Nothing in the production pipeline imports from `SmartParking/legacy/`.
Deleting the directory would not break the web app, training, or
tests — only this historical record.
