# Model Card — NeoSmart / Car_Detector

## Overview

Single-class YOLOv8-small detector that powers every computer-vision
stage of the NeoSmart web app: parking-space occupancy, wrong-parking
detection, suspicious-parking (long-dwell) flagging, and the plate
barrier's car-presence gating.

| Field | Value |
|-------|-------|
| Model name | `Car_Detector.pt` (production weights) |
| Backbone | Ultralytics YOLOv8s (`yolov8s.pt` starting weights) |
| Input | RGB image, `imgsz=640` |
| Output | `(x1, y1, x2, y2, confidence)` per detected car |
| Classes | 1 — `car` |
| Framework | Ultralytics 8.x |
| Runtime | PyTorch (GPU or CPU). ONNX / TFLite exports exist only for the plate-scanner sibling model, not for this one. |
| Confidence threshold (live) | `0.65` (see [SmartParking/web_app/detector.py](../../SmartParking/web_app/detector.py)) |

## Intended use

- Detect cars on top-down / oblique camera feeds of a parking lot or
  entry lane, as rendered by the NeoSmart UE5 scene.
- Serve as the input stage for the downstream NeoSmart components:
  polygon-based occupancy ([detector.py](../../SmartParking/web_app/detector.py)),
  SORT tracking ([SmartParking/web_app/sort.py](../../SmartParking/web_app/sort.py)),
  barrier state machine ([barrier_controller.py](../../SmartParking/web_app/barrier_controller.py)),
  traffic counting.

## Out-of-scope use

- **Safety-critical deployment.** Not validated on real-world footage,
  not tested on edge cases like occlusion, adverse weather, or
  night-time scenes.
- **Multi-class traffic analysis.** Only `car` is predicted; buses,
  trucks, bikes, and pedestrians will be missed or misclassified as
  cars at best.
- **Deployment without retraining on domain data.** Expected
  performance drop on real CCTV footage is not measured here — the
  current evaluation is purely in-distribution on UE5 renders. See
  [Documentation/thesis/05_limitations_and_future_work.md](../thesis/05_limitations_and_future_work.md).

## Training data

See the full [data card](data_card.md). In short: 576/164/82
train/val/test split of UE5 renders of a virtual parking lot, single
`car` class, 1821 total annotations. Labels generated from the scene
graph (no human labelling, zero label noise).

**Synthetic-data caveat (explicit on purpose):** all training and
evaluation data in this repository is synthetic. The pipeline is
explicitly designed so that swapping in real imagery is a zero-code
change (drop data under `Training/data/` with the same layout). Whether
the learned features transfer to real cameras is an open question and
is treated as future work, not as a solved problem.

## Training procedure

The production weights (`Models/Car_Detector.pt`) were produced by the
release reproduction run (`exp/01 — baseline_yolov8s_100ep`, see
[Documentation/reports/experiments_report.md](../reports/experiments_report.md)):

| Hyperparameter | Value |
|----------------|-------|
| Backbone | `yolov8s.pt` |
| Input size | 640 |
| Epochs | 100 |
| Batch size | 16 |
| Optimizer | `auto` (Ultralytics default, SGD at this dataset size) |
| LR schedule | linear decay (Ultralytics default), no cosine |
| Warmup epochs | 3 |
| Augmentations | Ultralytics defaults (mosaic=1.0, fliplr=0.5, hsv) |
| Seed | 0 (Ultralytics default) |
| Early stopping | disabled (patience=100 = no patience trigger) |
| Precision | AMP (mixed) |

All other knobs are Ultralytics defaults. The run is reproducible from
the checked-in config at
[config/experiments/01_baseline.yaml](../../config/experiments/01_baseline.yaml)
merged with [config/training.yaml](../../config/training.yaml).

## Evaluation

Populated from the results table in
[Documentation/reports/experiments_report.md](../reports/experiments_report.md)
after each run completes. The current entries are placeholders and
will be filled in when the experiment series executes on GPU.

Metrics are computed on the in-distribution validation split of the
synthetic dataset. Any generalisation claim outside this domain is
explicitly out-of-scope.

Planned additional evaluation artefacts per-model:

- `Validation/reports/<date>/metrics.json` — mAP50, mAP50-95, P, R, F1
  at multiple confidence thresholds (conf-sweep).
- `Validation/reports/<date>/pr_curve.png` — precision/recall curve.
- `Validation/reports/<date>/latency.json` — per-stage wall-clock
  latency (inference, SORT, plate OCR, WebSocket encode).
- `Validation/reports/<date>/debug_samples/` — 16 val frames with
  predictions overlaid.

## Known failure modes

- **Small / distant cars** on fish-eye cameras are easier to miss
  because the resolution budget is 640 across the whole frame. The
  wrong-parking logic compensates with per-camera / per-spot
  `OUTSIDE_THRESHOLD` overrides ([config/default.yaml](../../config/default.yaml)),
  not by adjusting the detector.
- **Stacked / overlapping cars** near the barrier reading zone can
  cause the plate-recognition state machine to switch to the wrong
  vehicle. Mitigated by the "pick the car with the largest bbox inside
  the reading zone" heuristic in
  [barrier_controller.py](../../SmartParking/web_app/barrier_controller.py).
- **Cars parked at an unusual angle** (e.g. trailers on angled spots)
  are labelled by bounding box, so a tight box on an angled vehicle
  will overlap neighbouring polygon cells and can raise a false
  wrong-parking alert. This is a data / representation limitation, not
  a detector bug.
- **No detection for heavy vehicles / two-wheelers.** Only the `car`
  class is modelled.

## Responsible use

- Do not deploy against real surveillance footage without a dedicated
  real-data evaluation. The synthetic-only training leaves an open
  question about sim-to-real transfer; reporting synthetic metrics as
  real-world metrics is misleading.
- This detector is one input to a larger pipeline that also performs
  plate recognition and writes to a barrier-access database. Treat it
  accordingly: errors here propagate to access-control decisions.

## Reproduction

```bash
# One-time: local ClearML server for tracking.
docker compose -f docker-compose.clearml.yml up -d
# Configure credentials via http://localhost:8080 — see Documentation/CLEARML_SETUP.md.

# Train a single experiment (e.g. exp01 = release reproduction).
python Training/train.py \
    --config config/training.yaml \
    --overrides config/experiments/01_baseline.yaml \
    --exp-name exp01_baseline_yolov8s_100ep

# Or run the whole exp/01..06 series.
python Training/run_experiments.py --all

# Promote the winning run.
python Training/promote_best.py \
    --run-dir Training/runs/exp06_final_best \
    --version v2
```

## Change log

| Date | Change |
|------|--------|
| 2026-04-20 | Initial model card. Describes `Car_Detector.pt` as produced by the exp/01 release-reproduction run. Metrics table to be filled after the experiment series executes. |
