# Data Card — NeoSmart / CarDetector training set

## Summary

A single-class (`car`) object-detection dataset composed entirely of
synthetic frames rendered from an Unreal Engine 5 virtual parking lot.
Used to train and evaluate `Models/Car_Detector.pt` — the YOLO model
that feeds every downstream component of the NeoSmart web app (parking
occupancy, wrong parking, suspicious parking, barrier access).

| Field | Value |
|-------|-------|
| Location in repo | [Training/data/](../../Training/data/) (gitignored, not shipped) |
| Manifest | [Training/data/data.yaml](../../Training/data/data.yaml) |
| Total images | 822 |
| Total annotations | 1821 objects |
| Classes | 1 — `car` |
| Image format | JPEG, filenames prefixed `unreal_*` |
| Label format | YOLO TXT (`<class> <cx> <cy> <w> <h>` normalised to `[0, 1]`) |
| License | Internal project asset — renders produced by this project's UE5 scene, not redistributed. |

## Splits

Ultralytics-style `train/ | val/ | test/` under `Training/data/`, each
with `images/` and `labels/`. Split breakdown (numbers match
[Documentation/eda/eda_report.md](../eda/eda_report.md)):

| split | images | objects | empty frames | µ obj/frame |
|-------|--------|---------|--------------|-------------|
| train | 576 | 1269 | 0 | 2.2 |
| val   | 164 | 373  | 0 | 2.3 |
| test  | 82  | 179  | 0 | 2.2 |

Every frame contains at least one car — no negative samples in the set.

## Provenance

The scene is a custom UE5 parking lot built for this project. A set of
fixed virtual cameras (mapping to the six dashboard cameras
`camera1`..`camera6`) export frames via the sidecar UE5 integration.
Ground-truth boxes are generated from the known placement of vehicle
actors in the scene, rather than manually labelled.

Consequences of that provenance:

- **Annotation quality is high and consistent** — boxes come from the
  scene graph, not from human click accuracy. Zero label noise.
- **Zero cost per additional image** — generating more data is a
  render pass, not a labelling campaign.
- **Coverage is as diverse as the scene is**. New diversity requires
  scene-level work (new camera placements, additional vehicle meshes,
  lighting variants) rather than data collection.

## Intended use

- Train and evaluate the car detector used in
  [SmartParking/web_app/detector.py](../../SmartParking/web_app/detector.py).
- Drive the exp/01–06 ClearML experiment series in
  [Documentation/reports/experiments_report.md](../reports/experiments_report.md).
- Benchmark the downstream stack (SORT tracking, wrong-parking logic,
  barrier state machine) against a controlled, reproducible input.

Any detection model trained exclusively on this set is intended for
**integration and end-to-end testing of the NeoSmart pipeline on
synthetic renders**, not as a production detector for real CCTV
footage.

## Out-of-scope use

- Generalising to real parking lots without domain adaptation or
  fine-tuning on real imagery. The sim-to-real gap is not measured
  here — see [Documentation/thesis/05_limitations_and_future_work.md](../thesis/05_limitations_and_future_work.md).
- Any deployment where failure modes have safety implications
  (autonomous vehicles, access control to critical assets, etc.). This
  data set is for a demonstration and research pipeline.

## Known biases and limitations

These follow from the synthetic provenance and are called out
explicitly so that evaluation metrics are read in the correct context:

- **No real-world footage.** Lighting, weather, motion blur, sensor
  noise, compression artefacts, and lens distortion do not match the
  behaviour of a physical camera.
- **Fixed camera poses.** The six cameras cover the same virtual lot
  from six fixed angles; distribution of object positions is
  camera-specific (see spatial heatmap in the EDA report).
- **No variation in weather / time of day** at the current scene
  version. Shadows and highlights are close to constant across frames.
- **Limited vehicle diversity.** The UE5 scene reuses a small set of
  vehicle meshes; the detector does not see the long tail of real-world
  vehicle shapes (vans, trucks, motorcycles, trailers).
- **Class coverage is trivial (nc=1).** The detector cannot be used as
  a multi-class traffic classifier without extending the label set and
  retraining.
- **Split is random, not stratified by camera.** Some drift between
  train and val is visible on bbox height (KS D=0.106, p≈0.003). The
  other three dimensions (width, aspect ratio, objects-per-image) show
  no significant drift.

## Real-data integration point

The pipeline is designed to swap in real imagery without code changes:
drop a real dataset under `Training/data/` (same Ultralytics layout) or
point `data.yaml` at a sibling folder (e.g. `Training/data_real/`) and
rerun `python Training/train.py --exp <name>`. No consumer of the
detector knows or cares whether the input came from UE5.

## Reproduction

The full EDA — splits, object counts, bounding-box geometry, spatial
heatmap, train/val KS-drift test, label sanity grid — is auto-generated
by [notebooks/01_eda.py](../../notebooks/01_eda.py):

```bash
python notebooks/01_eda.py              # refreshes Documentation/eda/
python notebooks/build_ipynb.py notebooks/01_eda.py   # regenerates .ipynb
```

Figures land in [Documentation/eda/figures/](../eda/figures/) and a
written report in [Documentation/eda/eda_report.md](../eda/eda_report.md).

## Change log

| Date | Change |
|------|--------|
| 2026-04-20 | Initial data card. Covers the UE5 synthetic snapshot used for the exp/01–06 training series. |
