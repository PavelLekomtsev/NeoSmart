# NeoSmart — CarDetector experiment series

## Scope

This report tracks the progressive improvement of the car detector
from the reference `Car_Detector.pt` checkpoint. Three experiments
are run sequentially under the ClearML project `NeoSmart/CarDetector`;
the best-performing run replaces the production weights as
`Car_Detector@v2` and is archived side-by-side with v1.

The series was consolidated from an earlier six-experiment plan. The
aug study and the LR-schedule study were merged (they are cheap,
non-interacting wins), and the AdamW / regularisation / long-horizon
studies were merged into a single "full stack" run so the promotion
candidate is trained once with every winning ingredient in place,
rather than stacking three separate runs that mostly re-measure the
same effect.

**Invariants across the series.** All experiments use the same
backbone (`yolov8s.pt`), the same input resolution (`imgsz=640`), the
same dataset (`Training/data/` — UE5-rendered synthetic frames), and
the same seed (`42`). Only
hyperparameters, optimizer, LR schedule, regularisation, and
augmentations are varied. This isolates the effect of training-time
choices from architectural ones and keeps the live-inference pipeline
(`SmartParking/web_app/detector.py`) unchanged — the resulting
weights are a drop-in replacement for the current `Car_Detector.pt`.

Synthetic data is a deliberate choice — see
[`Documentation/thesis/05_limitations_and_future_work.md`](../thesis/05_limitations_and_future_work.md)
for the rationale. The pipeline is ready to accept real footage at
`Training/data_real/` with no code changes.

## How to reproduce

```bash
# 1. Start the local ClearML server
docker compose -f docker-compose.clearml.yml up -d

# 2. Configure the SDK (one-time) — see Documentation/CLEARML_SETUP.md
#    (generates credentials via http://localhost:8080, writes ~/clearml.conf)

# 3. Run everything, sequentially
python Training/run_experiments.py --all

# Alternatives
python Training/run_experiments.py --only 03       # one experiment
python Training/run_experiments.py --from 04       # resume from id 04
python Training/run_experiments.py --all --continue-on-failure

# 4. Promote the winner
python Training/promote_best.py \
    --run-dir Training/runs/exp03_full_stack \
    --version v2
```

## Results

Metrics are read from the validation split at the final epoch of each
run. Latency is measured via `Validation/evaluate.py` after promotion.

Fill this table after each run completes.

Metrics are taken at the best epoch by `mAP50-95(B)` of each run.

| # | Name | Key change | mAP50 | mAP50-95 | P | R | F1@0.5 | Best ep. | ClearML |
|---|------|------------|-------|----------|---|---|--------|----------|---------|
| 01 | baseline_yolov8s_100ep | release repro of `Car_Detector.pt` (TrainYolo.py args + defaults) | 0.9945 | **0.9738** | 0.9942 | 0.9920 | 0.9931 | 98 / 100 | [task a14cf011](http://localhost:8080/projects/df3975eb30224f8b9b81af297652c3e7/experiments/a14cf011531b4586bb21ebf071bb220d/output/log) |
| 02 | aug_schedule           | mosaic+mixup+copy-paste+HSV + cosine LR + warmup + early stop, 150 ep | 0.9946 | 0.9728 | 0.9943 | 0.9866 | 0.9904 | 109 / 129 (EarlyStop) | [task 6459eb76](http://localhost:8080/projects/df3975eb30224f8b9b81af297652c3e7/experiments/6459eb767d9144368696ec9adcf7fa28/output/log) |
| 03 | full_stack             | AdamW + lower lr0 + label smoothing + weight decay + geometric aug, 100 ep | 0.9946 | 0.9377 | 0.9940 | 0.9866 | 0.9903 | 90 / 100 | [task b8aef48b](http://localhost:8080/projects/df3975eb30224f8b9b81af297652c3e7/experiments/b8aef48b902a4f4c994ffb1ebdcc718a/output/log) |

## Narrative

_Fill per-experiment observations here: what changed relative to the
previous run, whether the metric movement matched the hypothesis,
notable failure modes in `Debug Samples`, decisions carried forward
into the next run._

### exp/01 — release repro
_Re-enacts the original training call that produced
`Models/Car_Detector.pt` (former `Training/TrainYolo.py`): yolov8s,
`data/data.yaml`, 100 epochs, imgsz 640, every other knob at
Ultralytics default. Provides a fair head-to-head baseline for the
later runs — any improvement shown by exp/02–06 is measured against
these numbers, not against the v1 weights (which were trained with an
older Ultralytics release and without ClearML tracking)._

### exp/02 — augmentations + cosine LR schedule
_Hypothesis: mosaic + mixup + copy-paste + stronger HSV reduce
scene-specific overfitting on synthetic footage, and a cosine LR with
warmup descends more smoothly than the default step schedule. Merged
into one run because the two knobs do not interact pathologically and
the baseline already isolates the default-everything case — any gain
here is attributable to this combined change. Patience-based early
stop caps wasted compute at 20 non-improving epochs._

### exp/03 — full stack (AdamW + regularisation + long horizon)
_Hypothesis: on top of the exp/02 aug stack, switching to AdamW with
a ~10× smaller `lr0`, adding label smoothing + higher weight decay,
and introducing mild rotation/shear/perspective will find a flatter
minimum with better small-box recall. Horizon matched to exp/01
(100 epochs) for a fair head-to-head comparison; patience 20 lets
AdamW converge without burning budget on a plateau. Promoted as
`Car_Detector@v2` if it wins._

## Artifacts per run

Each ClearML task contains:

- **Configuration** — full training YAML (`config/training.yaml` +
  per-experiment overrides merged on top).
- **Hyperparameters** — scalar knobs (epochs, LR, optimizer, aug).
- **Scalars** — per-epoch loss, mAP50, mAP50-95, precision, recall
  (Ultralytics native) + single values for dataset sizes and final
  metrics.
- **Plots** — bbox width/height/aspect-ratio histograms,
  objects-per-image histogram, Ultralytics PR / P / R / F1 curves,
  confusion matrix.
- **Debug Samples** — 16 train images with GT overlays + 16 val
  images with GT overlays (uploaded before training); Ultralytics'
  `train_batch*.jpg` showing augmented batches actually fed to the
  model; `val_batch*_labels.jpg` and `val_batch*_pred.jpg` so GT vs.
  model predictions on the val split can be eyeballed side-by-side.
- **Artifacts** — `best.pt`, `last.pt`, `results.png`, `results.csv`,
  `args.yaml`.
- **Model Registry** — `best.pt` registered under the experiment
  name; `exp03_full_stack` is additionally registered as
  `Car_Detector@v2` after promotion.

## Promotion protocol

1. Identify the winning run by comparing the rows above (sort by
   mAP50-95, break ties by F1@0.5).
2. Run `python Training/promote_best.py --run-dir <path> --version v2
   --clearml-task-id <id>`.
3. The script copies the current `Models/Car_Detector.pt` to
   `Models/archive/Car_Detector_v1.pt`, installs the new weights, and
   re-registers them in ClearML as `Car_Detector@v2`.
4. Commit the promoted weights with `feat(models): promote
   Car_Detector@v2` so the git history pins which experiment produced
   production weights.
