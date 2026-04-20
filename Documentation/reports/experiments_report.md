# NeoSmart — CarDetector experiment series

## Scope

This report tracks the progressive improvement of the car detector
from the reference `Car_Detector.pt` checkpoint. Six experiments are
run sequentially under the ClearML project `NeoSmart/CarDetector`;
the best-performing run replaces the production weights as
`Car_Detector@v2` and is archived side-by-side with v1.

**Invariants across the series.** All experiments use the same
backbone (`yolov8s.pt`), the same input resolution (`imgsz=640`), the
same dataset (`Training/data2/`), and the same seed (`42`). Only
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
    --run-dir Training/runs/exp06_final_best \
    --version v2
```

## Results

Metrics are read from the validation split at the final epoch of each
run. Latency is measured via `Validation/evaluate.py` after promotion.

Fill this table after each run completes.

| # | Name | Key change | mAP50 | mAP50-95 | P | R | F1@0.5 | ClearML |
|---|------|------------|-------|----------|---|---|--------|---------|
| 01 | baseline_yolov8s_100ep | release repro of `Car_Detector.pt` (TrainYolo.py args + defaults) | — | — | — | — | — | `<link>` |
| 02 | aug_mosaic_mixup       | mosaic+mixup+copy-paste, stronger HSV | — | — | — | — | — | `<link>` |
| 03 | cosine_lr_warmup       | cosine LR + warmup + early stopping | — | — | — | — | — | `<link>` |
| 04 | optimizer_adamw        | AdamW optimizer + tuned lr/weight-decay | — | — | — | — | — | `<link>` |
| 05 | regularization         | label smoothing + weight decay + geometric aug | — | — | — | — | — | `<link>` |
| 06 | final_best             | combined best, long run | — | — | — | — | — | `<link>` |

## Narrative

_Fill per-experiment observations here: what changed relative to the
previous run, whether the metric movement matched the hypothesis,
notable failure modes in `Debug Samples`, decisions carried forward
into the next run._

### exp/01 — release repro
_Re-enacts the original training call that produced
`Models/Car_Detector.pt` (former `Training/TrainYolo.py`): yolov8s,
`data2/data.yaml`, 100 epochs, imgsz 640, every other knob at
Ultralytics default. Provides a fair head-to-head baseline for the
later runs — any improvement shown by exp/02–06 is measured against
these numbers, not against the v1 weights (which were trained with an
older Ultralytics release and without ClearML tracking)._

### exp/02 — aggressive augmentations
_Hypothesis: mosaic + mixup + copy-paste + stronger HSV reduce
scene-specific overfitting on synthetic footage._

### exp/03 — cosine LR + warmup + early stopping
_Hypothesis: smoother LR descent improves final mAP; patience-based
early stop prevents wasted compute._

### exp/04 — AdamW optimizer
_Hypothesis: AdamW with lower lr0 and higher weight decay finds a
flatter minimum than SGD + cosine on this dataset size._

### exp/05 — regularisation + geometric augmentation
_Hypothesis: label smoothing + stronger weight decay + mild rotation/
shear/perspective further improve generalisation, particularly for
small boxes._

### exp/06 — combined best
_Takes the winning ingredients from 02–05 and trains a longer horizon
with early stopping. Promoted as `Car_Detector@v2`._

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
  images with GT overlays; post-run sanity checks pull 16 val images
  with predictions overlaid via `Validation/evaluate.py`.
- **Artifacts** — `best.pt`, `last.pt`, `results.png`, `results.csv`,
  `args.yaml`.
- **Model Registry** — `best.pt` registered under the experiment
  name; `exp06_final_best` is additionally registered as
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
