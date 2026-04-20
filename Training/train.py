"""YOLO training CLI with rich ClearML integration.

Replaces the bare Ultralytics call in the old TrainYolo.py. Layered on
top of Ultralytics' native ClearML hooks so that, after a run, the
ClearML task contains:

  * Configuration  — full training YAML.
  * Scalars        — per-epoch loss/mAP/P/R/F1 (Ultralytics native)
                     plus Single Values (dataset sizes, best metrics).
  * Plots          — bbox size / aspect ratio / objects-per-image
                     histograms + Ultralytics PR/F1/confusion matrices.
  * Debug Samples  — 16 train + 16 val images with GT overlays.
  * Artifacts      — best.pt, last.pt, results.png, confusion matrices.
  * Model Registry — best.pt registered under the task name.

Usage:
    python Training/train.py --exp-name exp01_baseline
    python Training/train.py --exp-name exp02_aug_v1 \\
        --mosaic 1.0 --mixup 0.15 --copy-paste 0.1

Ultralytics auto-detects ``clearml`` on import and attaches its own
callbacks; `Task.init` must therefore run *before* `YOLO(...)`.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — we render figures, don't show them.
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "SmartParking"))

# ClearML must import before ultralytics so the integration registers.
from clearml import Logger, OutputModel, Task  # noqa: E402
from neosmart.logging_setup import (  # noqa: E402
    configure_logging,
    print_banner,
    print_section,
)
from ultralytics import YOLO  # noqa: E402

logger = logging.getLogger("neosmart.training")


# ---------------------------------------------------------------------
# CLI + config loading
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a YOLO car detector with ClearML logging.",
    )
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "training.yaml"),
        help="YAML config with training defaults.",
    )
    p.add_argument(
        "--overrides",
        default=None,
        help="Optional YAML with per-experiment overrides merged on top of --config.",
    )
    p.add_argument(
        "--exp-name",
        required=True,
        help="ClearML task name + Ultralytics run folder.",
    )
    # --- overrides (1:1 with training.yaml keys) ---
    p.add_argument("--data", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cos-lr", dest="cos_lr", action="store_true", default=None)
    p.add_argument("--mosaic", type=float, default=None)
    p.add_argument("--mixup", type=float, default=None)
    p.add_argument("--copy-paste", dest="copy_paste", type=float, default=None)
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--lrf", type=float, default=None)
    p.add_argument("--warmup-epochs", dest="warmup_epochs", type=float, default=None)
    p.add_argument("--tags", nargs="+", default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config + create ClearML task + report stats, skip training.",
    )
    return p.parse_args()


_OVERRIDE_KEYS = (
    "data", "model", "epochs", "imgsz", "batch", "patience", "device",
    "workers", "seed", "cos_lr", "mosaic", "mixup", "copy_paste",
    "lr0", "lrf", "warmup_epochs", "tags",
)


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    with path.open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    if args.overrides:
        ov_path = Path(args.overrides).resolve()
        with ov_path.open() as f:
            overrides: dict[str, Any] = yaml.safe_load(f) or {}
        # Tags merge as union; scalar keys are replaced.
        if overrides.get("tags"):
            cfg["tags"] = list(dict.fromkeys(cfg.get("tags", []) + overrides["tags"]))
            overrides.pop("tags")
        cfg.update(overrides)
    for key in _OVERRIDE_KEYS:
        v = getattr(args, key, None)
        if v is not None:
            cfg[key] = v
    return cfg


# ---------------------------------------------------------------------
# Dataset statistics + debug samples
# ---------------------------------------------------------------------


def collect_dataset_stats(data_yaml_path: Path) -> dict[str, Any]:
    """Walk train/val/test labels and build per-split summaries."""
    data_yaml = yaml.safe_load(data_yaml_path.read_text())
    root = data_yaml_path.parent
    out: dict[str, Any] = {
        "classes": data_yaml.get("names", []),
        "splits": {},
        "bbox_widths": [],
        "bbox_heights": [],
        "aspect_ratios": [],
        "objects_per_image": [],
    }
    for split in ("train", "val", "test"):
        rel = data_yaml.get(split)
        if not rel:
            continue
        img_dir = (root / rel).resolve()
        if not img_dir.exists():
            continue
        lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.exists():
            continue

        n_images = 0
        n_objects = 0
        class_counts: Counter[int] = Counter()
        for lbl_file in lbl_dir.glob("*.txt"):
            n_images += 1
            per_image = 0
            with lbl_file.open() as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    w = float(parts[3])
                    h = float(parts[4])
                    out["bbox_widths"].append(w)
                    out["bbox_heights"].append(h)
                    denom = max(min(w, h), 1e-6)
                    out["aspect_ratios"].append(max(w, h) / denom)
                    class_counts[cls] += 1
                    per_image += 1
                    n_objects += 1
            out["objects_per_image"].append(per_image)
        out["splits"][split] = {
            "images": n_images,
            "objects": n_objects,
            "class_counts": dict(class_counts),
        }
    return out


def _histogram_figure(values: np.ndarray, *, title: str, xlabel: str,
                      bins: int = 40) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    ax.hist(values, bins=bins, color="#3b82f6", edgecolor="#1e3a8a")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)
    mean = float(np.mean(values))
    median = float(np.median(values))
    ax.axvline(mean, color="#ef4444", linestyle="--", linewidth=1,
               label=f"mean = {mean:.3f}")
    ax.axvline(median, color="#10b981", linestyle="--", linewidth=1,
               label=f"median = {median:.3f}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _class_balance_figure(stats: dict[str, Any]) -> plt.Figure | None:
    classes: list[str] = list(stats.get("classes") or [])
    if not classes:
        return None
    splits = list(stats["splits"].keys())
    x = np.arange(len(classes))
    width = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    for i, split in enumerate(splits):
        cc = stats["splits"][split]["class_counts"]
        counts = [cc.get(ci, 0) for ci in range(len(classes))]
        ax.bar(x + (i - (len(splits) - 1) / 2) * width, counts,
               width=width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_title("Object counts per class, per split")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _split_size_figure(stats: dict[str, Any]) -> plt.Figure:
    splits = list(stats["splits"].keys())
    imgs = [stats["splits"][s]["images"] for s in splits]
    objs = [stats["splits"][s]["objects"] for s in splits]
    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    ax.bar(x - 0.2, imgs, width=0.4, label="images", color="#3b82f6")
    ax.bar(x + 0.2, objs, width=0.4, label="objects", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_title("Dataset split sizes")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    for xi, v in zip(x - 0.2, imgs, strict=False):
        ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + 0.2, objs, strict=False):
        ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def report_dataset_stats(stats: dict[str, Any]) -> None:
    """Upload dataset summary to ClearML.

    Writes single-value metrics AND matplotlib figures (explicit images
    in the Plots tab — ``report_histogram`` alone sometimes lands in a
    sub-section that's easy to miss).
    """
    cl = Logger.current_logger()

    total_images = sum(s["images"] for s in stats["splits"].values())
    total_objects = sum(s["objects"] for s in stats["splits"].values())
    cl.report_single_value("dataset_total_images", total_images)
    cl.report_single_value("dataset_total_objects", total_objects)
    for split, s in stats["splits"].items():
        cl.report_single_value(f"dataset_{split}_images", s["images"])
        cl.report_single_value(f"dataset_{split}_objects", s["objects"])

    fig = _split_size_figure(stats)
    cl.report_matplotlib_figure(title="dataset_split_sizes",
                                series="images_vs_objects",
                                figure=fig, iteration=0, report_image=True)
    plt.close(fig)

    cb = _class_balance_figure(stats)
    if cb is not None:
        cl.report_matplotlib_figure(title="class_balance",
                                    series="per_split",
                                    figure=cb, iteration=0, report_image=True)
        plt.close(cb)

    if stats["bbox_widths"]:
        widths = np.array(stats["bbox_widths"])
        heights = np.array(stats["bbox_heights"])
        aspects = np.array(stats["aspect_ratios"])
        areas = widths * heights  # normalised relative box area

        for title, values, xlabel in (
            ("bbox_normalized_width", widths, "bbox w / image w"),
            ("bbox_normalized_height", heights, "bbox h / image h"),
            ("bbox_aspect_ratio", aspects, "max(w,h) / min(w,h)"),
            ("bbox_normalized_area", areas, "bbox area / image area"),
        ):
            fig = _histogram_figure(values, title=title, xlabel=xlabel)
            cl.report_matplotlib_figure(title=title, series="all_splits",
                                        figure=fig, iteration=0,
                                        report_image=True)
            plt.close(fig)

    if stats["objects_per_image"]:
        opi = np.array(stats["objects_per_image"])
        fig = _histogram_figure(opi, title="objects_per_image",
                                xlabel="objects per image",
                                bins=int(max(opi.max() - opi.min() + 1, 10)))
        cl.report_matplotlib_figure(title="objects_per_image",
                                    series="all_splits",
                                    figure=fig, iteration=0,
                                    report_image=True)
        plt.close(fig)

    logger.info("Uploaded dataset plots: split sizes, class balance, "
                "bbox width/height/aspect/area histograms, objects-per-image.")


def _overlay_gt(img: Image.Image, label_file: Path,
                color: str = "yellow") -> Image.Image:
    """Draw YOLO-format GT boxes onto a copy of the image."""
    img = img.copy()
    if not label_file.exists():
        return img
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]))
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return img


def report_augmented_individual_samples(cfg: dict[str, Any],
                                        data_yaml_path: Path,
                                        *, n: int = 16) -> None:
    """Show what a single training sample looks like AFTER Ultralytics'
    per-image augmentations (HSV / flip / translate / scale / rotate /
    shear / perspective) but BEFORE mosaic/mixup/copy-paste are applied.

    The trainer's own ``train_batch*.jpg`` snapshots are useful to see
    mosaic composition, but when an experiment varies per-image aug
    (HSV, geometric) the mosaic hides what actually changed per sample.
    This function builds a fresh YOLODataset with mosaic=mixup=copy_paste=0
    (while inheriting every other augmentation knob from ``cfg``), pulls
    ``n`` samples, and uploads them with GT overlays to Debug Samples.
    """
    try:
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset
    except ImportError as exc:
        logger.warning("Could not import Ultralytics data utilities: %s", exc)
        return

    # Pass-through keys — only the ones Ultralytics' DEFAULT_CFG actually
    # recognises. Anything else would raise in get_cfg.
    passthrough = {
        k: cfg[k]
        for k in (
            "imgsz", "hsv_h", "hsv_s", "hsv_v",
            "degrees", "translate", "scale", "shear", "perspective",
            "flipud", "fliplr",
        )
        if k in cfg
    }
    passthrough.update({
        "task": "detect",
        "mode": "train",
        "data": str(data_yaml_path),
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
    })
    yolo_cfg = get_cfg(overrides=passthrough)
    data = check_det_dataset(str(data_yaml_path))
    try:
        dataset = build_yolo_dataset(yolo_cfg, data["train"], batch=1,
                                     data=data, mode="train", stride=32)
    except Exception as exc:
        logger.warning("Could not build augmented preview dataset: %s", exc)
        return

    rng = random.Random(cfg.get("seed", 0))
    picks = rng.sample(range(len(dataset)), min(n, len(dataset)))

    cl = Logger.current_logger()
    uploaded = 0
    for i, idx in enumerate(picks):
        try:
            sample = dataset[idx]
        except Exception as exc:
            logger.warning("Could not load augmented sample %d: %s", idx, exc)
            continue

        # Tensor [C,H,W] uint8 (RGB) → HWC uint8.
        img_t = sample["img"]
        if hasattr(img_t, "cpu"):
            img_t = img_t.cpu().numpy()
        img = np.ascontiguousarray(np.transpose(img_t, (1, 2, 0)))
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        H, W = img.shape[:2]

        bboxes = sample.get("bboxes")
        if bboxes is not None and len(bboxes):
            boxes = bboxes.cpu().numpy() if hasattr(bboxes, "cpu") else np.asarray(bboxes)
            for cx, cy, bw, bh in boxes:
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                x2 = (cx + bw / 2) * W
                y2 = (cy + bh / 2) * H
                draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)

        cl.report_image(
            title="train_augmented_individual",
            series=Path(sample["im_file"]).stem,
            iteration=i,
            image=np.array(pil),
        )
        uploaded += 1

    if uploaded:
        logger.info("Uploaded %d non-mosaic augmented samples → Debug Samples "
                    "(title: train_augmented_individual).", uploaded)


def report_debug_samples(data_yaml_path: Path, *, n: int = 16,
                         seed: int = 0) -> None:
    cl = Logger.current_logger()
    data_yaml = yaml.safe_load(data_yaml_path.read_text())
    root = data_yaml_path.parent
    rng = random.Random(seed)

    for split in ("train", "val"):
        rel = data_yaml.get(split)
        if not rel:
            continue
        img_dir = (root / rel).resolve()
        lbl_dir = img_dir.parent / "labels"
        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if not imgs:
            continue
        picks = rng.sample(imgs, min(n, len(imgs)))
        for i, img_path in enumerate(picks):
            img = Image.open(img_path).convert("RGB")
            overlaid = _overlay_gt(
                img, lbl_dir / (img_path.stem + ".txt"), color="yellow",
            )
            cl.report_image(
                title=f"{split}_samples_gt",
                series=img_path.stem,
                iteration=i,
                image=np.array(overlaid),
            )


# ---------------------------------------------------------------------
# Post-training model registration
# ---------------------------------------------------------------------


def locate_run_dir(project_dir: Path, exp_name: str) -> Path | None:
    """Return the run directory for ``exp_name`` under ``project_dir``.

    Ultralytics puts runs under ``<project_dir>/<exp_name>/`` (or under a
    ``train/``/``detect/`` subdir depending on version/task) and, with
    ``exist_ok=False``, auto-increments by appending a number — so a
    second run of the same experiment lands in ``<exp_name>2``,
    ``<exp_name>3``, etc. Pick the most recently modified match so a
    restart uploads artifacts from the new run rather than the stale
    one with the base name.
    """
    search_roots = (project_dir, project_dir / "train", project_dir / "detect")
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(root.glob(f"{exp_name}*"))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def upload_run_artifacts(task: Task, run_dir: Path) -> Path | None:
    """Attach best.pt / last.pt + standard Ultralytics plots as Artifacts."""
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    if best_pt.exists():
        task.upload_artifact("best.pt", artifact_object=best_pt)
    if last_pt.exists():
        task.upload_artifact("last.pt", artifact_object=last_pt)
    for name in (
        "results.png",
        "results.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "args.yaml",
    ):
        p = run_dir / name
        if p.exists():
            task.upload_artifact(name, artifact_object=p)
    return best_pt if best_pt.exists() else None


def register_best_model(task: Task, best_pt: Path, exp_name: str,
                        tags: list[str]) -> None:
    out = OutputModel(task=task, name=exp_name, framework="PyTorch", tags=tags)
    out.update_weights(weights_filename=str(best_pt))
    logger.info("Registered %s in ClearML Model Registry as %s", best_pt, exp_name)


def report_run_debug_images(run_dir: Path) -> None:
    """Push Ultralytics' train/val batch snapshots to ClearML Debug Samples.

    Ultralytics writes these to disk as part of normal training:
      * ``train_batch{0,1,2}.jpg`` — first three batches AFTER augmentation +
        mosaic, so an experiment's actual aug policy is visible. Lives in
        the train run dir.
      * ``val_batch{N}_labels.jpg`` — GT overlays on val batches.
      * ``val_batch{N}_pred.jpg``  — model predictions on val batches.

    Gotcha: in Ultralytics 8.x the in-training validator creates its own
    save_dir under ``<project_dir>/detect/train<N>/`` instead of writing
    into the trainer's run_dir, so the val images do NOT land in
    ``run_dir``. We look for them there too and pick the directory with
    the newest val_batch file (handles repeated runs of the same
    experiment).
    """
    cl = Logger.current_logger()

    train_files = sorted(run_dir.glob("train_batch*.jpg"))

    def find_val_dir() -> Path | None:
        # First preference: val plots already sitting in run_dir (works
        # if Ultralytics ever fixes the save_dir inheritance).
        if list(run_dir.glob("val_batch*_labels.jpg")):
            return run_dir
        # Fallback: sibling <project_dir>/detect/train*/ — pick the one
        # with the newest val_batch*_labels.jpg.
        aux_root = run_dir.parent / "detect"
        if not aux_root.exists():
            return None
        candidates = [d for d in aux_root.glob("train*") if d.is_dir()
                      and list(d.glob("val_batch*_labels.jpg"))]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda d: max((f.stat().st_mtime
                               for f in d.glob("val_batch*_labels.jpg")),
                              default=0.0),
        )

    val_dir = find_val_dir()
    if val_dir is None:
        logger.warning("No val_batch*.jpg files found under %s or %s/detect/train*/ "
                       "— val predictions will be missing from Debug Samples.",
                       run_dir, run_dir.parent)
    else:
        logger.info("Val plots directory: %s", val_dir)

    val_gt = sorted(val_dir.glob("val_batch*_labels.jpg")) if val_dir else []
    val_pred = sorted(val_dir.glob("val_batch*_pred.jpg")) if val_dir else []

    groups = (
        ("train_augmented_batches", train_files),
        ("val_ground_truth", val_gt),
        ("val_predictions", val_pred),
    )
    for title, files in groups:
        for i, fp in enumerate(files):
            try:
                img = np.array(Image.open(fp).convert("RGB"))
            except Exception as exc:
                logger.warning("Could not load %s: %s", fp, exc)
                continue
            cl.report_image(title=title, series=fp.stem, iteration=i, image=img)
        if files:
            logger.info("Uploaded %d %s → Debug Samples.", len(files), title)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main() -> int:
    configure_logging()
    print_banner("NeoSmart — YOLO training · ClearML")
    args = parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path, args)

    print_section("Configuration")
    for k, v in cfg.items():
        logger.info("  %-15s %s", k, v)

    print_section("ClearML task init")
    task = Task.init(
        project_name=cfg["project"],
        task_name=args.exp_name,
        tags=cfg.get("tags", []),
        output_uri=True,
        reuse_last_task_id=False,
    )
    task.connect_configuration(configuration=cfg, name="training_config")
    task.connect(
        {k: v for k, v in cfg.items() if k not in ("project", "tags")},
        name="hyperparameters",
    )
    logger.info("ClearML task page: %s", task.get_output_log_web_page())

    # Data paths — Ultralytics resolves `data` against CWD.
    data_yaml_abs = (REPO_ROOT / "Training" / cfg["data"]).resolve()
    if not data_yaml_abs.exists():
        logger.error("data.yaml not found: %s", data_yaml_abs)
        return 2

    print_section("Dataset statistics")
    stats = collect_dataset_stats(data_yaml_abs)
    for split, s in stats["splits"].items():
        logger.info("  %-5s  images=%d  objects=%d",
                    split, s["images"], s["objects"])
    report_dataset_stats(stats)

    print_section("Debug samples")
    report_debug_samples(data_yaml_abs, n=16, seed=cfg.get("seed", 0))
    logger.info("Uploaded 16 train + 16 val samples with GT overlays.")
    report_augmented_individual_samples(cfg, data_yaml_abs, n=16)

    if args.dry_run:
        print_section("Dry run — skipping training")
        return 0

    print_section("Training")
    # Ultralytics resolves `data: data/data.yaml` relative to CWD.
    os.chdir(REPO_ROOT / "Training")
    model = YOLO(cfg["model"])
    skip_keys = {"project", "tags", "model", "project_dir"}
    yolo_kwargs = {k: v for k, v in cfg.items() if k not in skip_keys}
    yolo_kwargs["project"] = cfg.get("project_dir", "runs")
    yolo_kwargs["name"] = args.exp_name
    yolo_kwargs["exist_ok"] = False
    model.train(**yolo_kwargs)

    print_section("Artifacts + Model Registry")
    project_dir = (REPO_ROOT / "Training" / cfg.get("project_dir", "runs")).resolve()
    run_dir = locate_run_dir(project_dir, args.exp_name)
    if run_dir is None:
        logger.warning("Could not locate run directory under %s — artifact upload skipped.",
                       project_dir)
    else:
        logger.info("Run directory: %s", run_dir)
        report_run_debug_images(run_dir)
        best_pt = upload_run_artifacts(task, run_dir)
        if best_pt is not None:
            register_best_model(task, best_pt, args.exp_name, cfg.get("tags", []))

    print_section("Done")
    logger.info("ClearML: %s", task.get_output_log_web_page())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
