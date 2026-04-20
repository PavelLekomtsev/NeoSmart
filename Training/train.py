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


def report_dataset_stats(stats: dict[str, Any]) -> None:
    cl = Logger.current_logger()
    total_images = sum(s["images"] for s in stats["splits"].values())
    total_objects = sum(s["objects"] for s in stats["splits"].values())
    cl.report_single_value("dataset_total_images", total_images)
    cl.report_single_value("dataset_total_objects", total_objects)
    for split, s in stats["splits"].items():
        cl.report_single_value(f"dataset_{split}_images", s["images"])
        cl.report_single_value(f"dataset_{split}_objects", s["objects"])

    if stats["bbox_widths"]:
        cl.report_histogram(
            title="bbox_normalized_width",
            series="all_splits",
            values=np.array(stats["bbox_widths"]),
            xaxis="bbox w / image w",
            yaxis="count",
        )
        cl.report_histogram(
            title="bbox_normalized_height",
            series="all_splits",
            values=np.array(stats["bbox_heights"]),
            xaxis="bbox h / image h",
            yaxis="count",
        )
        cl.report_histogram(
            title="bbox_aspect_ratio",
            series="all_splits",
            values=np.array(stats["aspect_ratios"]),
            xaxis="max(w,h) / min(w,h)",
            yaxis="count",
        )
    if stats["objects_per_image"]:
        cl.report_histogram(
            title="objects_per_image",
            series="all_splits",
            values=np.array(stats["objects_per_image"]),
            xaxis="objects per image",
            yaxis="count",
        )


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
    """Ultralytics puts runs under <project_dir>/<exp_name>/ or
    <project_dir>/train/<exp_name>/ or <project_dir>/detect/<exp_name>/
    depending on version and task — probe the known locations."""
    for candidate in (
        project_dir / exp_name,
        project_dir / "train" / exp_name,
        project_dir / "detect" / exp_name,
    ):
        if candidate.exists():
            return candidate
    return None


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

    if args.dry_run:
        print_section("Dry run — skipping training")
        return 0

    print_section("Training")
    # Ultralytics resolves `data: data2/data.yaml` relative to CWD.
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
        best_pt = upload_run_artifacts(task, run_dir)
        if best_pt is not None:
            register_best_model(task, best_pt, args.exp_name, cfg.get("tags", []))

    print_section("Done")
    logger.info("ClearML: %s", task.get_output_log_web_page())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
