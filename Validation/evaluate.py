"""CLI: evaluate a YOLO car detector and publish a ClearML eval task.

Usage:
    python Validation/evaluate.py \\
        --model Models/Car_Detector.pt \\
        --data Training/data2/valid

Outputs (under ``Validation/reports/YYYY-MM-DD/<exp-name>/``):
    metrics.json          — primary + sweep + best-F1 + latency
    pr_curve.png          — precision / recall / F1 vs. confidence
    latency.json          — per-stage latency breakdown
    debug_samples/        — 16 val images with prediction overlays

A matching ClearML task is created under ``NeoSmart/CarDetector/eval``
with Configuration / Artifacts / Plots / Debug Samples / Single Values.
Pass ``--no-clearml`` to skip ClearML for a local-only run.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display.
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "SmartParking"))

# ClearML import must precede ultralytics so its callbacks attach.
from clearml import Logger, Task  # noqa: E402
from neosmart.eval.compare import evaluate_model  # noqa: E402
from neosmart.eval.latency import benchmark  # noqa: E402
from neosmart.eval.threshold_sweep import SweepPoint, best_conf, sweep  # noqa: E402
from neosmart.logging_setup import (  # noqa: E402
    configure_logging,
    print_banner,
    print_section,
)
from ultralytics import YOLO  # noqa: E402

logger = logging.getLogger("neosmart.validation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a YOLO model and publish to ClearML.",
    )
    p.add_argument("--model", required=True, help="Path to .pt weights.")
    p.add_argument(
        "--data", required=True,
        help="Dataset split dir (expects images/ and labels/).",
    )
    p.add_argument(
        "--exp-name", default=None,
        help="ClearML task name; defaults to <model-stem>_<timestamp>.",
    )
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument(
        "--sweep-values", type=float, nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8],
    )
    p.add_argument("--latency-runs", type=int, default=100)
    p.add_argument("--latency-warmup", type=int, default=10)
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument(
        "--no-clearml", action="store_true",
        help="Skip ClearML task creation (local-only run).",
    )
    return p.parse_args()


def write_pr_curve(sweep_pts: list[SweepPoint], out_path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    confs = [p.conf for p in sweep_pts]
    ax.plot(confs, [p.precision for p in sweep_pts], marker="o", label="Precision")
    ax.plot(confs, [p.recall for p in sweep_pts], marker="s", label="Recall")
    ax.plot(confs, [p.f1 for p in sweep_pts], marker="^", label="F1")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs. confidence")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig


def write_debug_samples(
    model_path: Path, data_dir: Path, out_dir: Path, *,
    n: int = 16, conf: float = 0.5,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"
    imgs = sorted(
        list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")),
    )[:n]
    model = YOLO(str(model_path))
    saved: list[Path] = []
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        res = model(str(img_path), conf=conf, verbose=False)
        if res and res[0].boxes is not None and len(res[0].boxes):
            xyxy = res[0].boxes.xyxy.cpu().numpy()
            confs = res[0].boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, confs, strict=False):
                x1, y1, x2, y2 = (int(v) for v in b)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img, f"{c:.2f}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        saved.append(out_path)
    return saved


def main() -> int:
    configure_logging()
    print_banner("NeoSmart — model evaluation · ClearML")
    args = parse_args()

    model_path = Path(args.model).resolve()
    data_dir = Path(args.data).resolve()
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        return 2
    if not (data_dir / "images").exists() or not (data_dir / "labels").exists():
        logger.error("Expected images/ and labels/ under %s", data_dir)
        return 2

    ts = datetime.datetime.now()
    exp_name = args.exp_name or f"{model_path.stem}_{ts:%Y%m%d_%H%M%S}"
    report_dir = REPO_ROOT / "Validation" / "reports" / f"{ts:%Y-%m-%d}" / exp_name
    report_dir.mkdir(parents=True, exist_ok=True)

    print_section("Configuration")
    logger.info("  model:       %s", model_path)
    logger.info("  data:        %s", data_dir)
    logger.info("  conf:        %.2f", args.conf)
    logger.info("  iou:         %.2f", args.iou)
    logger.info("  sweep:       %s", args.sweep_values)
    logger.info("  report_dir:  %s", report_dir)

    task: Task | None = None
    if not args.no_clearml:
        print_section("ClearML task init")
        task = Task.init(
            project_name="NeoSmart/CarDetector/eval",
            task_name=exp_name,
            tags=["eval", model_path.stem],
            output_uri=True,
            reuse_last_task_id=False,
        )
        task.connect_configuration(
            configuration={
                "model": str(model_path),
                "data": str(data_dir),
                "conf": args.conf,
                "iou": args.iou,
                "sweep_values": list(args.sweep_values),
                "latency_runs": args.latency_runs,
                "latency_warmup": args.latency_warmup,
            },
            name="eval_config",
        )
        logger.info("ClearML: %s", task.get_output_log_web_page())

    print_section(f"Evaluation (conf={args.conf:.2f})")
    primary = evaluate_model(
        model_path, data_dir,
        conf_thresh=args.conf, iou_thresh=args.iou,
        model_name=model_path.stem,
    )
    logger.info("  TP=%d  FP=%d  FN=%d", primary.tp, primary.fp, primary.fn)
    logger.info(
        "  P=%.4f  R=%.4f  F1=%.4f",
        primary.precision, primary.recall, primary.f1,
    )

    print_section("Threshold sweep")
    sweep_pts = sweep(
        model_path, data_dir,
        conf_values=list(args.sweep_values), iou_thresh=args.iou,
    )
    best = best_conf(sweep_pts)
    logger.info(
        "  Best F1 @ conf=%.2f → P=%.4f R=%.4f F1=%.4f",
        best.conf, best.precision, best.recall, best.f1,
    )
    pr_path = report_dir / "pr_curve.png"
    fig = write_pr_curve(sweep_pts, pr_path)

    latency_reports = []
    if not args.skip_latency:
        print_section("Latency benchmark")
        images = sorted(
            list((data_dir / "images").glob("*.jpg"))
            + list((data_dir / "images").glob("*.png"))
        )
        if images:
            latency_reports = benchmark(
                model_path, images[: max(50, args.latency_runs)],
                warmup=args.latency_warmup, runs=args.latency_runs,
                conf=args.conf, include_sort=True,
            )
            for lr in latency_reports:
                logger.info(
                    "  %-22s  mean=%6.2fms  p50=%6.2fms  p95=%6.2fms",
                    lr.stage, lr.mean_ms, lr.p50_ms, lr.p95_ms,
                )

    print_section("Debug samples")
    debug_dir = report_dir / "debug_samples"
    debug_paths = write_debug_samples(
        model_path, data_dir, debug_dir, n=16, conf=args.conf,
    )
    logger.info("  wrote %d debug samples → %s", len(debug_paths), debug_dir)

    # --- persist --------------------------------------------------
    metrics = {
        "exp_name": exp_name,
        "timestamp": ts.isoformat(),
        "model": str(model_path),
        "data": str(data_dir),
        "conf": args.conf,
        "iou": args.iou,
        "primary": primary.to_dict(),
        "sweep": [p.to_dict() for p in sweep_pts],
        "best": best.to_dict(),
        "latency": [lr.to_dict() for lr in latency_reports],
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (report_dir / "latency.json").write_text(
        json.dumps([asdict(lr) for lr in latency_reports], indent=2),
    )

    if task is not None:
        print_section("ClearML reporting")
        cl = Logger.current_logger()
        cl.report_single_value("primary_precision", primary.precision)
        cl.report_single_value("primary_recall", primary.recall)
        cl.report_single_value("primary_f1", primary.f1)
        cl.report_single_value("primary_tp", primary.tp)
        cl.report_single_value("primary_fp", primary.fp)
        cl.report_single_value("primary_fn", primary.fn)
        cl.report_single_value("best_f1_conf", best.conf)
        cl.report_single_value("best_f1", best.f1)
        for p in sweep_pts:
            cl.report_scalar(
                title="sweep_precision", series="precision",
                value=p.precision, iteration=int(round(p.conf * 100)),
            )
            cl.report_scalar(
                title="sweep_recall", series="recall",
                value=p.recall, iteration=int(round(p.conf * 100)),
            )
            cl.report_scalar(
                title="sweep_f1", series="f1",
                value=p.f1, iteration=int(round(p.conf * 100)),
            )
        for lr in latency_reports:
            cl.report_single_value(f"latency_{lr.stage}_mean_ms", lr.mean_ms)
            cl.report_single_value(f"latency_{lr.stage}_p95_ms", lr.p95_ms)
        cl.report_matplotlib_figure(
            title="pr_curve", series="conf_sweep", figure=fig,
        )
        task.upload_artifact("metrics.json", artifact_object=report_dir / "metrics.json")
        task.upload_artifact("latency.json", artifact_object=report_dir / "latency.json")
        task.upload_artifact("pr_curve.png", artifact_object=pr_path)
        for d in debug_paths[:16]:
            task.upload_artifact(f"debug/{d.name}", artifact_object=d)

    print_section("Done")
    logger.info("Report: %s", report_dir)
    if task is not None:
        logger.info("ClearML: %s", task.get_output_log_web_page())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
