"""Promote an experiment's best.pt to production Car_Detector.pt.

Archives the current ``Models/Car_Detector.pt`` to
``Models/archive/Car_Detector_<version_prev>.pt`` (auto-named by
existing-version scan) and copies the new best into place. If a
ClearML task id is supplied, also tags the task and re-registers the
weights in the ClearML Model Registry under ``Car_Detector@<version>``.

Usage:
    python Training/promote_best.py \\
        --run-dir Training/runs/exp06_final_best \\
        --version v2

    # + register in Model Registry under the existing task
    python Training/promote_best.py \\
        --run-dir Training/runs/exp06_final_best \\
        --version v2 \\
        --clearml-task-id <task-id>
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "SmartParking"))

from neosmart.logging_setup import configure_logging, print_banner, print_section  # noqa: E402

logger = logging.getLogger("neosmart.promote")

MODELS_DIR = REPO_ROOT / "Models"
LIVE_WEIGHTS = MODELS_DIR / "Car_Detector.pt"
ARCHIVE_DIR = MODELS_DIR / "archive"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Promote experiment best.pt to production.")
    p.add_argument("--run-dir", required=True, help="Path to Ultralytics run dir (contains weights/best.pt).")
    p.add_argument("--version", required=True, help="Version label for the promoted weights (e.g. v2).")
    p.add_argument("--clearml-task-id", default=None, help="Optional ClearML task id to register model under.")
    p.add_argument("--force", action="store_true", help="Overwrite archive target if it already exists.")
    return p.parse_args()


def archive_current(version_label: str, *, force: bool) -> Path | None:
    if not LIVE_WEIGHTS.exists():
        logger.info("No live %s to archive.", LIVE_WEIGHTS)
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    target = ARCHIVE_DIR / f"Car_Detector_{version_label}.pt"
    if target.exists() and not force:
        raise SystemExit(f"Archive target already exists: {target} (use --force).")
    shutil.copy2(LIVE_WEIGHTS, target)
    logger.info("Archived live weights → %s", target)
    return target


def install_new(best_pt: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, LIVE_WEIGHTS)
    logger.info("Installed new weights → %s", LIVE_WEIGHTS)


def register_in_clearml(task_id: str, best_pt: Path, version: str) -> None:
    from clearml import OutputModel, Task

    task = Task.get_task(task_id=task_id)
    name = f"Car_Detector@{version}"
    tags = [*(task.get_tags() or []), "promoted", version]
    out = OutputModel(task=task, name=name, framework="PyTorch", tags=tags)
    out.update_weights(weights_filename=str(best_pt))
    task.add_tags(["promoted", version])
    logger.info("Registered in ClearML Model Registry as %s", name)


def main() -> int:
    configure_logging()
    print_banner("NeoSmart — model promotion")
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        logger.error("best.pt not found under %s", run_dir)
        return 2

    print_section("Archive current live weights")
    # Find next free slot for the OLD weights — label with predecessor version.
    # If version is v2, the old copy becomes Car_Detector_v1.pt; if v3, _v2.pt.
    stem = args.version.lstrip("v")
    try:
        prev = f"v{int(stem) - 1}" if stem.isdigit() else "prev"
    except ValueError:
        prev = "prev"
    archive_current(prev, force=args.force)

    print_section("Install new weights")
    install_new(best_pt)

    if args.clearml_task_id:
        print_section("ClearML Model Registry")
        register_in_clearml(args.clearml_task_id, best_pt, args.version)

    print_section("Done")
    logger.info("Live weights: %s", LIVE_WEIGHTS)
    logger.info("Archived under: %s", ARCHIVE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
