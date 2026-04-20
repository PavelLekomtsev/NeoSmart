"""Run the planned NeoSmart/CarDetector experiment suite.

Each experiment is a thin YAML override on top of ``config/training.yaml``
(see ``config/experiments/``). This script knows the canonical ordering
and names, and shells out to ``Training/train.py`` one experiment at a
time so a crash in the middle doesn't lose the runs that already
succeeded.

Usage:
    # Everything (sequential)
    python Training/run_experiments.py --all

    # A single experiment by id
    python Training/run_experiments.py --only 03

    # Start from a given id (useful after a crash)
    python Training/run_experiments.py --from 04

    # List without running
    python Training/run_experiments.py --list
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "SmartParking"))

from neosmart.logging_setup import (  # noqa: E402
    configure_logging,
    print_banner,
    print_section,
)

logger = logging.getLogger("neosmart.experiments")


@dataclass(frozen=True)
class Experiment:
    id: str
    exp_name: str
    overrides: str
    description: str


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        id="01",
        exp_name="exp01_baseline_yolov8s_100ep",
        overrides="config/experiments/01_baseline.yaml",
        description="Release repro of Car_Detector.pt (TrainYolo.py args + Ultralytics defaults).",
    ),
    Experiment(
        id="02",
        exp_name="exp02_aug_schedule",
        overrides="config/experiments/02_aug_schedule.yaml",
        description="Stronger aug (mosaic+mixup+copy-paste+HSV) + cosine LR + warmup, 150 ep.",
    ),
    Experiment(
        id="03",
        exp_name="exp03_full_stack",
        overrides="config/experiments/03_full_stack.yaml",
        description="AdamW + label smoothing + weight decay + geometric aug, 100 ep; promote candidate.",
    ),
)


def preflight_clearml() -> bool:
    """Verify that ClearML is installed and the server is reachable.

    Returns True on success, False on failure. Prints actionable hints
    so the caller (human or agent) can fix setup before burning GPU
    hours on a run that never logs.
    """
    try:
        from clearml import Task
    except ImportError:
        logger.error("clearml is not installed. Run: pip install -r requirements-dev.txt")
        return False

    try:
        probe = Task.init(
            project_name="NeoSmart/_preflight",
            task_name="clearml_probe",
            auto_connect_frameworks=False,
            auto_resource_monitoring=False,
            reuse_last_task_id=True,
        )
        probe.close()
    except Exception as exc:
        logger.error("ClearML preflight failed: %s", exc)
        logger.error("Start the server: docker compose -f docker-compose.clearml.yml up -d")
        logger.error("Configure SDK: see Documentation/CLEARML_SETUP.md")
        return False

    logger.info("ClearML preflight OK — server reachable, credentials valid.")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the CarDetector experiment suite.")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true", help="Run every experiment.")
    sel.add_argument("--only", metavar="ID", help="Run a single experiment by id (e.g. 03).")
    sel.add_argument("--from", dest="start_from", metavar="ID",
                     help="Run experiments from the given id to the end.")
    p.add_argument("--list", action="store_true", help="Print the plan and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Pass --dry-run through to train.py (no GPU training).")
    p.add_argument("--continue-on-failure", action="store_true",
                   help="Keep running remaining experiments after a failure.")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the ClearML connectivity check.")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra args forwarded to train.py (put after '--').")
    return p.parse_args()


def select(args: argparse.Namespace) -> list[Experiment]:
    if args.only:
        picked = [e for e in EXPERIMENTS if e.id == args.only]
        if not picked:
            raise SystemExit(f"unknown experiment id: {args.only}")
        return picked
    if args.start_from:
        idx = next((i for i, e in enumerate(EXPERIMENTS) if e.id == args.start_from), None)
        if idx is None:
            raise SystemExit(f"unknown experiment id: {args.start_from}")
        return list(EXPERIMENTS[idx:])
    return list(EXPERIMENTS)


def run_one(exp: Experiment, *, dry_run: bool, extra: list[str]) -> int:
    cmd: list[str] = [
        sys.executable,
        str(REPO_ROOT / "Training" / "train.py"),
        "--exp-name", exp.exp_name,
        "--overrides", str(REPO_ROOT / exp.overrides),
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(extra)
    logger.info("Command: %s", " ".join(cmd))
    # Strip ClearML master-process markers inherited from the preflight
    # Task.init — otherwise the subprocess's Task.init returns a StubObject
    # thinking it's a forked child of the preflight task.
    child_env = {k: v for k, v in os.environ.items()
                 if k not in ("CLEARML_PROC_MASTER_ID", "TRAINS_PROC_MASTER_ID")}
    return subprocess.call(cmd, env=child_env)


def main() -> int:
    configure_logging()
    print_banner("NeoSmart — CarDetector experiment suite")
    args = parse_args()

    if args.list or not (args.all or args.only or args.start_from):
        print_section("Plan")
        for e in EXPERIMENTS:
            logger.info("  exp/%s  %-40s  %s", e.id, e.exp_name, e.description)
        if args.list:
            return 0
        logger.info("")
        logger.info("Nothing selected. Pass --all, --only ID, or --from ID.")
        return 0

    plan = select(args)
    print_section(f"Running {len(plan)} experiment(s)")
    for e in plan:
        logger.info("  exp/%s  %s", e.id, e.exp_name)

    if not args.skip_preflight and not args.dry_run:
        print_section("ClearML preflight")
        if not preflight_clearml():
            return 3

    failures: list[tuple[Experiment, int]] = []
    for e in plan:
        print_section(f"exp/{e.id} — {e.exp_name}")
        rc = run_one(e, dry_run=args.dry_run, extra=args.extra)
        if rc != 0:
            logger.error("exp/%s failed with exit code %d.", e.id, rc)
            failures.append((e, rc))
            if not args.continue_on_failure:
                return rc

    if failures:
        print_section("Finished with failures")
        for e, rc in failures:
            logger.error("  exp/%s (%s) → exit %d", e.id, e.exp_name, rc)
        return 1

    print_section("All experiments finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
