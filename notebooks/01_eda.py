"""NeoSmart — CarDetector EDA.

Percent-format script: every ``# %%`` starts a new cell. Safe to run
as a plain script (``python notebooks/01_eda.py``) — side-effects are
idempotent and gated by ``if __name__ == '__main__'`` at the bottom.

The companion ``notebooks/build_ipynb.py`` converts this file into
``notebooks/01_eda.ipynb`` using nbformat, preserving the cell
boundaries and markdown blocks. Running the script also refreshes
``Documentation/eda/figures/`` and ``Documentation/eda/eda_report.md``.
"""

# %% [markdown]
# # NeoSmart — CarDetector EDA
#
# This notebook inspects the synthetic training dataset used for the
# NeoSmart car detector. **All images are UE5 renders of a virtual
# parking lot** (filenames start with `unreal_`); there is no real
# footage in the set. Synthetic data is a deliberate choice for the
# thesis — see `Documentation/thesis/05_limitations_and_future_work.md`
# for the defensive rationale. The pipeline is wired to accept real
# imagery at the same `Training/data/` location without code changes.
#
# Goals:
# 1. Quantify dataset sizes and class balance.
# 2. Describe bounding-box geometry (size, aspect ratio).
# 3. Spot train/val distribution drift before training.
# 4. Locate objects spatially on the frame (heatmap).
# 5. Sanity-check labels via a sample grid with GT overlays.

# %%
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw

def _find_repo_root() -> Path:
    """Locate the repo root from either a .py script or a Jupyter kernel.

    Under ``python notebooks/01_eda.py`` we have ``__file__``; under
    Jupyter we don't, and the kernel's cwd is usually the notebook
    folder — so walk upwards looking for ``Training/data/data.yaml``.
    """
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    cur = Path.cwd().resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "Training" / "data" / "data.yaml").exists():
            return candidate
    raise RuntimeError(
        "Cannot locate repo root — run this notebook from within the NeoSmart checkout.",
    )


REPO_ROOT = _find_repo_root()
DATA_YAML = REPO_ROOT / "Training" / "data" / "data.yaml"
FIGURES_DIR = REPO_ROOT / "Documentation" / "eda" / "figures"
REPORT_PATH = REPO_ROOT / "Documentation" / "eda" / "eda_report.md"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Dataset layout
#
# The split directory tree under `Training/data/` follows the
# Ultralytics convention: `train/`, `val/`, `test/`, each containing
# `images/` and `labels/` sub-folders. Labels are YOLO format
# (`<class> <cx> <cy> <w> <h>` normalised to [0, 1]). The root
# `data.yaml` declares a single class, `car`.

# %%
DATA_ROOT = REPO_ROOT / "Training" / "data"
SPLITS = ("train", "val", "test")


def split_dir(split: str) -> Path:
    return DATA_ROOT / split


def image_paths(split: str) -> list[Path]:
    d = split_dir(split) / "images"
    if not d.exists():
        return []
    return sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))


def label_file(split: str, image_path: Path) -> Path:
    return split_dir(split) / "labels" / f"{image_path.stem}.txt"


# %% [markdown]
# ## 2. Split sizes
#
# How many images and how many object annotations live in each split?

# %%
def collect_split_stats(split: str) -> dict:
    imgs = image_paths(split)
    n_images = len(imgs)
    n_objects = 0
    class_counts: Counter[int] = Counter()
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    aspect_ratios: list[float] = []
    centers: list[tuple[float, float]] = []
    per_image: list[int] = []
    for p in imgs:
        lf = label_file(split, p)
        if not lf.exists():
            per_image.append(0)
            continue
        count_here = 0
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = (
                int(parts[0]), float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]),
            )
            class_counts[cls] += 1
            bbox_widths.append(w)
            bbox_heights.append(h)
            aspect_ratios.append(max(w, h) / max(min(w, h), 1e-6))
            centers.append((cx, cy))
            count_here += 1
            n_objects += 1
        per_image.append(count_here)
    return {
        "n_images": n_images,
        "n_objects": n_objects,
        "class_counts": dict(class_counts),
        "bbox_widths": np.array(bbox_widths),
        "bbox_heights": np.array(bbox_heights),
        "aspect_ratios": np.array(aspect_ratios),
        "centers": np.array(centers) if centers else np.zeros((0, 2)),
        "objects_per_image": np.array(per_image),
    }


stats = {s: collect_split_stats(s) for s in SPLITS}
for s in SPLITS:
    st = stats[s]
    n_empty = int(np.sum(st["objects_per_image"] == 0))
    print(f"{s:>5}: {st['n_images']:>4} images, "
          f"{st['n_objects']:>5} objects, {n_empty} empty frames")

# %% [markdown]
# ## 3. Objects per image
#
# A wide spread here means the detector must handle very different
# scene densities; a tight distribution with many empty frames is a
# warning flag (model may learn "predict nothing and be right most of
# the time").

# %%
fig, axes = plt.subplots(1, len(SPLITS), figsize=(14, 4), sharey=True)
for ax, s in zip(axes, SPLITS, strict=True):
    data = stats[s]["objects_per_image"]
    if len(data) == 0:
        ax.set_title(f"{s} (empty)")
        continue
    bins = np.arange(0, int(data.max()) + 2) - 0.5
    ax.hist(data, bins=bins, color="steelblue", edgecolor="black")
    ax.set_title(f"{s} (n={len(data)}, µ={data.mean():.1f})")
    ax.set_xlabel("objects per image")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("frame count")
fig.suptitle("Objects per image by split")
fig.tight_layout()
_p = FIGURES_DIR / "objects_per_image.png"
fig.savefig(_p, dpi=120)
print(f"saved -> {_p.relative_to(REPO_ROOT)}")

# %% [markdown]
# ## 4. Bounding-box geometry
#
# Three views: normalised width, normalised height, and aspect ratio
# (≥ 1, computed as max(w,h)/min(w,h)). A distribution concentrated at
# small normalised widths/heights means many small targets per frame —
# relevant when choosing `imgsz`. A right-skewed aspect ratio means
# more elongated cars (side view), while near-1 values mean cars seen
# from a top-down angle.

# %%
def plot_bbox_geometry(stats: dict) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for ax, key, title, xlab in (
        (axes[0], "bbox_widths", "Normalised bbox width", "w / image_w"),
        (axes[1], "bbox_heights", "Normalised bbox height", "h / image_h"),
        (axes[2], "aspect_ratios", "Aspect ratio (max/min)", "max(w,h) / min(w,h)"),
    ):
        for s, color in zip(SPLITS, ("steelblue", "orange", "green"), strict=True):
            arr = stats[s][key]
            if len(arr) == 0:
                continue
            ax.hist(arr, bins=40, alpha=0.55, color=color, label=s, density=True)
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel("density")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "bbox_geometry.png"
    fig.savefig(out, dpi=120)
    return out


_p = plot_bbox_geometry(stats)
print(f"saved -> {_p.relative_to(REPO_ROOT)}")

# %% [markdown]
# ## 5. Spatial heatmap of object centres
#
# Where on the frame do cars usually appear? For UE5 renders with
# fixed camera placements we expect clusters along parking bays. If
# train and val heatmaps diverge dramatically, the validation split
# is measuring a different scenario than training — which usually
# means the random split captured a camera-specific bias.

# %%
def plot_spatial_heatmap(stats: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, s in zip(axes, ("train", "val"), strict=True):
        c = stats[s]["centers"]
        if len(c) == 0:
            ax.set_title(f"{s} (empty)")
            continue
        H, xedges, yedges = np.histogram2d(
            c[:, 0], c[:, 1], bins=40, range=[[0, 1], [0, 1]],
        )
        # Flip y so the plot matches image coordinates (origin top-left).
        ax.imshow(
            H.T[::-1], extent=(0, 1, 0, 1), aspect="equal", cmap="magma",
        )
        ax.set_title(f"{s} — object centre density")
        ax.set_xlabel("x (normalised)")
        ax.set_ylabel("y (normalised, image origin top-left)")
    fig.tight_layout()
    out = FIGURES_DIR / "spatial_heatmap.png"
    fig.savefig(out, dpi=120)
    return out


_p = plot_spatial_heatmap(stats)
print(f"saved -> {_p.relative_to(REPO_ROOT)}")

# %% [markdown]
# ## 6. Train/val drift: Kolmogorov–Smirnov tests
#
# A cheap check: are the bbox-size and aspect-ratio distributions in
# `val` drawn from the same law as in `train`? KS on each dimension.
# Large D (and tiny p) means the split doesn't mirror training — the
# model's mAP on `val` will be partially measuring distribution shift,
# not generalisation.

# %%
try:
    from scipy.stats import ks_2samp  # type: ignore[import-untyped]

    for key, label in (
        ("bbox_widths", "bbox width"),
        ("bbox_heights", "bbox height"),
        ("aspect_ratios", "aspect ratio"),
        ("objects_per_image", "objects per image"),
    ):
        a, b = stats["train"][key], stats["val"][key]
        if len(a) == 0 or len(b) == 0:
            print(f"{label:>22}: skipped (empty split)")
            continue
        res = ks_2samp(a, b)
        print(f"{label:>22}: D={res.statistic:.3f}  p={res.pvalue:.3g}")
except ImportError:
    print("scipy not installed — skipping KS drift test")

# %% [markdown]
# ## 7. Sample grid with GT overlays
#
# Eight random `val` images with ground-truth boxes drawn in yellow.
# This is the label-sanity check: if boxes are visibly misaligned or
# span whole cars + empty asphalt, the downstream training is fighting
# a labelling problem, not a modelling one.

# %%
def plot_sample_grid(split: str = "val", n: int = 8, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    imgs = image_paths(split)
    if not imgs:
        raise RuntimeError(f"no images under {split_dir(split)}")
    picks = list(rng.choice(imgs, size=min(n, len(imgs)), replace=False))
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(rows, cols)
    for idx, img_path in enumerate(picks):
        r, c = divmod(idx, cols)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        draw = ImageDraw.Draw(img)
        lf = label_file(split, img_path)
        if lf.exists():
            for line in lf.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = (int(parts[0]), *map(float, parts[1:5]))
                x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
                draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
        axes[r, c].imshow(np.asarray(img))
        axes[r, c].set_title(img_path.name[:22], fontsize=8)
        axes[r, c].axis("off")
    for idx in range(len(picks), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
    fig.suptitle(f"{split} — GT sample grid")
    fig.tight_layout()
    out = FIGURES_DIR / "val_sample_grid.png"
    fig.savefig(out, dpi=120)
    return out


_p = plot_sample_grid("val", n=8, seed=42)
print(f"saved -> {_p.relative_to(REPO_ROOT)}")

# %% [markdown]
# ## 8. Summary
#
# Numbers from the cells above should feed the data card and model
# card. Anything surprising in the drift test or sample grid is worth
# a line in the limitations chapter of the thesis.


# %%
def build_report() -> None:
    """Write Documentation/eda/eda_report.md with embedded figures.

    Uses the already-computed ``stats`` dict and the figure files
    produced by the cells above. Idempotent: overwrites the report.
    """
    lines: list[str] = []
    lines.append("# NeoSmart — CarDetector EDA report\n")
    lines.append(
        "_Auto-generated by [notebooks/01_eda.py](../../notebooks/01_eda.py). "
        "Do not edit by hand — rerun the script to refresh._\n",
    )
    lines.append("\n## Dataset\n")
    with DATA_YAML.open() as f:
        data_yaml = yaml.safe_load(f)
    lines.append(f"- Source: `{DATA_YAML.relative_to(REPO_ROOT).as_posix()}`")
    lines.append(
        "- Origin: UE5-rendered synthetic frames (`unreal_*.jpg`), no real footage.",
    )
    lines.append(f"- Classes: `{data_yaml.get('names')}` (nc={data_yaml.get('nc')})")
    totals_img = sum(stats[s]["n_images"] for s in SPLITS)
    totals_obj = sum(stats[s]["n_objects"] for s in SPLITS)
    lines.append(
        f"- Total: {totals_img} images, {totals_obj} annotated objects.\n",
    )

    lines.append("## Split sizes\n")
    lines.append("| split | images | objects | empty frames | µ obj/frame |")
    lines.append("|-------|--------|---------|--------------|-------------|")
    for s in SPLITS:
        st = stats[s]
        n_empty = int(np.sum(st["objects_per_image"] == 0))
        mean_obj = float(st["objects_per_image"].mean()) if st["n_images"] else 0.0
        lines.append(
            f"| {s} | {st['n_images']} | {st['n_objects']} | {n_empty} | "
            f"{mean_obj:.1f} |",
        )
    lines.append("")

    lines.append("## Objects per image")
    lines.append("![objects_per_image](figures/objects_per_image.png)\n")

    lines.append("## Bounding-box geometry")
    lines.append("![bbox_geometry](figures/bbox_geometry.png)\n")
    for s in SPLITS:
        st = stats[s]
        if st["n_objects"] == 0:
            continue
        w, h, a = st["bbox_widths"], st["bbox_heights"], st["aspect_ratios"]
        lines.append(
            f"- **{s}** — w: µ={w.mean():.3f} σ={w.std():.3f}, "
            f"h: µ={h.mean():.3f} σ={h.std():.3f}, "
            f"aspect µ={a.mean():.2f} (p95={np.percentile(a, 95):.2f})",
        )
    lines.append("")

    lines.append("## Spatial heatmap (object centres)")
    lines.append("![spatial_heatmap](figures/spatial_heatmap.png)\n")

    lines.append("## Train/val drift — KS test")
    try:
        from scipy.stats import ks_2samp  # type: ignore[import-untyped]

        lines.append("| dimension | D | p-value |")
        lines.append("|-----------|---|---------|")
        for key, label in (
            ("bbox_widths", "bbox width"),
            ("bbox_heights", "bbox height"),
            ("aspect_ratios", "aspect ratio"),
            ("objects_per_image", "objects per image"),
        ):
            a, b = stats["train"][key], stats["val"][key]
            if len(a) == 0 or len(b) == 0:
                continue
            r = ks_2samp(a, b)
            lines.append(f"| {label} | {r.statistic:.3f} | {r.pvalue:.3g} |")
    except ImportError:
        lines.append("_scipy not installed, KS test skipped._")
    lines.append("")

    lines.append("## Validation sample grid (ground truth)")
    lines.append("![val_sample_grid](figures/val_sample_grid.png)\n")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote -> {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_report()
