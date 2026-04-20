"""Convert a percent-format .py into a Jupyter .ipynb.

Minimal jupytext replacement tailored to this repo: splits the source
on ``# %%`` headers and emits code / markdown cells. No outputs are
stored — the notebook is meant to be re-executed on demand.

Usage:
    python notebooks/build_ipynb.py notebooks/01_eda.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat


def split_cells(source: str) -> list[tuple[str, str]]:
    """Return list of (kind, body) where kind is 'code' or 'markdown'."""
    pattern = re.compile(r"^# %%(?: \[(\w+)\])?\s*$", re.MULTILINE)
    matches = list(pattern.finditer(source))
    if not matches:
        return [("code", source)]
    cells: list[tuple[str, str]] = []
    preamble = source[: matches[0].start()]
    if preamble.strip():
        cells.append(("code", preamble))
    for i, m in enumerate(matches):
        kind = "markdown" if (m.group(1) == "markdown") else "code"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        body = source[start:end]
        if kind == "markdown":
            body = "\n".join(
                line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
                for line in body.splitlines()
            )
        cells.append((kind, body.strip("\n")))
    return cells


def build(py_path: Path) -> Path:
    ipynb_path = py_path.with_suffix(".ipynb")
    src = py_path.read_text(encoding="utf-8")
    nb = nbformat.v4.new_notebook()
    nb.cells = []
    for kind, body in split_cells(src):
        if not body.strip():
            continue
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(body))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbformat.write(nb, ipynb_path)
    return ipynb_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("py_file", type=Path)
    args = ap.parse_args()
    out = build(args.py_file)
    print(f"wrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
