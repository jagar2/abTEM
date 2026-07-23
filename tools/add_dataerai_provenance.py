"""Add (or check) Dataerai provenance tracing cells in abTEM notebooks.

Every notebook gets two tagged cells:

- a header that loads the ``%dataerai`` magic and starts a traced session
  filing into ``abTEM / notebook runs / <category>``, guarded so notebooks
  run unchanged when the Dataerai SDK or daemon is unavailable;
- a footer that publishes the execution trace, likewise guarded.

Cells carry the ``dataerai-provenance`` metadata tag, so the script is
idempotent: rerunning refreshes existing cells in place instead of
duplicating them. Categories derive from the notebook's directory
(walkthrough, tutorials, appendix, examples, articles, demos, misc).

Usage::

    python tools/add_dataerai_provenance.py <path> [<path> ...]
    python tools/add_dataerai_provenance.py --check <path> [...]

Paths may be notebooks or directories (searched recursively, skipping
checkpoints and ``_build``). ``--check`` exits non-zero if any notebook
lacks the cells, so it can serve as a CI or pre-commit gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell

TAG = "dataerai-provenance"
COLLECTION_ROOT = "abTEM / notebook runs"
CATEGORIES = ("walkthrough", "tutorials", "appendix", "examples", "articles", "demos")

HEADER_TEMPLATE = """\
# --- Dataerai provenance (optional) -----------------------------------------
# Traces this notebook run - cell source, outputs, logs, transfers, and
# environment - to the Dataerai platform when the SDK and daemon are
# available. Without them the notebook runs exactly as before.
try:
    %load_ext dataerai.magics
    %dataerai --trace --notebook {name} {collection}
except Exception as _dataerai_error:
    print(f"Dataerai tracing not active: {{_dataerai_error}}")
"""

FOOTER = """\
# --- Dataerai provenance: publish the execution trace, if one is active -----
try:
    %dataerai --finish
except Exception as _dataerai_error:
    print(f"Dataerai trace not published: {{_dataerai_error}}")
""".replace("{{", "{").replace("}}", "}")


def category_for(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    for category in CATEGORIES:
        if category in parts:
            return category
    return "misc"


def collection_for(path: Path) -> str:
    return f"{COLLECTION_ROOT} / {category_for(path)}"


def _tagged(cell) -> bool:
    return TAG in cell.get("metadata", {}).get("tags", [])


def _make_cell(source: str) -> nbformat.NotebookNode:
    cell = new_code_cell(source)
    cell.metadata["tags"] = [TAG]
    return cell


def integrate(path: Path) -> bool:
    """Ensure provenance cells in one notebook; return True if modified."""
    nb = nbformat.read(path, as_version=4)
    header = _make_cell(
        HEADER_TEMPLATE.format(name=path.name, collection=collection_for(path))
    )
    footer = _make_cell(FOOTER)

    original = [
        (cell.get("cell_type"), cell.get("source")) for cell in nb.cells
    ]
    kept = [cell for cell in nb.cells if not _tagged(cell)]
    nb.cells = [header, *kept, footer]

    changed = original != [
        (cell.get("cell_type"), cell.get("source")) for cell in nb.cells
    ]
    if changed:
        nbformat.write(nb, path)
    return changed


def has_integration(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    tagged = [cell for cell in nb.cells if _tagged(cell)]
    return len(tagged) >= 2


def find_notebooks(paths: list[Path]) -> list[Path]:
    notebooks: list[Path] = []
    for path in paths:
        if path.is_dir():
            notebooks.extend(
                candidate
                for candidate in sorted(path.rglob("*.ipynb"))
                if ".ipynb_checkpoints" not in candidate.parts
                and "_build" not in candidate.parts
                and ".venv" not in candidate.parts
            )
        else:
            notebooks.append(path)
    return notebooks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report notebooks missing integration; do not modify",
    )
    args = parser.parse_args(argv)

    notebooks = find_notebooks(args.paths)
    if not notebooks:
        print("no notebooks found", file=sys.stderr)
        return 1

    missing = 0
    changed = 0
    for notebook in notebooks:
        if args.check:
            if not has_integration(notebook):
                print(f"MISSING {notebook}")
                missing += 1
        else:
            if integrate(notebook):
                changed += 1
                print(f"updated {notebook} -> {collection_for(notebook)}")

    if args.check:
        print(f"checked {len(notebooks)} notebooks, {missing} missing")
        return 1 if missing else 0
    print(f"processed {len(notebooks)} notebooks, {changed} updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
