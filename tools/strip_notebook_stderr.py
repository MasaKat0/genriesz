"""Remove stderr stream outputs from executed experiment notebooks.

Re-running ``notebooks/experiments/`` writes every NumPy RuntimeWarning raised
inside the Monte Carlo loops into the notebook as an ``stderr`` output. One
execution of ``01_main_simulation_study.ipynb`` produced 6508 of them and grew
the file to 115 MB, past GitHub's 100 MB blob limit.

Everything else a cell emits is kept: stdout streams, ``display_data`` (the
figures), ``execute_result`` (the HTML tables), and error tracebacks.

    python tools/strip_notebook_stderr.py            # rewrite the working tree
    python tools/strip_notebook_stderr.py --check    # report; exit 1 if dirty
    python tools/strip_notebook_stderr.py --staged   # report on staged blobs only

``--staged`` is what the pre-commit hook uses. It judges the blobs recorded in
the index rather than the files on disk, because those are what the commit will
carry, and it ignores notebooks that are not part of this commit.

Scope is limited to ``notebooks/experiments/``. Notebooks elsewhere in the repo
were written by tools whose JSON layout nbformat does not reproduce byte for
byte, so rewriting them would produce unrelated reformatting noise.
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import sys

from _gitutil import blob_text, is_experiment_notebook, staged_blob_shas, staged_paths

try:
    import nbformat
except ImportError:  # pragma: no cover - depends on the environment
    sys.exit(
        "nbformat is required: pip install nbformat\n"
        "(it ships with jupyter, which you need to run these notebooks anyway)"
    )

TARGET_GLOB = "notebooks/experiments/*.ipynb"


def _is_stderr(output) -> bool:
    return output.get("output_type") == "stream" and output.get("name") == "stderr"


def count_stderr(nb) -> int:
    return sum(1 for cell in nb.cells for out in cell.get("outputs", []) if _is_stderr(out))


def strip_stderr(nb) -> int:
    """Drop stderr outputs in place. Returns the number removed."""
    removed = 0
    for cell in nb.cells:
        outputs = cell.get("outputs")
        if not outputs:
            continue
        kept = [out for out in outputs if not _is_stderr(out)]
        removed += len(outputs) - len(kept)
        cell["outputs"] = kept
    return removed


def _human(size: int) -> str:
    return f"{size / 1048576:.2f} MB" if size >= 1048576 else f"{size / 1024:.0f} KB"


def _rewrite(path: str) -> tuple[int, int, int]:
    """Strip `path` in place. Returns (removed, size before, size after)."""
    before = os.path.getsize(path)
    with open(path, encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    removed = strip_stderr(nb)
    if removed == 0:
        return 0, before, before

    buf = io.StringIO()
    nbformat.write(nb, buf)
    data = buf.getvalue()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(data)
    return removed, before, len(data.encode("utf-8"))


def _check_worktree() -> tuple[list[str], int]:
    paths = sorted(glob.glob(TARGET_GLOB))
    if not paths:
        print(f"no notebooks matched {TARGET_GLOB}", file=sys.stderr)
        sys.exit(1)

    dirty = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            nb = nbformat.read(fh, as_version=4)
        removed = count_stderr(nb)
        if removed:
            dirty.append(path)
            print(f"{path}: {removed} stderr output(s), file is {_human(os.path.getsize(path))}")
    return dirty, len(paths)


def _check_staged() -> tuple[list[str], int]:
    paths = [p for p in staged_paths() if is_experiment_notebook(p)]
    shas = staged_blob_shas(paths)

    dirty = []
    for path, sha in sorted(shas.items()):
        nb = nbformat.reads(blob_text(sha), as_version=4)
        removed = count_stderr(nb)
        if removed:
            dirty.append(path)
            print(f"{path}: {removed} stderr output(s) in the staged blob")
    return dirty, len(shas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if any notebook carries stderr output",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="inspect the staged blobs instead of the working tree (implies --check)",
    )
    args = parser.parse_args(argv)

    if args.staged:
        dirty, scanned = _check_staged()
        if not dirty:
            print(f"{scanned} staged notebook(s) clean: no stderr output")
            return 0
        print(
            f"\n{len(dirty)} staged notebook(s) carry stderr output. "
            "Run `make notebooks-strip`, re-stage them, and commit.",
            file=sys.stderr,
        )
        return 1

    if args.check:
        dirty, scanned = _check_worktree()
        if not dirty:
            print(f"{scanned} notebooks clean: no stderr output")
            return 0
        print(
            f"\n{len(dirty)} notebook(s) carry stderr output. "
            "Run `make notebooks-strip` and commit the result.",
            file=sys.stderr,
        )
        return 1

    paths = sorted(glob.glob(TARGET_GLOB))
    if not paths:
        print(f"no notebooks matched {TARGET_GLOB}", file=sys.stderr)
        return 1

    dirty = []
    for path in paths:
        removed, before, after = _rewrite(path)
        if removed:
            dirty.append(path)
            sizes = f"{_human(before)} -> {_human(after)}"
            print(f"{path}: removed {removed} stderr output(s), {sizes}")

    if not dirty:
        print(f"{len(paths)} notebooks clean: no stderr output")
        return 0
    print(f"\nstripped {len(dirty)} notebook(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
