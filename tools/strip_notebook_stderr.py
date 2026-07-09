"""Remove stderr stream outputs from executed experiment notebooks.

Re-running ``notebooks/experiments/`` writes every NumPy RuntimeWarning raised
inside the Monte Carlo loops into the notebook as an ``stderr`` output. One
execution of ``01_main_simulation_study.ipynb`` produced 6508 of them and grew
the file to 115 MB, past GitHub's 100 MB blob limit.

Everything else a cell emits is kept: stdout streams, ``display_data`` (the
figures), ``execute_result`` (the HTML tables), and error tracebacks.

    python tools/strip_notebook_stderr.py            # rewrite in place
    python tools/strip_notebook_stderr.py --check     # report only, exit 1 if dirty

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

try:
    import nbformat
except ImportError:  # pragma: no cover - depends on the environment
    sys.exit(
        "nbformat is required: pip install nbformat\n"
        "(it ships with jupyter, which you need to run these notebooks anyway)"
    )

TARGET_GLOB = "notebooks/experiments/*.ipynb"


def _strip(nb) -> int:
    """Drop stderr stream outputs from every cell. Returns the number removed."""
    removed = 0
    for cell in nb.cells:
        outputs = cell.get("outputs")
        if not outputs:
            continue
        kept = []
        for out in outputs:
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                removed += 1
            else:
                kept.append(out)
        cell["outputs"] = kept
    return removed


def _human(size: int) -> str:
    return f"{size / 1048576:.2f} MB" if size >= 1048576 else f"{size / 1024:.0f} KB"


def process(path: str, check_only: bool) -> tuple[int, int, int]:
    """Return (stderr outputs removed, size before, size after)."""
    before = os.path.getsize(path)
    with open(path, encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    removed = _strip(nb)
    if removed == 0:
        return 0, before, before

    buf = io.StringIO()
    nbformat.write(nb, buf)
    data = buf.getvalue()

    if not check_only:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(data)

    return removed, before, len(data.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if any notebook still carries stderr output",
    )
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(TARGET_GLOB))
    if not paths:
        print(f"no notebooks matched {TARGET_GLOB}", file=sys.stderr)
        return 1

    dirty = []
    for path in paths:
        removed, before, after = process(path, args.check)
        if removed:
            dirty.append(path)
            verb = "would remove" if args.check else "removed"
            print(f"{path}: {verb} {removed} stderr output(s), {_human(before)} -> {_human(after)}")

    if not dirty:
        print(f"{len(paths)} notebooks clean: no stderr output")
        return 0

    if args.check:
        print(
            f"\n{len(dirty)} notebook(s) carry stderr output. "
            "Run `make notebooks-strip` and commit the result.",
            file=sys.stderr,
        )
        return 1

    print(f"\nstripped {len(dirty)} notebook(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
