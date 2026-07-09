"""Fail when a file heading into git history is too large to push.

GitHub rejects any push containing a blob over 100 MB, and it rejects the whole
push even when a later commit deletes the blob -- the only remedy is rewriting
history. Catching an oversized file before it is committed is much cheaper.

    python tools/check_large_files.py             # tracked files in the working tree
    python tools/check_large_files.py --staged    # blobs staged for commit
"""

from __future__ import annotations

import argparse
import os
import sys

from _gitutil import blob_sizes, staged_blob_shas, staged_paths, tracked_paths

DEFAULT_MAX_MB = 50.0


def _staged_sizes() -> list[tuple[int, str]]:
    shas = staged_blob_shas(staged_paths())
    sizes = blob_sizes(list(shas.values()))
    return [(sizes[sha], path) for path, sha in shas.items() if sha in sizes]


def _tracked_sizes() -> list[tuple[int, str]]:
    """What a commit of the current working tree would capture."""
    return [
        (os.path.getsize(p), p)
        for p in tracked_paths()
        if os.path.isfile(p)  # skip deletions and submodules
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staged", action="store_true", help="check staged blobs only")
    parser.add_argument(
        "--max-mb",
        type=float,
        default=DEFAULT_MAX_MB,
        help=f"size limit in MB (default: {DEFAULT_MAX_MB:g}; GitHub hard-rejects at 100)",
    )
    args = parser.parse_args(argv)

    limit = int(args.max_mb * 1048576)
    entries = _staged_sizes() if args.staged else _tracked_sizes()

    oversized = sorted((e for e in entries if e[0] > limit), reverse=True)
    if not oversized:
        return 0

    scope = "staged" if args.staged else "tracked"
    print(f"{len(oversized)} {scope} file(s) exceed {args.max_mb:g} MB:", file=sys.stderr)
    for size, path in oversized:
        print(f"  {size / 1048576:8.2f} MB  {path}", file=sys.stderr)
    print(
        "\nGitHub rejects pushes containing blobs over 100 MB, and a later commit "
        "that deletes the blob does not help.\n"
        "For executed notebooks, `make notebooks-strip` usually removes the bulk.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
