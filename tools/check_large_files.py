"""Fail when a file heading into git history is too large to push.

GitHub rejects any push containing a blob over 100 MB, and it rejects the whole
push even when a later commit deletes the blob -- the only remedy is rewriting
history. Catching an oversized file before it is committed is much cheaper.

    python tools/check_large_files.py             # tracked files in the working tree
    python tools/check_large_files.py --staged    # files staged for commit
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

DEFAULT_MAX_MB = 50.0


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def _staged_sizes() -> list[tuple[int, str]]:
    """Sizes of the blobs recorded in the index, not of the working-tree files.

    A file can be staged and then shrunk on disk; it is the staged blob that the
    commit -- and therefore the push -- will carry.
    """
    staged = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    paths = [p for p in staged.split("\0") if p]
    if not paths:
        return []

    sizes = []
    for entry in _git("ls-files", "-s", "-z", "--", *paths).split("\0"):
        if not entry:
            continue
        # "<mode> <sha> <stage>\t<path>"
        meta, path = entry.split("\t", 1)
        sha = meta.split()[1]
        sizes.append((int(_git("cat-file", "-s", sha).strip()), path))
    return sizes


def _tracked_sizes() -> list[tuple[int, str]]:
    """Sizes of the working-tree files git tracks -- what a commit would capture."""
    sizes = []
    for path in _git("ls-files", "-z").split("\0"):
        if path and os.path.isfile(path):  # skip deletions and submodules
            sizes.append((os.path.getsize(path), path))
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staged", action="store_true", help="check staged files only")
    parser.add_argument(
        "--max-mb",
        type=float,
        default=DEFAULT_MAX_MB,
        help=f"size limit in MB (default: {DEFAULT_MAX_MB:g}; GitHub hard-rejects at 100)",
    )
    args = parser.parse_args(argv)

    limit = int(args.max_mb * 1048576)
    entries = _staged_sizes() if args.staged else _tracked_sizes()

    oversized = [(size, path) for size, path in entries if size > limit]
    if not oversized:
        return 0

    scope = "staged" if args.staged else "tracked"
    print(f"{len(oversized)} {scope} file(s) exceed {args.max_mb:g} MB:", file=sys.stderr)
    for size, path in sorted(oversized, reverse=True):
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
