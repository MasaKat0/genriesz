"""Thin wrappers over the git plumbing the repo-hygiene tools need.

A pre-commit hook must judge the *staged* content, not the working tree: a file
can be staged and then changed on disk, and it is the staged blob that the
commit -- and therefore the push -- will carry.
"""

from __future__ import annotations

import subprocess

EXPERIMENT_NOTEBOOK_PREFIX = "notebooks/experiments/"


def git(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, input=stdin
    ).stdout


def staged_paths() -> list[str]:
    """Paths added, copied, modified, or renamed in the index."""
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [p for p in out.split("\0") if p]


def staged_blob_shas(paths: list[str]) -> dict[str, str]:
    """Map each path to the blob sha recorded in the index."""
    if not paths:
        return {}
    shas = {}
    for entry in git("ls-files", "-s", "-z", "--", *paths).split("\0"):
        if not entry:
            continue
        # "<mode> <sha> <stage>\t<path>"; stage != 0 only during a merge conflict
        meta, path = entry.split("\t", 1)
        _, sha, stage = meta.split()
        if stage == "0":
            shas[path] = sha
    return shas


def blob_sizes(shas: list[str]) -> dict[str, int]:
    """Uncompressed size of each blob, in one `git cat-file` process."""
    if not shas:
        return {}
    out = git("cat-file", "--batch-check", stdin="\n".join(shas) + "\n")
    sizes = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "blob":
            sizes[parts[0]] = int(parts[2])
    return sizes


def blob_text(sha: str) -> str:
    return git("cat-file", "-p", sha)


def tracked_paths() -> list[str]:
    return [p for p in git("ls-files", "-z").split("\0") if p]


def is_experiment_notebook(path: str) -> bool:
    return path.startswith(EXPERIMENT_NOTEBOOK_PREFIX) and path.endswith(".ipynb")
