"""Tests for the repo-hygiene tools under tools/.

These guard `make verify` and the pre-commit hook. Both gates exist because an
executed notebook once reached 115 MB of stderr output and made `git push` fail
against GitHub's 100 MB blob limit, which could only be undone by rewriting
history.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import check_large_files  # noqa: E402
import strip_notebook_stderr as strip  # noqa: E402

NB_DIR = "notebooks/experiments"


def _notebook(*, stderr_outputs: int = 0, stdout: bool = False) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("print('hi')")
    cell.outputs = []
    for _ in range(stderr_outputs):
        cell.outputs.append(
            nbformat.v4.new_output("stream", name="stderr", text="RuntimeWarning\n")
        )
    if stdout:
        cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text="result\n"))
    cell.outputs.append(nbformat.v4.new_output("display_data", data={"text/plain": "figure"}))
    nb.cells = [cell]
    return nb


def _write(path: Path, nb: nbformat.NotebookNode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo, with cwd inside it."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _add(repo: Path, *paths: str) -> None:
    subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True)


# ---------------------------------------------------------------- pure logic


def test_strip_removes_only_stderr():
    nb = _notebook(stderr_outputs=3, stdout=True)
    assert strip.count_stderr(nb) == 3

    removed = strip.strip_stderr(nb)

    assert removed == 3
    assert strip.count_stderr(nb) == 0
    kinds = [o["output_type"] for o in nb.cells[0].outputs]
    assert kinds == ["stream", "display_data"], "stdout and figures must survive"
    assert nb.cells[0].outputs[0]["name"] == "stdout"


def test_strip_is_idempotent():
    nb = _notebook(stderr_outputs=2)
    strip.strip_stderr(nb)
    assert strip.strip_stderr(nb) == 0


def test_error_tracebacks_are_kept():
    nb = _notebook(stderr_outputs=1)
    nb.cells[0].outputs.append(
        nbformat.v4.new_output("error", ename="ValueError", evalue="x", traceback=["..."])
    )
    strip.strip_stderr(nb)
    assert [o["output_type"] for o in nb.cells[0].outputs] == ["display_data", "error"]


# ------------------------------------------------- staged vs working tree
# A file can be staged and then changed on disk. The commit carries the staged
# blob, so that is what both gates must judge.


def test_staged_check_catches_dirty_blob_when_worktree_is_clean(repo):
    path = repo / NB_DIR / "01_x.ipynb"
    _write(path, _notebook(stderr_outputs=5))
    _add(repo, f"{NB_DIR}/01_x.ipynb")
    _write(path, _notebook(stderr_outputs=0))  # clean the worktree, do not re-stage

    assert strip.main(["--staged"]) == 1, "the staged blob still carries stderr"


def test_staged_check_ignores_dirty_notebooks_outside_the_commit(repo):
    staged = repo / NB_DIR / "02_clean.ipynb"
    _write(staged, _notebook(stderr_outputs=0))
    _add(repo, f"{NB_DIR}/02_clean.ipynb")

    unrelated = repo / NB_DIR / "01_dirty.ipynb"
    _write(unrelated, _notebook(stderr_outputs=9))  # dirty on disk, never staged

    assert strip.main(["--staged"]) == 0, "an unstaged notebook must not block the commit"


def test_staged_check_passes_when_nothing_is_staged(repo):
    assert strip.main(["--staged"]) == 0


def test_large_file_check_reads_the_staged_blob_not_the_disk_file(repo):
    big = repo / "big.bin"
    big.write_bytes(b"x" * 200_000)
    _add(repo, "big.bin")
    big.write_bytes(b"x")  # shrink on disk only

    assert check_large_files.main(["--staged", "--max-mb", "0.1"]) == 1


def test_large_file_check_passes_under_the_limit(repo):
    small = repo / "small.bin"
    small.write_bytes(b"x" * 10)
    _add(repo, "small.bin")

    assert check_large_files.main(["--staged", "--max-mb", "0.1"]) == 0


def test_tracked_check_scans_the_working_tree(repo):
    big = repo / "big.bin"
    big.write_bytes(b"x" * 200_000)
    _add(repo, "big.bin")
    subprocess.run(["git", "commit", "-qm", "big"], cwd=repo, check=True)

    assert check_large_files.main(["--max-mb", "0.1"]) == 1
    assert check_large_files.main(["--max-mb", "10"]) == 0
