"""The worker count of grrexp.parallel.run_tasks can change during a run."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import parallel  # noqa: E402


def _square(x):
    return x * x


def _fail_on_three(x):
    if x == 3:
        raise ValueError("task three failed")
    return x


def test_results_keep_task_order_and_match_serial(monkeypatch):
    monkeypatch.delenv(parallel.MAX_WORKERS_FILE_ENV, raising=False)
    tasks = list(range(20))
    assert parallel.run_tasks(_square, tasks, 2) == [t * t for t in tasks]


def test_control_file_lowers_and_clamps(tmp_path, monkeypatch):
    f = tmp_path / "max_workers"
    monkeypatch.setenv(parallel.MAX_WORKERS_FILE_ENV, str(f))
    assert parallel.desired_workers(4) == 4  # no file yet
    f.write_text("2")
    assert parallel.desired_workers(4) == 2
    f.write_text("9")
    assert parallel.desired_workers(4) == 4
    f.write_text("0")
    assert parallel.desired_workers(4) == 1
    f.write_text("x")
    assert parallel.desired_workers(4) == 4
    f.write_text("1")
    tasks = list(range(12))
    assert parallel.run_tasks(_square, tasks, 4) == [t * t for t in tasks]


def test_worker_exception_propagates(monkeypatch):
    monkeypatch.delenv(parallel.MAX_WORKERS_FILE_ENV, raising=False)
    with pytest.raises(ValueError, match="task three failed"):
        parallel.run_tasks(_fail_on_three, list(range(8)), 2)


@pytest.mark.parametrize("jobs", [0, 5, 1.5, True])
def test_worker_limit(jobs):
    with pytest.raises(ValueError, match="1..4"):
        parallel.run_tasks(_square, [1], jobs)
