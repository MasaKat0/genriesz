"""Process-level parallelism of registration §1.2.

Workers are spawned (not forked) with ``OMP_NUM_THREADS``,
``OPENBLAS_NUM_THREADS`` and ``VECLIB_MAXIMUM_THREADS`` set to 1 in their
environment before Python starts, so the BLAS of every worker is single-threaded
from its first import. Every worker, and a serial run, then sets
``torch.set_num_threads(1)`` and ``torch.use_deterministic_algorithms(True)`` and
verifies with threadpoolctl that every loaded BLAS/OpenMP runtime uses one
thread before each task.

``run_tasks`` returns results in the order of ``tasks`` whatever the completion
order, so a run with 1 worker and a run with 12 workers can be compared bit for
bit (Stage 0.5, :func:`verify_parallel_identity`).
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import pickle
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


@contextmanager
def _single_thread_environment():
    saved = {k: os.environ.get(k) for k in THREAD_VARIABLES}
    os.environ.update({k: "1" for k in THREAD_VARIABLES})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def configure_worker() -> None:
    """Pin the thread counts inside a worker or a serial run, then verify them.

    torch is registered (§1.7), so an import failure stops the run.
    """
    os.environ.update({k: "1" for k in THREAD_VARIABLES})
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    verify_single_thread()


def verify_single_thread() -> None:
    """Raise unless the variables, every loaded BLAS/OpenMP runtime and torch use 1 thread."""
    if not all(os.environ.get(k) == "1" for k in THREAD_VARIABLES):
        raise RuntimeError("single-thread environment variables are not set")
    from threadpoolctl import threadpool_info

    busy = [
        (i["internal_api"], i["num_threads"]) for i in threadpool_info() if i["num_threads"] != 1
    ]
    if busy:
        raise RuntimeError(f"thread pools not pinned to 1: {busy}")
    import torch

    if torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("torch is not single-threaded and deterministic")


def _call(fn: Callable[[Any], Any], task: Any) -> Any:
    verify_single_thread()
    return fn(task)


def run_tasks(fn: Callable[[Any], Any], tasks: Sequence[Any], n_jobs: int) -> list[Any]:
    """Apply ``fn`` to each task; ``fn`` must be importable (defined in a module).

    ``n_jobs=1`` runs in this process after :func:`configure_worker`; the kernel
    must have been started with the single-thread variables, or the check of the
    loaded BLAS runtime fails. Exceptions propagate and stop the run.
    """
    tasks = list(tasks)
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    if not tasks:
        raise ValueError("no tasks")
    if n_jobs == 1:
        configure_worker()
        return [_call(fn, t) for t in tasks]
    with _single_thread_environment():
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_jobs, mp_context=ctx, initializer=configure_worker
        ) as pool:
            futures = [pool.submit(_call, fn, t) for t in tasks]
            return [f.result() for f in futures]


def result_digest(results: Iterable[Any]) -> str:
    """SHA-256 of the pickled results (protocol 5); equal digests mean equal bytes."""
    h = hashlib.sha256()
    for r in results:
        h.update(pickle.dumps(r, protocol=5))
    return h.hexdigest()


def assert_bit_identical(serial: Sequence[Any], parallel: Sequence[Any], *, expected: int) -> str:
    """Raise unless both runs returned ``expected`` (> 0) results with equal bytes."""
    if expected < 1:
        raise ValueError("expected must be positive")
    if len(serial) != expected or len(parallel) != expected:
        raise AssertionError(
            f"expected {expected} results, got {len(serial)} serial and {len(parallel)} parallel"
        )
    a, b = result_digest(serial), result_digest(parallel)
    if a != b:
        raise AssertionError(f"serial and parallel runs differ: {a} != {b}")
    return a


def verify_parallel_identity(
    fn: Callable[[Any], Any], tasks: Sequence[Any], n_jobs: int = 12
) -> str:
    """Stage 0.5: run every task with 1 and with ``n_jobs`` workers and compare bytes."""
    tasks = list(tasks)
    serial = run_tasks(fn, tasks, 1)
    parallel = run_tasks(fn, tasks, n_jobs)
    return assert_bit_identical(serial, parallel, expected=len(tasks))
