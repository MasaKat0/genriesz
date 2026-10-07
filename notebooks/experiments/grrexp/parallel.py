"""Process-level parallelism of registration §1.2.

Workers are spawned (not forked) with ``OMP_NUM_THREADS``,
``OPENBLAS_NUM_THREADS`` and ``VECLIB_MAXIMUM_THREADS`` set to 1 in their
environment before Python starts, so the BLAS of every worker is single-threaded
from its first import. If torch is importable, each worker sets
``torch.set_num_threads(1)`` and ``torch.use_deterministic_algorithms(True)``.

``run_tasks`` returns results in the order of ``tasks`` whatever the completion
order, so a run with 1 worker and a run with 12 workers can be compared bit for
bit with :func:`assert_bit_identical` (Stage 0.5).
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
    """Pin the thread counts inside a worker (also call it in a serial run)."""
    os.environ.update({k: "1" for k in THREAD_VARIABLES})
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def thread_environment_ok() -> bool:
    return all(os.environ.get(k) == "1" for k in THREAD_VARIABLES)


def _call(fn: Callable[[Any], Any], task: Any) -> Any:
    if not thread_environment_ok():
        raise RuntimeError("worker started without single-thread BLAS settings")
    return fn(task)


def run_tasks(fn: Callable[[Any], Any], tasks: Sequence[Any], n_jobs: int) -> list[Any]:
    """Apply ``fn`` to each task; ``fn`` must be importable (defined in a module).

    ``n_jobs=1`` runs in this process, which must already have the single-thread
    environment (start the kernel with it, or call :func:`configure_worker`
    before importing numpy). Exceptions propagate and stop the run.
    """
    tasks = list(tasks)
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    if n_jobs == 1:
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


def assert_bit_identical(serial: Sequence[Any], parallel: Sequence[Any]) -> str:
    a, b = result_digest(serial), result_digest(parallel)
    if a != b:
        raise AssertionError(f"serial and parallel runs differ: {a} != {b}")
    return a
