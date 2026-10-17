"""Elastic process execution with at most four single-threaded workers."""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import queue
import traceback
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
MAX_WORKERS_FILE_ENV = "GRREXP_MAX_WORKERS_FILE"


def desired_workers(n_jobs: int) -> int:
    """The number of workers to keep running: ``n_jobs``, or the integer in the control file
    named by ``GRREXP_MAX_WORKERS_FILE`` (clamped to ``1..n_jobs``) when it can be read."""
    path = os.environ.get(MAX_WORKERS_FILE_ENV)
    if not path:
        return n_jobs
    try:
        with open(path, encoding="utf-8") as f:
            value = int(f.read().strip())
    except (OSError, ValueError):
        return n_jobs
    return max(1, min(n_jobs, value))


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


def run_tasks(
    fn: Callable[[Any], Any], tasks: Sequence[Any], n_jobs: int,
    *, worker_records: list[dict] | None = None,
) -> list[Any]:
    """Apply ``fn`` to each task; ``fn`` must be importable (defined in a module).

    ``n_jobs=1`` runs in this process after :func:`configure_worker`; the kernel
    must have been started with the single-thread variables, or the check of the
    loaded BLAS runtime fails. Exceptions propagate and stop the run.
    """
    tasks = list(tasks)
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or not 1 <= n_jobs <= 4:
        raise ValueError("n_jobs must be an integer in 1..4")
    if not tasks:
        raise ValueError("no tasks")
    if n_jobs == 1:
        variables = {name: os.environ.get(name) for name in THREAD_VARIABLES}
        configure_worker()
        if worker_records is not None:
            worker_records.append(_worker_record(0, variables))
        return [_call(fn, t) for t in tasks]
    with _single_thread_environment():
        return _run_elastic(fn, tasks, n_jobs, mp.get_context("spawn"), worker_records)


def _worker_loop(fn, task_q, result_q, wid: int) -> None:
    try:
        variables = {name: os.environ.get(name) for name in THREAD_VARIABLES}
        configure_worker()
        record = _worker_record(wid, variables)
    except BaseException:
        result_q.put((
            -1, False,
            RuntimeError(f"worker {wid} initialization failed:\n{traceback.format_exc()}"),
            wid,
        ))
        return
    result_q.put((-2, True, record, wid))
    while True:
        item = task_q.get()
        if item is None:
            return
        i, task = item
        try:
            result_q.put((i, True, _call(fn, task), wid))
        except BaseException as e:  # sent to the parent, which re-raises it
            try:
                pickle.dumps(e)
            except Exception:
                e = RuntimeError(f"{type(e).__name__}: {e}")
            result_q.put((i, False, e, wid))
            return


def _run_elastic(fn, tasks, n_jobs, ctx, worker_records=None):
    """Dispatch one task at a time to spawned workers; keep ``desired_workers`` running."""
    result_q = ctx.Queue()
    workers: dict[int, tuple[Any, Any]] = {}
    results: list[Any] = [None] * len(tasks)
    state = {"next": 0, "wid": 0}

    def start() -> int:
        wid = state["wid"]
        state["wid"] += 1
        q = ctx.Queue()
        proc = ctx.Process(target=_worker_loop, args=(fn, q, result_q, wid), daemon=True)
        proc.start()
        workers[wid] = (proc, q)
        return wid

    def assign(wid: int) -> None:
        i = state["next"]
        state["next"] += 1
        workers[wid][1].put((i, tasks[i]))

    def retire(wid: int) -> None:
        proc, q = workers.pop(wid)
        q.put(None)
        proc.join()

    busy: set[int] = set()
    done = 0
    try:
        for _ in range(min(desired_workers(n_jobs), len(tasks))):
            w = start()
            assign(w)
            busy.add(w)
        while done < len(tasks):
            try:
                i, ok, payload, w = result_q.get(timeout=5)
            except queue.Empty:
                dead = [k for k in busy if not workers[k][0].is_alive()]
                if dead:
                    exits = {k: workers[k][0].exitcode for k in dead}
                    raise RuntimeError(
                        f"worker process(es) {dead} exited without returning a result; "
                        f"exit codes: {exits}"
                    ) from None
                continue
            if not ok:
                raise payload
            if i == -2:
                if worker_records is not None:
                    worker_records.append(payload)
                continue
            results[i] = payload
            done += 1
            busy.discard(w)
            target = desired_workers(n_jobs)
            if state["next"] < len(tasks) and len(busy) < target:
                assign(w)
                busy.add(w)
            else:
                retire(w)
            while state["next"] < len(tasks) and len(busy) < target:
                w = start()
                assign(w)
                busy.add(w)
    finally:
        for proc, q in workers.values():
            if proc.is_alive():
                q.put(None)
        for proc, _ in list(workers.values()):
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
    return results


def _worker_record(wid, variables):
    return {
        "worker_id": wid,
        "pid": os.getpid(),
        "scope": "replication process initialization",
        "variables_before_configuration": variables,
        "threads": thread_record({name: os.environ.get(name) for name in THREAD_VARIABLES}),
    }


def thread_record(variables_at_start: dict[str, str | None]) -> dict:
    """What the manifest records about threads (§1.2).

    threadpoolctl observes OpenBLAS, MKL, BLIS and OpenMP pools but not Apple
    Accelerate; for Accelerate the evidence is ``VECLIB_MAXIMUM_THREADS=1`` in the
    environment the runner gave the kernel before any import, which is recorded
    separately from the observed pools.
    """
    import numpy as np
    import torch
    from threadpoolctl import threadpool_info

    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    return {
        "variables": dict(sorted(variables_at_start.items())),
        "blas_backend": str(blas.get("name", "")),
        "observed_pools": [
            {"internal_api": i["internal_api"], "num_threads": i["num_threads"]}
            for i in threadpool_info()
        ],
        "torch": {
            "num_threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
        },
    }
