"""Outputs and manifest of registration §1.7.

Each stage of an experiment has its own directory under
``notebooks/experiments/results/E-xx/``::

    stage0/   Stage 0 frozen predictions (stage0_*.json); for the deterministic
              experiments E-16, E-17 and E-23 also summary.csv, tables, figures, macros
    pilot/    Stage 0.5: timing.csv and status_counts.csv (no estimates) and the
              1-versus-12-worker identity record
    stage1/   Stage 1: raw.parquet (one row per replication x arm, failures
              included), summary.csv, tables/*.tex, figures/*.pdf, macros_E-xx.tex

Registered runs are started only by :mod:`grrexp.runner`. The runner reserves
the stage directory (``mkdir`` fails if it exists), writes ``RUNNING.json`` with
a token, executes the committed notebook in a fresh kernel, and writes
``manifest.json`` after the kernel finished. Inside the kernel,
:class:`RunRecorder` attaches to that directory, writes the outputs, and leaves
``kernel.json`` (what the kernel loaded and counted). A stage is written once; a
finished or interrupted stage is moved aside, never deleted, with
:func:`archive_run` before a registered re-run (§1.8).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import env
from .parallel import THREAD_VARIABLES
from .seeds import CONFIRMATORY_ENTROPY, PILOT_ENTROPY

STAGE_DIRS = {"stage0": "stage0", "stage0.5": "pilot", "stage1": "stage1"}
STAGE_ENTROPY = {"stage0": None, "stage0.5": PILOT_ENTROPY, "stage1": CONFIRMATORY_ENTROPY}
MAX_FILE_BYTES = 50 * 1024 * 1024  # the repository's large-file guard
MACRO_NAME = re.compile(r"^[A-Za-z]+$")
STEM = re.compile(r"^[A-Za-z0-9_-]+$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RUN_ENV = "GRREXP_RUN"
PILOT_REPS = 10
PILOT_WORKERS = 12
RUNNER_FILES = ("RUNNING.json", "kernel.json", "manifest.json", "executed.ipynb", "FAILED.txt")

# §0.2: deterministic experiments are finished at Stage 0.
DETERMINISTIC = frozenset({16, 17, 23})
# §0.2: experiments whose predictions are frozen at Stage 0 before Stage 1.
STAGE0_BEFORE_STAGE1 = frozenset({12, 13, 14, 15, 18, 19, 20, 24})

# §2: registered replications and sample sizes per part of each Monte Carlo experiment.
# ``None`` for n: the registration fixes R but not the sample sizes of that part.
REGISTERED_RUNS = {
    12: {"main": (2000, [500, 1000, 2000, 4000])},
    13: {"main": (1000, [1000, 4000])},
    14: {"main": (200, [4000, 8000, 16000, 32000, 64000])},
    15: {"partB": (2000, [500, 1000, 2000, 4000, 8000])},
    18: {"18A": (500, [1000, 2000, 4000, 8000]), "18B": (1000, [1000, 2000, 4000, 8000])},
    19: {
        "inference": (2000, [1000, 2000, 4000]),
        "counterexamples": (5000, [1000, 4000]),
        "illustration": (2000, None),
    },
    20: {"main": (2000, [[500, 500], [2000, 2000], [8000, 8000], [8000, 1000], [1000, 8000]])},
    21: {"main": (100, [747])},
    22: {"main": (100, [614])},
    24: {"main": (1000, [1000, 2000, 4000, 8000])},
}

# §4: manuscript-facing outputs of each experiment (relative to the stage directory).
REGISTERED_OUTPUTS = {
    12: (
        "tables/tab_E12_main.tex",
        "tables/tab_E12_full.tex",
        "tables/tab_E12_bp.tex",
        "tables/tab_E12_ord.tex",
        "macros_E-12.tex",
    ),
    13: ("figures/fig_E13_tangent.pdf", "tables/tab_E13.tex", "tables/tab_E13_status.tex"),
    14: ("figures/fig_E14_rate.pdf", "tables/tab_E14.tex", "tables/tab_E14_rate.tex"),
    15: (
        "tables/tab_E15_exact.tex",
        "tables/tab_E15_mc.tex",
        "tables/tab_E15_status.tex",
        "figures/fig_E15_coverage.pdf",
    ),
    16: ("macros_E-16.tex", "tables/tab_E16.tex"),
    17: ("figures/fig_E17_bounds.pdf", "tables/tab_E17.tex"),
    18: ("figures/fig_E18_oracle.pdf", "tables/tab_E18_inference.tex", "macros_E-18.tex"),
    19: ("tables/tab_E19.tex", "figures/fig_E19_counterexamples.pdf"),
    20: ("tables/tab_E20.tex", "tables/tab_E20_status.tex"),
    21: (
        "tables/tab_E21_ate.tex",
        "tables/tab_E21_att.tex",
        "tables/tab_E21_status.tex",
        "macros_E-21.tex",
    ),
    22: (
        "tables/tab_E22_main.tex",
        "tables/tab_E22_full.tex",
        "tables/tab_E22_smd.tex",
        "figures/fig_E22_love.pdf",
        "tables/tab_E22_diag.tex",
        "tables/tab_E22_status.tex",
    ),
    23: ("tables/tab_E23.tex",),
    24: ("tables/tab_E24.tex", "figures/fig_E24_rmse.pdf", "macros_E-24.tex"),
}
PILOT_COLUMNS = {
    "timing.csv": ("cell", "workers", "seconds"),
    "status_counts.csv": ("cell", "arm", "status", "count"),
}


def recorded_statuses() -> tuple[str, ...]:
    """Statuses a run may record (§1.3 B): the genriesz solver statuses, the
    outcome-model failures ``outcome_<status>`` of ``grr_functional``, and the
    baseline failures of :mod:`grrexp.baselines`."""
    from genriesz import STATUSES

    from .baselines import BASELINE_STATUSES

    outcome = tuple(f"outcome_{s}" for s in (*STATUSES, "optimizer_failure") if s != "ok")
    return (*STATUSES, *outcome, *BASELINE_STATUSES)


MANIFEST_FIELDS = (
    "experiment",
    "stage",
    "parent_repository",
    "genriesz",
    "environment",
    "threads",
    "pip_freeze",
    "pip_freeze_sha256",
    "blas",
    "cpu",
    "data_sha256",
    "entropy",
    "R",
    "n",
    "cells",
    "prerequisites",
    "started_utc",
    "finished_utc",
    "wall_seconds",
    "warnings",
    "failures",
    "outputs_sha256",
)


def deliverable_stage(exp: int) -> str:
    """The stage whose outputs reach the manuscript."""
    return "stage0" if exp in DETERMINISTIC else "stage1"


def experiment_dir(exp: int) -> Path:
    if exp not in REGISTERED_OUTPUTS:
        raise ValueError(f"EXP must be in 12..24, got {exp}")
    return env.RESULTS_DIR / f"E-{exp}"


def stage_dir(exp: int, stage: str) -> Path:
    if stage not in STAGE_DIRS:
        raise ValueError(f"stage must be one of {tuple(STAGE_DIRS)}")
    return experiment_dir(exp) / STAGE_DIRS[stage]


def prerequisite_stages(exp: int, stage: str) -> tuple[str, ...]:
    if stage == "stage0":
        return ()
    if exp in DETERMINISTIC:
        raise ValueError(f"E-{exp} is deterministic and has only Stage 0")
    if stage == "stage0.5":
        return ()
    return ("stage0", "stage0.5") if exp in STAGE0_BEFORE_STAGE1 else ("stage0.5",)


def registered_conditions(exp: int, stage: str) -> tuple[dict | None, dict | None]:
    """``(R, n)`` that the manifest of this stage must carry (``None``: not applicable)."""
    if stage == "stage0":
        return None, None
    runs = REGISTERED_RUNS[exp]
    if stage == "stage0.5":
        return {p: PILOT_REPS for p in runs}, {p: n for p, (_, n) in runs.items()}
    return {p: r for p, (r, _) in runs.items()}, {p: n for p, (_, n) in runs.items()}


def required_outputs(exp: int, stage: str) -> tuple[str, ...]:
    if stage == "stage0.5":
        return tuple(PILOT_COLUMNS)
    if stage == deliverable_stage(exp):
        extra = ("summary.csv",) if exp in DETERMINISTIC else ("raw.parquet", "summary.csv")
        return REGISTERED_OUTPUTS[exp] + extra
    return ("stage0_*.json",)


def missing_outputs(exp: int, stage: str, names) -> list[str]:
    names = list(names)
    return [pat for pat in required_outputs(exp, stage) if not fnmatch.filter(names, pat)]


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _timestamp(value) -> datetime:
    try:
        t = datetime.fromisoformat(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"bad timestamp {value!r}") from e
    _require(t.tzinfo is not None, f"timestamp {value!r} has no time zone")
    return t


def _check_n(n_registered, n_recorded) -> bool:
    if n_registered is not None:
        return n_recorded == n_registered
    return (
        isinstance(n_recorded, list)
        and bool(n_recorded)
        and all(isinstance(v, int) and v > 0 for v in n_recorded)
    )


def validate_manifest(manifest: dict, exp: int, stage: str) -> None:
    """Raise unless ``manifest`` carries every §1.7 field, consistently, for this stage.

    Checks that need git objects (module blobs, the lock file at the run commit,
    committed prerequisite manifests) are made by the runner and the sync tool.
    """
    missing = [k for k in MANIFEST_FIELDS if k not in manifest]
    _require(not missing, f"manifest lacks {missing}")
    _require(manifest["experiment"] == f"E-{exp}", "manifest names another experiment")
    _require(manifest["stage"] == stage, f"manifest stage {manifest['stage']!r} != {stage!r}")

    src = manifest["genriesz"]
    _require(bool(HEX40.match(str(src.get("sha", "")))), "genriesz.sha is not a commit")
    _require(bool(HEX40.match(str(src.get("notebook_blob", "")))), "notebook blob missing")
    _require(
        bool(HEX64.match(str(src.get("notebook_code_cells_sha256", "")))), "notebook digest bad"
    )
    nb = str(src.get("notebook", ""))
    _require(nb.startswith("notebooks/experiments/") and nb.endswith(".ipynb"), "bad notebook")
    computation = src.get("computation") or {}
    _require(
        set(env.COMPUTATION_TREES) <= set(computation) and nb in computation,
        "computation inputs not recorded",
    )
    _require(all(HEX40.match(str(v)) for v in computation.values()), "bad computation object id")
    modules = src.get("loaded_modules") or {}
    _require(any(k.startswith("src/genriesz/") for k in modules), "no genriesz module recorded")
    _require(all(HEX40.match(str(v)) for v in modules.values()), "bad module blob hash")

    parent = manifest["parent_repository"]
    _require(bool(HEX40.match(str(parent.get("sha", "")))), "parent sha missing")
    _require(bool(HEX40.match(str(parent.get("gitlink", "")))), "parent gitlink missing")

    envr = manifest["environment"]
    _require(envr.get("platform") == list(env.REGISTERED_PLATFORM), "platform not verified")
    _require(envr.get("python") == env.REGISTERED_PYTHON, "python not verified")
    _require(bool(HEX64.match(str(envr.get("lock_sha256", "")))), "lock digest missing")
    _require(int(envr.get("lock_pins_verified", 0)) > 0, "lock not verified")
    _require(bool(str(envr.get("executable", "")).strip()), "kernel interpreter not recorded")

    threads = manifest["threads"]
    _require(
        threads.get("variables") == {k: "1" for k in sorted(threads.get("variables", {}))}
        and len(threads.get("variables", {})) == 3,
        "thread variables were not all 1 at kernel start",
    )
    _require(bool(threads.get("blas_backend")), "BLAS backend not recorded")
    _require(
        all(p.get("num_threads") == 1 for p in threads.get("observed_pools", [])),
        "an observed thread pool was not 1",
    )
    _require(threads.get("torch") == {"num_threads": 1, "deterministic": True}, "torch not pinned")

    freeze = manifest["pip_freeze"]
    _require(isinstance(freeze, str) and bool(freeze), "pip freeze empty")
    _require(
        manifest["pip_freeze_sha256"] == hashlib.sha256(freeze.encode()).hexdigest(),
        "pip freeze digest does not match",
    )
    _require(bool(manifest["blas"]) and bool(str(manifest["cpu"]).strip()), "BLAS or CPU empty")

    expected_data = set(env.REGISTERED_DATA.get(exp, ()))
    _require(set(manifest["data_sha256"]) == expected_data, "data checks differ from registration")
    for name in expected_data:
        _require(manifest["data_sha256"][name] == env.DATA_SHA256[name], f"{name} hash differs")
    _require(manifest["entropy"] == STAGE_ENTROPY[stage], "entropy differs from the stage's")

    R_reg, n_reg = registered_conditions(exp, stage)
    if R_reg is None:
        _require(manifest["R"] is None and manifest["n"] is None, "Stage 0 has no R or n")
        _require(manifest["cells"] is None, "Stage 0 has no Monte Carlo cells")
    else:
        _require(manifest["R"] == R_reg, f"R {manifest['R']} != registered {R_reg}")
        n_rec = manifest["n"]
        _require(
            isinstance(n_rec, dict)
            and set(n_rec) == set(n_reg)
            and all(_check_n(n_reg[p], n_rec[p]) for p in n_reg),
            f"n {n_rec} != registered {n_reg}",
        )
        cells = manifest["cells"]
        _require(isinstance(cells, list) and bool(cells), "registered cell inventory missing")
        _require(len(set(map(json.dumps, cells))) == len(cells), "duplicate cells")

    prereq = manifest["prerequisites"]
    _require(isinstance(prereq, dict), "prerequisites must be a mapping")
    _require(set(prereq) == set(prerequisite_stages(exp, stage)), "prerequisites differ")
    for value in prereq.values():
        _require(bool(HEX64.match(str(value.get("manifest_sha256", "")))), "prereq digest bad")
        _require(bool(HEX40.match(str(value.get("commit", "")))), "prereq commit bad")

    if stage == "stage0.5":
        ident = manifest.get("parallel_identity") or {}
        _require(ident.get("reps") == PILOT_REPS, "pilot must run 10 replications per cell")
        _require(ident.get("n_jobs") == PILOT_WORKERS, "pilot must compare 1 and 12 workers")
        _require(ident.get("cells") == manifest["cells"], "identity check skipped cells")
        _require(
            ident.get("tasks") == PILOT_REPS * len(manifest["cells"]), "identity check incomplete"
        )
        _require(bool(HEX64.match(str(ident.get("digest", "")))), "identity digest missing")

    started, finished = _timestamp(manifest["started_utc"]), _timestamp(manifest["finished_utc"])
    _require(finished >= started, "run finished before it started")
    wall = manifest["wall_seconds"]
    _require(isinstance(wall, (int, float)) and math.isfinite(wall) and wall >= 0, "bad wall time")
    for key in ("warnings", "failures"):
        value = manifest[key]
        _require(isinstance(value, int) and value >= 0, f"{key} must be an int >= 0")

    outputs = manifest["outputs_sha256"]
    _require(isinstance(outputs, dict) and bool(outputs), "no outputs recorded")
    gaps = missing_outputs(exp, stage, outputs)
    _require(not gaps, f"registered outputs missing: {gaps}")
    for name, digest in outputs.items():
        _require(bool(HEX64.match(str(digest))), f"bad SHA-256 for {name}")
        _require(name not in RUNNER_FILES[:3], f"{name} is not an output")


def archive_run(exp: int, stage: str, reason: str) -> Path:
    """Move a finished or interrupted stage aside before a registered re-run (§1.8)."""
    if not reason.strip():
        raise ValueError("give the reason for the re-run")
    src = stage_dir(exp, stage)
    if not src.is_dir():
        raise FileNotFoundError(f"nothing to archive in {src}")
    state = "finished" if (src / "manifest.json").is_file() else "interrupted"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dst = env.RESULTS_DIR / "superseded" / f"E-{exp}_{STAGE_DIRS[stage]}_{stamp}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), dst)
    (dst / "REASON.txt").write_text(f"{state}\n{reason.strip()}\n", encoding="utf-8")
    return dst


class RunRecorder:
    """Kernel side of a registered run: writes the outputs into the reserved stage directory.

    It refuses to start unless the runner reserved this experiment and stage
    for this process (``GRREXP_RUN`` and the token in ``RUNNING.json``) and the
    single-thread variables were set before the kernel started.
    """

    def __init__(
        self, exp: int, stage: str, *, cells: list | None = None, arms: list[str] | None = None
    ):
        raw = os.environ.get(RUN_ENV)
        if not raw:
            raise RuntimeError("registered runs start through grrexp.runner, not interactively")
        run = json.loads(raw)
        if run.get("exp") != exp or run.get("stage") != stage:
            raise RuntimeError(f"the runner reserved {run.get('exp')}/{run.get('stage')}")
        self.exp, self.stage = exp, stage
        self.dir = stage_dir(exp, stage)
        running = json.loads((self.dir / "RUNNING.json").read_text(encoding="utf-8"))
        if running.get("token") != run.get("token"):
            raise RuntimeError("RUNNING.json does not belong to this run")
        if (self.dir / "kernel.json").exists():
            raise FileExistsError("a recorder already ran in this stage directory")
        self.threads_at_start = {k: os.environ.get(k) for k in sorted(THREAD_VARIABLES)}
        if any(v != "1" for v in self.threads_at_start.values()):
            raise RuntimeError(f"thread variables at kernel start: {self.threads_at_start}")
        from .parallel import configure_worker

        configure_worker()  # torch single-threaded and deterministic in this kernel
        if run.get("executable") != sys.executable:
            raise RuntimeError(f"kernel runs {sys.executable}, the runner {run.get('executable')}")
        self.environment = {**env.check_environment(), "executable": sys.executable}
        self.data_sha256 = env.verify_data(env.REGISTERED_DATA.get(exp, ()))
        if stage == "stage0":
            if cells is not None:
                raise ValueError("Stage 0 has no Monte Carlo cells")
        elif not cells:
            raise ValueError("give the registered cell inventory")
        if stage == "stage0.5" and not arms:
            raise ValueError("give the registered arm labels")
        if arms is not None and not all(isinstance(a, str) and STEM.match(a) for a in arms):
            raise ValueError("arm labels are [A-Za-z0-9_-]")
        self.arms = None if arms is None else list(arms)
        self.cells = (
            None if cells is None else [list(c) if isinstance(c, tuple) else c for c in cells]
        )
        self.files: dict[str, Path] = {}
        self.parallel_identity: dict | None = None
        self.closed = False

    # -- writers -------------------------------------------------------------

    def _path(self, stem: str, folder: str | None, suffix: str) -> tuple[str, Path]:
        if self.closed:
            raise RuntimeError("this run is finalized")
        if not STEM.match(stem):
            raise ValueError(f"output names are single stems [A-Za-z0-9_-], got {stem!r}")
        rel = f"{folder}/{stem}{suffix}" if folder else f"{stem}{suffix}"
        if rel in self.files or rel in RUNNER_FILES:
            raise FileExistsError(f"{rel} was already written in this run")
        path = (self.dir / rel).resolve()
        if self.dir.resolve() not in path.parents:
            raise ValueError(f"{rel} leaves the stage directory")
        if path.exists():
            raise FileExistsError(f"{rel} exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        return rel, path

    def _register(self, rel: str, path: Path) -> Path:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f"{rel} is {size} bytes, above the 50 MB guard")
        self.files[rel] = path
        return path

    def _deliverable(self) -> None:
        if self.stage != deliverable_stage(self.exp):
            raise ValueError(f"{self.stage} of E-{self.exp} writes no manuscript outputs")

    def write_stage0(self, name: str, payload: dict) -> Path:
        if self.stage != "stage0":
            raise ValueError("stage0_*.json is written only in Stage 0")
        rel, path = self._path(f"stage0_{name}", None, ".json")
        with open(path, "x", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        return self._register(rel, path)

    def write_pilot(self, name: str, frame) -> Path:
        """Stage 0.5: ``timing.csv`` or ``status_counts.csv`` with the registered columns."""
        if self.stage != "stage0.5":
            raise ValueError("pilot files are written only in Stage 0.5")
        if name not in PILOT_COLUMNS or tuple(frame.columns) != PILOT_COLUMNS[name]:
            raise ValueError(f"Stage 0.5 keeps only {PILOT_COLUMNS}")
        cells = {json.dumps(c) for c in self.cells}
        if not set(frame["cell"].map(json.dumps)) <= cells:
            raise ValueError("pilot rows name unregistered cells")
        if name == "timing.csv":
            seconds = frame["seconds"].astype(float)
            if not (seconds.map(math.isfinite).all() and (seconds >= 0).all()):
                raise ValueError("seconds must be finite and >= 0")
            if not frame["workers"].isin([1, PILOT_WORKERS]).all():
                raise ValueError(f"workers must be 1 or {PILOT_WORKERS}")
        else:
            counts = frame["count"]
            if not (counts.map(lambda v: isinstance(v, int) or float(v).is_integer()).all()):
                raise ValueError("count must be an integer")
            if not (counts >= 0).all():
                raise ValueError("count must be >= 0")
            statuses = recorded_statuses()
            if not frame["status"].isin(statuses).all():
                raise ValueError(f"status must be one of {statuses}")
            if not frame["arm"].isin(self.arms).all():
                raise ValueError(f"arm must be one of the registered arms {self.arms}")
        rel, path = self._path(name.removesuffix(".csv"), None, ".csv")
        frame.to_csv(path, index=False)
        return self._register(rel, path)

    def record_parallel_identity(self, *, cells, reps: int, n_jobs: int, digest: str) -> None:
        """Stage 0.5: the result of :func:`grrexp.parallel.verify_parallel_identity`."""
        if self.stage != "stage0.5":
            raise ValueError("the identity check belongs to Stage 0.5")
        cells = [list(c) if isinstance(c, tuple) else c for c in cells]
        if cells != self.cells or reps != PILOT_REPS or n_jobs != PILOT_WORKERS:
            raise ValueError("the identity check must cover every cell, 10 reps, 1 vs 12 workers")
        self.parallel_identity = {
            "cells": cells,
            "reps": reps,
            "n_jobs": n_jobs,
            "tasks": reps * len(cells),
            "digest": digest,
        }

    def write_raw(self, frame) -> Path:
        if self.stage != "stage1":
            raise ValueError("raw.parquet is written only in Stage 1")
        rel, path = self._path("raw", None, ".parquet")
        frame.to_parquet(path, index=False, compression="zstd")
        return self._register(rel, path)

    def write_summary(self, frame) -> Path:
        """``summary.csv``, the only input of the final cell that makes tables and figures (§4)."""
        self._deliverable()
        rel, path = self._path("summary", None, ".csv")
        frame.to_csv(path, index=False)
        return self._register(rel, path)

    def write_table(self, name: str, tex: str) -> Path:
        self._deliverable()
        rel, path = self._path(name, "tables", ".tex")
        with open(path, "x", encoding="utf-8") as f:
            f.write(tex)
        return self._register(rel, path)

    def save_figure(self, fig, name: str) -> Path:
        self._deliverable()
        rel, path = self._path(name, "figures", ".pdf")
        fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None})
        return self._register(rel, path)

    def write_macros(self, values: dict[str, str]) -> Path:
        """``macros_E-xx.tex``: one ``\\newcommand`` per number the text cites."""
        self._deliverable()
        if not values:
            raise ValueError("no macros given")
        lines = []
        for name, value in values.items():
            if not MACRO_NAME.match(name):
                raise ValueError(f"macro name must be letters only, got {name!r}")
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
        rel, path = self._path(f"macros_E-{self.exp}", None, ".tex")
        with open(path, "x", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return self._register(rel, path)

    # -- finish ----------------------------------------------------------------

    def finalize(self, *, R, n, warnings: int, failures: int, extra: dict | None = None) -> Path:
        """Write ``kernel.json`` for the runner; the runner writes the manifest."""
        if self.closed:
            raise RuntimeError("this run is already finalized")
        gaps = missing_outputs(self.exp, self.stage, self.files)
        if gaps:
            raise ValueError(f"registered outputs not written: {gaps}")
        if self.stage == "stage0.5" and self.parallel_identity is None:
            raise ValueError("Stage 0.5 needs record_parallel_identity")
        from . import parallel

        freeze, freeze_hash = env.pip_freeze()
        record = {
            "R": R,
            "n": n,
            "cells": self.cells,
            "warnings": warnings,
            "failures": failures,
            "outputs": sorted(self.files),
            "loaded_modules": env.loaded_module_blobs(),
            "threads": parallel.thread_record(self.threads_at_start),
            "environment": self.environment,
            "data_sha256": self.data_sha256,
            "pip_freeze": freeze,
            "pip_freeze_sha256": freeze_hash,
            "blas": env.blas_config(),
            "cpu": env.cpu_description(),
            "extra": extra or {},
        }
        if self.parallel_identity is not None:
            record["parallel_identity"] = self.parallel_identity
        with open(self.dir / "kernel.json", "x", encoding="utf-8") as f:
            json.dump(record, f, indent=2, allow_nan=False)
        self.closed = True
        return self.dir / "kernel.json"


def read_summary(recorder: RunRecorder):
    """Read back ``summary.csv`` as written in this run (the final cell's only input, §4)."""
    import pandas as pd

    if "summary.csv" not in recorder.files:
        raise ValueError("write summary.csv before making tables and figures")
    return pd.read_csv(recorder.files["summary.csv"])
