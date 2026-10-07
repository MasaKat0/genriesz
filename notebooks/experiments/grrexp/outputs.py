"""Outputs and manifest of registration §1.7.

Each stage of an experiment has its own directory under
``notebooks/experiments/results/E-xx/``::

    stage0/   Stage 0 frozen predictions (stage0_*.json); for the deterministic
              experiments E-16, E-17 and E-23 also their tables, figures, macros
    pilot/    Stage 0.5: timing.csv and status_counts.csv only (no estimates)
    stage1/   Stage 1: raw.parquet (one row per replication x arm, failures
              included), summary.csv, tables/*.tex, figures/*.pdf, macros_E-xx.tex

A stage directory is reserved by creating it (``mkdir`` fails if it exists),
holds ``RUNNING.json`` while the run computes and ``manifest.json`` once it is
finished, and is written once. A finished or interrupted stage is moved aside,
never deleted, with :func:`archive_run` before a registered re-run (§1.8).

The manuscript never reads these files: the parent repository's
``tools/sync_experiment_outputs.py`` validates the manifest with
:func:`validate_manifest` and copies the committed outputs.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from . import env
from .seeds import CONFIRMATORY_ENTROPY, PILOT_ENTROPY

STAGE_DIRS = {"stage0": "stage0", "stage0.5": "pilot", "stage1": "stage1"}
STAGE_ENTROPY = {"stage0": None, "stage0.5": PILOT_ENTROPY, "stage1": CONFIRMATORY_ENTROPY}
MAX_FILE_BYTES = 50 * 1024 * 1024  # the repository's large-file guard
MACRO_NAME = re.compile(r"^[A-Za-z]+$")
OUTPUT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
PILOT_FILES = ("timing.csv", "status_counts.csv")
RESERVED_NAMES = ("manifest.json", "RUNNING.json")

# §0.2: deterministic experiments are finished at Stage 0.
DETERMINISTIC = frozenset({16, 17, 23})
# §0.2: experiments whose predictions are frozen at Stage 0 before Stage 1.
STAGE0_BEFORE_STAGE1 = frozenset({12, 13, 14, 15, 18, 19, 20, 24})

# §4: manuscript-facing outputs of each experiment (patterns relative to the stage directory).
REGISTERED_OUTPUTS = {
    12: (
        "tables/tab_E12_main.tex",
        "tables/tab_E12_full.tex",
        "tables/tab_E12_bp.tex",
        "tables/tab_E12_ord.tex",
        "macros_E-12.tex",
    ),
    13: ("figures/fig_E13_tangent.pdf", "tables/tab_E13.tex"),
    14: ("figures/fig_E14_rate.pdf", "tables/tab_E14.tex"),
    15: ("tables/tab_E15_exact.tex", "tables/tab_E15_mc.tex", "figures/fig_E15_coverage.pdf"),
    16: ("macros_E-16.tex", "tables/tab_E16.tex"),
    17: ("figures/fig_E17_bounds.pdf", "tables/tab_E17.tex"),
    18: ("figures/fig_E18_oracle.pdf", "tables/tab_E18_inference.tex", "macros_E-18.tex"),
    19: ("tables/tab_E19.tex", "figures/fig_E19_counterexamples.pdf"),
    20: ("tables/tab_E20.tex",),
    21: ("tables/tab_E21_ate.tex", "tables/tab_E21_att.tex"),
    22: ("tables/tab_E22_*.tex", "figures/fig_E22_love.pdf"),
    23: ("tables/tab_E23.tex",),
    24: ("tables/tab_E24.tex", "figures/fig_E24_rmse.pdf", "macros_E-24.tex"),
}

MANIFEST_FIELDS = (
    "experiment",
    "stage",
    "parent_repository",
    "genriesz",
    "environment",
    "pip_freeze",
    "pip_freeze_sha256",
    "blas",
    "cpu",
    "data_sha256",
    "entropy",
    "R",
    "n",
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


def required_outputs(exp: int, stage: str) -> tuple[str, ...]:
    if stage == "stage0.5":
        return PILOT_FILES
    if stage == deliverable_stage(exp):
        extra = () if exp in DETERMINISTIC else ("raw.parquet", "summary.csv")
        return REGISTERED_OUTPUTS[exp] + extra
    return ("stage0_*.json",)


def missing_outputs(exp: int, stage: str, names) -> list[str]:
    names = list(names)
    return [pat for pat in required_outputs(exp, stage) if not fnmatch.filter(names, pat)]


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def validate_manifest(manifest: dict, exp: int, stage: str) -> None:
    """Raise unless ``manifest`` has every §1.7 field for this experiment and stage."""
    missing = [k for k in MANIFEST_FIELDS if k not in manifest]
    _require(not missing, f"manifest lacks {missing}")
    _require(manifest["experiment"] == f"E-{exp}", "manifest names another experiment")
    _require(manifest["stage"] == stage, f"manifest stage {manifest['stage']!r} != {stage!r}")
    src = manifest["genriesz"]
    for key in ("sha", "notebook", "notebook_code_cells_sha256", "loaded_modules"):
        _require(bool(src.get(key)), f"genriesz.{key} is empty")
    _require(
        any(k.startswith("src/genriesz/") for k in src["loaded_modules"]),
        "no genriesz module recorded",
    )
    parent = manifest["parent_repository"]
    _require(bool(parent.get("sha")) and bool(parent.get("gitlink")), "parent state is empty")
    envr = manifest["environment"]
    _require(envr.get("platform") == list(env.REGISTERED_PLATFORM), "platform not verified")
    _require(envr.get("python") == env.REGISTERED_PYTHON, "python not verified")
    _require(int(envr.get("lock_pins_verified", 0)) > 0, "lock not verified")
    _require(bool(manifest["pip_freeze"]) and bool(manifest["blas"]), "pip freeze or BLAS empty")
    expected_data = set(env.REGISTERED_DATA.get(exp, ()))
    _require(set(manifest["data_sha256"]) == expected_data, "data checks differ from registration")
    for name in expected_data:
        _require(manifest["data_sha256"][name] == env.DATA_SHA256[name], f"{name} hash differs")
    _require(manifest["entropy"] == STAGE_ENTROPY[stage], "entropy differs from the stage's")
    if stage != "stage0":
        _require(isinstance(manifest["R"], int) and manifest["R"] > 0, "R must be a positive int")
        _require(manifest["n"] is not None, "n is missing")
    for key in ("warnings", "failures"):
        _require(isinstance(manifest[key], int) and manifest[key] >= 0, f"{key} must be >= 0")
    _require(manifest["wall_seconds"] >= 0, "wall time is negative")
    outputs = manifest["outputs_sha256"]
    _require(bool(outputs), "no outputs recorded")
    gaps = missing_outputs(exp, stage, outputs)
    _require(not gaps, f"registered outputs missing: {gaps}")
    for name, digest in outputs.items():
        _require(isinstance(digest, str) and len(digest) == 64, f"bad SHA-256 for {name}")


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
    """Reserves one stage of one experiment, collects its outputs, writes its manifest.

    Create it after the imports and before any computation: it checks the
    environment, the data and the source, and fixes the source snapshot that
    :meth:`finalize` requires to be unchanged.
    """

    def __init__(self, exp: int, stage: str, *, notebook, R: int | None, n):
        self.exp = exp
        self.stage = stage
        self.R = R
        self.n = n
        self.dir = stage_dir(exp, stage)
        if stage == "stage1":
            if exp in DETERMINISTIC:
                raise ValueError(f"E-{exp} is deterministic and finished at Stage 0")
            needed = ["stage0.5"] + (["stage0"] if exp in STAGE0_BEFORE_STAGE1 else [])
            for prior in needed:
                if not (stage_dir(exp, prior) / "manifest.json").is_file():
                    raise RuntimeError(f"Stage 1 of E-{exp} needs a finished {prior}")
        if stage == "stage0.5" and exp in DETERMINISTIC:
            raise ValueError(f"E-{exp} has no Monte Carlo pilot")
        if stage != "stage0" and (not isinstance(R, int) or R <= 0):
            raise ValueError("R must be a positive int")
        self.environment = env.check_environment()
        self.data_sha256 = env.verify_data(env.REGISTERED_DATA.get(exp, ()))
        self.source = env.source_snapshot(notebook)
        self._notebook = notebook
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        self.dir.mkdir()  # reservation: fails if this stage was run before
        self.started = datetime.now(timezone.utc)
        self._t0 = time.perf_counter()
        running = {"started_utc": self.started.isoformat(), "genriesz": self.source["sha"]}
        (self.dir / "RUNNING.json").write_text(json.dumps(running), encoding="utf-8")
        self.files: dict[str, Path] = {}
        self.closed = False

    # -- writers -------------------------------------------------------------

    def _path(self, relative: str) -> Path:
        if self.closed:
            raise RuntimeError("this run is finalized")
        parts = relative.split("/")
        valid = all(OUTPUT_NAME.match(p) for p in parts) and relative not in RESERVED_NAMES
        if not valid:
            raise ValueError(f"invalid output name {relative!r}")
        if relative in self.files:
            raise FileExistsError(f"{relative} was already written in this run")
        path = self.dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _register(self, relative: str, path: Path) -> Path:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f"{relative} is {size} bytes, above the 50 MB guard")
        self.files[relative] = path
        return path

    def _deliverable(self) -> None:
        if self.stage != deliverable_stage(self.exp):
            raise ValueError(f"{self.stage} of E-{self.exp} writes no manuscript outputs")

    def write_stage0(self, name: str, payload: dict) -> Path:
        if self.stage != "stage0":
            raise ValueError("stage0_*.json is written only in Stage 0")
        rel = f"stage0_{name}.json"
        path = self._path(rel)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return self._register(rel, path)

    def write_pilot(self, name: str, frame) -> Path:
        """Stage 0.5: ``timing.csv`` or ``status_counts.csv`` (no estimates)."""
        if self.stage != "stage0.5":
            raise ValueError("pilot files are written only in Stage 0.5")
        if name not in PILOT_FILES:
            raise ValueError(f"Stage 0.5 keeps only {PILOT_FILES}")
        path = self._path(name)
        frame.to_csv(path, index=False)
        return self._register(name, path)

    def write_raw(self, frame) -> Path:
        if self.stage != "stage1":
            raise ValueError("raw.parquet is written only in Stage 1")
        path = self._path("raw.parquet")
        frame.to_parquet(path, index=False, compression="zstd")
        return self._register("raw.parquet", path)

    def write_summary(self, frame) -> Path:
        if self.stage != "stage1":
            raise ValueError("summary.csv is written only in Stage 1")
        path = self._path("summary.csv")
        frame.to_csv(path, index=False)
        return self._register("summary.csv", path)

    def write_table(self, name: str, tex: str) -> Path:
        self._deliverable()
        rel = f"tables/{name}.tex"
        path = self._path(rel)
        path.write_text(tex, encoding="utf-8")
        return self._register(rel, path)

    def save_figure(self, fig, name: str) -> Path:
        self._deliverable()
        rel = f"figures/{name}.pdf"
        path = self._path(rel)
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
        rel = f"macros_E-{self.exp}.tex"
        path = self._path(rel)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self._register(rel, path)

    # -- finish ----------------------------------------------------------------

    def finalize(self, *, warnings: int, failures: int, extra: dict | None = None) -> Path:
        if self.closed:
            raise RuntimeError("this run is already finalized")
        gaps = missing_outputs(self.exp, self.stage, self.files)
        if gaps:
            raise ValueError(f"registered outputs not written: {gaps}")
        if env.source_snapshot(self._notebook) != self.source:
            raise RuntimeError("source changed during the run; archive it and re-run")
        freeze, freeze_hash = env.pip_freeze()
        manifest = {
            "experiment": f"E-{self.exp}",
            "stage": self.stage,
            "parent_repository": env.parent_state(),
            "genriesz": self.source,
            "environment": self.environment,
            "pip_freeze": freeze,
            "pip_freeze_sha256": freeze_hash,
            "blas": env.blas_config(),
            "cpu": env.cpu_description(),
            "data_sha256": self.data_sha256,
            "entropy": STAGE_ENTROPY[self.stage],
            "R": self.R,
            "n": self.n,
            "started_utc": self.started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - self._t0,
            "warnings": int(warnings),
            "failures": int(failures),
            "outputs_sha256": {
                rel: env.sha256_file(path) for rel, path in sorted(self.files.items())
            },
        }
        if extra:
            clash = set(extra) & set(manifest)
            if clash:
                raise ValueError(f"extra keys clash with registered fields: {sorted(clash)}")
            manifest.update(extra)
        validate_manifest(manifest, self.exp, self.stage)
        path = self.dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (self.dir / "RUNNING.json").unlink()
        self.closed = True
        return path
