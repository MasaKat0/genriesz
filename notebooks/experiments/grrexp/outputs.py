"""Outputs and manifest of registration §1.7.

Layout under ``notebooks/experiments/results/E-xx/``::

    stage0_*.json             Stage 0 frozen predictions   (manifest_stage0.json)
    pilot/                    Stage 0.5 timing and status counts only (pilot/manifest.json)
    raw.parquet               Stage 1: one row per replication x arm, failures included
    summary.csv, tables/*.tex, figures/*.pdf, macros_E-xx.tex
    manifest.json             Stage 1

A Stage 1 directory is written once. Re-running after a bug fix first moves the
previous run aside with :func:`archive_run` (it is kept, not deleted).
The manuscript never reads these files directly: the parent repository's
``tools/sync_experiment_outputs.py`` checks the manifest hashes and copies them.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from . import env
from .seeds import CONFIRMATORY_ENTROPY, PILOT_ENTROPY

STAGES = ("stage0", "stage0.5", "stage1")
MAX_FILE_BYTES = 50 * 1024 * 1024  # the repository's large-file guard
MACRO_NAME = re.compile(r"^[A-Za-z]+$")
STAGE_ENTROPY = {"stage0": None, "stage0.5": PILOT_ENTROPY, "stage1": CONFIRMATORY_ENTROPY}


def experiment_dir(exp: int) -> Path:
    if exp not in range(12, 25):
        raise ValueError(f"EXP must be in 12..24, got {exp}")
    return env.RESULTS_DIR / f"E-{exp}"


def archive_run(exp: int, reason: str) -> Path:
    """Move a finished Stage 1 run aside before a registered re-run (§1.8)."""
    src = experiment_dir(exp)
    if not (src / "manifest.json").is_file():
        raise FileNotFoundError(f"no finished Stage 1 run in {src}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = env.RESULTS_DIR / "superseded" / f"E-{exp}_{stamp}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    keep = {
        p.name
        for p in src.iterdir()
        if p.name.startswith("stage0") or p.name == "manifest_stage0.json"
    }
    dst.mkdir()
    for p in src.iterdir():
        if p.name not in keep:
            shutil.move(str(p), dst / p.name)
    (dst / "REASON.txt").write_text(reason.strip() + "\n", encoding="utf-8")
    return dst


class RunRecorder:
    """Collects the outputs of one stage of one experiment and writes its manifest."""

    def __init__(self, exp: int, stage: str, *, R: int | None, n, data_files=()):
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        self.exp = exp
        self.stage = stage
        self.R = R
        self.n = n
        base = experiment_dir(exp)
        self.dir = base / "pilot" if stage == "stage0.5" else base
        self.manifest_path = self.dir / (
            "manifest_stage0.json" if stage == "stage0" else "manifest.json"
        )
        if self.manifest_path.exists():
            raise FileExistsError(f"{self.manifest_path} exists; archive the previous run first")
        self.data_sha256 = env.verify_data(data_files)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files: list[Path] = []
        self.started = datetime.now(timezone.utc)
        self._t0 = time.perf_counter()

    def _register(self, path: Path) -> Path:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f"{path.name} is {size} bytes, above the 50 MB guard")
        self.files.append(path)
        return path

    def _path(self, relative: str) -> Path:
        path = self.dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_stage0(self, name: str, payload: dict) -> Path:
        if self.stage != "stage0":
            raise ValueError("stage0_*.json is written only in Stage 0")
        path = self._path(f"stage0_{name}.json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return self._register(path)

    def write_raw(self, frame) -> Path:
        if self.stage == "stage0.5":
            raise ValueError("Stage 0.5 keeps no estimates (registration §0.2)")
        path = self._path("raw.parquet")
        frame.to_parquet(path, index=False, compression="zstd")
        return self._register(path)

    def write_csv(self, name: str, frame) -> Path:
        path = self._path(name)
        frame.to_csv(path, index=False)
        return self._register(path)

    def write_table(self, name: str, tex: str) -> Path:
        path = self._path(f"tables/{name}.tex")
        path.write_text(tex, encoding="utf-8")
        return self._register(path)

    def save_figure(self, fig, name: str) -> Path:
        path = self._path(f"figures/{name}.pdf")
        fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None})
        return self._register(path)

    def write_macros(self, values: dict[str, str]) -> Path:
        """``macros_E-xx.tex``: one ``\\newcommand`` per number the text cites."""
        lines = []
        for name, value in values.items():
            if not MACRO_NAME.match(name):
                raise ValueError(f"macro name must be letters only, got {name!r}")
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
        path = self._path(f"macros_E-{self.exp}.tex")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self._register(path)

    def finalize(self, *, warnings: int, failures: int, extra: dict | None = None) -> Path:
        freeze, freeze_hash = env.pip_freeze()
        parent = env.GENRIESZ_ROOT.parent
        manifest = {
            "experiment": f"E-{self.exp}",
            "stage": self.stage,
            "parent_repository": env.git_state(parent),
            "parent_gitlink": env._git(parent, "ls-tree", "HEAD", env.GENRIESZ_ROOT.name).split()[
                2
            ],
            "genriesz": env.git_state(env.GENRIESZ_ROOT),
            "pip_freeze": freeze,
            "pip_freeze_sha256": freeze_hash,
            "blas": env.blas_config(),
            "python": platform.python_version(),
            "platform": platform.platform(),
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
                str(p.relative_to(self.dir)): env.sha256_file(p) for p in sorted(self.files)
            },
        }
        if extra:
            clash = set(extra) & set(manifest)
            if clash:
                raise ValueError(f"extra keys clash with registered fields: {sorted(clash)}")
            manifest.update(extra)
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return self.manifest_path
