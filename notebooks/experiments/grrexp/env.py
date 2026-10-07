"""Provenance of a run (registration §1.7) and the genriesz source check.

The ambient ``python3`` on the author's machine imports genriesz from an
editable install outside this repository. :func:`use_submodule_genriesz` puts
this checkout's ``src/`` first on ``sys.path`` and refuses to continue if
genriesz was already imported from elsewhere, so a run can only use the source
whose SHA the manifest records.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import platform
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
GENRIESZ_ROOT = EXPERIMENTS_DIR.parents[1]
GENRIESZ_SRC = GENRIESZ_ROOT / "src"
DATA_DIR = EXPERIMENTS_DIR / "data"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# Registration §1.7: snapshots verified before every run that reads them.
DATA_SHA256 = {
    "lalonde/lalonde.csv": "0e07558a87643b1a44bd3fc1d634890f0e9175e9c2bc02d7aeed5139ad771557",
    "ihdp/ihdp_npci_1-100.train.npz": (
        "750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240"
    ),
    "ihdp/ihdp_npci_1-100.test.npz": (
        "a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c"
    ),
}

# Registration §1.7: the registered environment. ``check_environment`` stops on a mismatch.
REGISTERED_PYTHON = "3.13.6"
REGISTERED_PACKAGES = {
    "numpy": "2.3.5",
    "scipy": "1.16.3",
    "pandas": "2.3.3",
    "scikit-learn": "1.7.2",
    "matplotlib": "3.10.8",
    "pyarrow": "22.0.0",
    "torch": "2.8.0",
    "cvxpy": "1.7.3",
}


def use_submodule_genriesz():
    """Import genriesz from this checkout's ``src/`` and return the module."""
    src = str(GENRIESZ_SRC)
    loaded = sys.modules.get("genriesz")
    if loaded is not None and Path(loaded.__file__).resolve().parents[1] != GENRIESZ_SRC:
        raise RuntimeError(
            f"genriesz was already imported from {loaded.__file__}; restart the kernel"
        )
    if sys.path[0] != src:
        sys.path.insert(0, src)
    import genriesz

    if Path(genriesz.__file__).resolve().parents[1] != GENRIESZ_SRC:
        raise RuntimeError(f"genriesz resolved to {genriesz.__file__}, not {GENRIESZ_SRC}")
    return genriesz


def check_environment() -> None:
    """Stop unless Python and the registered packages have the registered versions."""
    problems = []
    if platform.python_version() != REGISTERED_PYTHON:
        problems.append(f"python {platform.python_version()} != {REGISTERED_PYTHON}")
    for name, version in REGISTERED_PACKAGES.items():
        found = importlib.metadata.version(name)
        if found != version:
            problems.append(f"{name} {found} != {version}")
    if problems:
        raise RuntimeError("environment differs from registration §1.7: " + "; ".join(problems))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_data(names) -> dict[str, str]:
    """Check the registered SHA-256 of each data file; a missing or changed file stops the run."""
    out = {}
    for name in names:
        expected = DATA_SHA256[name]
        path = DATA_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"data snapshot missing: {path}")
        found = sha256_file(path)
        if found != expected:
            raise ValueError(f"{name}: SHA-256 {found} != registered {expected}")
        out[name] = found
    return out


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def git_state(root: Path) -> dict:
    return {
        "sha": _git(root, "rev-parse", "HEAD"),
        "dirty_tracked": bool(_git(root, "status", "--porcelain", "--untracked-files=no")),
    }


def pip_freeze() -> tuple[str, str]:
    text = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
    ).stdout
    return text, hashlib.sha256(text.encode()).hexdigest()


def blas_config() -> str:
    import numpy as np

    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    return buf.getvalue()


def cpu_description() -> str:
    if sys.platform == "darwin":
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    return platform.processor() or platform.machine()
