"""Source, uv environment and data provenance for the paper experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import tomllib

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
GENRIESZ_ROOT = EXPERIMENTS_DIR.parent
GENRIESZ_SRC = GENRIESZ_ROOT / "src"
DATA_DIR = GENRIESZ_ROOT / "notebooks" / "experiments" / "data"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LEGACY_RESULTS_DIR = GENRIESZ_ROOT / "notebooks" / "experiments" / "results"
LOCK_FILE = EXPERIMENTS_DIR / "uv.lock"
RESULTS_PREFIXES = ("experiments/results/", "notebooks/experiments/results/")

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
# Registration §1.7 and §3: the data each experiment reads (D-01 IHDP, D-02 Lalonde).
REGISTERED_DATA = {
    21: ("ihdp/ihdp_npci_1-100.train.npz", "ihdp/ihdp_npci_1-100.test.npz"),
    22: ("lalonde/lalonde.csv",),
}

REGISTERED_PYTHON = "3.13.6"
REGISTERED_PLATFORM = ("Darwin", "arm64")
# The registered packages of §1.7 (the roots of the lock's dependency closure).
LOCK_ROOTS = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "pyarrow",
    "torch",
    "cvxpy",
    "clarabel",
    "sympy",
    "marimo",
)


def use_submodule_genriesz():
    """Import genriesz from this checkout's ``src/`` and return the module."""
    src = str(GENRIESZ_SRC)
    loaded = sys.modules.get("genriesz")
    if loaded is not None and Path(loaded.__file__).resolve().parents[1] != GENRIESZ_SRC:
        raise RuntimeError(
            f"genriesz was already imported from {loaded.__file__}; restart the process"
        )
    if sys.path[0] != src:
        sys.path.insert(0, src)
    import genriesz

    if Path(genriesz.__file__).resolve().parents[1] != GENRIESZ_SRC:
        raise RuntimeError(f"genriesz resolved to {genriesz.__file__}, not {GENRIESZ_SRC}")
    return genriesz


# ------------------------------------------------------------------ environment


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dependency_closure(roots=LOCK_ROOTS) -> dict[str, str]:
    """Installed versions of ``roots`` and of every installed requirement they pull in.

    Requirements whose environment marker is false, or that are only extras, are
    not followed.
    """
    from packaging.requirements import Requirement

    seen: dict[str, str] = {}
    stack = [_canonical(r) for r in roots]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        dist = importlib.metadata.distribution(name)
        seen[name] = dist.version
        for spec in dist.requires or ():
            req = Requirement(spec)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            stack.append(_canonical(req.name))
    return dict(sorted(seen.items()))


def read_lock(path: Path = LOCK_FILE) -> dict[str, str]:
    """Read registry package versions selected for this environment in uv.lock."""
    from packaging.markers import Marker

    pins = {}
    with Path(path).open("rb") as stream:
        lock = tomllib.load(stream)
    for package in lock["package"]:
        if "registry" not in package.get("source", {}):
            continue
        markers = package.get("resolution-markers", [])
        if markers and not any(Marker(marker).evaluate() for marker in markers):
            continue
        name, version = _canonical(package["name"]), package["version"]
        if name in pins and pins[name] != version:
            raise ValueError(f"uv.lock selects multiple versions of {name}")
        pins[name] = version
    if not pins:
        raise ValueError(f"{path} pins nothing")
    return pins


def check_environment() -> dict:
    """Stop unless the platform, Python and every lock pin match; return the record."""
    problems = []
    system = (platform.system(), platform.machine())
    if system != REGISTERED_PLATFORM:
        problems.append(f"platform {system} != {REGISTERED_PLATFORM}")
    if platform.python_version() != REGISTERED_PYTHON:
        problems.append(f"python {platform.python_version()} != {REGISTERED_PYTHON}")
    pins = read_lock()
    missing_roots = {_canonical(r) for r in LOCK_ROOTS} - set(pins)
    if missing_roots:
        problems.append(f"lock lacks registered packages {sorted(missing_roots)}")
    installed = dependency_closure()
    for name in installed:
        if name not in pins:
            problems.append(f"uv.lock lacks installed dependency {name}")
            continue
        version = pins[name]
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name} not installed (lock {version})")
            continue
        if found != version:
            problems.append(f"{name} {found} != lock {version}")
    if problems:
        raise RuntimeError("environment differs from registration §1.7: " + "; ".join(problems))
    return {
        "platform": list(system),
        "python": platform.python_version(),
        "executable": sys.executable,
        "lock_sha256": sha256_file(LOCK_FILE),
        "lock_pins_verified": len(installed),
    }


# ------------------------------------------------------------------ data


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


# ------------------------------------------------------------------ source


def _git(root: Path, *args: str, text: bool = True):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=text
    ).stdout


def git_blob_sha1(content: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def _head_blobs(root: Path) -> dict[str, str]:
    out = {}
    for line in _git(root, "ls-tree", "-r", "HEAD").splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind == "blob":
            out[path] = sha
    return out


def head_blobs(root: Path | None = None) -> dict[str, str]:
    """``path -> blob SHA-1`` of every file committed at ``HEAD``."""
    return _head_blobs(GENRIESZ_ROOT if root is None else root)


def checkout_state(app_file: Path) -> dict:
    """Record the actual checkout, or verify an explicitly supplied source snapshot."""
    root = GENRIESZ_ROOT
    app_file = Path(app_file).resolve()
    app_rel = app_file.relative_to(root).as_posix()
    if not (root / ".git").exists():
        path, snapshot = _source_snapshot()
        return {
            "kind": "source_snapshot",
            "root": str(root),
            "sha": None,
            "source_git_state": snapshot["source_genriesz"],
            "source_manifest_sha256": sha256_file(path),
            "app": app_rel,
            "app_sha256": sha256_file(app_file),
        }
    if Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve() != root:
        raise RuntimeError(f"{root} is not the root of the genriesz checkout")
    blobs = _head_blobs(root)
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    return {
        "kind": "git_checkout",
        "root": str(root),
        "sha": _git(root, "rev-parse", "HEAD").strip(),
        "status_porcelain": status,
        "dirty": bool(status),
        "dirty_tracked": bool(_git(root, "status", "--porcelain", "--untracked-files=no")),
        "app": app_rel,
        "app_committed_blob": blobs.get(app_rel),
        "app_sha256": sha256_file(app_file),
    }


def _source_snapshot() -> tuple[Path, dict]:
    path = GENRIESZ_ROOT.parent / "SOURCE_SHA256.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    for name in ("files", "source_genriesz", "source_parent_repository"):
        if name not in snapshot or not snapshot[name]:
            raise ValueError(f"{path} lacks {name}")
    required = (
        list(GENRIESZ_SRC.rglob("*.py"))
        + list((EXPERIMENTS_DIR / "grrexp").glob("*.py"))
        + list((EXPERIMENTS_DIR / "apps").glob("*.py"))
        + [
            EXPERIMENTS_DIR / "run_registered.py",
            EXPERIMENTS_DIR / "pyproject.toml",
            EXPERIMENTS_DIR / "uv.lock",
            GENRIESZ_ROOT / "pyproject.toml",
        ]
    )
    missing = {file.relative_to(GENRIESZ_ROOT).as_posix() for file in required} - set(
        snapshot["files"]
    )
    if missing:
        raise ValueError(f"{path} lacks source files: {sorted(missing)}")
    for name, digest in snapshot["files"].items():
        file = (GENRIESZ_ROOT / name).resolve()
        if not file.is_relative_to(GENRIESZ_ROOT) or sha256_file(file) != digest:
            raise ValueError(f"source snapshot differs from its manifest: {name}")
    return path, snapshot


def loaded_module_blobs() -> dict[str, str]:
    """``path -> blob SHA-1`` of every loaded module file inside this checkout."""
    out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not isinstance(f, str) or not Path(f).is_absolute():
            continue  # built-in, namespace, or synthetic (e.g. torch's "_ops.py")
        p = Path(f).resolve()
        if GENRIESZ_ROOT in p.parents:
            out[p.relative_to(GENRIESZ_ROOT).as_posix()] = git_blob_sha1(p.read_bytes())
    return dict(sorted(out.items()))


def check_modules_at_head(loaded: dict[str, str], blobs: dict[str, str]) -> None:
    """Raise unless every loaded module is committed and equal to its blob."""
    bad = [rel for rel, sha in loaded.items() if blobs.get(rel) != sha]
    if bad:
        raise RuntimeError(f"loaded modules differ from the commit: {bad}")
    if not any(rel.startswith("src/genriesz/") for rel in loaded):
        raise RuntimeError("genriesz from this checkout was not loaded")


COMPUTATION_TREES = ("src/genriesz", "experiments/grrexp")


def computation_inputs(commit: str, app_rel: str, root: Path | None = None) -> dict:
    """Git object ids of what a run computes with: the genriesz and grrexp trees,
    the marimo app and the uv environment at ``commit``."""
    root = GENRIESZ_ROOT if root is None else root
    out = {t: _git(root, "rev-parse", f"{commit}:{t}").strip() for t in COMPUTATION_TREES}
    for name in (app_rel, "experiments/uv.lock", "experiments/pyproject.toml", "pyproject.toml"):
        out[name] = _git(root, "rev-parse", f"{commit}:{name}").strip()
    return dict(sorted(out.items()))


PARENT_ENV = "GRREXP_PARENT_REPO"


def parent_state() -> dict:
    """The parent repository's commit and its recorded genriesz gitlink (read only).

    A run from a ``git worktree`` of genriesz outside the parent repository names the
    parent with ``GRREXP_PARENT_REPO``; its submodule path is then ``genriesz`` and the
    worktree path is recorded.
    """
    if not (GENRIESZ_ROOT / ".git").exists():
        path, snapshot = _source_snapshot()
        return {
            "kind": "source_snapshot",
            "sha": None,
            "gitlink": None,
            "source_git_state": snapshot["source_parent_repository"],
            "source_manifest_sha256": sha256_file(path),
        }
    override = os.environ.get(PARENT_ENV)
    parent = Path(override) if override else GENRIESZ_ROOT.parent
    name = "genriesz" if override else GENRIESZ_ROOT.name
    if Path(_git(parent, "rev-parse", "--show-toplevel").strip()).resolve() != parent.resolve():
        raise RuntimeError(f"{parent} is not the root of the parent repository")
    gitlink = _git(parent, "ls-tree", "HEAD", name).split()
    if len(gitlink) != 4 or gitlink[:2] != ["160000", "commit"]:
        raise RuntimeError(f"{name} is not a committed gitlink in {parent}")
    status = _git(parent, "status", "--porcelain", "--untracked-files=all")
    state = {
        "kind": "git_checkout",
        "root": str(parent),
        "sha": _git(parent, "rev-parse", "HEAD").strip(),
        "gitlink": gitlink[2],
        "status_porcelain": status,
        "dirty": bool(status),
        "dirty_tracked": bool(
            _git(parent, "status", "--porcelain", "--untracked-files=no").strip()
        ),
    }
    if override:
        state["genriesz_worktree"] = str(GENRIESZ_ROOT)
    return state


def pip_freeze() -> tuple[str, str]:
    text = subprocess.run(
        ["uv", "--no-cache", "pip", "freeze", "--python", sys.executable],
        check=True, capture_output=True, text=True,
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
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    return platform.processor() or platform.machine()
