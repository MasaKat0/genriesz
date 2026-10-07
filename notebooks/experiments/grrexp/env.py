"""Provenance of a run (registration §1.7): source, environment and data checks.

The ambient ``python3`` on the author's machine imports genriesz from an
editable install outside this repository. :func:`use_submodule_genriesz` puts
this checkout's ``src/`` first on ``sys.path`` and refuses to continue if
genriesz was already imported from elsewhere.

:func:`checkout_state` ties a registered run to a commit (no change and no
untracked file outside ``results/``); :mod:`grrexp.runner` checks it before and
after the run and compares the modules the kernel loaded
(:func:`loaded_module_blobs`) with the blobs of that commit.

:func:`check_environment` enforces the registered platform and every pin of
``requirements-lock.txt`` (the registered packages and their dependency
closure, see :func:`dependency_closure`).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
GENRIESZ_ROOT = EXPERIMENTS_DIR.parents[1]
GENRIESZ_SRC = GENRIESZ_ROOT / "src"
DATA_DIR = EXPERIMENTS_DIR / "data"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOCK_FILE = EXPERIMENTS_DIR / "requirements-lock.txt"
RESULTS_PREFIX = "notebooks/experiments/results/"

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
    "nbformat",
    "nbconvert",
    "ipykernel",
    "notebook",
)


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
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if not sep or not version:
            raise ValueError(f"lock line is not 'name==version': {line!r}")
        pins[_canonical(name)] = version.strip()
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
    for name, version in pins.items():
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
        "lock_sha256": sha256_file(LOCK_FILE),
        "lock_pins_verified": len(pins),
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


def _code_cells(raw: bytes) -> list[str]:
    nb = json.loads(raw.decode("utf-8"))
    out = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = cell["source"]
            out.append("".join(src) if isinstance(src, list) else src)
    return out


def head_blobs(root: Path | None = None) -> dict[str, str]:
    """``path -> blob SHA-1`` of every file committed at ``HEAD``."""
    return _head_blobs(GENRIESZ_ROOT if root is None else root)


def checkout_state(notebook: Path) -> dict:
    """The commit a registered run starts from; raises unless the checkout is clean.

    Nothing may be modified or untracked outside ``results/`` (the executing
    notebook included: a registered run executes the committed notebook into
    its stage directory and never saves into the tracked file).
    """
    root = GENRIESZ_ROOT
    notebook = Path(notebook).resolve()
    nb_rel = notebook.relative_to(root).as_posix()
    blobs = _head_blobs(root)
    if nb_rel not in blobs:
        raise RuntimeError(f"{nb_rel} is not committed at HEAD")
    dirty = []
    for line in _git(root, "status", "--porcelain", "--untracked-files=all").splitlines():
        path = line[3:].strip('"')
        if not path.startswith(RESULTS_PREFIX):
            dirty.append(line)
    if dirty:
        raise RuntimeError("genriesz checkout is not clean: " + "; ".join(dirty))
    raw = _git(root, "cat-file", "blob", blobs[nb_rel], text=False)
    return {
        "sha": _git(root, "rev-parse", "HEAD").strip(),
        "notebook": nb_rel,
        "notebook_blob": blobs[nb_rel],
        "notebook_code_cells_sha256": hashlib.sha256(
            json.dumps(_code_cells(raw)).encode("utf-8")
        ).hexdigest(),
    }


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


def parent_state() -> dict:
    parent = GENRIESZ_ROOT.parent
    return {
        "sha": _git(parent, "rev-parse", "HEAD").strip(),
        "gitlink": _git(parent, "ls-tree", "HEAD", GENRIESZ_ROOT.name).split()[2],
        "dirty_tracked": bool(
            _git(parent, "status", "--porcelain", "--untracked-files=no").strip()
        ),
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
