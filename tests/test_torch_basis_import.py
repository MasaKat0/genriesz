"""torch_basis must degrade cleanly when PyTorch is absent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = """
import sys
sys.modules["torch"] = None  # simulate a missing optional dependency
sys.path.insert(0, {src!r})

import genriesz.torch_basis as tb  # must import without crashing

try:
    tb.MLPEmbeddingNet(input_dim=2)
except ImportError:
    print("IMPORTERROR-OK")
else:
    print("NO-ERROR")
"""


def test_torch_basis_imports_and_raises_importerror_without_torch():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(src=str(SRC))],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "IMPORTERROR-OK" in proc.stdout
