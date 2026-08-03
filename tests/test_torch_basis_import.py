"""torch_basis must propagate a missing optional PyTorch dependency."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = """
import importlib.abc
import sys

sys.path.insert(0, {src!r})
import genriesz  # load required dependencies before blocking the optional one


class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("PyTorch blocked for optional-dependency test")
        return None


sys.meta_path.insert(0, BlockTorch())
import genriesz.torch_basis as tb

tb.MLPEmbeddingNet(input_dim=2)
"""


def test_torch_basis_import_propagates_missing_pytorch_dependency():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(src=str(SRC))],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "ModuleNotFoundError" in proc.stderr
    assert "PyTorch blocked for optional-dependency test" in proc.stderr
