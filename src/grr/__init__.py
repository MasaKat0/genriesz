"""Generalized Riesz Regression (GRR).

This repository provides reference implementations used in the paper
"Riesz Representer Fitting under Bregman Divergence: A Unified Framework for
Debiased Machine Learning".

The public API is intentionally small:

- :class:`grr.GRR_ATE`: ATE estimation via GRR (DM / IPW / AIPW) with cross-fitting.
- :class:`grr.RKHS_GRR`: RKHS-based Riesz representer learner.
- :class:`grr.NN_GRR`: Neural-network-based Riesz representer learner (optional; requires PyTorch).
"""

from __future__ import annotations

from .grr import GRR_ATE
from .rkhs_grr import RKHS_GRR

try:
    # Optional dependency (PyTorch)
    from .nn_grr import NN_GRR
except ImportError:  # pragma: no cover
    NN_GRR = None  # type: ignore[assignment]

__all__ = [
    "GRR_ATE",
    "RKHS_GRR",
    "NN_GRR",
]
