"""Convenience re-exports for a DML-style workflow.

Historically this package exposed a small DML-style API in :mod:`genriesz.dml`.

The recommended entry point is :func:`genriesz.grr_functional`.

This module is kept as a lightweight re-export so that users can write
``from genriesz.dml import grr_functional`` if they prefer.
"""

from __future__ import annotations

from .estimate_functional import (
    FunctionalEstimateResult,
    LinearOutcomeModel,
    grr_ame,
    grr_ate,
    grr_att,
    grr_did,
    grr_functional,
)

__all__ = [
    "grr_functional",
    "grr_ate",
    "grr_att",
    "grr_did",
    "grr_ame",
    "LinearOutcomeModel",
    "FunctionalEstimateResult",
]
