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
    grr_functional,
    grr_policy_effect,
)

__all__ = [
    "grr_functional",
    "grr_ate",
    "grr_ame",
    "grr_policy_effect",
    "LinearOutcomeModel",
    "FunctionalEstimateResult",
]
