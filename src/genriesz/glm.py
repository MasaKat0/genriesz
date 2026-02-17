"""Backward-compatibility re-exports.

Historically, the GLM-style GRR solver lived in :mod:`genriesz.glm` as
:class:`genriesz.glm.GRRGLM`.

The core implementation has since moved to :mod:`genriesz.grr` and the main class is
now :class:`genriesz.GRR`.

This module re-exports the public symbols so that existing user code that
imports from ``genriesz.glm`` keeps working.
"""

from __future__ import annotations

from .grr import ARBLink, GRR, GRRGLM, run_grr_glm_arb

__all__ = [
    "GRR",
    "GRRGLM",
    "ARBLink",
    "run_grr_glm_arb",
]
