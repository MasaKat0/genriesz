"""Backward-compatibility re-exports.

Historically, the GLM-style GRR solver lived in :mod:`genriesz.glm` as
:class:`genriesz.glm.GRRGLM`.

The core implementation has since moved to :mod:`genriesz.grr` and the main class is
now :class:`genriesz.genriesz.GRR`.

This module re-exports the public symbols so that existing user code that
imports from ``genriesz.glm`` keeps working.
"""

from __future__ import annotations

from .grr import ACBLink, GRR, GRRGLM, run_grr_glm_acb

__all__ = [
    "GRR",
    "GRRGLM",
    "ACBLink",
    "run_grr_glm_acb",
]
