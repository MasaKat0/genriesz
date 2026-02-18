"""genriesz

Generalized Riesz Regression (GRR) and applications to causal inference.

The package provides:

- General-purpose estimation of linear functionals via :func:`genriesz.grr_functional`.
- Convenience wrappers for common causal estimands: ATE / ATT / DID / AME.
- Built-in Bregman generators (SQ / UKL / BP) that induce **Automatic Regressor Balancing (ARB)**
  when paired with the canonical link.
- A small basis/feature-map library, including polynomial features and optional
  random Fourier features / tree leaves / neural embeddings.

Public API is re-exported here for convenience.
"""

from __future__ import annotations

__version__ = "0.2.0"

# High-level estimation
from .estimation import grr_ame, grr_ate, grr_att, grr_did, grr_functional

# Functionals
from .functionals import AMEFunctional, ATEFunctional, ATTFunctional, DIDFunctional, LinearFunctional

# Bases
from .basis import BaseBasis, KNNCatchmentBasis, PolynomialBasis, RBFRandomFourierBasis, TreatmentInteractionBasis

# Generators
from .generators import BPGenerator, BregmanGenerator, SquaredGenerator, UKLGenerator

# Matching (NN / local polynomial NN-LSIF)
from .matching import (
    local_polynomial_nn_lsif_density_ratio,
    local_polynomial_nn_lsif_inverse_propensity_weights,
    nn_matching_inverse_propensity_weights,
)

# Results
from .results import FunctionalEstimate, SingleEstimate

__all__ = [
    "__version__",
    # Estimation
    "grr_functional",
    "grr_ate",
    "grr_att",
    "grr_did",
    "grr_ame",
    # Functionals
    "LinearFunctional",
    "ATEFunctional",
    "ATTFunctional",
    "DIDFunctional",
    "AMEFunctional",
    # Bases
    "BaseBasis",
    "PolynomialBasis",
    "TreatmentInteractionBasis",
    "RBFRandomFourierBasis",
    "KNNCatchmentBasis",
    # Generators
    "BregmanGenerator",
    "SquaredGenerator",
    "UKLGenerator",
    "BPGenerator",
    # Matching
    "nn_matching_inverse_propensity_weights",
    "local_polynomial_nn_lsif_density_ratio",
    "local_polynomial_nn_lsif_inverse_propensity_weights",
    # Results
    "FunctionalEstimate",
    "SingleEstimate",
]
