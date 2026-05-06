"""genriesz

Generalized Riesz Regression (GRR) under Bregman divergences.

This package exposes a small, research-friendly API for:

- estimating linear functionals via generalized Riesz regression,
- common causal estimands (ATE / ATT / DID / AME),
- nearest-neighbor matching as LSIF/Riesz regression (and local-polynomial extensions),
- density ratio estimation (covariate shift) via generalized Bregman divergence minimization.

Public symbols are re-exported from submodules for convenience.
"""

from __future__ import annotations

__version__ = "0.2.5"

# High-level estimation
from .estimation import grr_ame, grr_ate, grr_att, grr_did, grr_functional

# Low-level solver (advanced)
from .glm import GRRGLM

# Density ratio and covariate shift
from .density_ratio import DensityRatioResult, fit_density_ratio

# Functionals
from .functionals import (
    AMEFunctional,
    ATEFunctional,
    ATTFunctional,
    DIDFunctional,
    CallableFunctional,
    LinearFunctional,
)

# Bases
from .basis import (
    BaseBasis,
    CallableBasis,
    GaussianRKHSBasis,
    KNNCatchmentBasis,
    PolynomialBasis,
    RBFRandomFourierBasis,
    RBFNystromBasis,
    TreatmentInteractionBasis,
)

# Generators
from .generators import BPGenerator, BKLGenerator, PUGenerator, BregmanGenerator, SquaredGenerator, UKLGenerator

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
    "GRRGLM",
    # Density ratio
    "fit_density_ratio",
    "DensityRatioResult",
    # Functionals
    "LinearFunctional",
    "CallableFunctional",
    "ATEFunctional",
    "ATTFunctional",
    "DIDFunctional",
    "AMEFunctional",
    # Bases
    "BaseBasis",
    "CallableBasis",
    "PolynomialBasis",
    "TreatmentInteractionBasis",
    "RBFRandomFourierBasis",
    "GaussianRKHSBasis",
    "RBFNystromBasis",
    "KNNCatchmentBasis",
    # Generators
    "BregmanGenerator",
    "SquaredGenerator",
    "UKLGenerator",
    "BKLGenerator",
    "BPGenerator",
    "PUGenerator",
    # Matching
    "nn_matching_inverse_propensity_weights",
    "local_polynomial_nn_lsif_density_ratio",
    "local_polynomial_nn_lsif_inverse_propensity_weights",
    # Results
    "FunctionalEstimate",
    "SingleEstimate",
]