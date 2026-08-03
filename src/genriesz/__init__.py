"""genriesz

Generalized Riesz Regression (GRR) under Bregman divergences.

This package exposes a small, research-friendly API for:

- estimating linear functionals via generalized Riesz regression,
- common causal estimands (ATE, ATT, DID, and AME),
- nearest-neighbor matching as LSIF/Riesz regression (and local-polynomial extensions),
- density ratio estimation (covariate shift) via generalized Bregman divergence minimization.

Public symbols are re-exported from submodules for convenience.
"""

from __future__ import annotations

__version__ = "0.3.0"

# High-level estimation
# Bases
from .basis import (
    BaseBasis,
    CallableBasis,
    GaussianRKHSBasis,
    KNNCatchmentBasis,
    PolynomialBasis,
    RBFNystromBasis,
    RBFRandomFourierBasis,
    TreatmentInteractionBasis,
)

# Density ratio and covariate shift
from .density_ratio import DensityRatioResult, fit_density_ratio
from .estimation import grr_ame, grr_ate, grr_att, grr_did, grr_functional

# Functionals
from .functionals import (
    AMEFunctional,
    ATEFunctional,
    ATTFunctional,
    CallableFunctional,
    DIDFunctional,
    LinearFunctional,
)

# Generators
from .generators import (
    BKLGenerator,
    BoundedBKLGenerator,
    BoundedUKLGenerator,
    BPGenerator,
    BregmanGenerator,
    DomainError,
    PUGenerator,
    SquaredGenerator,
    UKLGenerator,
)

# Low-level solver (advanced)
from .glm import GRRGLM, OutcomeGLM

# Matching (NN / local polynomial NN-LSIF)
from .matching import (
    local_polynomial_nn_lsif_density_ratio,
    local_polynomial_nn_lsif_inverse_propensity_weights,
    nn_matching_inverse_propensity_weights,
)

# Inner cross-validation for Riesz hyper-parameters
from .model_selection import GRRCVConfig, GRRCVResult, select_grr_hyperparams

# Results
from .results import FunctionalEstimate, SingleEstimate

# Diagnostics helpers (coverage-failure tables)
from .utils import bias_proxy, coverage_decomposition, oracle_decomposition


def load_scorematchingriesz():
    """Return the optional :mod:`genriesz.scorematchingriesz` module."""

    from importlib import import_module

    return import_module("genriesz.scorematchingriesz")

__all__ = [
    "__version__",
    # Estimation
    "grr_functional",
    "grr_ate",
    "grr_att",
    "grr_did",
    "grr_ame",
    "GRRGLM",
    "OutcomeGLM",
    # Density ratio
    "fit_density_ratio",
    "DensityRatioResult",
    # ScoreMatchingRiesz optional module
    "load_scorematchingriesz",
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
    "BoundedBKLGenerator",
    "BoundedUKLGenerator",
    "BPGenerator",
    "PUGenerator",
    "DomainError",
    # Matching
    "nn_matching_inverse_propensity_weights",
    "local_polynomial_nn_lsif_density_ratio",
    "local_polynomial_nn_lsif_inverse_propensity_weights",
    # Results
    "FunctionalEstimate",
    "SingleEstimate",
    # Diagnostics helpers
    "bias_proxy",
    "coverage_decomposition",
    "oracle_decomposition",
    # Model selection (inner CV)
    "GRRCVConfig",
    "GRRCVResult",
    "select_grr_hyperparams",
]