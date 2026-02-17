"""genriesz: Generalized Riesz Regression under Bregman divergences.

This package provides:

- A general GLM-style GRR solver (:class:`genriesz.GRR`) with **automatic regressor
  balancing (ARB)** via Bregman-generator-induced link functions.
- A high-level functional estimation interface (:func:`genriesz.grr_functional`)
  that can report RA/RW/ARW/TMLE estimates with cross-fitting, confidence intervals,
  and p-values.
- Basis functions (polynomial, RKHS-style random features, Nyström, kNN catchment
  areas, random forest leaves, neural embeddings, ...).
- Common causal estimands (ATE, ATT, panel DID, AME).

See the README and the documentation for end-to-end examples.
"""

from .grr import ARBLink, GRR, GRRGLM, run_grr_glm_arb
from .estimate_functional import (
    FunctionalEstimateResult,
    LinearOutcomeModel,
    grr_ame,
    grr_ate,
    grr_att,
    grr_did,
    grr_functional,
)
from .bregman import (
    BregmanGenerator,
    BPGenerator,
    BKLGenerator,
    SquaredGenerator,
    UKLGenerator,
)
from .basis import (
    PolynomialBasis,
    RBFRandomFourierBasis,
    RBFNystromBasis,
    GaussianRKHSBasis,
    TreatmentInteractionBasis,
)
from .density_ratio import DensityRatioResult, grr_density_ratio
from .knn_basis import KNNCatchmentBasis
from .nnlsif import LocalPolynomialNNLSIF, local_polynomial_nnlsif_weights, nn_matching_weights
from .functionals import (
    ATEFunctional,
    ATTFunctional,
    AverageDerivativeFunctional,
)

__all__ = [
    # Core GRR
    "GRR",
    "GRRGLM",
    "ARBLink",
    "run_grr_glm_arb",
    # Functional estimation API
    "grr_functional",
    "grr_ate",
    "grr_att",
    "grr_did",
    "grr_ame",
    "grr_density_ratio",
    "DensityRatioResult",
    "FunctionalEstimateResult",
    "LinearOutcomeModel",
    # Generators
    "BregmanGenerator",
    "SquaredGenerator",
    "UKLGenerator",
    "BKLGenerator",
    "BPGenerator",
    # Bases
    "PolynomialBasis",
    "RBFRandomFourierBasis",
    "RBFNystromBasis",
    "GaussianRKHSBasis",
    "TreatmentInteractionBasis",
    "KNNCatchmentBasis",
    # Matching / NN-LSIF utilities
    "nn_matching_weights",
    "LocalPolynomialNNLSIF",
    "local_polynomial_nnlsif_weights",
    # Common functionals
    "ATEFunctional",
    "ATTFunctional",
    "AverageDerivativeFunctional",
]

__version__ = "0.2.6"
