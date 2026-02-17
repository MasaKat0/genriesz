"""genriesz: Generalized Riesz Regression under Bregman divergences.

This package provides:

- A general GLM-style GRR solver (:class:`genriesz.genriesz.GRR`) with *automatic covariate
 - A general GLM-style GRR solver (:class:`genriesz.GRR`) with *automatic covariate
  balancing* via Bregman-generator-induced link functions.
- A high-level functional estimation interface (:func:`genriesz.grr_functional`)
  that can report DM, IPW, and AIPW estimates with cross-fitting, confidence intervals,
  and p-values.
- Basis functions (polynomial, RKHS-style random features, Nystrom, random forest leaves, ...)
  and common causal estimands (ATE, AME, policy effect).

See the README for usage examples.
"""

from .grr import ACBLink, GRR, GRRGLM, run_grr_glm_acb
from .estimate_functional import (
    FunctionalEstimateResult,
    LinearOutcomeModel,
    grr_ame,
    grr_ate,
    grr_functional,
    grr_policy_effect,
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
    TreatmentInteractionBasis,
)
from .knn_basis import KNNCatchmentBasis
from .functionals import (
    ATEFunctional,
    AverageDerivativeFunctional,
    PolicyEffectFunctional,
)

__all__ = [
    # Core GRR
    "GRR",
    "GRRGLM",
    "ACBLink",
    "run_grr_glm_acb",
    # Functional estimation API
    "grr_functional",
    "grr_ate",
    "grr_ame",
    "grr_policy_effect",
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
    "TreatmentInteractionBasis",
    "KNNCatchmentBasis",
    # Common functionals
    "ATEFunctional",
    "PolicyEffectFunctional",
    "AverageDerivativeFunctional",
]

__version__ = "0.1.9"
