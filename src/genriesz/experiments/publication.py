"""Estimators and data-generating processes used by the manuscript notebooks.

The default UKL and BKL specifications are truncated models whose links
saturate at representer bounds stated before fitting (see
:func:`make_compatible_generator`); the squared and BP links are exact. The
functions in this module do not rewrite a fitted representer, replace a
failed candidate, download substitute data, or continue after an unexpected
programming error.  Expected numerical outcomes are returned through explicit
status fields.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from genriesz.basis import (
    BaseBasis,
    GaussianRKHSBasis,
    PolynomialBasis,
    RBFRandomFourierBasis,
    TreatmentInteractionBasis,
)
from genriesz.functionals import ATEFunctional, ATTFunctional, LinearFunctional
from genriesz.generators import (
    BKLGenerator,
    BoundedBKLGenerator,
    BoundedUKLGenerator,
    BPGenerator,
    BregmanGenerator,
    SquaredGenerator,
)
from genriesz.glm import GRRGLM, OutcomeGLM
from genriesz.matching import nn_matching_inverse_propensity_weights
from genriesz.sklearn_basis import RandomForestLeafBasis
from genriesz.utils import Fold, stratified_kfold_splits

RANDOM_SEED = 20260211
TREATMENT_INDEX = 0
ESTIMANDS = ("ATE", "ATT")
ESTIMATORS_ALL = ("ra", "rw", "arw", "tmle")
COMPATIBLE_LOSSES = (
    {"loss": "SQ", "omega": None, "label": "SQ"},
    {"loss": "UKL", "omega": None, "label": "UKL"},
    {"loss": "BKL", "omega": None, "label": "BKL"},
    {"loss": "BP", "omega": 0.5, "label": "BP(0.5)"},
)
DATA_DIR = Path(__file__).resolve().parents[3] / "notebooks" / "experiments" / "data"

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExperimentFailure:
    """Expected failure of an experiment specification."""

    status: str
    message: str
    fold: int | None = None


@dataclass(frozen=True)
class FunctionalFit:
    """Cross-fitted estimates and diagnostics for one specification."""

    estimates: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]
    failure: ExperimentFailure | None


class ZOnlyBasis(BaseBasis):
    """Lift a covariate basis to ``X=(D,Z)`` by removing treatment."""

    def __init__(self, base_basis: BaseBasis, treatment_index: int = 0):
        self.base_basis = base_basis
        self.treatment_index = int(treatment_index)
        self._n_features: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> ZOnlyBasis:
        X_ = np.asarray(X, dtype=float)
        Z = np.delete(X_, self.treatment_index, axis=1)
        self.base_basis.fit(Z, y=y)
        self._n_features = int(self.base_basis.n_features)
        return self

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            raise RuntimeError("ZOnlyBasis must be fit before n_features is available.")
        return self._n_features

    def __call__(self, X: ArrayLike) -> FloatArray:
        X_ = np.asarray(X, dtype=float)
        single = X_.ndim == 1
        if single:
            X_ = X_.reshape(1, -1)
        if self._n_features is None:
            raise RuntimeError("ZOnlyBasis must be fit before prediction.")
        Z = np.delete(X_, self.treatment_index, axis=1)
        Phi = np.asarray(self.base_basis(Z), dtype=float)
        return Phi[0] if single else Phi

    def derivative(self, X: ArrayLike, coordinate: int) -> FloatArray:
        X_ = np.asarray(X, dtype=float)
        single = X_.ndim == 1
        if single:
            X_ = X_.reshape(1, -1)
        if int(coordinate) == self.treatment_index:
            out = np.zeros((X_.shape[0], self.n_features), dtype=float)
            return out[0] if single else out
        z_coordinate = int(coordinate) - int(int(coordinate) > self.treatment_index)
        Z = np.delete(X_, self.treatment_index, axis=1)
        out = np.asarray(self.base_basis.derivative(Z, z_coordinate), dtype=float)
        return out[0] if single else out


class SelectedColumnsBasis(BaseBasis):
    """Treatment-interaction basis using the first selected covariates."""

    def __init__(self, n_active: int, treatment_index: int = 0):
        self.n_active = int(n_active)
        self.treatment_index = int(treatment_index)
        self._basis = TreatmentInteractionBasis(
            PolynomialBasis(degree=1, include_bias=True),
            treatment_index=self.treatment_index,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> SelectedColumnsBasis:
        X_ = np.asarray(X, dtype=float)
        self._basis.fit(self._subset(X_), y=y)
        return self

    @property
    def n_features(self) -> int:
        return int(self._basis.n_features)

    def _subset(self, X: FloatArray) -> FloatArray:
        D = X[:, [self.treatment_index]]
        Z = np.delete(X, self.treatment_index, axis=1)
        return np.column_stack((D, Z[:, : self.n_active]))

    def __call__(self, X: ArrayLike) -> FloatArray:
        X_ = np.asarray(X, dtype=float)
        single = X_.ndim == 1
        if single:
            X_ = X_.reshape(1, -1)
        out = np.asarray(self._basis(self._subset(X_)), dtype=float)
        return out[0] if single else out


def branch_treated(x: FloatArray) -> int:
    return 1 if float(np.asarray(x, dtype=float).reshape(-1)[TREATMENT_INDEX]) >= 0.5 else -1


def make_base_basis(
    kind: str,
    *,
    seed: int = 0,
    n_features: int = 120,
    degree: int = 2,
    sigma: str | float = "auto",
) -> BaseBasis:
    kind_ = str(kind).lower()
    if kind_ == "polynomial":
        return PolynomialBasis(degree=degree, include_bias=True)
    if kind_ == "rkhs":
        return GaussianRKHSBasis(
            n_centers=n_features,
            sigma=sigma,
            include_bias=True,
            standardize=True,
            random_state=seed,
        )
    if kind_ == "rff":
        return RBFRandomFourierBasis(
            n_features=n_features,
            sigma=sigma,
            include_bias=True,
            standardize=True,
            random_state=seed,
        )
    if kind_ == "rf":
        forest = RandomForestRegressor(
            n_estimators=max(50, n_features // 2),
            max_depth=4,
            min_samples_leaf=10,
            random_state=seed,
            n_jobs=1,
        )
        return RandomForestLeafBasis(forest, include_bias=True, normalize=True)
    raise ValueError(f"Unknown basis kind: {kind}")


def make_basis(
    kind: str,
    *,
    mode: str = "regressor",
    seed: int = 0,
    n_features: int = 120,
    degree: int = 2,
    sigma: str | float = "auto",
) -> BaseBasis:
    base = make_base_basis(
        kind,
        seed=seed,
        n_features=n_features,
        degree=degree,
        sigma=sigma,
    )
    if mode == "regressor":
        return TreatmentInteractionBasis(base_basis=base, treatment_index=TREATMENT_INDEX)
    if mode == "covariate":
        return ZOnlyBasis(base, treatment_index=TREATMENT_INDEX)
    raise ValueError("mode must be 'regressor' or 'covariate'")


#: Propensity window stated by the default truncated UKL model for ATE. The
#: arm-wise representer magnitudes 1/e and 1/(1-e) then lie in
#: [1/UKL_ATE_E_MAX, 1/UKL_ATE_E_MIN].
UKL_ATE_E_MIN = 0.01
UKL_ATE_E_MAX = 0.99
#: Representer-magnitude cap stated by the default truncated UKL model for
#: ATT (same cap as the ATE window's upper end 1/UKL_ATE_E_MIN). The control
#: branch can approach zero for ATT, so only the upper bound is a model
#: choice; the lower clamp stays at the float64 representability floor.
UKL_ATT_ALPHA_MAX = 100.0
#: Representer-magnitude cap stated by the default truncated BKL model.
BKL_ALPHA_MAX = 50.0


def generator_shift_for_estimand(loss: str, estimand: str) -> float:
    loss_ = str(loss).upper()
    estimand_ = str(estimand).upper()
    if loss_ == "SQ":
        return 0.0
    if estimand_ == "ATE":
        return 1.0
    if estimand_ == "ATT" and loss_ in {"UKL", "BP"}:
        return 0.0
    if estimand_ == "ATT" and loss_ == "BKL":
        # The BKL generator needs C > 0; the ATT control branch can approach
        # zero, so the shift is small and the truncated link's lower clamp
        # (the float64 representability floor above C) absorbs the tail.
        return 0.05
    raise ValueError(f"Unknown loss or estimand: {loss}, {estimand}")


def make_compatible_generator(
    loss: str,
    *,
    estimand: str,
    omega: float | None = None,
) -> BregmanGenerator:
    """Build the default generator for a loss/estimand pair.

    UKL and BKL are fitted with their truncated links
    (:class:`~genriesz.BoundedUKLGenerator` / :class:`~genriesz.BoundedBKLGenerator`):
    the exact links diverge as the dual index approaches the domain boundary,
    and the stated bounds absorb that numerical instability as part of the
    model. The bounds are fixed constants of the experimental design
    (module-level ``UKL_ATE_E_MIN``/``UKL_ATE_E_MAX``, ``UKL_ATT_ALPHA_MAX``
    and ``BKL_ALPHA_MAX``); binding rates surface in the result diagnostics.
    SQ and BP keep their exact links: SQ is defined on the whole line, and
    BP's restricted dual domain is enforced by explicit linear constraints
    during fitting.
    """

    loss_ = str(loss).upper()
    estimand_ = str(estimand).upper()
    C = generator_shift_for_estimand(loss_, estimand_)
    if loss_ == "SQ":
        return SquaredGenerator(C=C)
    if loss_ == "UKL":
        if estimand_ == "ATE":
            return BoundedUKLGenerator.from_propensity_bounds(
                UKL_ATE_E_MIN,
                UKL_ATE_E_MAX,
                C=C,
                branch_fn=branch_treated,
            )
        return BoundedUKLGenerator(
            C=C,
            alpha_max=UKL_ATT_ALPHA_MAX,
            branch_fn=branch_treated,
        )
    if loss_ == "BKL":
        return BoundedBKLGenerator(
            C=C,
            alpha_max=BKL_ALPHA_MAX,
            branch_fn=branch_treated,
        )
    if loss_ == "BP":
        return BPGenerator(
            C=C,
            omega=0.5 if omega is None else float(omega),
            branch_fn=branch_treated,
        )
    raise ValueError(f"Unknown loss: {loss}")


def _fit_linear_link_ukl(
    Phi: FloatArray,
    M: FloatArray,
    sign: FloatArray,
    *,
    C: float,
    lam: float,
    max_iter: int,
) -> tuple[FloatArray, str]:
    """Fit the deliberately mismatched pair: UKL loss under the linear link.

    The representer model is the linear link ``alpha = Phi beta`` while the
    loss stays UKL, so the fit minimizes the primal Bregman--Riesz objective

        mean(g_UKL(alpha)) - mean(M beta) + (lam / 2) * ||beta||^2

    on the branchwise UKL domain ``sign * alpha > C``, imposed as explicit
    linear constraints (the same mechanism the exact solvers use). The UKL
    dual coordinate is not linear in this model, so the balance equations of
    the compatible pairs fail by construction; that failure is what the
    incompatible comparison measures. An infeasible or non-converged fit is
    reported as a failure and is never replaced by another value.
    """

    A = sign[:, None] * Phi
    m_bar = M.mean(axis=0)
    margin = 1e-6
    floor = C + margin

    target = np.full(A.shape[0], C + 1.0)
    beta0 = np.linalg.lstsq(A, target, rcond=None)[0]
    if not np.all(A @ beta0 >= floor):
        phase = optimize.linprog(
            np.zeros(A.shape[1], dtype=float),
            A_ub=-A,
            b_ub=np.full(A.shape[0], -floor),
            bounds=[(None, None)] * A.shape[1],
            method="highs",
        )
        if not phase.success or phase.x is None:
            return np.full(A.shape[1], np.nan), "linear_link_domain_infeasible"
        beta0 = np.asarray(phase.x, dtype=float)

    def objective(beta: FloatArray) -> float:
        u = A @ beta
        t = u - C
        if np.any(t <= 0.0):
            return float("inf")
        return float(
            np.mean(t * np.log(t) - u) - float(m_bar @ beta) + 0.5 * lam * float(beta @ beta)
        )

    def gradient(beta: FloatArray) -> FloatArray:
        t = (A @ beta) - C
        return (np.log(t)[:, None] * A).mean(axis=0) - m_bar + lam * beta

    result = optimize.minimize(
        objective,
        beta0,
        jac=gradient,
        method="SLSQP",
        constraints=(optimize.LinearConstraint(A, lb=np.full(A.shape[0], floor)),),
        options={"maxiter": int(max_iter), "ftol": 1e-10},
    )
    beta_hat = np.asarray(result.x, dtype=float)
    if not (
        bool(result.success)
        and np.all(np.isfinite(beta_hat))
        and np.all(A @ beta_hat > C)
        and np.isfinite(objective(beta_hat))
    ):
        return beta_hat, "optimizer_failure"
    grad_hat = gradient(beta_hat)
    slack = (A @ beta_hat) - floor
    active = A[slack <= 1e-8 * max(1.0, abs(floor))]
    if active.shape[0]:
        multipliers, _ = optimize.nnls(active.T, grad_hat)
        kkt_residual = float(np.max(np.abs(grad_hat - active.T @ multipliers)))
    else:
        kkt_residual = float(np.max(np.abs(grad_hat)))
    if not (np.isfinite(kkt_residual) and kkt_residual <= 1e-2):
        return beta_hat, "kkt_failure"
    return beta_hat, "converged"


def standardize_columns(Z: ArrayLike) -> FloatArray:
    Z_ = np.asarray(Z, dtype=float)
    mean = Z_.mean(axis=0)
    sd = Z_.std(axis=0, ddof=0)
    sd = np.where(sd > 0.0, sd, 1.0)
    return (Z_ - mean) / sd


#: Fixed large draw used to compute population estimands. The influence-function
#: standard errors target the population ATE/ATT, so the stored truth must be a
#: population value that does not vary across Monte Carlo replications.
POPULATION_TRUTH_DRAW = 1_000_000
POPULATION_TRUTH_SEED = 20_260_803
_population_truth_cache: dict[tuple, tuple[float, float]] = {}


def _population_thetas(key: tuple, draw) -> tuple[float, float]:
    if key not in _population_truth_cache:
        data = draw()
        tau = np.asarray(data["tau"], dtype=float)
        e = np.asarray(data["e"], dtype=float)
        theta_ate = float(np.mean(tau))
        theta_att = float(np.mean(e * tau) / np.mean(e))
        _population_truth_cache[key] = (theta_ate, theta_att)
    return _population_truth_cache[key]


def make_simulation_data(
    dgp: str, *, n: int, seed: int, _population: bool = True
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if dgp == "DGP 1: smooth heterogeneous effects":
        d = 6
        Z = rng.normal(size=(n, d))
        logits = 0.6 * Z[:, 0] - 0.4 * Z[:, 1] + 0.25 * np.sin(Z[:, 2])
        noise_sd = 1.0
        tau = 1.0 + 0.45 * Z[:, 0] + 0.25 * np.sin(Z[:, 2])
        mu0 = 0.5 * Z[:, 0] + 0.4 * np.sin(Z[:, 1]) + 0.2 * Z[:, 2] ** 2
    elif dgp == "DGP 2: weak overlap nonlinear design":
        d = 8
        Z = rng.normal(size=(n, d))
        logits = (
            1.10 * Z[:, 0]
            - 0.90 * Z[:, 1]
            + 0.45 * np.sin(Z[:, 2])
            + 0.25 * np.tanh(Z[:, 3] * Z[:, 4])
        )
        noise_sd = 1.10
        tau = 1.0 + 0.30 * Z[:, 0] - 0.20 * Z[:, 1] + 0.20 * np.sin(Z[:, 3])
        mu0 = 0.35 * Z[:, 0] + 0.20 * np.tanh(Z[:, 1] ** 2) + 0.35 * np.sin(Z[:, 2])
    elif dgp == "DGP 3: high-dimensional sparse confounding":
        d = 20
        idx = np.arange(d)
        Sigma = 0.5 ** np.abs(idx[:, None] - idx[None, :])
        Z = rng.multivariate_normal(np.zeros(d), Sigma, size=n)
        logits = (
            0.75 * Z[:, 0]
            - 0.65 * Z[:, 1]
            + 0.45 * Z[:, 2]
            - 0.35 * Z[:, 3]
            + 0.25 * np.sin(Z[:, 4])
        )
        noise_sd = 1.0
        tau = 1.0 + 0.30 * Z[:, 0] + 0.30 * np.sin(Z[:, 1]) + 0.20 * Z[:, 2] * Z[:, 3]
        mu0 = Z[:, :8] @ np.array([0.5, -0.4, 0.35, 0.25, -0.20, 0.15, 0.10, -0.10])
        mu0 = mu0 + 0.2 * np.sin(Z[:, 8])
    else:
        raise ValueError(dgp)
    # The bounds are part of the data-generating process used in the original
    # notebooks. They are not post-fit weight caps or numerical substitutes.
    e = np.clip(expit(logits), 0.05, 0.95)
    D = rng.binomial(1, e).astype(float)
    if not (np.any(D == 1.0) and np.any(D == 0.0)):
        raise RuntimeError("The generated sample does not contain both treatment groups.")
    Y0 = mu0 + rng.normal(0.0, noise_sd, size=n)
    Y1 = mu0 + tau + rng.normal(0.0, noise_sd, size=n)
    Y = D * Y1 + (1.0 - D) * Y0
    if _population:
        theta_ate, theta_att = _population_thetas(
            ("make_simulation_data", dgp),
            lambda: make_simulation_data(
                dgp, n=POPULATION_TRUTH_DRAW, seed=POPULATION_TRUTH_SEED, _population=False
            ),
        )
    else:
        theta_ate = float(np.mean(tau))
        theta_att = float(np.mean(tau[D == 1]))
    return {
        "X": np.column_stack((D, Z)),
        "Y": Y,
        "D": D,
        "Z": Z,
        "mu0": mu0,
        "mu1": mu0 + tau,
        "tau": tau,
        "e": e,
        "theta_ate": theta_ate,
        "theta_att": theta_att,
    }


def make_dimension_data(
    *, n: int, d: int, seed: int, _population: bool = True
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    logits = 0.8 * Z[:, 0] - 0.6 * Z[:, 1] + 0.35 * np.sin(Z[:, 2])
    e = np.clip(expit(logits), 0.03, 0.97)
    D = rng.binomial(1, e).astype(float)
    if not (np.any(D == 1.0) and np.any(D == 0.0)):
        raise RuntimeError("The generated sample does not contain both treatment groups.")
    tau = 1.0 + 0.35 * Z[:, 0] + 0.25 * np.sin(Z[:, 1])
    beta = np.zeros(d)
    values = np.array([0.5, -0.4, 0.35, 0.25, -0.2, 0.15, 0.1, -0.1])
    beta[: min(8, d)] = values[: min(8, d)]
    mu0 = Z @ beta
    Y = mu0 + D * tau + rng.normal(scale=1.0, size=n)
    if _population:
        theta_ate, theta_att = _population_thetas(
            ("make_dimension_data", d),
            lambda: make_dimension_data(
                n=POPULATION_TRUTH_DRAW, d=d, seed=POPULATION_TRUTH_SEED, _population=False
            ),
        )
    else:
        theta_ate = float(np.mean(tau))
        theta_att = float(np.mean(tau[D == 1]))
    return {
        "X": np.column_stack((D, Z)),
        "Y": Y,
        "D": D,
        "Z": Z,
        "tau": tau,
        "e": e,
        "theta_ate": theta_ate,
        "theta_att": theta_att,
    }


class CoverageDiagnosticBasis(BaseBasis):
    """Treatment-specific dictionary for the coverage diagnostic design."""

    def __init__(self, *, include_quadratic: bool):
        self.include_quadratic = bool(include_quadratic)
        self._n_features = 8 if self.include_quadratic else 6

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> CoverageDiagnosticBasis:
        del X, y
        return self

    @property
    def n_features(self) -> int:
        return self._n_features

    def __call__(self, X: ArrayLike) -> FloatArray:
        X_ = np.asarray(X, dtype=float)
        single = X_.ndim == 1
        if single:
            X_ = X_.reshape(1, -1)
        D = X_[:, [0]]
        Z = X_[:, 1:]
        base_parts = [np.ones(X_.shape[0]), Z[:, 0], Z[:, 1]]
        if self.include_quadratic:
            base_parts.append(Z[:, 1] ** 2 - 1.0)
        base = np.column_stack(base_parts)
        out = np.column_stack((D * base, (1.0 - D) * base))
        return out[0] if single else out


def make_coverage_diagnostic_data(
    *,
    n: int,
    seed: int,
    overlap_scale: float,
) -> dict[str, Any]:
    """Data-generating process for the coverage decomposition experiment."""

    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    index = Z[:, 0] - 0.5 * Z[:, 1] + 0.6 * (Z[:, 1] ** 2 - 1.0)
    e = expit(float(overlap_scale) * index)
    D = rng.binomial(1, e).astype(float)
    if not (np.any(D == 1.0) and np.any(D == 0.0)):
        raise RuntimeError("The generated sample does not contain both treatment groups.")
    mu0 = Z[:, 0] + 0.5 * (Z[:, 1] ** 2 - 1.0)
    tau = np.ones(n)
    Y = mu0 + D * tau + rng.normal(size=n)
    return {
        "X": np.column_stack((D, Z)),
        "Y": Y,
        "D": D,
        "Z": Z,
        "e": e,
        "tau": tau,
        "theta_ate": 1.0,
        "theta_att": 1.0,
    }


def make_score_guided_data(
    *, n: int, seed: int, heterogeneous: bool = True, _population: bool = True
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    e = expit(0.5 * Z[:, 0] - 0.4 * Z[:, 1] + 0.2 * np.sin(Z[:, 2]))
    D = rng.binomial(1, e).astype(float)
    if not (np.any(D == 1.0) and np.any(D == 0.0)):
        raise RuntimeError("The generated sample does not contain both treatment groups.")
    psi = np.column_stack(
        (np.sin(Z[:, 0]), np.cos(Z[:, 1]), Z[:, 0] * Z[:, 2], Z[:, 1] ** 2, np.sin(Z[:, 2]))
    )
    beta0 = np.array([0.4, -0.3, 0.2, 0.1, 0.2])
    betat = np.array([0.5, 0.2, -0.2, 0.15, 0.2]) if heterogeneous else np.zeros(5)
    mu0 = psi @ beta0
    tau = 1.0 + psi @ betat
    Y = mu0 + D * tau + rng.normal(scale=0.1, size=n)
    if _population:
        theta_ate, theta_att = _population_thetas(
            ("make_score_guided_data", bool(heterogeneous)),
            lambda: make_score_guided_data(
                n=POPULATION_TRUTH_DRAW,
                seed=POPULATION_TRUTH_SEED,
                heterogeneous=heterogeneous,
                _population=False,
            ),
        )
    else:
        theta_ate = float(np.mean(tau))
        theta_att = float(np.mean(tau[D == 1]))
    return {
        "X": np.column_stack((D, Z)),
        "Y": Y,
        "D": D,
        "Z": Z,
        "tau": tau,
        "e": e,
        "theta_ate": theta_ate,
        "theta_att": theta_att,
    }


def make_kang_schafer_data(*, n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 4))
    logits = -Z[:, 0] + 0.5 * Z[:, 1] - 0.25 * Z[:, 2] - 0.1 * Z[:, 3]
    e = expit(logits)
    D = rng.binomial(1, e).astype(float)
    if not (np.any(D == 1.0) and np.any(D == 0.0)):
        raise RuntimeError("The generated sample does not contain both treatment groups.")
    X1 = np.exp(Z[:, 0] / 2.0)
    X2 = Z[:, 1] / (1.0 + np.exp(Z[:, 0])) + 10.0
    X3 = (Z[:, 0] * Z[:, 2] / 25.0 + 0.6) ** 3
    X4 = (Z[:, 1] + Z[:, 3] + 20.0) ** 2
    features = standardize_columns(
        np.column_stack((X1, X2, X3, X4, X1**2, X2**2, X3**2, X4**2))
    )
    mu0 = 210.0 + 27.4 * Z[:, 0] + 13.7 * Z[:, 1] + 13.7 * Z[:, 2] + 13.7 * Z[:, 3]
    tau = np.ones(n)
    Y = mu0 + D * tau + rng.normal(scale=1.0, size=n)
    return {
        "X": np.column_stack((D, features)),
        "Y": Y,
        "D": D,
        "Z": features,
        "theta_ate": 1.0,
        "theta_att": 1.0,
        "e": e,
    }


def make_kernel_gp_data(*, n: int, d: int, seed: int, outcome_kernel: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    logits = 0.7 * Z[:, 0] - 0.4 * Z[:, 1] + 0.25 * np.sin(Z[:, 2])
    e = np.clip(expit(logits), 0.03, 0.97)
    D = rng.binomial(1, e).astype(float)
    if not (np.any(D == 1.0) and np.any(D == 0.0)):
        raise RuntimeError("The generated sample does not contain both treatment groups.")
    if outcome_kernel == "polynomial":
        mu0 = 0.4 * Z[:, 0] + 0.35 * Z[:, 1] ** 2 - 0.25 * Z[:, 2] * Z[:, 3]
    elif outcome_kernel == "sinusoidal":
        mu0 = 0.5 * np.sin(Z[:, 0]) + 0.4 * np.cos(Z[:, 1])
        mu0 = mu0 + 0.25 * np.sin(Z[:, 2] * Z[:, 3])
    else:
        raise ValueError(outcome_kernel)
    tau = np.ones(n)
    Y = mu0 + D * tau + rng.normal(scale=1.0, size=n)
    return {
        "X": np.column_stack((D, Z)),
        "Y": Y,
        "D": D,
        "Z": Z,
        "tau": tau,
        "theta_ate": 1.0,
        "theta_att": 1.0,
    }


def true_theta(data: dict[str, Any], estimand: str) -> float | None:
    key = "theta_ate" if str(estimand).upper() == "ATE" else "theta_att"
    return None if key not in data else float(data[key])


def load_ihdp_replication(
    replication: int,
    *,
    data_dir: Path = DATA_DIR / "ihdp",
) -> dict[str, Any]:
    train_path = data_dir / "ihdp_npci_1-100.train.npz"
    test_path = data_dir / "ihdp_npci_1-100.test.npz"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "IHDP files are missing. Place ihdp_npci_1-100.train.npz and "
            f"ihdp_npci_1-100.test.npz under {data_dir}."
        )
    train = np.load(train_path)
    test = np.load(test_path)
    j = int(replication) - 1
    n_reps = int(train["x"].shape[2])
    if j < 0 or j >= n_reps:
        raise ValueError(f"replication must be between 1 and {n_reps}")
    Xcov = np.vstack((train["x"][:, :, j], test["x"][:, :, j]))
    D = np.concatenate((train["t"][:, j], test["t"][:, j])).astype(float)
    Y = np.concatenate((train["yf"][:, j], test["yf"][:, j])).astype(float)
    mu1 = np.concatenate((train["mu1"][:, j], test["mu1"][:, j])).astype(float)
    mu0 = np.concatenate((train["mu0"][:, j], test["mu0"][:, j])).astype(float)
    tau = mu1 - mu0
    return {
        "X": np.column_stack((D, Xcov)),
        "Y": Y,
        "D": D,
        "Z": Xcov,
        "mu1": mu1,
        "mu0": mu0,
        "tau": tau,
        "theta_ate": float(np.mean(tau)),
        "theta_att": float(np.mean(tau[D == 1])),
        "replication": int(replication),
        "source": train_path.name,
    }


def load_lalonde(*, data_dir: Path = DATA_DIR / "lalonde") -> dict[str, Any]:
    csv_path = data_dir / "lalonde.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Lalonde data are missing: {csv_path}")
    df = pd.read_csv(csv_path)
    for id_col in ("rownames", "ID"):
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
    if "race" in df.columns:
        race = pd.get_dummies(df["race"], prefix="race", drop_first=False, dtype=float)
        df = pd.concat((df.drop(columns=["race"]), race), axis=1)
    required = ("treat", "age", "educ", "married", "nodegree", "re74", "re75", "re78")
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Lalonde data are missing columns: {missing}")
    D = df["treat"].to_numpy(dtype=float)
    Y = df["re78"].to_numpy(dtype=float)
    covariate_names = [name for name in df.columns if name not in {"treat", "re78"}]
    Z = df[covariate_names].to_numpy(dtype=float)
    return {
        "X": np.column_stack((D, Z)),
        "Y": Y,
        "D": D,
        "Z": Z,
        "covariate_names": covariate_names,
        "source": str(csv_path),
    }


def _functional(estimand: str, D: FloatArray) -> LinearFunctional:
    if str(estimand).upper() == "ATE":
        return ATEFunctional(treatment_index=TREATMENT_INDEX)
    if str(estimand).upper() == "ATT":
        pi = float(np.mean(D))
        if not 0.0 < pi < 1.0:
            raise ValueError("ATT requires both treated and control observations.")
        return ATTFunctional(
            treatment_index=TREATMENT_INDEX,
            pi=pi,
            pi_is_estimated=True,
        )
    raise ValueError(f"Unknown estimand: {estimand}")


def _folds(D: FloatArray, *, cross_fit: bool, folds: int, random_state: int) -> list[Fold]:
    if cross_fit:
        return list(stratified_kfold_splits(D, folds=folds, random_state=random_state))
    idx = np.arange(D.shape[0])
    return [Fold(train=idx, test=idx)]


def _counterfactuals(X: FloatArray) -> tuple[FloatArray, FloatArray]:
    X1 = X.copy()
    X0 = X.copy()
    X1[:, TREATMENT_INDEX] = 1.0
    X0[:, TREATMENT_INDEX] = 0.0
    return X1, X0


def _m_values(
    functional: LinearFunctional,
    X: FloatArray,
    values1: FloatArray,
    values0: FloatArray,
) -> FloatArray:
    if isinstance(functional, ATEFunctional):
        return values1 - values0
    if isinstance(functional, ATTFunctional):
        D = X[:, functional.treatment_index]
        return (D / functional.pi) * (values1 - values0)
    raise TypeError(type(functional).__name__)


def _bkl_initial_beta(basis: BaseBasis, X: FloatArray) -> FloatArray:
    Phi = np.asarray(basis(X), dtype=float)
    sign = np.where(X[:, TREATMENT_INDEX] >= 0.5, 1.0, -1.0)
    target = -2.0 * sign
    ridge = 1e-6 * np.eye(Phi.shape[1])
    return np.linalg.solve(Phi.T @ Phi + ridge, Phi.T @ target)


def _failure_rows(
    label: dict[str, Any],
    failure: ExperimentFailure,
    true_value: float | None,
) -> list[dict[str, Any]]:
    row = dict(label)
    row.update(
        {
            "estimator": "failed",
            "estimate": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "status": failure.status,
            "message": failure.message,
            "failure_fold": failure.fold,
        }
    )
    if true_value is not None:
        row["true_value"] = true_value
    return [row]


def _weighted_mean(values: FloatArray, weights: FloatArray) -> float:
    denom = float(np.sum(weights))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / denom)


def _effective_sample_size(weights: FloatArray) -> float:
    weights_ = np.asarray(weights, dtype=float)
    total = float(np.sum(weights_))
    squares = float(np.sum(weights_ * weights_))
    if not np.isfinite(total) or not np.isfinite(squares) or total <= 0.0 or squares <= 0.0:
        return float("nan")
    return total * total / squares


def _balance_table(Z: FloatArray, D: FloatArray, alpha: FloatArray, estimand: str) -> pd.DataFrame:
    treated = D == 1.0
    control = D == 0.0
    var_t = np.var(Z[treated], axis=0, ddof=1)
    var_c = np.var(Z[control], axis=0, ddof=1)
    pooled = np.sqrt(0.5 * (var_t + var_c))
    pooled = np.where(pooled > 0.0, pooled, np.nan)
    unweighted = (np.mean(Z[treated], axis=0) - np.mean(Z[control], axis=0)) / pooled
    if str(estimand).upper() == "ATE":
        wt = np.abs(alpha[treated])
        wc = np.abs(alpha[control])
        mt = np.array([_weighted_mean(Z[treated, j], wt) for j in range(Z.shape[1])])
    else:
        wc = np.abs(alpha[control])
        mt = np.mean(Z[treated], axis=0)
    mc = np.array([_weighted_mean(Z[control, j], wc) for j in range(Z.shape[1])])
    weighted = (mt - mc) / pooled
    return pd.DataFrame(
        {
            "abs_smd_unweighted": np.abs(unweighted),
            "abs_smd_weighted": np.abs(weighted),
        }
    )


def _normal_estimate(theta: float, psi: FloatArray) -> dict[str, Any]:
    n = int(psi.shape[0])
    se = float(np.std(psi, ddof=1) / math.sqrt(n))
    z = float(theta / se) if se > 0.0 else float("nan")
    p = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    critical = float(norm.ppf(0.975))
    return {
        "estimate": float(theta),
        "se": se,
        "ci_low": float(theta - critical * se),
        "ci_high": float(theta + critical * se),
        "p_value": p,
        "inference": "influence_normal",
    }


def _point_only_estimate(theta: float) -> dict[str, Any]:
    """Report a point estimate without inference.

    The plug-in scores of regression adjustment, Riesz weighting, and the
    matching comparison are not Neyman orthogonal, so a sample-variance
    standard error would ignore the first-order effect of the estimated
    nuisances. No standard error, interval, p-value, or coverage is attached
    until the estimator-specific influence function is implemented.
    """

    return {
        "estimate": float(theta),
        "se": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": float("nan"),
        "inference": "point_only",
    }


def _recenter_influence(
    functional: LinearFunctional,
    D: FloatArray,
    theta: float,
    psi: FloatArray,
) -> FloatArray:
    """Account for an estimated treatment share in ATT influence values."""

    if isinstance(functional, ATTFunctional) and functional.pi_is_estimated:
        return np.asarray(psi, dtype=float) - (theta / functional.pi) * (D - functional.pi)
    return np.asarray(psi, dtype=float)




@dataclass(frozen=True)
class RieszCoefficientFit:
    """Coefficient fit for one Riesz representer optimization."""

    beta: FloatArray
    success: bool
    status: str
    message: str
    kkt_residual: float
    domain_margin: float
    clip_binding_rate: float = float("nan")
    binding_rate_lower: float = float("nan")
    binding_rate_upper: float = float("nan")


def _fit_riesz_coefficients(
    X: FloatArray,
    *,
    basis: BaseBasis,
    functional: LinearFunctional,
    generator: BregmanGenerator,
    penalty: str,
    lam: float,
    max_iter: int,
    tolerance: float,
    kkt_tolerance: float,
    dual_margin: float = 1e-10,
) -> RieszCoefficientFit:
    """Fit a linear dual index for one generator.

    Generators whose links are defined for every finite dual coordinate
    (squared, truncated UKL/BKL) fit unconstrained through ``GRRGLM``. Exact
    BKL and BP have restricted dual domains, enforced here as explicit linear
    constraints with the conjugate evaluated as an extended-value function.
    """

    Phi = np.asarray(basis(X), dtype=float)
    M = np.asarray(functional.m_basis_matrix(X, basis), dtype=float)
    n, p = Phi.shape
    penalty_ = None if penalty in {"none", "", None} else str(penalty).lower()
    if penalty_ not in {None, "l2", "ridge"}:
        raise ValueError("The manuscript experiments use an l2 penalty or no penalty.")

    if not isinstance(generator, (BKLGenerator, BPGenerator)):
        model = GRRGLM(
            basis=basis,
            generator=copy.deepcopy(generator),
            functional=functional,
            penalty=penalty_,
            lam=float(lam),
        )
        fit = model.fit(X, max_iter=max_iter, tol=tolerance, fit_basis=False)
        beta = np.asarray(fit.beta, dtype=float)
        kkt_residual = float(fit.kkt_residual)
        kkt_ok = np.isfinite(kkt_residual) and kkt_residual <= float(kkt_tolerance)
        success = bool(fit.success) and kkt_ok
        # A fit that failed on its own terms keeps its own status; the
        # experiment's KKT criterion renames only a converged fit whose
        # gradient residual is too large.
        status = str(fit.status)
        message = str(fit.message)
        if bool(fit.success) and not kkt_ok:
            status = "kkt_failure"
            message = (
                f"The fitted gradient residual {kkt_residual:.6g} exceeds "
                f"the tolerance {float(kkt_tolerance):.6g}."
            )
        return RieszCoefficientFit(
            beta=beta,
            success=success,
            status=status,
            message=message,
            kkt_residual=kkt_residual,
            domain_margin=float("inf"),
            clip_binding_rate=float(fit.clip_binding_rate),
            binding_rate_lower=float(fit.binding_rate_lower),
            binding_rate_upper=float(fit.binding_rate_upper),
        )

    sign = np.where(X[:, TREATMENT_INDEX] >= 0.5, 1.0, -1.0)
    signed_Phi = sign[:, None] * Phi
    if isinstance(generator, BKLGenerator):
        beta0 = np.linalg.solve(
            Phi.T @ Phi + 1e-8 * np.eye(p),
            Phi.T @ (-2.0 * sign),
        )
        constraint = optimize.LinearConstraint(
            signed_Phi,
            np.full(n, -np.inf),
            np.full(n, -dual_margin),
        )
    else:
        beta0 = np.zeros(p, dtype=float)
        k = 1.0 + 1.0 / generator.omega
        constraint = optimize.LinearConstraint(
            signed_Phi,
            np.full(n, k * (dual_margin - 1.0)),
            np.full(n, np.inf),
        )

    def objective(beta: FloatArray) -> float:
        v = Phi @ beta
        evaluation = generator.conjugate_status(X, v)
        if not np.all(evaluation.valid):
            # The exact conjugate is +inf outside the dual domain. Reporting
            # the extended value makes the line search reject an infeasible
            # trial step instead of the evaluation raising mid-solve; no
            # substitute value enters the fit.
            return float("inf")
        value = float(np.mean(evaluation.conjugate - M @ beta))
        if penalty_ is not None:
            value += 0.5 * float(lam) * float(np.sum(beta * beta))
        return value

    def gradient(beta: FloatArray) -> FloatArray:
        v = Phi @ beta
        evaluation = generator.conjugate_status(X, v)
        if not np.all(evaluation.valid):
            # Undefined outside the dual domain. The infinite objective has
            # already rejected such a step; NaN keeps any consumer visibly
            # broken rather than steering it with a fabricated direction.
            return np.full(beta.shape, np.nan)
        value = np.mean(evaluation.alpha[:, None] * Phi - M, axis=0)
        if penalty_ is not None:
            value = value + float(lam) * beta
        return np.asarray(value, dtype=float)

    result = optimize.minimize(
        objective,
        beta0,
        jac=gradient,
        method="SLSQP",
        constraints=(constraint,),
        options={"maxiter": int(max_iter), "ftol": float(tolerance), "disp": False},
    )
    beta = np.asarray(result.x, dtype=float)
    signed_index = signed_Phi @ beta
    if isinstance(generator, BKLGenerator):
        margin = float(np.min(-signed_index))
    else:
        k = 1.0 + 1.0 / generator.omega
        margin = float(np.min(1.0 + signed_index / k))
    grad = gradient(beta)
    if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(grad))):
        # The optimizer returned a point outside the exact dual domain; the
        # KKT diagnostics are undefined there and the fit is a failure.
        return RieszCoefficientFit(
            beta=beta,
            success=False,
            status="constrained_optimizer_failure",
            message=(
                "The constrained optimizer returned a point outside the "
                "exact dual domain."
            ),
            kkt_residual=float("inf"),
            domain_margin=margin,
            clip_binding_rate=0.0,
        )
    if isinstance(generator, BKLGenerator):
        slack = -dual_margin - signed_index
        stationarity_target = -grad
    else:
        k = 1.0 + 1.0 / generator.omega
        lower = k * (dual_margin - 1.0)
        slack = signed_index - lower
        stationarity_target = grad
    active = slack <= max(1e-7, 10.0 * float(tolerance))
    if np.any(active):
        multipliers, _ = optimize.nnls(signed_Phi[active].T, stationarity_target)
        if isinstance(generator, BKLGenerator):
            stationarity = grad + signed_Phi[active].T @ multipliers
        else:
            stationarity = grad - signed_Phi[active].T @ multipliers
        complementarity = multipliers * slack[active]
        complementarity_residual = float(np.max(np.abs(complementarity)))
    else:
        stationarity = grad
        complementarity_residual = 0.0
    stationarity_residual = float(np.max(np.abs(stationarity)))
    primal_residual = float(np.max(np.maximum(-slack, 0.0)))
    kkt = max(stationarity_residual, complementarity_residual, primal_residual)
    success = bool(result.success) and np.isfinite(margin) and margin >= 0.5 * dual_margin
    success = success and np.isfinite(kkt) and kkt <= float(kkt_tolerance)
    status = "converged" if success else "constrained_optimizer_failure"
    message = str(result.message)
    if bool(result.success) and not success:
        message = (
            f"The constrained KKT residual {kkt:.6g} exceeds the tolerance "
            f"{float(kkt_tolerance):.6g}, or the exact dual-domain margin is too small."
        )
    return RieszCoefficientFit(
        beta=beta,
        success=success,
        status=status,
        message=message,
        kkt_residual=kkt,
        domain_margin=margin,
        # The exact dual domain is enforced by constraints; no clamp exists,
        # so nothing binds and the per-side clamp rates do not apply.
        clip_binding_rate=0.0,
    )


def _fit_functional(
    X: FloatArray,
    Y: FloatArray,
    *,
    functional: LinearFunctional,
    generator: BregmanGenerator,
    representer_basis: BaseBasis,
    outcome_basis: BaseBasis,
    cross_fit: bool,
    folds: int,
    random_state: int,
    riesz_penalty: str,
    riesz_lam: float,
    outcome_lam: float,
    estimators: Sequence[str],
    max_iter: int,
    tolerance: float,
    kkt_tolerance: float = 1e-2,
) -> FunctionalFit:
    n = X.shape[0]
    D = X[:, TREATMENT_INDEX]
    alpha = np.full(n, np.nan)
    alpha1 = np.full(n, np.nan)
    alpha0 = np.full(n, np.nan)
    mu = np.full(n, np.nan)
    mu1 = np.full(n, np.nan)
    mu0 = np.full(n, np.nan)
    held_out_max: list[float] = []
    fit_statuses: list[str] = []
    fit_gradients: list[float] = []
    fit_binding_rates: list[float] = []
    fit_binding_lower: list[float] = []
    fit_binding_upper: list[float] = []
    split_list = _folds(D, cross_fit=cross_fit, folds=folds, random_state=random_state)

    for fold_id, fold in enumerate(split_list):
        train = fold.train
        test = fold.test
        D_train = D[train]
        if np.sum(D_train == 1.0) == 0 or np.sum(D_train == 0.0) == 0:
            return FunctionalFit(
                estimates={},
                diagnostics={},
                failure=ExperimentFailure(
                    "degenerate_treatment_fold",
                    "The training fold does not contain both treatment groups.",
                    fold_id,
                ),
            )
        basis_r = representer_basis.copy().fit(X[train], Y[train])
        fit_r = _fit_riesz_coefficients(
            X[train],
            basis=basis_r,
            functional=functional,
            generator=copy.deepcopy(generator),
            penalty=riesz_penalty,
            lam=float(riesz_lam),
            max_iter=max_iter,
            tolerance=tolerance,
            kkt_tolerance=kkt_tolerance,
        )
        fit_statuses.append(fit_r.status)
        fit_gradients.append(float(fit_r.kkt_residual))
        fit_binding_rates.append(float(fit_r.clip_binding_rate))
        fit_binding_lower.append(float(fit_r.binding_rate_lower))
        fit_binding_upper.append(float(fit_r.binding_rate_upper))
        if not fit_r.success:
            return FunctionalFit(
                estimates={},
                diagnostics={"riesz_fit_statuses": tuple(fit_statuses)},
                failure=ExperimentFailure(fit_r.status, fit_r.message, fold_id),
            )
        if not np.isfinite(fit_r.kkt_residual):
            return FunctionalFit(
                estimates={},
                diagnostics={"riesz_fit_statuses": tuple(fit_statuses)},
                failure=ExperimentFailure(
                    "kkt_failure",
                    "The fitted gradient is not finite.",
                    fold_id,
                ),
            )
        X_test = X[test]
        X1_test, X0_test = _counterfactuals(X_test)
        Phi_test = np.asarray(basis_r(X_test), dtype=float)
        Phi_one = np.asarray(basis_r(X1_test), dtype=float)
        Phi_zero = np.asarray(basis_r(X0_test), dtype=float)
        obs_link = generator.inv_grad_status(X_test, Phi_test @ fit_r.beta)
        one_link = generator.inv_grad_status(X1_test, Phi_one @ fit_r.beta)
        zero_link = generator.inv_grad_status(X0_test, Phi_zero @ fit_r.beta)
        if not (np.all(obs_link.valid) and np.all(one_link.valid) and np.all(zero_link.valid)):
            return FunctionalFit(
                estimates={},
                diagnostics={"riesz_fit_statuses": tuple(fit_statuses)},
                failure=ExperimentFailure(
                    "held_out_domain_failure",
                    "The exact representer link is undefined on a held-out observation.",
                    fold_id,
                ),
            )
        alpha[test] = obs_link.values
        alpha1[test] = one_link.values
        alpha0[test] = zero_link.values
        M_test = np.asarray(functional.m_basis_matrix(X_test, basis_r), dtype=float)
        delta = np.mean(obs_link.values[:, None] * Phi_test - M_test, axis=0)
        held_out_max.append(float(np.max(np.abs(delta))))

        basis_y = outcome_basis.copy().fit(X[train], Y[train])
        model_y = OutcomeGLM(
            basis=basis_y,
            link="identity",
            penalty="l2",
            lam=float(outcome_lam),
        )
        fit_y = model_y.fit(X[train], Y[train], max_iter=max_iter, tol=tolerance)
        if not fit_y.success:
            return FunctionalFit(
                estimates={},
                diagnostics={"riesz_fit_statuses": tuple(fit_statuses)},
                failure=ExperimentFailure(f"outcome_{fit_y.status}", fit_y.message, fold_id),
            )
        mu[test] = model_y.predict(X_test)
        mu1[test] = model_y.predict(X1_test)
        mu0[test] = model_y.predict(X0_test)

    if not all(np.all(np.isfinite(v)) for v in (alpha, alpha1, alpha0, mu, mu1, mu0)):
        return FunctionalFit(
            estimates={},
            diagnostics={},
            failure=ExperimentFailure(
                "nonfinite_score_input", "A nuisance prediction is nonfinite."
            ),
        )

    m_mu = _m_values(functional, X, mu1, mu0)
    m_alpha = _m_values(functional, X, alpha1, alpha0)
    estimates: dict[str, dict[str, Any]] = {}

    if "rw" in estimators:
        estimates["rw"] = _point_only_estimate(float(np.mean(alpha * Y)))
    if "ra" in estimators:
        estimates["ra"] = _point_only_estimate(float(np.mean(m_mu)))
    if "arw" in estimators:
        score = m_mu + alpha * (Y - mu)
        theta = float(np.mean(score))
        estimates["arw"] = _normal_estimate(
            theta,
            _recenter_influence(functional, D, theta, score - theta),
        )
    if "tmle" in estimators:
        denominator = float(np.sum(alpha * alpha))
        if not np.isfinite(denominator) or denominator <= 0.0:
            return FunctionalFit(
                estimates={},
                diagnostics={},
                failure=ExperimentFailure(
                    "tmle_denominator_failure", "The TMLE denominator is not positive."
                ),
            )
        epsilon = float(np.sum(alpha * (Y - mu)) / denominator)
        mu_star = mu + epsilon * alpha
        m_mu_star = m_mu + epsilon * m_alpha
        score = m_mu_star + alpha * (Y - mu_star)
        theta = float(np.mean(m_mu_star))
        estimates["tmle"] = _normal_estimate(
            theta,
            _recenter_influence(functional, D, theta, score - theta),
        )

    abs_alpha = np.abs(alpha)
    balance = _balance_table(np.delete(X, TREATMENT_INDEX, axis=1), D, alpha, functional.name)
    treated = D == 1.0
    control = D == 0.0
    diagnostics: dict[str, Any] = {
        "alpha_values": alpha,
        "alpha_abs_mean": float(np.mean(abs_alpha)),
        "alpha_abs_p95": float(np.percentile(abs_alpha, 95)),
        "alpha_abs_max": float(np.max(abs_alpha)),
        "max_abs_smd_unweighted": float(balance["abs_smd_unweighted"].max()),
        "max_abs_smd_weighted": float(balance["abs_smd_weighted"].max()),
        "ess_treated": _effective_sample_size(abs_alpha[treated]),
        "ess_control": _effective_sample_size(abs_alpha[control]),
        "held_out_imbalance_max": float(np.max(held_out_max)),
        "riesz_fit_statuses": tuple(fit_statuses),
        "riesz_gradient_norm_max": float(np.max(fit_gradients)),
        "riesz_clip_binding_rate_max": float(np.max(fit_binding_rates)),
        "riesz_binding_rate_lower_max": (
            float(np.max(fit_binding_lower))
            if np.all(np.isfinite(fit_binding_lower))
            else float("nan")
        ),
        "riesz_binding_rate_upper_max": (
            float(np.max(fit_binding_upper))
            if np.all(np.isfinite(fit_binding_upper))
            else float("nan")
        ),
        "love_plot": balance,
    }
    return FunctionalFit(estimates=estimates, diagnostics=diagnostics, failure=None)


def result_to_rows(
    fit: FunctionalFit,
    *,
    label_info: dict[str, Any],
    true_value: float | None,
) -> list[dict[str, Any]]:
    if fit.failure is not None:
        return _failure_rows(label_info, fit.failure, true_value)
    rows: list[dict[str, Any]] = []
    for estimator, estimate in fit.estimates.items():
        row = dict(label_info)
        row.update({"estimator": estimator, "status": "ok", **estimate})
        if true_value is not None:
            row["true_value"] = true_value
            row["error"] = float(estimate["estimate"] - true_value)
            row["squared_error"] = float((estimate["estimate"] - true_value) ** 2)
            if np.isfinite(estimate["ci_low"]) and np.isfinite(estimate["ci_high"]):
                row["covered"] = bool(
                    estimate["ci_low"] <= true_value <= estimate["ci_high"]
                )
            else:
                row["covered"] = float("nan")
        for key, value in fit.diagnostics.items():
            if key in {"love_plot", "alpha_values"}:
                continue
            if isinstance(value, (int, float, bool, str)) or value is None:
                row[key] = value
        rows.append(row)
    return rows


def fit_one_grr(
    data: dict[str, Any],
    *,
    estimand: str,
    loss_spec: dict[str, Any],
    basis_kind: str,
    basis_mode: str = "regressor",
    outcome_basis_kind: str | None = None,
    outcome_basis_mode: str = "regressor",
    cross_fit: bool = True,
    lam: float = 1e-2,
    penalty: str = "l2",
    basis_features: int = 120,
    degree: int = 2,
    sigma: str | float = "auto",
    folds: int = 5,
    estimators: Sequence[str] = ESTIMATORS_ALL,
    max_iter: int = 500,
    random_state: int = 0,
) -> list[dict[str, Any]]:
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    label = {
        "estimand": str(estimand).upper(),
        "loss": str(loss_spec["label"]),
        "basis": basis_kind,
        "basis_mode": basis_mode,
        "outcome_basis": outcome_basis_kind or basis_kind,
        "outcome_basis_mode": outcome_basis_mode,
        "cross_fit": bool(cross_fit),
        "lambda_riesz": float(lam),
        "penalty": penalty,
    }
    generator = make_compatible_generator(
        loss_spec["loss"],
        estimand=estimand,
        omega=loss_spec.get("omega"),
    )
    basis_r = make_basis(
        basis_kind,
        mode=basis_mode,
        seed=random_state,
        n_features=basis_features,
        degree=degree,
        sigma=sigma,
    )
    basis_y = make_basis(
        outcome_basis_kind or basis_kind,
        mode=outcome_basis_mode,
        seed=random_state + 7919,
        n_features=basis_features,
        degree=degree,
        sigma=sigma,
    )
    functional = _functional(estimand, X[:, TREATMENT_INDEX])
    fit = _fit_functional(
        X,
        Y,
        functional=functional,
        generator=generator,
        representer_basis=basis_r,
        outcome_basis=basis_y,
        cross_fit=cross_fit,
        folds=folds,
        random_state=random_state,
        riesz_penalty=penalty,
        riesz_lam=lam,
        outcome_lam=1e-3,
        estimators=tuple(str(e).lower() for e in estimators),
        max_iter=max_iter,
        tolerance=1e-8,
    )
    return result_to_rows(fit, label_info=label, true_value=true_theta(data, estimand))




def fit_one_grr_with_basis(
    data: dict[str, Any],
    *,
    estimand: str,
    loss_spec: dict[str, Any],
    representer_basis: BaseBasis,
    outcome_basis: BaseBasis | None = None,
    cross_fit: bool = True,
    lam: float = 1e-2,
    penalty: str = "l2",
    folds: int = 5,
    estimators: Sequence[str] = ESTIMATORS_ALL,
    max_iter: int = 500,
    random_state: int = 0,
    label_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fit one specification with a caller-supplied basis."""

    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    label = {
        "estimand": str(estimand).upper(),
        "loss": str(loss_spec["label"]),
        "basis": representer_basis.__class__.__name__,
        "cross_fit": bool(cross_fit),
        "lambda_riesz": float(lam),
        "penalty": penalty,
    }
    if label_info is not None:
        label.update(label_info)
    generator = make_compatible_generator(
        loss_spec["loss"], estimand=estimand, omega=loss_spec.get("omega")
    )
    fit = _fit_functional(
        X,
        Y,
        functional=_functional(estimand, X[:, TREATMENT_INDEX]),
        generator=generator,
        representer_basis=representer_basis,
        outcome_basis=copy.deepcopy(representer_basis) if outcome_basis is None else outcome_basis,
        cross_fit=cross_fit,
        folds=folds,
        random_state=random_state,
        riesz_penalty=penalty,
        riesz_lam=lam,
        outcome_lam=1e-3,
        estimators=tuple(str(e).lower() for e in estimators),
        max_iter=max_iter,
        tolerance=1e-8,
    )
    return result_to_rows(fit, label_info=label, true_value=true_theta(data, estimand))


def fit_grr_love_plot_data(
    data: dict[str, Any],
    *,
    estimand: str,
    loss_spec: dict[str, Any],
    basis_kind: str,
    basis_features: int,
    lam: float,
    folds: int,
    random_state: int,
) -> pd.DataFrame:
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    generator = make_compatible_generator(
        loss_spec["loss"],
        estimand=estimand,
        omega=loss_spec.get("omega"),
    )
    functional = _functional(estimand, X[:, TREATMENT_INDEX])
    basis_r = make_basis(
        basis_kind,
        mode="regressor",
        seed=random_state,
        n_features=basis_features,
    )
    basis_y = make_basis(
        basis_kind,
        mode="regressor",
        seed=random_state + 7919,
        n_features=basis_features,
    )
    fit = _fit_functional(
        X,
        Y,
        functional=functional,
        generator=generator,
        representer_basis=basis_r,
        outcome_basis=basis_y,
        cross_fit=True,
        folds=folds,
        random_state=random_state,
        riesz_penalty="l2",
        riesz_lam=lam,
        outcome_lam=1e-3,
        estimators=("arw",),
        max_iter=1000,
        tolerance=1e-8,
    )
    if fit.failure is not None:
        raise RuntimeError(f"{fit.failure.status}: {fit.failure.message}")
    balance = fit.diagnostics["love_plot"].copy()
    names = data.get("covariate_names")
    if names is None:
        names = [f"Z{j + 1}" for j in range(balance.shape[0])]
    balance.insert(0, "covariate", list(names))
    return balance


def fit_matching_ate(data: dict[str, Any], *, rep: int, M: int = 1) -> list[dict[str, Any]]:
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    D = X[:, TREATMENT_INDEX]
    matching = nn_matching_inverse_propensity_weights(
        np.delete(X, TREATMENT_INDEX, axis=1),
        D,
        int(M),
    )
    # ``matching.w`` contains positive inverse-propensity magnitudes. The ATE
    # Riesz representer is signed by treatment status.
    alpha = np.where(D == 1.0, matching.w, -matching.w)
    theta = float(np.mean(alpha * Y))
    estimate = _point_only_estimate(theta)
    label = {
        "estimand": "ATE",
        "loss": "matching",
        "basis": "matching",
        "basis_mode": "matching",
        "outcome_basis": "none",
        "outcome_basis_mode": "none",
        "cross_fit": False,
        "lambda_riesz": np.nan,
        "penalty": "none",
        "replication": int(rep),
        "estimator": "rw",
        "status": "ok",
        **estimate,
    }
    target = true_theta(data, "ATE")
    if target is not None:
        label["true_value"] = target
        label["error"] = theta - target
        label["squared_error"] = (theta - target) ** 2
        label["covered"] = float("nan")
    return [label]


def _fit_propensity_index(
    Phi: FloatArray, D: FloatArray, *, pair_name: str, lam: float
) -> tuple[FloatArray, str]:
    if pair_name == "BKL loss + logit link":
        model = LogisticRegression(
            C=1.0 / max(float(lam), 1e-12),
            fit_intercept=False,
            solver="lbfgs",
            max_iter=2000,
        )
        model.fit(Phi, D.astype(int))
        return np.asarray(model.coef_, dtype=float).reshape(-1), "converged"
    if pair_name == "SQ loss + logit link":
        beta0 = np.zeros(Phi.shape[1], dtype=float)

        def objective(beta: FloatArray) -> float:
            p = expit(Phi @ beta)
            return float(np.mean((D - p) ** 2) + 0.5 * lam * np.sum(beta * beta))

        def gradient(beta: FloatArray) -> FloatArray:
            p = expit(Phi @ beta)
            return np.mean(2.0 * ((p - D) * p * (1.0 - p))[:, None] * Phi, axis=0) + lam * beta

        result = optimize.minimize(
            objective,
            beta0,
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-10},
        )
        return (
            np.asarray(result.x, dtype=float),
            "converged" if result.success else "optimizer_failure",
        )
    raise ValueError(pair_name)


def fit_one_incompatible(
    data: dict[str, Any],
    *,
    estimand: str,
    pair_name: str,
    cross_fit: bool = True,
    lam: float = 1e-2,
    basis_features: int = 120,
    folds: int = 5,
    max_iter: int = 500,
    random_state: int = 0,
) -> list[dict[str, Any]]:
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    D = X[:, TREATMENT_INDEX]
    n = X.shape[0]
    alpha = np.full(n, np.nan)
    alpha1 = np.full(n, np.nan)
    alpha0 = np.full(n, np.nan)
    mu = np.full(n, np.nan)
    mu1 = np.full(n, np.nan)
    mu0 = np.full(n, np.nan)
    functional = _functional(estimand, D)
    for fold_id, fold in enumerate(
        _folds(D, cross_fit=cross_fit, folds=folds, random_state=random_state)
    ):
        train, test = fold.train, fold.test
        if np.sum(D[train] == 1.0) == 0 or np.sum(D[train] == 0.0) == 0:
            return _failure_rows(
                {"estimand": str(estimand).upper(), "loss_link_pair": pair_name},
                ExperimentFailure(
                    "degenerate_treatment_fold",
                    "The training fold does not contain both treatment groups.",
                    fold_id,
                ),
                true_theta(data, estimand),
            )
        if pair_name == "UKL loss + linear link":
            rep_basis = make_basis(
                "rkhs",
                mode="regressor",
                seed=random_state + fold_id,
                n_features=basis_features,
            ).fit(X[train])
            Phi_train = np.asarray(rep_basis(X[train]), dtype=float)
            M_train = np.asarray(
                functional.m_basis_matrix(X[train], rep_basis), dtype=float
            )
            beta, status = _fit_linear_link_ukl(
                Phi_train,
                M_train,
                np.where(D[train] == 1.0, 1.0, -1.0),
                C=generator_shift_for_estimand("UKL", estimand),
                lam=lam,
                max_iter=max_iter,
            )
            if status != "converged":
                return _failure_rows(
                    {"estimand": str(estimand).upper(), "loss_link_pair": pair_name},
                    ExperimentFailure(
                        status,
                        "The mismatched linear-link fit did not converge.",
                        fold_id,
                    ),
                    true_theta(data, estimand),
                )
            alpha[test] = np.asarray(rep_basis(X[test]), dtype=float) @ beta
            X1_test, X0_test = _counterfactuals(X[test])
            alpha1[test] = np.asarray(rep_basis(X1_test), dtype=float) @ beta
            alpha0[test] = np.asarray(rep_basis(X0_test), dtype=float) @ beta
            shift = generator_shift_for_estimand("UKL", estimand)
            sign_test = np.where(D[test] == 1.0, 1.0, -1.0)
            if not (
                np.all(sign_test * alpha[test] > shift)
                and np.all(alpha1[test] > shift)
                and np.all(-alpha0[test] > shift)
            ):
                return _failure_rows(
                    {"estimand": str(estimand).upper(), "loss_link_pair": pair_name},
                    ExperimentFailure(
                        "held_out_domain_failure",
                        "The linear-link representer leaves the UKL domain "
                        "on a held-out observation.",
                        fold_id,
                    ),
                    true_theta(data, estimand),
                )
        else:
            prop_basis = make_base_basis(
                "rkhs",
                seed=random_state + fold_id,
                n_features=basis_features,
            ).fit(np.delete(X[train], TREATMENT_INDEX, axis=1), D[train])
            Phi = np.asarray(prop_basis(np.delete(X[train], TREATMENT_INDEX, axis=1)), dtype=float)
            beta, status = _fit_propensity_index(Phi, D[train], pair_name=pair_name, lam=lam)
            if status != "converged":
                return _failure_rows(
                    {"estimand": str(estimand).upper(), "loss_link_pair": pair_name},
                    ExperimentFailure(
                        status, "The propensity-index optimizer did not converge.", fold_id
                    ),
                    true_theta(data, estimand),
                )
            Z_test = np.delete(X[test], TREATMENT_INDEX, axis=1)
            e = expit(np.asarray(prop_basis(Z_test), dtype=float) @ beta)
            if not np.all(np.isfinite(e)) or np.any(e <= 0.0) or np.any(e >= 1.0):
                return _failure_rows(
                    {"estimand": str(estimand).upper(), "loss_link_pair": pair_name},
                    ExperimentFailure(
                        "propensity_domain_failure",
                        "A fitted propensity is outside (0,1).",
                        fold_id,
                    ),
                    true_theta(data, estimand),
                )
            if isinstance(functional, ATEFunctional):
                alpha[test] = D[test] / e - (1.0 - D[test]) / (1.0 - e)
                alpha1[test] = 1.0 / e
                alpha0[test] = -1.0 / (1.0 - e)
            else:
                alpha[test] = D[test] / functional.pi
                alpha[test] -= (1.0 - D[test]) * e / (functional.pi * (1.0 - e))
                alpha1[test] = 1.0 / functional.pi
                alpha0[test] = -e / (functional.pi * (1.0 - e))
        basis_y = make_basis(
            "rkhs",
            mode="regressor",
            seed=random_state + fold_id + 7919,
            n_features=basis_features,
        ).fit(X[train], Y[train])
        outcome = OutcomeGLM(basis=basis_y, link="identity", penalty="l2", lam=1e-3)
        outcome_fit = outcome.fit(
            X[train], Y[train], max_iter=max_iter, tol=1e-8, fit_basis=False
        )
        if not outcome_fit.success:
            return _failure_rows(
                {"estimand": str(estimand).upper(), "loss_link_pair": pair_name},
                ExperimentFailure(f"outcome_{outcome_fit.status}", outcome_fit.message, fold_id),
                true_theta(data, estimand),
            )
        X1, X0 = _counterfactuals(X[test])
        mu[test] = outcome.predict(X[test])
        mu1[test] = outcome.predict(X1)
        mu0[test] = outcome.predict(X0)
    if not all(np.all(np.isfinite(v)) for v in (alpha, alpha1, alpha0, mu, mu1, mu0)):
        return _failure_rows(
            {"estimand": str(estimand).upper(), "loss_link_pair": pair_name},
            ExperimentFailure(
                "nonfinite_score_input",
                "A propensity-based nuisance prediction is nonfinite.",
            ),
            true_theta(data, estimand),
        )
    m_mu = _m_values(functional, X, mu1, mu0)
    m_alpha = _m_values(functional, X, alpha1, alpha0)
    estimates: dict[str, dict[str, Any]] = {}
    score = m_mu + alpha * (Y - mu)
    theta_arw = float(np.mean(score))
    estimates["arw"] = _normal_estimate(
        theta_arw,
        _recenter_influence(functional, D, theta_arw, score - theta_arw),
    )
    estimates["rw"] = _point_only_estimate(float(np.mean(alpha * Y)))
    estimates["ra"] = _point_only_estimate(float(np.mean(m_mu)))
    denominator = float(np.sum(alpha * alpha))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return _failure_rows(
            {"estimand": str(estimand).upper(), "loss_link_pair": pair_name},
            ExperimentFailure(
                "tmle_denominator_failure",
                "The propensity-based TMLE denominator is not positive.",
            ),
            true_theta(data, estimand),
        )
    epsilon = float(np.sum(alpha * (Y - mu)) / denominator)
    mu_star = mu + epsilon * alpha
    m_mu_star = m_mu + epsilon * m_alpha
    tmle_score = m_mu_star + alpha * (Y - mu_star)
    theta_tmle = float(np.mean(m_mu_star))
    estimates["tmle"] = _normal_estimate(
        theta_tmle,
        _recenter_influence(functional, D, theta_tmle, tmle_score - theta_tmle),
    )
    fit = FunctionalFit(estimates=estimates, diagnostics={}, failure=None)
    label = {
        "estimand": str(estimand).upper(),
        "loss_link_pair": pair_name,
        "basis": "rkhs",
        "cross_fit": bool(cross_fit),
        "lambda_riesz": float(lam),
        "penalty": "l2",
    }
    return result_to_rows(fit, label_info=label, true_value=true_theta(data, estimand))


#: Propensity clip window of the plug-in logistic baseline. The clip is part
#: of the declared comparison method (the textbook IPW/AIPW practice), not a
#: rewrite of a genriesz estimator; the rate at which it is active is
#: reported as ``propensity_clip_rate`` in the baseline's result rows.
PLUGIN_PROPENSITY_CLIP = 0.01


def fit_one_plugin_logistic(
    data: dict[str, Any],
    *,
    estimand: str,
    folds: int = 5,
    basis_features: int = 120,
    random_state: int = 0,
    outcome_lam: float = 1e-3,
    max_iter: int = 500,
) -> list[dict[str, Any]]:
    """Cross-fitted logistic-propensity IPW/AIPW baseline on raw covariates.

    The propensity model is an effectively unpenalized logistic regression of
    the treatment on the raw covariates (standardized with training-fold
    moments), the textbook plug-in baseline; it is deliberately not the
    RKHS-feature propensity index of the incompatible loss--link pairs.
    Fitted propensities are clipped to ``[PLUGIN_PROPENSITY_CLIP, 1 -
    PLUGIN_PROPENSITY_CLIP]``: the clip is part of the declared baseline and
    its activation rate is reported as ``propensity_clip_rate``. The IPW arm
    is a plug-in weighting score, so it is reported point-only; the AIPW arm
    carries the influence-function inference. The outcome model matches the
    RKHS specification of the GRR arms.
    """

    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    D = X[:, TREATMENT_INDEX]
    Z = np.delete(X, TREATMENT_INDEX, axis=1)
    n = X.shape[0]
    label = {
        "estimand": str(estimand).upper(),
        "loss": "Logistic",
        "basis": "raw Z",
        "basis_mode": "regressor",
        "cross_fit": True,
        "lambda_riesz": float("nan"),
        "penalty": "none",
    }
    functional = _functional(estimand, D)
    alpha = np.full(n, np.nan)
    mu = np.full(n, np.nan)
    mu1 = np.full(n, np.nan)
    mu0 = np.full(n, np.nan)
    clip_hits = 0
    basis_y = make_basis(
        "rkhs",
        mode="regressor",
        seed=random_state + 7919,
        n_features=basis_features,
    )
    for fold_id, fold in enumerate(
        stratified_kfold_splits(D, folds=int(folds), random_state=int(random_state))
    ):
        train = fold.train
        test = fold.test
        D_train = D[train]
        if np.sum(D_train == 1.0) == 0 or np.sum(D_train == 0.0) == 0:
            return _failure_rows(
                label,
                ExperimentFailure(
                    "degenerate_treatment_fold",
                    "The training fold does not contain both treatment groups.",
                    fold_id,
                ),
                true_theta(data, estimand),
            )
        Z_train = Z[train]
        mean_train = Z_train.mean(axis=0)
        sd_train = Z_train.std(axis=0)
        sd_train = np.where(sd_train > 0.0, sd_train, 1.0)
        model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=2000)
        model.fit((Z_train - mean_train) / sd_train, D_train.astype(int))
        e_raw = model.predict_proba((Z[test] - mean_train) / sd_train)[:, 1]
        clip_hits += int(
            np.sum(
                (e_raw < PLUGIN_PROPENSITY_CLIP)
                | (e_raw > 1.0 - PLUGIN_PROPENSITY_CLIP)
            )
        )
        e = np.clip(e_raw, PLUGIN_PROPENSITY_CLIP, 1.0 - PLUGIN_PROPENSITY_CLIP)
        D_test = D[test]
        if isinstance(functional, ATEFunctional):
            alpha[test] = D_test / e - (1.0 - D_test) / (1.0 - e)
        else:
            alpha[test] = (D_test - (1.0 - D_test) * e / (1.0 - e)) / functional.pi
        fold_basis = basis_y.copy().fit(X[train], Y[train])
        model_y = OutcomeGLM(
            basis=fold_basis, link="identity", penalty="l2", lam=float(outcome_lam)
        )
        fit_y = model_y.fit(
            X[train], Y[train], max_iter=max_iter, tol=1e-8, fit_basis=False
        )
        if not fit_y.success:
            return _failure_rows(
                label,
                ExperimentFailure(f"outcome_{fit_y.status}", fit_y.message, fold_id),
                true_theta(data, estimand),
            )
        X1_test, X0_test = _counterfactuals(X[test])
        mu[test] = model_y.predict(X[test])
        mu1[test] = model_y.predict(X1_test)
        mu0[test] = model_y.predict(X0_test)
    if not all(np.all(np.isfinite(v)) for v in (alpha, mu, mu1, mu0)):
        return _failure_rows(
            label,
            ExperimentFailure(
                "nonfinite_score_input", "A plug-in nuisance prediction is nonfinite."
            ),
            true_theta(data, estimand),
        )
    m_mu = _m_values(functional, X, mu1, mu0)
    theta_ipw = float(np.mean(alpha * Y))
    score = m_mu + alpha * (Y - mu)
    theta_aipw = float(np.mean(score))
    estimates = {
        "ipw": _point_only_estimate(theta_ipw),
        "aipw": _normal_estimate(
            theta_aipw,
            _recenter_influence(functional, D, theta_aipw, score - theta_aipw),
        ),
    }
    abs_alpha = np.abs(alpha)
    balance = _balance_table(Z, D, alpha, functional.name)
    diagnostics = {
        "alpha_abs_mean": float(np.mean(abs_alpha)),
        "alpha_abs_p95": float(np.percentile(abs_alpha, 95)),
        "alpha_abs_max": float(np.max(abs_alpha)),
        "max_abs_smd_unweighted": float(balance["abs_smd_unweighted"].max()),
        "max_abs_smd_weighted": float(balance["abs_smd_weighted"].max()),
        "ess_treated": _effective_sample_size(abs_alpha[D == 1.0]),
        "ess_control": _effective_sample_size(abs_alpha[D == 0.0]),
        "propensity_clip_rate": float(clip_hits) / float(n),
    }
    fit = FunctionalFit(estimates=estimates, diagnostics=diagnostics, failure=None)
    return result_to_rows(fit, label_info=label, true_value=true_theta(data, estimand))


def summarize_estimates(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    ok = df[(df["status"] == "ok") & (df["estimator"] != "failed")].copy()
    if ok.empty:
        return ok
    agg: dict[str, Any] = {"estimate": ["mean", "std"], "se": "mean"}
    if "squared_error" in ok.columns:
        agg["squared_error"] = ["mean"]
    for name in (
        "alpha_abs_p95",
        "alpha_abs_max",
        "max_abs_smd_weighted",
        "ess_treated",
        "ess_control",
        "held_out_imbalance_max",
    ):
        if name in ok.columns:
            agg[name] = "mean"
    for name in (
        "riesz_clip_binding_rate_max",
        "riesz_binding_rate_lower_max",
        "riesz_binding_rate_upper_max",
    ):
        if name in ok.columns:
            agg[name] = ["mean", "max"]
    out = ok.groupby(group_cols, dropna=False).agg(agg)
    out.columns = ["_".join(str(x) for x in col if x != "").strip("_") for col in out.columns]
    out = out.reset_index()
    if "squared_error_mean" in out.columns:
        out = out.rename(columns={"squared_error_mean": "mse"})
        out["rmse"] = np.sqrt(out["mse"])
    if "covered" in ok.columns:
        coverage = ok.groupby(group_cols, dropna=False)["covered"].mean().rename("coverage")
        out = out.merge(coverage.reset_index(), on=group_cols, how="left")
    return out


__all__ = [
    "COMPATIBLE_LOSSES",
    "CoverageDiagnosticBasis",
    "DATA_DIR",
    "ESTIMANDS",
    "ESTIMATORS_ALL",
    "RANDOM_SEED",
    "SelectedColumnsBasis",
    "TREATMENT_INDEX",
    "ZOnlyBasis",
    "fit_grr_love_plot_data",
    "fit_matching_ate",
    "fit_one_grr",
    "fit_one_grr_with_basis",
    "fit_one_incompatible",
    "fit_one_plugin_logistic",
    "generator_shift_for_estimand",
    "load_ihdp_replication",
    "load_lalonde",
    "make_base_basis",
    "make_basis",
    "make_compatible_generator",
    "make_coverage_diagnostic_data",
    "make_dimension_data",
    "make_kang_schafer_data",
    "make_kernel_gp_data",
    "make_score_guided_data",
    "make_simulation_data",
    "result_to_rows",
    "standardize_columns",
    "summarize_estimates",
    "true_theta",
]
