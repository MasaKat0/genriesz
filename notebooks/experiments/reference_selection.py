"""Reference-based loss-link selection experiment.

The module implements the simulation design in the generalized Riesz regression
manuscript. It separates training, diagnostic, and evaluation observations.
Candidate Riesz representer estimators are compared through held-out score
differences relative to a pre-specified reference estimator.

The publication configurations at the end of the file contain the Monte Carlo
designs reported in the manuscript.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from hashlib import blake2b
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg, optimize, sparse, stats
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from genriesz.basis import BaseBasis, RBFRandomFourierBasis
from genriesz.functionals import ATEFunctional
from genriesz.generators import BKLGenerator, BPGenerator, SquaredGenerator, UKLGenerator
from genriesz.glm import GRRGLM, OutcomeGLM

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CandidateSpec:
    """A representer specification compared by the selection rule."""

    loss: Literal["SQ", "UKL", "BKL", "BP"]
    dictionary: Literal["linear", "second_order", "rich"]
    penalty_multiplier: float
    omega: float | None = None

    @property
    def label(self) -> str:
        loss_label = self.loss if self.loss != "BP" else f"BP({self.omega:g})"
        return f"{loss_label}|{self.dictionary}|c={self.penalty_multiplier:g}"


@dataclass(frozen=True)
class SimulationConfig:
    """Monte Carlo configuration for one set of data-generating processes."""

    name: str
    design: Literal["low", "high"]
    sample_sizes: tuple[int, ...]
    overlap_scales: tuple[float, ...]
    replications: int
    diagnostic_delta: float = 0.01
    interval_miscoverage: float = 0.05
    multiplier_draws: int = 2000
    integration_size: int = 100_000
    base_seed: int = 20260720
    n_folds: int = 5
    max_iter: int = 1000
    optimization_tolerance: float = 1e-8
    optimization_gradient_tolerance: float = 1e-2
    reference_modes: tuple[Literal["truth", "estimated"], ...] = ("truth", "estimated")
    high_dim_reference_constants: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    max_workers: int | None = None
    batch_size: int = 10


@dataclass(frozen=True)
class GeneratedData:
    X: FloatArray
    y: FloatArray
    gamma0: FloatArray
    alpha0: FloatArray
    tau: FloatArray
    propensity: FloatArray


@dataclass(frozen=True)
class FoldRoles:
    training: IntArray
    diagnostic: IntArray
    evaluation: IntArray


@dataclass(frozen=True)
class RepresenterFit:
    spec: CandidateSpec
    basis: "ExperimentBasis"
    model: GRRGLM | None
    success: bool
    status: str
    objective: float
    gradient_norm: float
    kkt_residual: float
    binding_rate: float

    def predict(self, X: FloatArray) -> FloatArray:
        if not self.success or self.model is None:
            raise RuntimeError(f"Representer fit is unavailable: {self.status}")
        return np.asarray(self.model.predict_alpha(X), dtype=float)


@dataclass(frozen=True)
class OutcomeFit:
    kind: str
    model: OutcomeGLM | HistGradientBoostingRegressor
    basis: BaseBasis | None

    def predict(self, X: FloatArray) -> FloatArray:
        if isinstance(self.model, OutcomeGLM):
            return np.asarray(self.model.predict(X), dtype=float)
        return np.asarray(self.model.predict(X), dtype=float)


@dataclass(frozen=True)
class ReferenceFit:
    mode: Literal["truth", "estimated"]
    representer: RepresenterFit | "RFFSquaredReference" | "LogisticReferenceRepresenter" | None
    outcome: OutcomeFit | None
    bias_allowance: float
    status: str


@dataclass(frozen=True)
class RFFSquaredReference:
    basis: RBFRandomFourierBasis
    beta: FloatArray
    success: bool
    status: str

    def predict(self, X: FloatArray) -> FloatArray:
        if not self.success:
            raise RuntimeError(f"RFF reference is unavailable: {self.status}")
        phi = np.asarray(self.basis(X), dtype=float)
        return 0.5 * (phi @ self.beta)


@dataclass(frozen=True)
class LogisticReferenceRepresenter:
    model: LogisticRegression
    success: bool
    status: str

    def _features(self, X: FloatArray) -> FloatArray:
        Z = np.asarray(X, dtype=float)[:, 1:]
        return np.column_stack((Z[:, 0], Z[:, 1], Z[:, 2] ** 2 - 1.0, np.sin(Z[:, 3])))

    def predict_propensity(self, X: FloatArray) -> FloatArray:
        if not self.success:
            raise RuntimeError(f"Logistic reference is unavailable: {self.status}")
        return np.asarray(self.model.predict_proba(self._features(X))[:, 1], dtype=float)

    def predict(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        D = X[:, 0]
        e = self.predict_propensity(X)
        return D / e - (1.0 - D) / (1.0 - e)


class LowOutcomeBasis(BaseBasis):
    """Correctly specified outcome series for the low-dimensional design."""

    @property
    def n_features(self) -> int:
        return 12

    def __call__(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        D = X[:, [0]]
        Z = X[:, 1:]
        base = np.column_stack(
            (
                np.ones(X.shape[0]),
                Z[:, 0],
                Z[:, 1],
                Z[:, 1] ** 2 - 1.0,
                np.sin(Z[:, 2]),
                Z[:, 3] * Z[:, 4],
            )
        )
        return np.column_stack((D * base, (1.0 - D) * base))



class ExperimentBasis(BaseBasis):
    """Treatment-specific series basis standardized on the fitting sample."""

    def __init__(self, kind: Literal["linear", "second_order", "rich"]):
        self.kind = kind
        self._mean: FloatArray | None = None
        self._scale: FloatArray | None = None
        self._n_features: int | None = None

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            raise RuntimeError("ExperimentBasis must be fit before n_features is available.")
        return self._n_features

    def _base_features(self, Z: FloatArray) -> FloatArray:
        n, d = Z.shape
        blocks: list[FloatArray] = [np.ones((n, 1), dtype=float), Z]
        if self.kind in {"second_order", "rich"}:
            blocks.append(Z * Z)
        if self.kind == "rich":
            q = min(10, d)
            blocks.append(np.sin(Z[:, :q]))
            blocks.append(np.abs(Z[:, :q]))
            interactions = [
                (Z[:, j] * Z[:, k]).reshape(-1, 1)
                for j in range(q)
                for k in range(j + 1, q)
            ]
            if interactions:
                blocks.append(np.column_stack(interactions))
        return np.column_stack(blocks)

    def _raw_features(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        D = X[:, [0]]
        Z = X[:, 1:]
        base = self._base_features(Z)
        return np.column_stack((D * base, (1.0 - D) * base))

    def fit(self, X: FloatArray, y: FloatArray | None = None) -> "ExperimentBasis":
        raw = self._raw_features(np.asarray(X, dtype=float))
        mean = raw.mean(axis=0)
        scale = raw.std(axis=0, ddof=0)
        constant = scale <= 1e-12
        mean[constant] = 0.0
        scale[constant] = 1.0
        self._mean = mean
        self._scale = scale
        self._n_features = raw.shape[1]
        return self

    def __call__(self, X: FloatArray) -> FloatArray:
        if self._mean is None or self._scale is None:
            raise RuntimeError("ExperimentBasis must be fit on training data before evaluation.")
        raw = self._raw_features(np.asarray(X, dtype=float))
        return (raw - self._mean) / self._scale


class TruthOutcome:
    """Outcome regression used only for simulation calibration and audit."""

    def __init__(self, design: Literal["low", "high"]):
        self.design = design

    def predict(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        D = X[:, 0]
        Z = X[:, 1:]
        return _mu0(Z, self.design) + D * _tau(Z, self.design)


class TruthRepresenter:
    """ATE Riesz representer used only for simulation calibration and audit."""

    def __init__(self, scale: float, design: Literal["low", "high"]):
        self.scale = scale
        self.design = design

    def predict(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        D = X[:, 0]
        Z = X[:, 1:]
        e = _propensity(Z, self.scale, self.design)
        return D / e - (1.0 - D) / (1.0 - e)


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(str(x) for x in (base_seed, *parts)).encode("utf-8")
    digest = blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _covariance_matrix(d: int, rho: float = 0.5) -> FloatArray:
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _treatment_index(Z: FloatArray, design: Literal["low", "high"]) -> FloatArray:
    if design == "low":
        return 0.6 * Z[:, 0] - 0.4 * Z[:, 1] + 0.5 * (Z[:, 2] ** 2 - 1.0) + 0.3 * np.sin(Z[:, 3])
    return (
        0.5 * Z[:, 0]
        - 0.4 * Z[:, 1]
        + 0.3 * Z[:, 2] * Z[:, 3]
        + 0.4 * np.sin(Z[:, 4])
        + 0.2 * (Z[:, 5] ** 2 - 1.0)
    )


def _mu0(Z: FloatArray, design: Literal["low", "high"]) -> FloatArray:
    out = 1.0 + Z[:, 0] + 0.5 * (Z[:, 1] ** 2 - 1.0) + 0.5 * np.sin(Z[:, 2]) + 0.25 * Z[:, 3] * Z[:, 4]
    if design == "high":
        out = out + 0.25 * np.abs(Z[:, 5])
    return out


def _tau(Z: FloatArray, design: Literal["low", "high"]) -> FloatArray:
    if design == "low":
        return 1.0 + 0.5 * Z[:, 0] - 0.25 * Z[:, 1]
    return 1.0 + 0.4 * Z[:, 0] - 0.2 * Z[:, 1] + 0.2 * np.sin(Z[:, 2])


def _propensity(Z: FloatArray, scale: float, design: Literal["low", "high"]) -> FloatArray:
    index = np.clip(scale * _treatment_index(Z, design), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-index))


def generate_data(
    *,
    n: int,
    design: Literal["low", "high"],
    overlap_scale: float,
    seed: int,
) -> GeneratedData:
    """Generate one sample from a manuscript data-generating process."""

    rng = np.random.default_rng(seed)
    d = 5 if design == "low" else 50
    if design == "low":
        Z = rng.normal(size=(n, d))
    else:
        Z = rng.multivariate_normal(np.zeros(d), _covariance_matrix(d), size=n)
    e = _propensity(Z, overlap_scale, design)
    D = rng.binomial(1, e, size=n).astype(float)
    tau = _tau(Z, design)
    gamma0 = _mu0(Z, design) + D * tau
    y = gamma0 + rng.normal(size=n)
    alpha0 = D / e - (1.0 - D) / (1.0 - e)
    X = np.column_stack((D, Z))
    return GeneratedData(X=X, y=y, gamma0=gamma0, alpha0=alpha0, tau=tau, propensity=e)


def make_fold_roles(n: int, n_folds: int, seed: int) -> tuple[FoldRoles, ...]:
    """Create rotating training, diagnostic, and evaluation roles."""

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = tuple(np.asarray(x, dtype=np.int64) for x in np.array_split(perm, n_folds))
    roles: list[FoldRoles] = []
    all_idx = np.arange(n, dtype=np.int64)
    for k in range(n_folds):
        evaluation = folds[k]
        diagnostic = folds[(k + 1) % n_folds]
        excluded = np.zeros(n, dtype=bool)
        excluded[evaluation] = True
        excluded[diagnostic] = True
        training = all_idx[~excluded]
        roles.append(FoldRoles(training=training, diagnostic=diagnostic, evaluation=evaluation))
    return tuple(roles)


def candidate_grid() -> tuple[CandidateSpec, ...]:
    """Return the full candidate set from the manuscript."""

    losses: tuple[tuple[str, float | None], ...] = (
        ("SQ", None),
        ("UKL", None),
        ("BKL", None),
        ("BP", 0.25),
        ("BP", 0.5),
    )
    dictionaries = ("linear", "second_order", "rich")
    multipliers = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    return tuple(
        CandidateSpec(loss=loss, dictionary=dictionary, penalty_multiplier=c, omega=omega)
        for loss, omega in losses
        for dictionary in dictionaries
        for c in multipliers
    )


def _ate_branch(x: FloatArray) -> float:
    row = np.asarray(x, dtype=float).reshape(-1)
    return 1.0 if row[0] >= 0.5 else -1.0


def _make_generator(spec: CandidateSpec):
    if spec.loss == "SQ":
        return SquaredGenerator(C=0.0)
    if spec.loss == "UKL":
        return UKLGenerator(C=1.0, branch_fn=_ate_branch)
    if spec.loss == "BKL":
        return BKLGenerator(C=1.0, branch_fn=_ate_branch)
    if spec.loss == "BP":
        if spec.omega is None:
            raise ValueError("BP candidate requires omega.")
        return BPGenerator(C=1.0, omega=spec.omega, branch_fn=_ate_branch)
    raise ValueError(f"Unknown loss: {spec.loss}")


def fit_candidate(
    X_train: FloatArray,
    spec: CandidateSpec,
    *,
    max_iter: int,
    tolerance: float,
    gradient_tolerance: float,
) -> RepresenterFit:
    """Fit one candidate with the genriesz finite-dimensional estimator."""

    basis = ExperimentBasis(spec.dictionary).fit(X_train)
    p = basis.n_features
    lam = spec.penalty_multiplier * np.sqrt(np.log(max(p, 2)) / X_train.shape[0])
    model = GRRGLM(
        basis=basis,
        generator=_make_generator(spec),
        functional=ATEFunctional(treatment_index=0),
        penalty="l1",
        lam=lam,
        p_norm=1.0,
    )
    fit = model.fit(X_train, max_iter=max_iter, tol=tolerance)
    gradient = float(fit.gradient_norm)
    kkt = float(fit.kkt_residual)
    binding = float(fit.clip_binding_rate)
    diagnostics_ok = (
        fit.success
        and np.isfinite(gradient)
        and gradient <= gradient_tolerance
        and (not np.isfinite(binding) or binding == 0.0)
    )
    status = fit.status if fit.success else fit.status
    if fit.success and not diagnostics_ok:
        status = "diagnostic_failure"
    return RepresenterFit(
        spec=spec,
        basis=basis,
        model=model if diagnostics_ok else None,
        success=bool(diagnostics_ok),
        status=status,
        objective=float(fit.objective_value),
        gradient_norm=gradient,
        kkt_residual=kkt,
        binding_rate=binding,
    )


def fit_outcome(
    X_train: FloatArray,
    y_train: FloatArray,
    *,
    design: Literal["low", "high"],
    seed: int,
) -> OutcomeFit:
    """Fit the common outcome estimator used by all candidates in a fold."""

    if design == "low":
        basis = LowOutcomeBasis().fit(X_train)
        model = OutcomeGLM(basis=basis, link="identity", penalty="l2", lam=0.0)
        fit = model.fit(X_train, y_train)
        if not fit.success:
            raise RuntimeError(f"Low-dimensional outcome estimator failed: {fit.status}")
        return OutcomeFit(kind="series", model=model, basis=basis)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return OutcomeFit(kind="gradient_boosting", model=model, basis=None)


def _m_ate(X: FloatArray, outcome: OutcomeFit | TruthOutcome) -> FloatArray:
    X = np.asarray(X, dtype=float)
    X1 = X.copy()
    X0 = X.copy()
    X1[:, 0] = 1.0
    X0[:, 0] = 0.0
    return outcome.predict(X1) - outcome.predict(X0)


def score_contribution(
    X: FloatArray,
    y: FloatArray,
    representer: RepresenterFit | RFFSquaredReference | LogisticReferenceRepresenter | TruthRepresenter,
    outcome: OutcomeFit | TruthOutcome,
) -> FloatArray:
    alpha = representer.predict(X)
    gamma = outcome.predict(X)
    return _m_ate(X, outcome) + alpha * (y - gamma)


def _symmetric_matrix_sqrt(A: FloatArray) -> FloatArray:
    vals, vecs = np.linalg.eigh((A + A.T) / 2.0)
    vals = np.maximum(vals, 0.0)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _ellipsoid_l2_radius(covariance: FloatArray, gram: FloatArray, probability: float) -> float:
    if covariance.size == 0:
        return 0.0
    root = _symmetric_matrix_sqrt(covariance)
    matrix = root @ gram @ root
    largest = float(np.max(np.linalg.eigvalsh((matrix + matrix.T) / 2.0)))
    quantile = float(stats.chi2.ppf(probability, covariance.shape[0]))
    return float(np.sqrt(max(quantile * largest, 0.0)))


def _outcome_coefficient_radius(
    outcome: OutcomeFit,
    X_train: FloatArray,
    y_train: FloatArray,
    probability: float,
) -> float:
    if not isinstance(outcome.model, OutcomeGLM) or outcome.basis is None:
        raise ValueError("Coefficient radius requires a linear outcome estimator.")
    phi = np.asarray(outcome.basis(X_train), dtype=float)
    residual = y_train - outcome.predict(X_train)
    bread = np.linalg.pinv(phi.T @ phi)
    meat = phi.T @ ((residual * residual)[:, None] * phi)
    covariance = bread @ meat @ bread
    gram = (phi.T @ phi) / phi.shape[0]
    return _ellipsoid_l2_radius(covariance, gram, probability)


def _representer_coefficient_radius(
    representer: RepresenterFit,
    X_train: FloatArray,
    probability: float,
) -> float:
    if not representer.success or representer.model is None:
        raise ValueError("Representer radius requires a successful fit.")
    model = representer.model
    phi = np.asarray(representer.basis(X_train), dtype=float)
    m_matrix = np.asarray(ATEFunctional(0).m_basis_matrix(X_train, representer.basis), dtype=float)
    alpha = model.predict_alpha(X_train)
    curvature = np.asarray(model.generator.grad2(X_train, alpha), dtype=float)
    derivative = 1.0 / curvature
    score = alpha[:, None] * phi - m_matrix
    hessian = (phi.T @ (derivative[:, None] * phi)) / phi.shape[0]
    score_cov = (score.T @ score) / phi.shape[0]
    hessian_inv = np.linalg.pinv(hessian)
    covariance = hessian_inv @ score_cov @ hessian_inv / phi.shape[0]
    jacobian = derivative[:, None] * phi
    gram = (jacobian.T @ jacobian) / phi.shape[0]
    return _ellipsoid_l2_radius(covariance, gram, probability)


def fit_low_dimensional_reference(
    X_train: FloatArray,
    y_train: FloatArray,
    *,
    outcome: OutcomeFit,
    confidence_probability: float,
    max_iter: int,
    tolerance: float,
    gradient_tolerance: float,
) -> ReferenceFit:
    Z = X_train[:, 1:]
    features = np.column_stack((Z[:, 0], Z[:, 1], Z[:, 2] ** 2 - 1.0, np.sin(Z[:, 3])))
    logistic = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=max_iter,
        tol=tolerance,
        fit_intercept=True,
    )
    logistic.fit(features, X_train[:, 0])
    representer = LogisticReferenceRepresenter(logistic, True, "converged")

    design = np.column_stack((np.ones(X_train.shape[0]), features))
    e = representer.predict_propensity(X_train)
    information = design.T @ ((e * (1.0 - e))[:, None] * design)
    covariance_beta = np.linalg.pinv(information)
    D = X_train[:, 0]
    derivative = -(D * (1.0 - e) / e + (1.0 - D) * e / (1.0 - e))
    jacobian = derivative[:, None] * design
    gram_alpha = (jacobian.T @ jacobian) / X_train.shape[0]
    alpha_radius = _ellipsoid_l2_radius(
        covariance_beta,
        gram_alpha,
        confidence_probability,
    )
    gamma_radius = _outcome_coefficient_radius(
        outcome,
        X_train,
        y_train,
        confidence_probability,
    )
    return ReferenceFit(
        mode="estimated",
        representer=representer,
        outcome=outcome,
        bias_allowance=alpha_radius * gamma_radius,
        status="converged",
    )


def fit_rff_squared_reference(
    X_train: FloatArray,
    *,
    n_features: int,
    lam: float,
    seed: int,
    tolerance: float = 1e-8,
    max_iter: int = 3000,
) -> RFFSquaredReference:
    """Fit the high-dimensional squared reference with conjugate gradients."""

    basis = RBFRandomFourierBasis(
        n_features=n_features,
        sigma="auto",
        include_bias=True,
        standardize=True,
        random_state=seed,
    ).fit(X_train)
    phi = np.asarray(basis(X_train), dtype=float)
    m_matrix = np.asarray(ATEFunctional(0).m_basis_matrix(X_train, basis), dtype=float)
    n, p = phi.shape
    rhs = m_matrix.mean(axis=0)

    def matvec(beta: FloatArray) -> FloatArray:
        return 0.5 * (phi.T @ (phi @ beta)) / n + lam * beta

    operator = LinearOperator((p, p), matvec=matvec, dtype=float)
    beta, info = cg(operator, rhs, atol=tolerance, rtol=tolerance, maxiter=max_iter)
    success = bool(info == 0 and np.all(np.isfinite(beta)))
    status = "converged" if info == 0 else f"cg_status_{info}"
    return RFFSquaredReference(basis=basis, beta=np.asarray(beta, dtype=float), success=success, status=status)


def fit_high_dimensional_reference(
    X_train: FloatArray,
    *,
    outcome: OutcomeFit,
    seed: int,
    reference_constant: float,
    n_evaluation: int,
) -> ReferenceFit:
    representer = fit_rff_squared_reference(
        X_train,
        n_features=2000,
        lam=1e-3,
        seed=seed,
    )
    return ReferenceFit(
        mode="estimated",
        representer=representer if representer.success else None,
        outcome=outcome,
        bias_allowance=reference_constant / np.sqrt(n_evaluation),
        status=representer.status,
    )


def truth_reference(
    *,
    design: Literal["low", "high"],
    overlap_scale: float,
) -> ReferenceFit:
    return ReferenceFit(
        mode="truth",
        representer=TruthRepresenter(overlap_scale, design),
        outcome=TruthOutcome(design),
        bias_allowance=0.0,
        status="truth",
    )


def gaussian_multiplier_mean_radii(
    values: FloatArray,
    *,
    delta: float,
    draws: int,
    seed: int,
) -> FloatArray:
    """Return simultaneous Gaussian-multiplier radii for column means."""

    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    centered = values - values.mean(axis=0)
    sd = centered.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    rng = np.random.default_rng(seed)
    multipliers = rng.normal(size=(draws, n))
    boot_means = (multipliers @ centered) / np.sqrt(n)
    standardized = np.abs(boot_means / sd_safe)
    critical = float(np.quantile(np.max(standardized, axis=1), 1.0 - delta))
    return critical * sd / np.sqrt(n)


def gaussian_multiplier_variance_upper(
    values: FloatArray,
    *,
    delta: float,
    draws: int,
    seed: int,
) -> FloatArray:
    """Return simultaneous upper bounds for the column variances."""

    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    centered = values - values.mean(axis=0)
    squared = centered * centered
    variances = squared.mean(axis=0)
    squared_centered = squared - variances
    sd_squared = squared_centered.std(axis=0, ddof=1)
    sd_safe = np.where(sd_squared > 1e-12, sd_squared, 1.0)
    rng = np.random.default_rng(seed)
    multipliers = rng.normal(size=(draws, n))
    boot = (multipliers @ squared_centered) / np.sqrt(n)
    standardized = np.abs(boot / sd_safe)
    critical = float(np.quantile(np.max(standardized, axis=1), 1.0 - delta))
    upper = variances + critical * sd_squared / np.sqrt(n)
    return np.maximum(upper, 0.0)


def bias_aware_critical_value(t: float, coverage: float) -> float:
    """Critical value for a normal mean bounded in absolute value by t."""

    if t < 0:
        raise ValueError("t must be nonnegative.")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0, 1).")

    def equation(c: float) -> float:
        probability = stats.norm.cdf(c - t) - stats.norm.cdf(-c - t)
        return probability - coverage

    upper = t + stats.norm.ppf((1.0 + coverage) / 2.0) + 10.0
    return float(optimize.brentq(equation, 0.0, upper))


def _candidate_scores(
    candidates: Sequence[RepresenterFit],
    X: FloatArray,
    y: FloatArray,
    outcome: OutcomeFit,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    n_candidates = len(candidates)
    scores = np.full((X.shape[0], n_candidates), np.nan, dtype=float)
    admissible = np.zeros(n_candidates, dtype=bool)
    max_weight = np.full(n_candidates, np.nan, dtype=float)
    m_gamma = _m_ate(X, outcome)
    gamma = outcome.predict(X)
    residual = y - gamma
    for j, candidate in enumerate(candidates):
        if not candidate.success:
            continue
        alpha = candidate.predict(X)
        finite = np.all(np.isfinite(alpha))
        if not finite:
            continue
        score = m_gamma + alpha * residual
        if not np.all(np.isfinite(score)):
            continue
        scores[:, j] = score
        admissible[j] = True
        max_weight[j] = float(np.max(np.abs(alpha)))
    return scores, admissible, max_weight


def _reference_score(
    reference: ReferenceFit,
    X: FloatArray,
    y: FloatArray,
) -> FloatArray:
    if reference.representer is None or reference.outcome is None:
        raise RuntimeError(f"Reference estimator is unavailable: {reference.status}")
    return score_contribution(X, y, reference.representer, reference.outcome)


def _audit_scores(
    candidates: Sequence[RepresenterFit],
    outcome: OutcomeFit,
    integration: GeneratedData,
) -> tuple[FloatArray, FloatArray]:
    scores, admissible, _ = _candidate_scores(candidates, integration.X, integration.y, outcome)
    bias = np.full(len(candidates), np.nan, dtype=float)
    variance = np.full(len(candidates), np.nan, dtype=float)
    for j in range(len(candidates)):
        if not admissible[j]:
            continue
        bias[j] = float(np.mean(scores[:, j]) - 1.0)
        variance[j] = float(np.var(scores[:, j], ddof=0))
    return bias, variance


def _fold_result(
    *,
    config: SimulationConfig,
    data: GeneratedData,
    integration: GeneratedData,
    roles: FoldRoles,
    candidates_specs: tuple[CandidateSpec, ...],
    reference_mode: Literal["truth", "estimated"],
    reference_constant: float,
    scenario_seed: int,
    repetition: int,
    fold_index: int,
    sample_size: int,
    overlap_scale: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    X_train = data.X[roles.training]
    y_train = data.y[roles.training]
    X_diag = data.X[roles.diagnostic]
    y_diag = data.y[roles.diagnostic]
    X_eval = data.X[roles.evaluation]
    y_eval = data.y[roles.evaluation]

    outcome = fit_outcome(
        X_train,
        y_train,
        design=config.design,
        seed=_stable_seed(scenario_seed, repetition, fold_index, "outcome"),
    )
    candidates = tuple(
        fit_candidate(
            X_train,
            spec,
            max_iter=config.max_iter,
            tolerance=config.optimization_tolerance,
            gradient_tolerance=config.optimization_gradient_tolerance,
        )
        for spec in candidates_specs
    )

    fold_delta = config.diagnostic_delta / config.n_folds

    if reference_mode == "truth":
        reference = truth_reference(design=config.design, overlap_scale=overlap_scale)
    elif config.design == "low":
        reference = fit_low_dimensional_reference(
            X_train,
            y_train,
            outcome=outcome,
            confidence_probability=1.0 - fold_delta / 4.0,
            max_iter=config.max_iter,
            tolerance=config.optimization_tolerance,
            gradient_tolerance=config.optimization_gradient_tolerance,
        )
    else:
        reference = fit_high_dimensional_reference(
            X_train,
            outcome=outcome,
            seed=_stable_seed(scenario_seed, repetition, fold_index, "reference"),
            reference_constant=reference_constant,
            n_evaluation=len(roles.evaluation),
        )

    diagnostic_scores, admissible, max_weight = _candidate_scores(candidates, X_diag, y_diag, outcome)
    reference_diag = _reference_score(reference, X_diag, y_diag)
    valid_columns = np.where(admissible)[0]
    if valid_columns.size == 0:
        raise RuntimeError("Every candidate failed on a diagnostic fold.")
    differences = diagnostic_scores[:, valid_columns] - reference_diag[:, None]
    q_valid = gaussian_multiplier_mean_radii(
        differences,
        delta=fold_delta / 2.0,
        draws=config.multiplier_draws,
        seed=_stable_seed(scenario_seed, repetition, fold_index, reference_mode, "mean_bootstrap"),
    )
    relative_bias = differences.mean(axis=0)
    upper_bias = np.abs(relative_bias) + q_valid + reference.bias_allowance

    candidate_diag_score = diagnostic_scores[:, valid_columns]
    variance_upper = gaussian_multiplier_variance_upper(
        candidate_diag_score,
        delta=fold_delta / 2.0,
        draws=config.multiplier_draws,
        seed=_stable_seed(scenario_seed, repetition, fold_index, reference_mode, "variance_bootstrap"),
    )
    criterion = upper_bias * upper_bias + variance_upper / len(roles.evaluation)
    selected_local = int(np.argmin(criterion))
    selected_index = int(valid_columns[selected_local])

    evaluation_scores, evaluation_admissible, _ = _candidate_scores(candidates, X_eval, y_eval, outcome)
    if not evaluation_admissible[selected_index]:
        raise RuntimeError("The selected candidate became nonfinite on the evaluation fold.")
    selected_scores = evaluation_scores[:, selected_index]
    theta_fold = float(selected_scores.mean())
    se_fold = float(selected_scores.std(ddof=1) / np.sqrt(selected_scores.size))
    u_fold = float(upper_bias[selected_local])

    audit_bias, audit_variance = _audit_scores(candidates, outcome, integration)
    audit_risk = audit_bias * audit_bias + audit_variance / len(roles.evaluation)
    oracle_index = int(np.nanargmin(audit_risk))

    rows: list[dict[str, object]] = []
    valid_lookup = {int(global_index): j for j, global_index in enumerate(valid_columns)}
    for j, candidate in enumerate(candidates):
        diag_position = valid_lookup.get(j)
        row = {
            "experiment": config.name,
            "design": config.design,
            "sample_size": sample_size,
            "overlap_scale": overlap_scale,
            "reference_mode": reference_mode,
            "reference_constant": reference_constant,
            "repetition": repetition,
            "fold": fold_index,
            "candidate": candidate.spec.label,
            "loss": candidate.spec.loss,
            "omega": candidate.spec.omega,
            "dictionary": candidate.spec.dictionary,
            "penalty_multiplier": candidate.spec.penalty_multiplier,
            "fit_success": candidate.success,
            "fit_status": candidate.status,
            "objective": candidate.objective,
            "gradient_norm": candidate.gradient_norm,
            "kkt_residual": candidate.kkt_residual,
            "binding_rate": candidate.binding_rate,
            "max_abs_alpha_diagnostic": max_weight[j],
            "relative_score_difference": np.nan if diag_position is None else relative_bias[diag_position],
            "diagnostic_radius": np.nan if diag_position is None else q_valid[diag_position],
            "bias_upper_bound": np.nan if diag_position is None else upper_bias[diag_position],
            "variance_upper_bound": np.nan if diag_position is None else variance_upper[diag_position],
            "selection_criterion": np.nan if diag_position is None else criterion[diag_position],
            "audit_bias": audit_bias[j],
            "audit_variance": audit_variance[j],
            "audit_risk": audit_risk[j],
            "bias_bound_covers": bool(np.isfinite(audit_bias[j]) and diag_position is not None and abs(audit_bias[j]) <= upper_bias[diag_position]),
            "selected": j == selected_index,
            "oracle": j == oracle_index,
            "reference_bias_allowance": reference.bias_allowance,
            "reference_status": reference.status,
        }
        rows.append(row)

    fold_summary = {
        "experiment": config.name,
        "design": config.design,
        "sample_size": sample_size,
        "overlap_scale": overlap_scale,
        "reference_mode": reference_mode,
        "reference_constant": reference_constant,
        "repetition": repetition,
        "fold": fold_index,
        "selected_candidate": candidates[selected_index].spec.label,
        "selected_theta": theta_fold,
        "selected_se": se_fold,
        "selected_bias_upper_bound": u_fold,
        "selected_audit_bias": audit_bias[selected_index],
        "selected_audit_risk": audit_risk[selected_index],
        "oracle_candidate": candidates[oracle_index].spec.label,
        "oracle_audit_risk": audit_risk[oracle_index],
        "reference_bias_allowance": reference.bias_allowance,
        "reference_status": reference.status,
        "n_evaluation": len(roles.evaluation),
    }
    return rows, fold_summary


def _one_repetition(
    job: tuple[SimulationConfig, int, float, str, float, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config, sample_size, overlap_scale, reference_mode, reference_constant, repetition = job
    scenario_seed = _stable_seed(config.base_seed, config.name, sample_size, overlap_scale)
    data = generate_data(
        n=sample_size,
        design=config.design,
        overlap_scale=overlap_scale,
        seed=_stable_seed(scenario_seed, repetition, "sample"),
    )
    integration = generate_data(
        n=config.integration_size,
        design=config.design,
        overlap_scale=overlap_scale,
        seed=_stable_seed(scenario_seed, repetition, "integration"),
    )
    roles = make_fold_roles(
        sample_size,
        config.n_folds,
        _stable_seed(scenario_seed, repetition, "folds"),
    )
    specs = candidate_grid()
    candidate_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    evaluation_scores: list[FloatArray] = []
    fold_bias_bounds: list[float] = []
    fold_standard_errors: list[float] = []

    for k, fold_roles in enumerate(roles):
        rows, fold_summary = _fold_result(
            config=config,
            data=data,
            integration=integration,
            roles=fold_roles,
            candidates_specs=specs,
            reference_mode=reference_mode,
            reference_constant=reference_constant,
            scenario_seed=scenario_seed,
            repetition=repetition,
            fold_index=k,
            sample_size=sample_size,
            overlap_scale=overlap_scale,
        )
        candidate_rows.extend(rows)
        fold_rows.append(fold_summary)
        selected_label = str(fold_summary["selected_candidate"])
        selected_spec_index = next(j for j, spec in enumerate(specs) if spec.label == selected_label)
        X_train = data.X[fold_roles.training]
        y_train = data.y[fold_roles.training]
        outcome = fit_outcome(
            X_train,
            y_train,
            design=config.design,
            seed=_stable_seed(scenario_seed, repetition, k, "outcome"),
        )
        selected_fit = fit_candidate(
            X_train,
            specs[selected_spec_index],
            max_iter=config.max_iter,
            tolerance=config.optimization_tolerance,
            gradient_tolerance=config.optimization_gradient_tolerance,
        )
        score = score_contribution(
            data.X[fold_roles.evaluation],
            data.y[fold_roles.evaluation],
            selected_fit,
            outcome,
        )
        evaluation_scores.append(score)
        fold_bias_bounds.append(float(fold_summary["selected_bias_upper_bound"]))
        fold_standard_errors.append(float(fold_summary["selected_se"]))

    all_scores = np.concatenate(evaluation_scores)
    estimate = float(all_scores.mean())
    ordinary_se = float(all_scores.std(ddof=1) / np.sqrt(all_scores.size))
    ordinary_z = float(stats.norm.ppf(1.0 - config.interval_miscoverage / 2.0))
    ordinary_low = estimate - ordinary_z * ordinary_se
    ordinary_high = estimate + ordinary_z * ordinary_se

    weights = np.asarray([len(r.evaluation) / sample_size for r in roles], dtype=float)
    total_bias_bound = float(np.sum(weights * np.asarray(fold_bias_bounds)))
    fold_alpha = (config.interval_miscoverage - config.diagnostic_delta) / (2.0 * config.n_folds)
    conservative_z = float(stats.norm.ppf(1.0 - fold_alpha))
    conservative_half = float(
        np.sum(weights * (np.asarray(fold_bias_bounds) + conservative_z * np.asarray(fold_standard_errors)))
    )
    exact_coverage = 1.0 - (config.interval_miscoverage - config.diagnostic_delta)
    exact_critical = bias_aware_critical_value(total_bias_bound / ordinary_se, exact_coverage)
    exact_half = exact_critical * ordinary_se

    summary = {
        "experiment": config.name,
        "design": config.design,
        "sample_size": sample_size,
        "overlap_scale": overlap_scale,
        "reference_mode": reference_mode,
        "reference_constant": reference_constant,
        "repetition": repetition,
        "estimate": estimate,
        "bias": estimate - 1.0,
        "squared_error": (estimate - 1.0) ** 2,
        "ordinary_se": ordinary_se,
        "ordinary_ci_low": ordinary_low,
        "ordinary_ci_high": ordinary_high,
        "ordinary_coverage": ordinary_low <= 1.0 <= ordinary_high,
        "bias_upper_bound": total_bias_bound,
        "bias_aware_ci_low": estimate - exact_half,
        "bias_aware_ci_high": estimate + exact_half,
        "bias_aware_coverage": estimate - exact_half <= 1.0 <= estimate + exact_half,
        "conservative_ci_low": estimate - conservative_half,
        "conservative_ci_high": estimate + conservative_half,
        "conservative_coverage": estimate - conservative_half <= 1.0 <= estimate + conservative_half,
        "bias_aware_length": 2.0 * exact_half,
        "conservative_length": 2.0 * conservative_half,
        "ordinary_length": 2.0 * ordinary_z * ordinary_se,
    }
    return pd.DataFrame(candidate_rows), pd.DataFrame(fold_rows), pd.DataFrame([summary])


def _jobs(config: SimulationConfig) -> list[tuple[SimulationConfig, int, float, str, float, int]]:
    jobs: list[tuple[SimulationConfig, int, float, str, float, int]] = []
    for sample_size in config.sample_sizes:
        for scale in config.overlap_scales:
            for reference_mode in config.reference_modes:
                constants = (0.0,)
                if reference_mode == "estimated" and config.design == "high":
                    constants = config.high_dim_reference_constants
                for reference_constant in constants:
                    for repetition in range(config.replications):
                        jobs.append(
                            (
                                config,
                                sample_size,
                                scale,
                                reference_mode,
                                reference_constant,
                                repetition,
                            )
                        )
    return jobs


def run_experiment(config: SimulationConfig, output_dir: str | Path) -> None:
    """Run a full publication configuration and write batched Parquet files."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame([asdict(config)])
    metadata.to_json(output / "configuration.json", orient="records", indent=2)

    jobs = _jobs(config)
    batches = [jobs[i : i + config.batch_size] for i in range(0, len(jobs), config.batch_size)]
    for batch_index, batch in enumerate(batches):
        candidate_path = output / f"candidate_results_{batch_index:05d}.parquet"
        fold_path = output / f"fold_results_{batch_index:05d}.parquet"
        repetition_path = output / f"repetition_results_{batch_index:05d}.parquet"
        if candidate_path.exists() and fold_path.exists() and repetition_path.exists():
            continue
        if config.max_workers == 1:
            results = [_one_repetition(job) for job in batch]
        else:
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                results = list(executor.map(_one_repetition, batch))
        pd.concat([x[0] for x in results], ignore_index=True).to_parquet(candidate_path, index=False)
        pd.concat([x[1] for x in results], ignore_index=True).to_parquet(fold_path, index=False)
        pd.concat([x[2] for x in results], ignore_index=True).to_parquet(repetition_path, index=False)


def load_experiment(output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load every completed batch from an experiment directory."""

    output = Path(output_dir)
    candidate_files = sorted(output.glob("candidate_results_*.parquet"))
    fold_files = sorted(output.glob("fold_results_*.parquet"))
    repetition_files = sorted(output.glob("repetition_results_*.parquet"))
    if not candidate_files or not fold_files or not repetition_files:
        raise FileNotFoundError(f"No completed experiment batches were found in {output}.")
    candidates = pd.concat([pd.read_parquet(path) for path in candidate_files], ignore_index=True)
    folds = pd.concat([pd.read_parquet(path) for path in fold_files], ignore_index=True)
    repetitions = pd.concat([pd.read_parquet(path) for path in repetition_files], ignore_index=True)
    return candidates, folds, repetitions


def summarize_repetitions(repetitions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bias, RMSE, coverage, and interval length."""

    group = ["experiment", "design", "sample_size", "overlap_scale", "reference_mode", "reference_constant"]
    summary = (
        repetitions.groupby(group, dropna=False)
        .agg(
            replications=("repetition", "nunique"),
            bias=("bias", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            ordinary_coverage=("ordinary_coverage", "mean"),
            bias_aware_coverage=("bias_aware_coverage", "mean"),
            conservative_coverage=("conservative_coverage", "mean"),
            ordinary_length=("ordinary_length", "mean"),
            bias_aware_length=("bias_aware_length", "mean"),
            conservative_length=("conservative_length", "mean"),
            mean_bias_bound=("bias_upper_bound", "mean"),
        )
        .reset_index()
    )
    for column in ("ordinary_coverage", "bias_aware_coverage", "conservative_coverage"):
        summary[f"{column}_mcse"] = np.sqrt(summary[column] * (1.0 - summary[column]) / summary["replications"])
    return summary


PRIMARY_LOW_DIMENSIONAL = SimulationConfig(
    name="reference_selection_low",
    design="low",
    sample_sizes=(1000, 3000),
    overlap_scales=(0.5, 1.5, 2.5),
    replications=2000,
)

PRIMARY_HIGH_DIMENSIONAL = SimulationConfig(
    name="reference_selection_high",
    design="high",
    sample_sizes=(3000,),
    overlap_scales=(0.75, 2.0),
    replications=1000,
)

SENSITIVITY_CONFIGURATIONS = (
    replace(PRIMARY_LOW_DIMENSIONAL, name="reference_selection_low_n1500", sample_sizes=(1500,), replications=1000),
    replace(PRIMARY_HIGH_DIMENSIONAL, name="reference_selection_high_n5000", sample_sizes=(5000,), replications=1000),
)
