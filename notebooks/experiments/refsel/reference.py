"""Reference estimators and their bias allowances.

Section 9 of ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``. Four references
are available. ``misspecified`` is the important addition: it drops two terms
from the propensity index while keeping the *same* allowance formula, so the
stated ``b_r`` no longer bounds the reference drift. That is the situation a
referee asks about, and section 9.3 measures both the damage and whether the
pairwise reference check detects it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.linear_model import LogisticRegression

from genriesz.basis import BaseBasis, RBFRandomFourierBasis
from genriesz.functionals import ATEFunctional
from genriesz.glm import OutcomeGLM

from .candidates import OutcomeFit
from .dgp import Design, FloatArray, hidden_direction, true_outcome, true_representer

ReferenceName = Literal["truth", "correct", "misspecified", "rff"]


def _symmetric_matrix_sqrt(A: FloatArray) -> FloatArray:
    vals, vecs = np.linalg.eigh((A + A.T) / 2.0)
    vals = np.maximum(vals, 0.0)
    return (vecs * np.sqrt(vals)) @ vecs.T


def ellipsoid_l2_radius(covariance: FloatArray, gram: FloatArray, probability: float) -> float:
    """Return ``sqrt(c^2 lambda_max(Omega^{1/2} G Omega^{1/2}))``.

    ``covariance`` is the coefficient covariance, ``gram`` the prediction Gram
    matrix built from the Jacobian of the prediction map, and ``probability``
    the ellipsoid coverage level.
    """

    if covariance.size == 0:
        return 0.0
    root = _symmetric_matrix_sqrt(covariance)
    matrix = root @ gram @ root
    largest = float(np.max(np.linalg.eigvalsh((matrix + matrix.T) / 2.0)))
    quantile = float(stats.chi2.ppf(probability, covariance.shape[0]))
    return float(np.sqrt(max(quantile * largest, 0.0)))


def outcome_coefficient_radius(
    outcome: OutcomeFit,
    X_train: FloatArray,
    y_train: FloatArray,
    probability: float,
) -> float:
    """Sandwich-based prediction radius for a linear outcome series."""

    model = outcome.model
    basis = getattr(model, "basis", None)
    if outcome.kind != "series" or basis is None:
        raise ValueError("A coefficient radius requires the linear outcome estimator.")
    phi = np.asarray(basis(X_train), dtype=float)
    residual = y_train - outcome.predict(X_train)
    bread = np.linalg.pinv(phi.T @ phi)
    meat = phi.T @ ((residual * residual)[:, None] * phi)
    covariance = bread @ meat @ bread
    gram = (phi.T @ phi) / phi.shape[0]
    return ellipsoid_l2_radius(covariance, gram, probability)


class ReferenceEstimator:
    """Common interface for the reference score."""

    name: str
    status: str
    honest_allowance: float
    #: Whether the fit met its convergence and finiteness conditions. A reference
    #: that failed carries no allowance guarantee, so the fold must not use it.
    success: bool = True

    def alpha(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def gamma(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def contrast(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def score(self, X: FloatArray, y: FloatArray) -> FloatArray:
        """Orthogonal score contribution of the reference on a held-out sample."""

        return self.contrast(X) + self.alpha(X) * (y - self.gamma(X))

    def allowance(self, rho: float) -> float:
        """Return the allowance scaled by ``rho``.

        ``rho = 1`` is the honest allowance; smaller values quantify how much
        the coverage guarantee leans on it.
        """

        return float(rho) * self.honest_allowance


class TruthReference(ReferenceEstimator):
    """Infeasible reference built from simulation truth. ``b_r = 0``."""

    name = "truth"

    def __init__(self, *, design: Design, overlap_scale: float, hidden_scale: float):
        self.design = design
        self.overlap_scale = overlap_scale
        self.hidden_scale = hidden_scale
        self.status = "truth"
        self.honest_allowance = 0.0
        self.success = True

    def alpha(self, X: FloatArray) -> FloatArray:
        return true_representer(
            X,
            design=self.design,
            overlap_scale=self.overlap_scale,
            hidden_scale=self.hidden_scale,
        )

    def gamma(self, X: FloatArray) -> FloatArray:
        return true_outcome(X, design=self.design, hidden_scale=self.hidden_scale)

    def contrast(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        X1 = X.copy()
        X0 = X.copy()
        X1[:, 0] = 1.0
        X0[:, 0] = 0.0
        return self.gamma(X1) - self.gamma(X0)


def _correct_features(X: FloatArray) -> FloatArray:
    """Regressors spanning the true low-dimensional treatment index.

    The hidden direction is included. The reference is the estimator whose bias
    the procedure claims to bound, so it must remain correctly specified as the
    calibrated hidden scale grows; otherwise the premise of Theorem
    ``data_dependent_bias`` fails for every candidate at once. The candidate
    dictionaries never contain it, which is what creates the bias being swept.
    """

    Z = np.asarray(X, dtype=float)[:, 1:]
    return np.column_stack(
        (Z[:, 0], Z[:, 1], Z[:, 2] ** 2 - 1.0, np.sin(Z[:, 3]), hidden_direction(Z))
    )


def _misspecified_features(X: FloatArray) -> FloatArray:
    """Linear terms only: quadratic, sine, and hidden components are omitted."""

    Z = np.asarray(X, dtype=float)[:, 1:]
    return np.column_stack((Z[:, 0], Z[:, 1]))


class ReferenceOutcomeBasis(BaseBasis):
    """Correctly specified reference outcome series, including the hidden term."""

    @property
    def n_features(self) -> int:
        return 14

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
                hidden_direction(Z),
            )
        )
        return np.column_stack((D * base, (1.0 - D) * base))


class MisspecifiedOutcomeBasis(BaseBasis):
    """Intercept and raw covariates only."""

    @property
    def n_features(self) -> int:
        return 12

    def __call__(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        D = X[:, [0]]
        Z = X[:, 1:]
        base = np.column_stack((np.ones(X.shape[0]), Z))
        return np.column_stack((D * base, (1.0 - D) * base))


class LogisticReference(ReferenceEstimator):
    """Logistic propensity reference with its own outcome series.

    The reference carries its own outcome estimator rather than sharing the
    fold's. The fold's estimator is deliberately unable to represent the hidden
    direction, because that is what makes the candidates biased; a reference
    built on it would inherit the same bias and no allowance could bound it.

    The allowance is the product of two linearized confidence-ellipsoid radii.
    It bounds sampling error around the pseudo-true coefficient, not
    approximation error, so for the ``misspecified`` variant the stated
    allowance is deliberately too small.
    """

    def __init__(
        self,
        name: ReferenceName,
        model: LogisticRegression,
        outcome: OutcomeFit,
        *,
        feature_map,
        honest_allowance: float,
        status: str,
    ):
        self.name = name
        self.success = status == "converged"
        self.model = model
        self.outcome = outcome
        self._feature_map = feature_map
        self.honest_allowance = honest_allowance
        self.status = status

    def propensity(self, X: FloatArray) -> FloatArray:
        return np.asarray(
            self.model.predict_proba(self._feature_map(X))[:, 1], dtype=float
        )

    def alpha(self, X: FloatArray) -> FloatArray:
        X = np.asarray(X, dtype=float)
        D = X[:, 0]
        e = self.propensity(X)
        return D / e - (1.0 - D) / (1.0 - e)

    def gamma(self, X: FloatArray) -> FloatArray:
        return self.outcome.predict(X)

    def contrast(self, X: FloatArray) -> FloatArray:
        return self.outcome.contrast(X)


def fit_reference_outcome(
    X_train: FloatArray, y_train: FloatArray, *, name: ReferenceName
) -> OutcomeFit:
    """Fit the reference's own outcome series."""

    basis = ReferenceOutcomeBasis() if name == "correct" else MisspecifiedOutcomeBasis()
    model = OutcomeGLM(basis=basis, link="identity", penalty="l2", lam=0.0)
    fit = model.fit(X_train, y_train)
    if not fit.success:
        raise RuntimeError(f"The {name} reference outcome estimator failed: {fit.status}")
    return OutcomeFit(kind="series", model=model)


def fit_logistic_reference(
    X_train: FloatArray,
    y_train: FloatArray,
    *,
    name: ReferenceName,
    ellipsoid_probability: float,
    max_iter: int,
    tolerance: float,
) -> LogisticReference:
    """Fit a logistic propensity reference, its outcome series, and its allowance."""

    outcome = fit_reference_outcome(X_train, y_train, name=name)
    feature_map = _correct_features if name == "correct" else _misspecified_features
    features = feature_map(X_train)
    # scikit-learn 1.8 deprecates ``penalty=None`` and directs callers to
    # ``C=np.inf``, which then emits a self-contradictory UserWarning about
    # ignoring C. The message is filtered narrowly rather than with a blanket
    # suppression so that genuine convergence warnings still surface.
    model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=max_iter,
        tol=tolerance,
        fit_intercept=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Setting penalty=None will ignore the C and l1_ratio parameters",
            category=UserWarning,
        )
        model.fit(features, X_train[:, 0])

    design = np.column_stack((np.ones(X_train.shape[0]), features))
    e = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    information = design.T @ ((e * (1.0 - e))[:, None] * design)
    covariance_beta = np.linalg.pinv(information)
    D = X_train[:, 0]
    # d alpha / d beta = -(D (1-e)/e + (1-D) e/(1-e)) * design
    derivative = -(D * (1.0 - e) / e + (1.0 - D) * e / (1.0 - e))
    jacobian = derivative[:, None] * design
    gram_alpha = (jacobian.T @ jacobian) / X_train.shape[0]
    alpha_radius = ellipsoid_l2_radius(covariance_beta, gram_alpha, ellipsoid_probability)
    gamma_radius = outcome_coefficient_radius(outcome, X_train, y_train, ellipsoid_probability)
    allowance = float(alpha_radius * gamma_radius)

    # The allowance is the quantity experiments E2 and E5 measure, so a silent
    # failure here would fail open on exactly the wrong thing. Under weak overlap
    # the unpenalized logistic fit approaches separation, e goes to zero or one,
    # and the Jacobian -(D(1-e)/e + (1-D)e/(1-e)) diverges.
    iterations = int(np.max(np.atleast_1d(model.n_iter_)))
    if iterations >= max_iter:
        status = "logistic_max_iter"
    elif not np.all(np.isfinite(e)) or np.min(np.minimum(e, 1.0 - e)) <= 1e-10:
        status = "logistic_separation"
    elif not np.isfinite(allowance):
        status = "nonfinite_allowance"
    else:
        status = "converged"
    return LogisticReference(
        name,
        model,
        outcome,
        feature_map=feature_map,
        honest_allowance=allowance,
        status=status,
    )


class RFFSquaredReference(ReferenceEstimator):
    """High-dimensional reference: squared Riesz regression on random features."""

    name: ReferenceName = "rff"

    def __init__(
        self,
        basis: RBFRandomFourierBasis,
        beta: FloatArray,
        outcome: OutcomeFit,
        *,
        honest_allowance: float,
        status: str,
        success: bool,
    ):
        self.basis = basis
        self.beta = beta
        self.outcome = outcome
        self.honest_allowance = honest_allowance
        self.status = status
        self.success = success

    def alpha(self, X: FloatArray) -> FloatArray:
        if not self.success:
            raise RuntimeError(f"The random-feature reference is unavailable: {self.status}")
        phi = np.asarray(self.basis(X), dtype=float)
        return 0.5 * (phi @ self.beta)

    def gamma(self, X: FloatArray) -> FloatArray:
        return self.outcome.predict(X)

    def contrast(self, X: FloatArray) -> FloatArray:
        return self.outcome.contrast(X)


def fit_rff_reference(
    X_train: FloatArray,
    *,
    outcome: OutcomeFit,
    n_features: int,
    lam: float,
    seed: int,
    reference_constant: float,
    n_evaluation: int,
    tolerance: float = 1e-8,
    max_iter: int = 3000,
) -> RFFSquaredReference:
    """Solve the random-feature squared Riesz problem by conjugate gradients."""

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
    beta = np.asarray(beta, dtype=float)
    success = bool(info == 0 and np.all(np.isfinite(beta)))
    return RFFSquaredReference(
        basis,
        beta,
        outcome,
        honest_allowance=float(reference_constant / np.sqrt(n_evaluation)),
        status="converged" if info == 0 else f"cg_status_{info}",
        success=success,
    )


@dataclass(frozen=True)
class ReferenceCheck:
    """Result of the pairwise consistency check of Proposition ``several_references``."""

    first: str
    second: str
    difference: float
    radius: float
    allowance_sum: float

    @property
    def checkable(self) -> bool:
        """Whether every ingredient is finite.

        Without this the comparison fails open: ``abs(nan) > nan`` is ``False``,
        so a reference whose score blew up would be recorded as having passed the
        check rather than as undecidable.
        """

        return bool(
            np.isfinite(self.difference)
            and np.isfinite(self.radius)
            and np.isfinite(self.allowance_sum)
        )

    @property
    def violated(self) -> bool | None:
        if not self.checkable:
            return None
        return bool(abs(self.difference) > self.radius + self.allowance_sum)


def reference_check(
    first: ReferenceEstimator,
    second: ReferenceEstimator,
    X_diag: FloatArray,
    y_diag: FloatArray,
    *,
    delta: float,
    rho: float = 1.0,
) -> ReferenceCheck:
    """Evaluate ``|D_{r,s}| <= q_{r,s} + b_r + b_s`` on the diagnostic sample.

    A violation proves that at least one allowance or the concentration bound
    has failed, without saying which.

    The radius is a single-comparison normal radius for the mean of one
    difference, not a simultaneous radius over the candidate family. Reusing the
    candidate radius here would inflate the threshold by more than an order of
    magnitude and the check could never fire.
    """

    values = first.score(X_diag, y_diag) - second.score(X_diag, y_diag)
    difference = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    radius = float(stats.norm.ppf(1.0 - delta / 2.0) * standard_error)
    return ReferenceCheck(
        first=first.name,
        second=second.name,
        difference=difference,
        radius=radius,
        allowance_sum=first.allowance(rho) + second.allowance(rho),
    )
