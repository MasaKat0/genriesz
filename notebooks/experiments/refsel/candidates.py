"""Candidate representer specifications and their shared fold-level fitting.

Section 7 of ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``. All ninety
candidates in a fold share three dictionaries, so the basis is built three times
per fold rather than ninety times, and every selection rule reads the same
fitted library.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor

from genriesz.basis import BaseBasis
from genriesz.functionals import ATEFunctional
from genriesz.generators import (
    BKLGenerator,
    BPGenerator,
    BregmanGenerator,
    DomainError,
    SquaredGenerator,
    UKLGenerator,
)
from genriesz.glm import GRRGLM, OutcomeGLM

from .dgp import Design, FloatArray

Dictionary = Literal["linear", "second_order", "rich"]
Loss = Literal["SQ", "UKL", "BKL", "BP"]

#: Gradient infinity-norm above which a converged fit is still rejected.
GRADIENT_TOLERANCE = 1e-2


class ScaledGenerator(BregmanGenerator):
    """Wrap a generator ``g`` and represent ``kappa * g``.

    Used only by the rescaling demonstration (plan section 4). With ``lam = 0``
    the fitted representer is invariant while the objective value and the
    held-out Bregman criterion are multiplied by ``kappa``, which is why raw
    Bregman objectives cannot rank candidates across generators.

    Subclassing :class:`BregmanGenerator` rather than the concrete generator is
    deliberate: ``GRRGLM`` dispatches a closed form on
    ``isinstance(generator, SquaredGenerator)`` whose algebra assumes an
    unscaled squared generator.
    """

    def __init__(self, inner: BregmanGenerator, kappa: float):
        if kappa <= 0.0:
            raise ValueError("kappa must be positive.")
        super().__init__(name=f"{kappa:g}x{inner.name}", C=inner.C, branch_fn=inner.branch_fn)
        self.inner = inner
        self.kappa = float(kappa)
        self.modifies_estimand = inner.modifies_estimand

    def g(self, X, alpha):  # type: ignore[override]
        return self.kappa * np.asarray(self.inner.g(X, alpha), dtype=float)

    def grad(self, X, alpha):  # type: ignore[override]
        return self.kappa * np.asarray(self.inner.grad(X, alpha), dtype=float)

    def grad2(self, X, alpha):  # type: ignore[override]
        return self.kappa * np.asarray(self.inner.grad2(X, alpha), dtype=float)

    def inv_grad(self, X, v):  # type: ignore[override]
        return self.inner.inv_grad(X, np.asarray(v, dtype=float) / self.kappa)

    def domain_binding(self, X, v):  # type: ignore[override]
        return self.inner.domain_binding(X, np.asarray(v, dtype=float) / self.kappa)


@dataclass(frozen=True)
class CandidateSpec:
    """One representer specification in the candidate library."""

    loss: Loss
    dictionary: Dictionary
    penalty_multiplier: float
    omega: float | None = None

    @property
    def label(self) -> str:
        loss_label = self.loss if self.loss != "BP" else f"BP({self.omega:g})"
        return f"{loss_label}|{self.dictionary}|c={self.penalty_multiplier:g}"


def ate_branch(x: FloatArray) -> float:
    """Branch selector for the ATE representer: treated positive, control negative."""

    return 1.0 if np.asarray(x, dtype=float).reshape(-1)[0] >= 0.5 else -1.0


def make_generator(spec: CandidateSpec) -> BregmanGenerator:
    if spec.loss == "SQ":
        return SquaredGenerator(C=0.0)
    if spec.loss == "UKL":
        return UKLGenerator(C=1.0, branch_fn=ate_branch)
    if spec.loss == "BKL":
        return BKLGenerator(C=1.0, branch_fn=ate_branch)
    if spec.loss == "BP":
        if spec.omega is None:
            raise ValueError("A BP candidate requires omega.")
        return BPGenerator(C=1.0, omega=spec.omega, branch_fn=ate_branch)
    raise ValueError(f"Unknown loss: {spec.loss}")


def candidate_grid() -> tuple[CandidateSpec, ...]:
    """Return the ninety-candidate library.

    ``BP(1)`` is excluded: its second derivative is constantly two on a fixed
    branch, so it carries the same Bregman geometry as a branchwise squared
    generator. ``tests/`` pins that identity.
    """

    losses: tuple[tuple[Loss, float | None], ...] = (
        ("SQ", None),
        ("UKL", None),
        ("BKL", None),
        ("BP", 0.25),
        ("BP", 0.5),
    )
    dictionaries: tuple[Dictionary, ...] = ("linear", "second_order", "rich")
    multipliers = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    return tuple(
        CandidateSpec(loss=loss, dictionary=dictionary, penalty_multiplier=c, omega=omega)
        for loss, omega in losses
        for dictionary in dictionaries
        for c in multipliers
    )


#: The benchmark specification used to calibrate the hidden-direction scale.
BENCHMARK_SPEC = CandidateSpec(loss="SQ", dictionary="rich", penalty_multiplier=0.0)

#: Default practitioner choices reported as fixed-specification benchmarks.
FIXED_BENCHMARKS: dict[str, CandidateSpec] = {
    "fixed_sq": CandidateSpec("SQ", "rich", 1.0),
    "fixed_ukl": CandidateSpec("UKL", "rich", 1.0),
    "fixed_bkl": CandidateSpec("BKL", "rich", 1.0),
    "fixed_bp05": CandidateSpec("BP", "rich", 1.0, omega=0.5),
}


class ExperimentBasis(BaseBasis):
    """Treatment-specific series basis standardized on the fitting sample.

    Deterministic in ``(kind, X_train)``, which is what lets a fold share one
    instance across the thirty candidates that use the same dictionary.
    """

    def __init__(self, kind: Dictionary):
        self.kind = kind
        self._mean: FloatArray | None = None
        self._scale: FloatArray | None = None
        self._n_features: int | None = None

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            raise RuntimeError("ExperimentBasis must be fit before n_features is available.")
        return self._n_features

    def base_features(self, Z: FloatArray) -> FloatArray:
        n, d = Z.shape
        blocks: list[FloatArray] = [np.ones((n, 1), dtype=float), Z]
        if self.kind in {"second_order", "rich"}:
            blocks.append(Z * Z)
        if self.kind == "rich":
            q = min(10, d)
            blocks.append(np.sin(Z[:, :q]))
            blocks.append(np.abs(Z[:, :q]))
            interactions = [
                (Z[:, j] * Z[:, k]).reshape(-1, 1) for j in range(q) for k in range(j + 1, q)
            ]
            if interactions:
                blocks.append(np.column_stack(interactions))
        return np.column_stack(blocks)

    def raw_features(self, X: FloatArray) -> FloatArray:
        """Treatment-specific raw features, before standardization.

        Public because the audit caches these for the shared integration sample
        and applies each fold's affine standardization separately.
        """

        X = np.asarray(X, dtype=float)
        D = X[:, [0]]
        Z = X[:, 1:]
        base = self.base_features(Z)
        return np.column_stack((D * base, (1.0 - D) * base))

    def fit(self, X: FloatArray, y: FloatArray | None = None) -> ExperimentBasis:
        raw = self.raw_features(np.asarray(X, dtype=float))
        mean = raw.mean(axis=0)
        scale = raw.std(axis=0, ddof=0)
        constant = scale <= 1e-12
        mean[constant] = 0.0
        scale[constant] = 1.0
        self._mean = mean
        self._scale = scale
        self._n_features = int(raw.shape[1])
        return self

    def standardize(self, raw: FloatArray) -> FloatArray:
        """Apply the fitted affine map to pre-computed raw features."""

        if self._mean is None or self._scale is None:
            raise RuntimeError("ExperimentBasis must be fit before it is applied.")
        return (raw - self._mean) / self._scale

    def __call__(self, X: FloatArray) -> FloatArray:
        return self.standardize(self.raw_features(np.asarray(X, dtype=float)))


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


@dataclass(frozen=True)
class OutcomeFit:
    """Outcome estimator shared by every candidate within a fold."""

    kind: str
    model: OutcomeGLM | HistGradientBoostingRegressor

    def predict(self, X: FloatArray) -> FloatArray:
        return np.asarray(self.model.predict(np.asarray(X, dtype=float)), dtype=float)

    def contrast(self, X: FloatArray) -> FloatArray:
        """Return ``gamma_hat(1, Z) - gamma_hat(0, Z)``."""

        X = np.asarray(X, dtype=float)
        X1 = X.copy()
        X0 = X.copy()
        X1[:, 0] = 1.0
        X0[:, 0] = 0.0
        return self.predict(X1) - self.predict(X0)


def fit_outcome(
    X_train: FloatArray,
    y_train: FloatArray,
    *,
    design: Design,
    seed: int,
) -> OutcomeFit:
    """Fit the fold's outcome estimator.

    Hyperparameters are fixed rather than cross-validated, so that the only
    thing varying across candidates is the representer specification. Plan
    section 17 records that the manuscript must say so.
    """

    if design == "low":
        model = OutcomeGLM(basis=LowOutcomeBasis(), link="identity", penalty="l2", lam=0.0)
        fit = model.fit(X_train, y_train)
        if not fit.success:
            raise RuntimeError(f"Low-dimensional outcome estimator failed: {fit.status}")
        return OutcomeFit(kind="series", model=model)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return OutcomeFit(kind="gradient_boosting", model=model)


def _has_domain_clip(generator: BregmanGenerator) -> bool:
    """Whether the generator overrides the no-op base ``domain_binding``."""

    return type(generator).domain_binding is not BregmanGenerator.domain_binding


@dataclass(frozen=True)
class RepresenterFit:
    """One fitted candidate together with its numerical diagnostics."""

    spec: CandidateSpec
    beta: FloatArray | None
    success: bool
    status: str
    objective: float
    gradient_norm: float
    kkt_residual: float
    binding_rate: float


def fit_candidate_beta(
    X_train: FloatArray,
    spec: CandidateSpec,
    basis: ExperimentBasis,
    generator: BregmanGenerator,
    *,
    max_iter: int,
    tolerance: float,
    gradient_tolerance: float = GRADIENT_TOLERANCE,
) -> RepresenterFit:
    """Fit one candidate on a pre-built, already standardized basis."""

    p = basis.n_features
    lam = spec.penalty_multiplier * np.sqrt(np.log(max(p, 2)) / X_train.shape[0])
    model = GRRGLM(
        basis=basis,
        generator=generator,
        functional=ATEFunctional(treatment_index=0),
        penalty="l1",
        lam=lam,
        p_norm=1.0,
    )
    fit = model.fit(X_train, max_iter=max_iter, tol=tolerance)
    gradient = float(fit.gradient_norm)
    binding = float(fit.clip_binding_rate)
    if not np.isfinite(binding):
        # A generator without a domain clip never reports a rate. Treat the
        # missing value as "no binding" only in that case; otherwise fail closed.
        binding_ok = not _has_domain_clip(generator)
    else:
        binding_ok = binding == 0.0
    beta = np.asarray(fit.beta, dtype=float) if fit.beta is not None else None
    ok = bool(
        fit.success
        and beta is not None
        and np.all(np.isfinite(beta))
        and np.isfinite(gradient)
        and gradient <= gradient_tolerance
        and binding_ok
    )
    status = fit.status if not fit.success else (fit.status if ok else "diagnostic_failure")
    return RepresenterFit(
        spec=spec,
        beta=beta if ok else None,
        success=ok,
        status=status,
        objective=float(fit.objective_value),
        gradient_norm=gradient,
        kkt_residual=float(fit.kkt_residual),
        binding_rate=binding,
    )


class FoldLibrary:
    """The ninety fitted candidates of one fold, with shared dictionaries.

    Every selection rule in :mod:`refsel.selection` reads this object, so adding
    a rule costs no additional fitting.
    """

    def __init__(
        self,
        X_train: FloatArray,
        specs: Sequence[CandidateSpec],
        *,
        max_iter: int,
        tolerance: float,
        gradient_tolerance: float = GRADIENT_TOLERANCE,
    ):
        self.specs = tuple(specs)
        self.bases: dict[Dictionary, ExperimentBasis] = {}
        for kind in dict.fromkeys(spec.dictionary for spec in self.specs):
            self.bases[kind] = ExperimentBasis(kind).fit(X_train)
        # One generator instance per (loss, omega) rather than one per candidate.
        # The branch signs depend only on X, so sharing lets a single cached
        # evaluation serve all candidates that use the same generator. Sharing is
        # safe because fits and evaluations here are strictly sequential.
        pool: dict[tuple[Loss, float | None], BregmanGenerator] = {}
        generators: list[BregmanGenerator] = []
        for spec in self.specs:
            key = (spec.loss, spec.omega)
            if key not in pool:
                pool[key] = make_generator(spec)
            generators.append(pool[key])
        self.generators = tuple(generators)
        self._generator_pool = tuple(pool.values())
        self.fits = tuple(
            fit_candidate_beta(
                X_train,
                spec,
                self.bases[spec.dictionary],
                generator,
                max_iter=max_iter,
                tolerance=tolerance,
                gradient_tolerance=gradient_tolerance,
            )
            for spec, generator in zip(self.specs, self.generators, strict=True)
        )
        self._columns: dict[Dictionary, list[int]] = {}
        for j, spec in enumerate(self.specs):
            self._columns.setdefault(spec.dictionary, []).append(j)

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def success(self) -> NDArray[np.bool_]:
        return np.asarray([fit.success for fit in self.fits], dtype=bool)

    @property
    def dictionary_columns(self) -> dict[Dictionary, list[int]]:
        """Candidate indices grouped by the dictionary they share."""

        return {kind: list(columns) for kind, columns in self._columns.items()}

    @contextmanager
    def branch_caches(self) -> Iterator[None]:
        """Memoize branch signs for every pooled generator inside the block.

        Without this, evaluating ninety candidates on one sample calls the
        per-row branch selector once per candidate per row. Profiling a single
        low-dimensional replication showed 13.7 million such calls, more than a
        third of the total runtime.

        Do not mutate the arrays passed to the generators inside the block.
        """

        with ExitStack() as stack:
            for generator in self._generator_pool:
                stack.enter_context(generator.branch_cache())
            yield

    def _safe_inv_grad(self, index: int, X: FloatArray, v: FloatArray) -> FloatArray | None:
        """Evaluate one candidate's link, returning ``None`` on a domain error.

        ``BKLGenerator.inv_grad`` raises rather than clipping when the linear
        predictor leaves its domain, and the forced-branch evaluations used by
        the squared held-out risk make that likely. This is an expected numerical
        status, not a programming error, so it is recorded as a missing value
        instead of aborting the batch.
        """

        try:
            alpha = np.asarray(self.generators[index].inv_grad(X, v), dtype=float)
        except DomainError:
            return None
        return alpha if np.all(np.isfinite(alpha)) else None

    def _linear_predictors(
        self, X: FloatArray, kind: Dictionary, columns: Sequence[int]
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``(Phi, V)`` where ``V[:, i]`` is the predictor of ``columns[i]``."""

        basis = self.bases[kind]
        Phi = np.asarray(basis(X), dtype=float)
        beta = np.zeros((Phi.shape[1], len(columns)), dtype=float)
        for i, j in enumerate(columns):
            fit = self.fits[j]
            if fit.beta is not None:
                beta[:, i] = fit.beta
        return Phi, Phi @ beta

    def alpha_matrix(self, X: FloatArray) -> FloatArray:
        """Return the ``(n, n_candidates)`` matrix of fitted representers.

        Failed candidates receive a column of ``nan``.
        """

        X = np.asarray(X, dtype=float)
        out = np.full((X.shape[0], len(self.specs)), np.nan, dtype=float)
        with self.branch_caches():
            for kind, columns in self._columns.items():
                _, V = self._linear_predictors(X, kind, columns)
                for i, j in enumerate(columns):
                    if not self.fits[j].success:
                        continue
                    alpha = self._safe_inv_grad(j, X, V[:, i])
                    if alpha is not None:
                        out[:, j] = alpha
        return out

    def heldout_bregman(self, X: FloatArray) -> FloatArray:
        """Unpenalized Bregman criterion of each candidate on a held-out sample.

        This is the naive cross-validation objective. Plan section 4 shows it is
        not comparable across generators.
        """

        X = np.asarray(X, dtype=float)
        out = np.full(len(self.specs), np.nan, dtype=float)
        functional = ATEFunctional(treatment_index=0)
        with self.branch_caches():
            for kind, columns in self._columns.items():
                basis = self.bases[kind]
                _, V = self._linear_predictors(X, kind, columns)
                M = np.asarray(functional.m_basis_matrix(X, basis), dtype=float)
                for i, j in enumerate(columns):
                    fit = self.fits[j]
                    if not fit.success or fit.beta is None:
                        continue
                    try:
                        g_star, _ = self.generators[j].conjugate(X, V[:, i])
                    except DomainError:
                        continue
                    value = float(np.mean(np.asarray(g_star, dtype=float) - M @ fit.beta))
                    out[j] = value if np.isfinite(value) else np.nan
        return out

    def heldout_lsif(self, X: FloatArray) -> FloatArray:
        """Generator-agnostic squared (LSIF) risk on a held-out sample.

        ``0.5 E[alpha^2] - E[alpha(1, Z) - alpha(0, Z)]``. Unlike
        :meth:`heldout_bregman` this is comparable across generators, which
        makes it the strong competitor in the selection horse race.
        """

        X = np.asarray(X, dtype=float)
        X1 = X.copy()
        X0 = X.copy()
        X1[:, 0] = 1.0
        X0[:, 0] = 0.0
        out = np.full(len(self.specs), np.nan, dtype=float)
        with self.branch_caches():
            for kind, columns in self._columns.items():
                _, V = self._linear_predictors(X, kind, columns)
                _, V1 = self._linear_predictors(X1, kind, columns)
                _, V0 = self._linear_predictors(X0, kind, columns)
                for i, j in enumerate(columns):
                    if not self.fits[j].success:
                        continue
                    alpha = self._safe_inv_grad(j, X, V[:, i])
                    alpha1 = self._safe_inv_grad(j, X1, V1[:, i])
                    alpha0 = self._safe_inv_grad(j, X0, V0[:, i])
                    if alpha is None or alpha1 is None or alpha0 is None:
                        continue
                    value = float(0.5 * np.mean(alpha * alpha) - np.mean(alpha1 - alpha0))
                    out[j] = value if np.isfinite(value) else np.nan
        return out

    def coefficient_matrix(self, kind: Dictionary) -> tuple[list[int], FloatArray]:
        """Return the candidate columns and stacked coefficients for one dictionary."""

        columns = self._columns[kind]
        basis = self.bases[kind]
        beta = np.zeros((basis.n_features, len(columns)), dtype=float)
        for i, j in enumerate(columns):
            fit = self.fits[j]
            if fit.beta is not None:
                beta[:, i] = fit.beta
        return columns, beta
