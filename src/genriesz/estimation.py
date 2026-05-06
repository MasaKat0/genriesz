"""High-level estimation API.

The public API is centered around:

- :func:`genriesz.grr_functional`  (general linear functional)
- :func:`genriesz.grr_ate`         (ATE convenience wrapper)
- :func:`genriesz.grr_att`         (ATT convenience wrapper)
- :func:`genriesz.grr_did`         (panel DID as ΔY-ATT)
- :func:`genriesz.grr_ame`         (average marginal effect)

Estimators (naming convention):

- RA   : regression adjustment (plug-in)
- RW   : Riesz weighting (weighting only)
- ARW  : augmented Riesz weighting (orthogonal / doubly-robust)
- TMLE : targeted maximum likelihood estimator

TMLE likelihood is inferred from the *outcome regression link*:

- ``link='identity'`` => Gaussian targeting
- ``link='logit'``    => Bernoulli targeting

When ``link`` is not given, we default to identity unless the outcome is bounded in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from .basis import BaseBasis, Basis, CallableBasis
from .functionals import (
    ATEFunctional,
    AMEFunctional,
    ATTFunctional,
    DIDFunctional,
    CallableFunctional,
    LinearFunctional,
)
from .generators import BregmanGenerator
from .glm import GRRGLM, OutcomeGLM
from .matching import (
    LocalPolynomialLSIFWeights,
    NNMatchingWeights,
    local_polynomial_nn_lsif_inverse_propensity_weights,
    nn_matching_inverse_propensity_weights,
)
from .results import FunctionalEstimate, SingleEstimate
from .utils import kfold_splits, se_ci_pvalue, Fold


EstimatorName = Literal["ra", "rw", "arw", "tmle"]
OutcomeModels = Literal["shared", "separate", "both", "none", "auto"]
RieszMethod = Literal["grr", "nn_matching", "local_poly_nn_lsif"]


def _effective_sample_size(w: NDArray[np.float64]) -> float:
    """Compute the Kish effective sample size for nonnegative weights."""

    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size == 0:
        return float("nan")
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 0.0:
        return float("nan")
    sw2 = float(np.sum(w * w))
    if not np.isfinite(sw2) or sw2 <= 0.0:
        return float("nan")
    return float((sw * sw) / sw2)


def _covariate_balance_smd(
    *,
    Z: NDArray[np.float64],
    D: NDArray[np.float64],
    w_treated: NDArray[np.float64] | None = None,
    w_control: NDArray[np.float64] | None = None,
    target: Literal["ate", "att"] = "ate",
) -> dict[str, object]:
    """Compute standardized mean differences (SMDs) for a binary treatment.

    Parameters
    ----------
    Z:
        Covariate matrix excluding the treatment column.
    D:
        Binary treatment indicator in {0,1}.
    w_treated, w_control:
        Nonnegative weights for treated / control groups. If None, unweighted
        means are used.
    target:
        - "ate": compare weighted treated vs weighted control
        - "att": compare treated (unweighted) vs weighted control
    """

    Z = np.asarray(Z, dtype=float)
    D = np.asarray(D, dtype=float).reshape(-1)
    if Z.ndim != 2:
        raise ValueError("Z must be 2D")
    if D.shape[0] != Z.shape[0]:
        raise ValueError("D and Z must have the same number of rows")

    treated = D == 1.0
    control = D == 0.0
    if treated.sum() == 0 or control.sum() == 0:
        raise ValueError("Both treated and control groups must be nonempty")

    Z1 = Z[treated]
    Z0 = Z[control]

    mean1 = np.nanmean(Z1, axis=0)
    mean0 = np.nanmean(Z0, axis=0)
    var1 = np.nanvar(Z1, axis=0, ddof=1) if Z1.shape[0] > 1 else np.zeros(Z.shape[1])
    var0 = np.nanvar(Z0, axis=0, ddof=1) if Z0.shape[0] > 1 else np.zeros(Z.shape[1])
    sd_pooled = np.sqrt(0.5 * (var1 + var0))
    sd_pooled = np.where(sd_pooled > 0, sd_pooled, np.nan)

    smd_unweighted = (mean1 - mean0) / sd_pooled

    def wmean(A: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
        w = np.asarray(w, dtype=float).reshape(-1)
        if A.shape[0] != w.shape[0]:
            raise ValueError("weight length mismatch")
        sw = np.sum(w)
        if not np.isfinite(sw) or sw <= 0:
            return np.full(A.shape[1], np.nan)
        return (A * w.reshape(-1, 1)).sum(axis=0) / sw

    # Weighted means
    mean1_w = mean1.copy()
    if w_treated is not None:
        mean1_w = wmean(Z1, w_treated)

    mean0_w = mean0.copy()
    if w_control is not None:
        mean0_w = wmean(Z0, w_control)

    if target == "att":
        mean1_bal = mean1
        mean0_bal = mean0_w
    else:
        mean1_bal = mean1_w
        mean0_bal = mean0_w

    smd_weighted = (mean1_bal - mean0_bal) / sd_pooled

    return {
        "smd_unweighted": smd_unweighted,
        "smd_weighted": smd_weighted,
        "abs_smd_unweighted": np.abs(smd_unweighted),
        "abs_smd_weighted": np.abs(smd_weighted),
        "mean_treated": mean1,
        "mean_control": mean0,
        "mean_treated_weighted": mean1_w,
        "mean_control_weighted": mean0_w,
        "sd_pooled": sd_pooled,
        "n_treated": int(treated.sum()),
        "n_control": int(control.sum()),
    }


def _as_2d(X: ArrayLike) -> NDArray[np.float64]:
    X_ = np.asarray(X, dtype=float)
    if X_.ndim != 2:
        raise ValueError(f"X must be 2D. Got shape {X_.shape}.")
    return X_


def _as_1d(y: ArrayLike, *, n: int, name: str) -> NDArray[np.float64]:
    y_ = np.asarray(y, dtype=float).reshape(-1)
    if y_.shape[0] != n:
        raise ValueError(f"{name} must have length {n}. Got shape {y_.shape}.")
    return y_


def _canonical_estimators(estimators: Iterable[str]) -> tuple[EstimatorName, ...]:
    mapping = {
        "ra": "ra",
        "rw": "rw",
        "arw": "arw",
        "tmle": "tmle",
    }
    out: list[EstimatorName] = []
    for e in estimators:
        key = str(e).lower()
        if key not in mapping:
            raise ValueError(f"Unknown estimator: {e}")
        canon = mapping[key]
        if canon not in out:
            out.append(canon)  # preserve order
    return tuple(out)


def _coerce_basis(basis: Basis | Callable) -> Basis:
    """Coerce a raw callable into a Basis implementation.

    The public API documents that users may pass either a Basis instance
    or a plain callable ``basis(X) -> Phi``. The latter is wrapped in
    :class:`genriesz.CallableBasis`.

    Notes
    -----
    Do **not** probe ``basis.n_features`` here. Many bases (e.g.
    :class:`~genriesz.PolynomialBasis` and
    :class:`~genriesz.TreatmentInteractionBasis`) expose ``n_features`` as
    a property that is only valid *after* ``fit()``. Accessing it early
    would raise, and ``hasattr(obj, 'n_features')`` would inadvertently
    trigger that property.
    """

    # If it already behaves like a Basis, keep it.
    if isinstance(basis, CallableBasis):
        return basis

    # All built-in bases inherit from BaseBasis.
    if isinstance(basis, BaseBasis):
        return basis

    # Accept user-defined basis objects via duck typing, without touching
    # the potentially-unfitted ``n_features`` property.
    if hasattr(basis, 'fit') and hasattr(basis, 'copy') and callable(basis):  # type: ignore[arg-type]
        return basis  # type: ignore[return-value]

    # Otherwise, interpret it as a raw callable feature map.
    if callable(basis):
        return CallableBasis(basis)

    raise TypeError('basis must be a Basis instance or a callable basis(X)->Phi')


def _coerce_functional(m: LinearFunctional | Callable) -> LinearFunctional:
    """Coerce a functional argument into a :class:`LinearFunctional`.

    The public API supports either:

    - a :class:`~genriesz.functionals.LinearFunctional` instance (recommended), or
    - a plain callable ``m(x_row, gamma) -> float``.

    The callable case is wrapped in :class:`~genriesz.functionals.CallableFunctional`.
    """

    if isinstance(m, LinearFunctional):
        return m

    if callable(m):
        # Type: user-provided callable; we assume it follows the README signature.
        return CallableFunctional(m)  # type: ignore[arg-type]

    raise TypeError("m must be a LinearFunctional or a callable m(x_row, gamma)->float")


def _logit(p: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _expit(z: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _tmle_epsilon_gaussian(H: NDArray[np.float64], y: NDArray[np.float64], mu: NDArray[np.float64]) -> float:
    resid = y - mu
    denom = float(np.sum(H * H))
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return float(np.sum(H * resid) / denom)


def _tmle_epsilon_bernoulli(H: NDArray[np.float64], y: NDArray[np.float64], mu: NDArray[np.float64]) -> float:
    mu = np.clip(mu, 1e-6, 1.0 - 1e-6)
    offset = _logit(mu)

    def score(eps: float) -> float:
        mu_eps = _expit(offset + eps * H)
        return float(np.mean(H * (y - mu_eps)))

    # Newton with derivative (monotone score)
    eps = 0.0
    for _ in range(60):
        s = score(eps)
        if abs(s) < 1e-10:
            return float(eps)
        mu_eps = _expit(offset + eps * H)
        deriv = -float(np.mean((H * H) * mu_eps * (1.0 - mu_eps)))
        if deriv == 0 or not np.isfinite(deriv):
            break
        step = s / deriv
        eps_new = eps - step
        if abs(eps_new - eps) < 1e-10:
            return float(eps_new)
        eps = eps_new

    # Fallback: root_scalar with a bracket (if possible)
    left, right = -1.0, 1.0
    s_left, s_right = score(left), score(right)
    for _ in range(20):
        if s_left * s_right <= 0:
            res = optimize.root_scalar(score, bracket=[left, right], method="brentq")
            if getattr(res, "converged", False):
                return float(res.root)
            break
        left *= 2.0
        right *= 2.0
        s_left, s_right = score(left), score(right)
    return float(eps)


def grr_functional(
    *,
    X: ArrayLike,
    Y: ArrayLike,
    m: LinearFunctional | Callable,
    basis: Basis | Callable,
    generator: BregmanGenerator | None = None,
    g: Callable | None = None,
    grad_g: Callable | None = None,
    inv_grad_g: Callable | None = None,
    grad2_g: Callable | None = None,
    # Riesz estimation options
    riesz_method: RieszMethod = "grr",
    riesz_penalty: str | None = "l2",
    riesz_lam: float = 1e-3,
    riesz_p_norm: float | None = None,
    # Matching-only options (ATE only)
    M: int = 1,
    local_poly_degree: int = 1,
    standardize_for_matching: bool = True,
    # Outcome model options
    outcome_models: OutcomeModels = "auto",
    outcome_basis: Basis | Callable | None = None,
    outcome_link: str | None = None,
    outcome_penalty: str | None = "l2",
    outcome_lam: float = 1e-3,
    outcome_p_norm: float | None = None,
    # Cross-fitting
    cross_fit: bool = True,
    folds: int = 5,
    random_state: int | None = 0,
    # Output / inference
    estimators: Sequence[str] = ("ra", "rw", "arw", "tmle"),
    alpha: float = 0.05,
    null: float = 0.0,
    # Optimizers
    max_iter: int = 500,
    tol: float = 1e-8,
    verbose: bool = False,
) -> FunctionalEstimate:
    """Estimate a linear functional using generalized Riesz regression.

    Parameters
    ----------
    X, Y:
        Regressors and outcome.
    m:
        Either a :class:`~genriesz.functionals.LinearFunctional` instance (recommended)
        or a plain callable ``m(x_row, gamma) -> float``.
    basis:
        Basis used for Riesz regression (and for the outcome regression when
        ``outcome_models='shared'``).
    generator:
        Bregman generator used for GRR. If None, you can pass a generator function via ``g`` (and optionally ``grad_g``, ``inv_grad_g``, ``grad2_g``).
    riesz_method:
        - "grr"            : solve the GRR optimization problem
        - "nn_matching"    : NN-matching inverse propensity weights (**ATE-only** convenience)
        - "local_poly_nn_lsif" : local polynomial NN-LSIF weights (**ATE-only** convenience)
        Matching-based Riesz methods currently require ``cross_fit=False`` and
        do not support ``TMLE`` (because they do not provide a function-valued
        representer that can be evaluated counterfactually).
    outcome_link:
        If None, inferred as 'logit' for outcomes bounded in [0, 1], else 'identity'.
        TMLE likelihood is inferred from this link.
    """

    X_ = _as_2d(X)
    n = X_.shape[0]
    y_ = _as_1d(Y, n=n, name="Y")

    # Coerce raw callables into LinearFunctional (README-friendly)
    m = _coerce_functional(m)

    # Coerce raw callables into Basis objects (README-friendly)
    basis = _coerce_basis(basis)
    if outcome_basis is not None:
        outcome_basis = _coerce_basis(outcome_basis)

    ests = _canonical_estimators(estimators)

    riesz_method_ = str(riesz_method).lower()

    # Guard rails: matching-based Riesz methods are currently implemented only
    # for the ATE and only without cross-fitting.
    if riesz_method_ in {"nn_matching", "local_poly_nn_lsif"}:
        if not isinstance(m, ATEFunctional):
            raise ValueError(
                "riesz_method='nn_matching'/'local_poly_nn_lsif' is implemented only for ATE. "
                "Use riesz_method='grr' for other estimands."
            )
        if cross_fit:
            raise ValueError("cross_fit=True is not supported for matching-based Riesz methods.")
        if "tmle" in ests:
            raise ValueError(
                "TMLE requires a functional evaluation m(alpha_hat) and is not supported for "
                "matching-based Riesz methods. Use riesz_method='grr' for TMLE."
            )

    # ------------------------------------------------------------------
    # Generator inference (either pass `generator` or a raw `g`)
    # ------------------------------------------------------------------
    if riesz_method_ == "grr":
        if generator is not None and g is not None:
            raise ValueError('Pass either generator=... or g=... (not both).')
        if generator is None:
            if g is None:
                raise ValueError("When riesz_method='grr', you must provide generator or g.")
            generator = BregmanGenerator(g=g, grad=grad_g, inv_grad=inv_grad_g, grad2=grad2_g)


    # Outcome link inference
    if outcome_link is None:
        outcome_link_ = "logit" if (np.nanmin(y_) >= 0.0 and np.nanmax(y_) <= 1.0) else "identity"
    else:
        outcome_link_ = str(outcome_link).lower()
        if outcome_link_ not in {"identity", "logit"}:
            raise ValueError("outcome_link must be 'identity' or 'logit'")

    need_outcome = any(e in {"ra", "arw", "tmle"} for e in ests)

    if outcome_models in {None, "auto"}:
        outcome_models_ = "shared" if need_outcome else "none"
    else:
        outcome_models_ = str(outcome_models).lower()

    if outcome_models_ == "none" and need_outcome:
        raise ValueError("RA/ARW/TMLE require an outcome model; set outcome_models!='none'.")

    # ------------------------------------------------------------------
    # Cross-fitting splits
    # ------------------------------------------------------------------
    if cross_fit:
        splits = list(kfold_splits(n, folds=folds, random_state=random_state))
    else:
        all_idx = np.arange(n)
        splits = [Fold(train=all_idx, test=all_idx)]

    # Storage for nuisances (cross-fit predictions)
    alpha_obs = np.zeros(n, dtype=float)

    # For GRR-based TMLE (Gaussian): we only need m(alpha) for each observation.
    m_alpha = np.zeros(n, dtype=float)

    # Outcome regression predictions on observed X
    mu_obs: dict[str, NDArray[np.float64]] = {}
    m_mu: dict[str, NDArray[np.float64]] = {}

    # For Bernoulli TMLE with ATE/ATT/DID we need counterfactual mu/alpha.
    cf_cache: dict[str, NDArray[np.float64]] = {}

    # ------------------------------------------------------------------
    # Fit nuisances fold-by-fold
    # ------------------------------------------------------------------
    for fold_id, fold in enumerate(splits):
        train_idx, test_idx = fold.train, fold.test
        X_tr, y_tr = X_[train_idx], y_[train_idx]
        X_te, y_te = X_[test_idx], y_[test_idx]

        # ----- Riesz representer
        if riesz_method_ == "grr":
            if generator is None:
                raise ValueError("generator is required when riesz_method='grr'")

            basis_r = basis.copy().fit(X_tr, y_tr)
            grr = GRRGLM(
                basis=basis_r,
                generator=generator,
                functional=m,
                penalty=riesz_penalty,
                lam=riesz_lam,
                p_norm=riesz_p_norm,
            )
            grr.fit(X_tr, max_iter=max_iter, tol=tol, verbose=verbose)

            alpha_obs[test_idx] = grr.predict_alpha(X_te)

            # m(alpha) is needed for Gaussian TMLE update for any functional.
            # For AME we need derivatives; others only need predict().
            try:
                m_alpha[test_idx] = m.m_from_function(
                    X_te,
                    predict=grr.predict_alpha,
                    derivative=getattr(grr, "derivative_alpha", None),
                )
            except NotImplementedError:
                # If the functional cannot be applied to alpha, TMLE will be unavailable.
                m_alpha[test_idx] = np.nan

            # For Bernoulli TMLE with treatment-type functionals, cache cf values.
            if "tmle" in ests and outcome_link_ == "logit" and isinstance(
                m, (ATEFunctional, ATTFunctional, DIDFunctional)
            ):
                # Construct counterfactual regressors by toggling the treatment column.
                t_idx = getattr(m, "treatment_index", 0)
                X1 = X_te.copy()
                X1[:, t_idx] = 1.0
                X0 = X_te.copy()
                X0[:, t_idx] = 0.0
                cf_cache.setdefault("alpha1", np.zeros(n, dtype=float))[test_idx] = grr.predict_alpha(X1)
                cf_cache.setdefault("alpha0", np.zeros(n, dtype=float))[test_idx] = grr.predict_alpha(X0)

        elif riesz_method_ in {"nn_matching", "local_poly_nn_lsif"}:
            # Matching-based Riesz methods: currently implemented only for the ATE.
            # Guard rails for mis-use are also enforced near the top of grr_functional().
            if not isinstance(m, ATEFunctional):
                raise ValueError(
                    "Matching-based Riesz methods are implemented only for the ATE. "
                    "Use riesz_method='grr' for other estimands."
                )
            t_idx = getattr(m, "treatment_index", 0)
            D = X_tr[:, t_idx].astype(int)
            Z_tr = np.delete(X_tr, t_idx, axis=1)

            if riesz_method_ == "nn_matching":
                wobj: NNMatchingWeights = nn_matching_inverse_propensity_weights(
                    Z_tr,
                    D,
                    M,
                    standardize=standardize_for_matching,
                )
                w = wobj.w
            else:
                wobj2: LocalPolynomialLSIFWeights = local_polynomial_nn_lsif_inverse_propensity_weights(
                    Z_tr,
                    D,
                    M,
                    degree=local_poly_degree,
                    standardize=standardize_for_matching,
                    verbose=verbose,
                )
                w = wobj2.w

            # Matching-style Riesz representer for the ATE:
            #
            #   alpha_i = (2D_i - 1) * w_i,
            #
            # where w_i >= 0 are the matching inverse-propensity weights.
            alpha_tr = D * w - (1 - D) * w
            alpha_obs[:] = alpha_tr
            m_alpha[:] = np.nan

        else:
            raise ValueError(f"Unknown riesz_method: {riesz_method_}")

        # ----- Outcome regression(s)
        if not need_outcome:
            continue

        variants: dict[str, Basis] = {}
        if outcome_models_ == "shared":
            if riesz_method_ == "grr":
                variants = {"shared": basis_r}
            else:
                variants = {"shared": basis.copy().fit(X_tr, y_tr)}
        elif outcome_models_ == "separate":
            ob = basis if outcome_basis is None else outcome_basis
            variants = {"separate": ob.copy().fit(X_tr, y_tr)}
        elif outcome_models_ == "both":
            if riesz_method_ == "grr":
                variants = {
                    "shared": basis_r,
                    "separate": (basis if outcome_basis is None else outcome_basis).copy().fit(X_tr, y_tr),
                }
            else:
                variants = {
                    "shared": basis.copy().fit(X_tr, y_tr),
                    "separate": (basis if outcome_basis is None else outcome_basis).copy().fit(X_tr, y_tr),
                }
        else:
            raise ValueError(f"Unknown outcome_models: {outcome_models}")

        for tag, b_out in variants.items():
            out = OutcomeGLM(
                basis=b_out,
                link=outcome_link_,
                penalty=outcome_penalty,
                lam=outcome_lam,
                p_norm=outcome_p_norm,
            )
            out.fit(X_tr, y_tr, max_iter=max_iter, tol=tol, verbose=verbose)

            mu_obs.setdefault(tag, np.zeros(n, dtype=float))[test_idx] = out.predict(X_te)

            # m(gamma_hat)
            try:
                m_mu.setdefault(tag, np.zeros(n, dtype=float))[test_idx] = m.m_from_function(
                    X_te,
                    predict=out.predict,
                    derivative=getattr(out, "derivative", None),
                )
            except NotImplementedError:
                m_mu.setdefault(tag, np.zeros(n, dtype=float))[test_idx] = np.nan

            # Cache cf values for Bernoulli TMLE if needed
            if "tmle" in ests and outcome_link_ == "logit" and isinstance(
                m, (ATEFunctional, ATTFunctional, DIDFunctional)
            ):
                t_idx = getattr(m, "treatment_index", 0)
                X1 = X_te.copy(); X1[:, t_idx] = 1.0
                X0 = X_te.copy(); X0[:, t_idx] = 0.0
                cf_cache.setdefault(f"mu1_{tag}", np.zeros(n, dtype=float))[test_idx] = out.predict(X1)
                cf_cache.setdefault(f"mu0_{tag}", np.zeros(n, dtype=float))[test_idx] = out.predict(X0)

    # ------------------------------------------------------------------
    # Compute estimators + inference
    # ------------------------------------------------------------------
    estimates: dict[str, SingleEstimate] = {}

    def add_est(key: str, name: str, est: float, psi: NDArray[np.float64]) -> None:
        se, lo, hi, p = se_ci_pvalue(est, psi, alpha=alpha, null=null)
        estimates[key] = SingleEstimate(name=name, estimate=float(est), se=float(se), ci_low=float(lo), ci_high=float(hi), p_value=float(p))

    # RW always available when we have alpha
    if "rw" in ests:
        theta = float(np.mean(alpha_obs * y_))
        psi = alpha_obs * y_ - theta
        add_est("rw", "RW", theta, psi)

    if need_outcome:
        # Choose the primary outcome model variant
        tags = list(mu_obs.keys())
        if outcome_models_ == "shared":
            primary = "shared"
        elif outcome_models_ == "separate":
            primary = "separate"
        else:
            primary = "shared"  # default

        def compute_for_tag(tag: str, suffix: str = "") -> None:
            mu = mu_obs[tag]
            m_mu_tag = m_mu[tag]

            if "ra" in ests:
                theta_ra = float(np.mean(m_mu_tag))
                psi_ra = m_mu_tag - theta_ra
                add_est(f"ra{suffix}", f"RA{suffix}", theta_ra, psi_ra)

            if "arw" in ests:
                theta_arw = float(np.mean(m_mu_tag + alpha_obs * (y_ - mu)))
                psi_arw = m_mu_tag + alpha_obs * (y_ - mu) - theta_arw
                add_est(f"arw{suffix}", f"ARW{suffix}", theta_arw, psi_arw)

            if "tmle" in ests:
                # If m(alpha) is not available, TMLE is not available.
                if not np.all(np.isfinite(m_alpha)):
                    raise RuntimeError(
                        "TMLE requires applying the functional m to the Riesz representer alpha. "
                        "This functional / basis combination does not support it."
                    )

                if outcome_link_ == "identity":
                    eps_hat = _tmle_epsilon_gaussian(alpha_obs, y_, mu)
                    mu_star = mu + eps_hat * alpha_obs
                    m_mu_star = m_mu_tag + eps_hat * m_alpha
                else:
                    # Bernoulli targeting
                    if not (np.nanmin(y_) >= 0.0 and np.nanmax(y_) <= 1.0):
                        raise ValueError("Bernoulli TMLE requires Y bounded in [0, 1].")
                    eps_hat = _tmle_epsilon_bernoulli(alpha_obs, y_, mu)
                    mu_star = _expit(_logit(mu) + eps_hat * alpha_obs)

                    # Treatment-type functionals need counterfactual evaluation.
                    if not isinstance(m, (ATEFunctional, ATTFunctional, DIDFunctional)):
                        raise ValueError("Bernoulli TMLE is only implemented for ATE/ATT/DID.")
                    mu1 = cf_cache[f"mu1_{tag}"]
                    mu0 = cf_cache[f"mu0_{tag}"]
                    a1 = cf_cache.get("alpha1")
                    a0 = cf_cache.get("alpha0")
                    if a1 is None or a0 is None:
                        raise RuntimeError("Missing alpha counterfactual cache for Bernoulli TMLE")
                    mu1_star = _expit(_logit(mu1) + eps_hat * a1)
                    mu0_star = _expit(_logit(mu0) + eps_hat * a0)
                    if isinstance(m, ATEFunctional):
                        m_mu_star = mu1_star - mu0_star
                    else:
                        # ATT / DID
                        D = X_[:, getattr(m, "treatment_index", 0)].astype(float)
                        pi = getattr(m, "pi", float(np.mean(D)))
                        m_mu_star = (D / pi) * (mu1_star - mu0_star)

                theta_tmle = float(np.mean(m_mu_star))
                psi_tmle = m_mu_star + alpha_obs * (y_ - mu_star) - theta_tmle
                add_est(f"tmle{suffix}", f"TMLE{suffix}", theta_tmle, psi_tmle)

        if outcome_models_ in {"shared", "separate"}:
            compute_for_tag(primary)
        else:
            # both
            compute_for_tag("shared", suffix=" (shared)")
            compute_for_tag("separate", suffix=" (separate)")

    # ------------------------------------------------------------------
    # Diagnostics (Love plot / balance table)
    # ------------------------------------------------------------------
    diagnostics: dict[str, object] = {}

    alpha_abs = np.abs(alpha_obs)
    diagnostics["alpha_abs_mean"] = float(np.mean(alpha_abs))
    diagnostics["alpha_abs_p95"] = float(np.percentile(alpha_abs, 95))
    diagnostics["alpha_abs_max"] = float(np.max(alpha_abs))

    if isinstance(m, (ATEFunctional, ATTFunctional, DIDFunctional)):
        t_idx = getattr(m, "treatment_index", 0)
        D = X_[:, t_idx].astype(float)
        uniq = np.unique(D)
        if not np.all(np.isin(uniq, [0.0, 1.0])):
            raise ValueError("Treatment indicator must be binary (0/1) to compute a Love plot.")

        Z = np.delete(X_, t_idx, axis=1)
        if Z.shape[1] > 0:
            cov_names = [f"X[{j}]" for j in range(X_.shape[1]) if j != t_idx]
            w_abs = np.abs(alpha_obs)
            treated = D == 1.0
            control = D == 0.0
            w1 = w_abs[treated]
            w0 = w_abs[control]

            target = "ate" if isinstance(m, ATEFunctional) else "att"
            bal = _covariate_balance_smd(Z=Z, D=D, w_treated=w1, w_control=w0, target=target)

            # Store summary scalars for easy printing.
            diagnostics["max_abs_smd_unweighted"] = float(np.nanmax(bal["abs_smd_unweighted"]))
            diagnostics["max_abs_smd_weighted"] = float(np.nanmax(bal["abs_smd_weighted"]))
            diagnostics["ess_treated"] = _effective_sample_size(w1)
            diagnostics["ess_control"] = _effective_sample_size(w0)

            # Store full data for plotting.
            diagnostics["love_plot"] = {
                "covariate_names": cov_names,
                "smd_unweighted": np.asarray(bal["smd_unweighted"], dtype=float).tolist(),
                "smd_weighted": np.asarray(bal["smd_weighted"], dtype=float).tolist(),
                "abs_smd_unweighted": np.asarray(bal["abs_smd_unweighted"], dtype=float).tolist(),
                "abs_smd_weighted": np.asarray(bal["abs_smd_weighted"], dtype=float).tolist(),
                "n_treated": int(bal["n_treated"]),
                "n_control": int(bal["n_control"]),
            }

    return FunctionalEstimate(estimand=m.name, n=n, alpha=alpha, null=null, estimates=estimates, diagnostics=diagnostics)


# ----------------------------------------------------------------------
# Convenience wrappers
# ----------------------------------------------------------------------

def grr_ate(
    *,
    X: ArrayLike,
    Y: ArrayLike,
    treatment_index: int = 0,
    basis: Basis | Callable,
    generator: BregmanGenerator | None = None,
    **kwargs,
) -> FunctionalEstimate:
    """Estimate ATE with the GRR API."""

    m = ATEFunctional(treatment_index=treatment_index)
    return grr_functional(X=X, Y=Y, m=m, basis=basis, generator=generator, **kwargs)


def grr_att(
    *,
    X: ArrayLike,
    Y: ArrayLike,
    treatment_index: int = 0,
    basis: Basis | Callable,
    generator: BregmanGenerator | None = None,
    **kwargs,
) -> FunctionalEstimate:
    """Estimate ATT with the GRR API."""

    X_ = _as_2d(X)
    D = X_[:, treatment_index]
    pi = float(np.mean(D))
    if pi <= 0 or pi >= 1:
        raise ValueError("ATT requires both treatment groups to be non-empty")
    m = ATTFunctional(treatment_index=treatment_index, pi=pi)
    return grr_functional(X=X, Y=Y, m=m, basis=basis, generator=generator, **kwargs)


def grr_did(
    *,
    X: ArrayLike,
    Y0: ArrayLike,
    Y1: ArrayLike,
    treatment_index: int = 0,
    basis: Basis | Callable,
    generator: BregmanGenerator | None = None,
    **kwargs,
) -> FunctionalEstimate:
    """Panel DID implemented as ATT on ΔY = Y1-Y0."""

    X_ = _as_2d(X)
    n = X_.shape[0]
    y0 = _as_1d(Y0, n=n, name="Y0")
    y1 = _as_1d(Y1, n=n, name="Y1")
    dy = y1 - y0

    D = X_[:, treatment_index]
    pi = float(np.mean(D))
    if pi <= 0 or pi >= 1:
        raise ValueError("DID requires both treatment groups to be non-empty")
    m = DIDFunctional(treatment_index=treatment_index, pi=pi)
    return grr_functional(X=X, Y=dy, m=m, basis=basis, generator=generator, **kwargs)


def grr_ame(
    *,
    X: ArrayLike,
    Y: ArrayLike,
    coordinate: int = 0,
    basis: Basis | Callable,
    generator: BregmanGenerator | None = None,
    **kwargs,
) -> FunctionalEstimate:
    """Estimate an average marginal effect (average derivative) wrt x_coordinate."""

    m = AMEFunctional(coordinate=coordinate)
    return grr_functional(X=X, Y=Y, m=m, basis=basis, generator=generator, **kwargs)
