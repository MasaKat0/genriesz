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
- ARW  : augmented Riesz weighting (orthogonal and doubly robust)
- TMLE : targeted maximum likelihood estimator

TMLE likelihood is inferred from the *outcome regression link*:

- ``link='identity'`` => Gaussian targeting
- ``link='logit'``    => Bernoulli targeting

When ``link`` is not given, we default to identity unless the outcome is bounded in [0, 1].
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from .basis import Basis, coerce_basis
from .functionals import (
    AMEFunctional,
    ATEFunctional,
    ATTFunctional,
    CallableFunctional,
    DIDFunctional,
    LinearFunctional,
    _check_treatment_index,
    _validate_treatment_index_arg,
)
from .generators import (
    BKLGenerator,
    BoundedBKLGenerator,
    BPGenerator,
    BregmanGenerator,
    UKLGenerator,
    coerce_generator,
)
from .glm import GRRGLM, OutcomeGLM
from .matching import (
    LocalPolynomialLSIFWeights,
    NNMatchingWeights,
    local_polynomial_nn_lsif_inverse_propensity_weights,
    nn_matching_inverse_propensity_weights,
)
from .model_selection import GRRCVConfig, select_grr_hyperparams
from .results import FunctionalEstimate, SingleEstimate
from .utils import (
    Fold,
    as_1d_of_length,
    as_2d,
    is_binary_y,
    kfold_splits,
    se_ci_pvalue,
    sigmoid,
    stratified_kfold_splits,
)

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
) -> dict[str, Any]:
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


def _canonical_estimators(estimators: Iterable[str]) -> tuple[EstimatorName, ...]:
    mapping: dict[str, EstimatorName] = {
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
    if not out:
        # An empty tuple would silently fit every nuisance and return a result
        # object with no estimates in it.
        raise ValueError(
            "estimators must contain at least one of 'ra', 'rw', 'arw', 'tmle'."
        )
    return tuple(out)


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


def _open_unit_interval(
    values: NDArray[np.float64], *, name: str
) -> NDArray[np.float64]:
    """Validate probabilities without changing their values."""

    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array) & (array > 0.0) & (array < 1.0)
    if not bool(np.all(valid)):
        n_bad = int(np.sum(~valid))
        raise ValueError(
            f"{name} must lie strictly inside (0, 1); {n_bad}/{array.size} "
            "value(s) are nonfinite or on the boundary."
        )
    return array


def _logit(p: NDArray[np.float64]) -> NDArray[np.float64]:
    probability = _open_unit_interval(p, name="Probability")
    return np.log(probability) - np.log1p(-probability)


def _tmle_epsilon_gaussian(
    H: NDArray[np.float64],
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
) -> float:
    H_ = np.asarray(H, dtype=float)
    resid = np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)
    denom = float(np.sum(H_ * H_))
    numerator = float(np.sum(H_ * resid))
    if not np.isfinite(denom) or not np.isfinite(numerator):
        raise ValueError("Gaussian TMLE received a nonfinite targeting equation.")
    if denom < 0.0:
        raise ValueError("Gaussian TMLE direction norm cannot be negative.")
    if denom == 0.0:
        return 0.0
    return numerator / denom


def _tmle_epsilon_bernoulli(
    H: NDArray[np.float64],
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
) -> float:
    H_ = np.asarray(H, dtype=float)
    y_ = np.asarray(y, dtype=float)
    mu_ = _open_unit_interval(np.asarray(mu, dtype=float), name="Bernoulli prediction")
    if not bool(np.all(np.isfinite(H_))) or not bool(np.all(np.isfinite(y_))):
        raise ValueError("Bernoulli TMLE received nonfinite observations or directions.")
    if bool(np.all(H_ == 0.0)):
        return 0.0
    offset = _logit(mu_)

    def score(epsilon: float) -> float:
        mu_epsilon = sigmoid(offset + float(epsilon) * H_)
        value = float(np.mean(H_ * (y_ - mu_epsilon)))
        if not np.isfinite(value):
            raise ValueError("Bernoulli TMLE score is nonfinite.")
        return value

    zero_score = score(0.0)
    if abs(zero_score) <= 1e-12:
        return 0.0

    left = -1.0
    right = 1.0
    left_score = score(left)
    right_score = score(right)
    for _ in range(64):
        if left_score >= 0.0 and right_score <= 0.0:
            result = optimize.root_scalar(
                score, bracket=(left, right), method="brentq", xtol=1e-12, rtol=1e-12
            )
            if not bool(result.converged):
                raise RuntimeError("Bernoulli TMLE root solver did not converge.")
            return float(result.root)
        left *= 2.0
        right *= 2.0
        left_score = score(left)
        right_score = score(right)
    raise RuntimeError(
        "Bernoulli TMLE has no finite fluctuation parameter within the searched "
        "range; the targeting equation is separated or numerically degenerate."
    )


def _raise_if_fit_failed(*, result, what: str, fold_id: int, tag: str | None = None) -> None:
    """Raise when a nuisance optimization failed to converge."""

    if bool(getattr(result, "success", False)):
        return
    label = what if tag is None else f"{what} ({tag})"
    message = str(getattr(result, "message", "unknown optimizer failure"))
    n_iter = int(getattr(result, "n_iter", -1))
    raise RuntimeError(
        f"{label} optimization failed in fold {fold_id}: {message} (n_iter={n_iter})."
    )


def grr_functional(
    *,
    X: ArrayLike,
    Y: ArrayLike,
    m: LinearFunctional | Callable,
    basis: Basis | Callable,
    generator: BregmanGenerator | str | None = None,
    g: Callable | None = None,
    grad_g: Callable | None = None,
    inv_grad_g: Callable | None = None,
    grad2_g: Callable | None = None,
    # Riesz estimation options
    riesz_method: RieszMethod = "grr",
    riesz_penalty: str | None = "l2",
    riesz_lam: float = 1e-3,
    riesz_p_norm: float | None = None,
    # Riesz inner cross-validation (optional; backward-compatible when all None)
    riesz_lam_grid: object | None = None,
    riesz_sigma_grid: object | None = None,
    riesz_n_centers_grid: object | None = None,
    riesz_cv_folds: int = 3,
    riesz_strict_nested: bool = True,
    riesz_selection_score: str = "bias_variance",
    riesz_admissibility_thresholds: dict | None = None,
    return_riesz_cv_path: bool = False,
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
    stratify_folds: bool | None = None,
    random_state: int | None = 0,
    # Output and inference
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
        Bregman generator used for GRR. If None, you can pass a generator
        function via ``g`` (and optionally ``grad_g``, ``inv_grad_g``,
        ``grad2_g``). The name ``'sq'`` (or ``'squared'``, ``'lsif'``) is
        accepted as a shorthand for ``SquaredGenerator(C=0.0)``. Branch-wise
        generators cannot be named here: a Riesz representer is negative on
        the control units, so the branch depends on the estimand and must be
        supplied explicitly, e.g.
        ``BKLGenerator(C=1.0, branch_fn=lambda x: int(x[treatment_index] == 1.0))``.
        Contrast :func:`genriesz.fit_density_ratio`, where a density ratio is
        nonnegative and every branch-wise name is therefore well defined.
    riesz_method:
        - "grr"            : solve the GRR optimization problem
        - "nn_matching"    : NN-matching inverse propensity weights (**ATE-only** convenience)
        - "local_poly_nn_lsif" : local polynomial NN-LSIF weights (**ATE-only** convenience)

        Matching-based Riesz methods currently require ``cross_fit=False`` and
        do not support ``TMLE`` (because they do not provide a function-valued
        representer that can be evaluated counterfactually).
    riesz_lam_grid, riesz_sigma_grid, riesz_n_centers_grid:
        Inner cross-validation grids for the Riesz hyper-parameters; each may be
        ``"auto"``, a scalar, or a list. ``None`` (the default) means *do not
        vary that dimension*: lambda stays at ``riesz_lam``, and the bandwidth
        and the number of centers stay at the basis' own values. The inner CV
        runs only when at least one grid is supplied, and only on the outer
        training fold.
    riesz_strict_nested:
        Retained as a compatibility argument and required to be ``True``. The
        inner Riesz CV fits standardization, the ``"auto"`` bandwidth, and kernel
        centers on each inner fold's training observations only.
    outcome_link:
        If None, inferred as 'logit' for outcomes bounded in [0, 1], else 'identity'.
        TMLE likelihood is inferred from this link. An explicit ``'logit'``
        requires Y bounded in [0, 1] whenever an outcome model is fitted.
    stratify_folds:
        If None (default), cross-fitting folds are stratified on the treatment
        indicator for the treatment-type functionals (ATE/ATT/DID) and are
        plain K-fold otherwise. Stratification keeps each fold's treated and
        control shares as balanced as the counts allow; a training fold that
        still ends up missing one group raises instead of fitting a degenerate
        Riesz representer. Pass False to force plain (unstratified) K-fold.
        Note that stratified and plain folds partition the sample differently,
        so estimates change (in distribution, not in validity) relative to
        releases before stratification was the default.
    """

    X_ = as_2d(X)
    n = X_.shape[0]
    y_ = as_1d_of_length(Y, n=n, name="Y")

    # Coerce raw callables into LinearFunctional (README-friendly)
    m = _coerce_functional(m)

    # Coerce raw callables into Basis objects (README-friendly)
    basis = coerce_basis(basis)
    if outcome_basis is not None:
        outcome_basis = coerce_basis(outcome_basis)

    ests = _canonical_estimators(estimators)

    # Fail on an invalid inference request before any nuisance is fitted;
    # se_ci_pvalue would only raise after all the cross-fitting work is done.
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha (significance level) must be in (0, 1). Got {alpha!r}.")
    if not np.isfinite(null):
        raise ValueError(f"null must be finite. Got {null!r}.")

    riesz_method_ = str(riesz_method).lower()

    if isinstance(m, (ATEFunctional, ATTFunctional, DIDFunctional)):
        t_idx = getattr(m, "treatment_index", 0)
        _check_treatment_index(X_, t_idx)
        D = X_[:, t_idx]
        uniq = np.unique(D)
        if not np.all(np.isin(uniq, [0.0, 1.0])):
            raise ValueError("Treatment indicator must be binary (0/1).")
        # Fail before any nuisance is fitted: a single-group sample cannot
        # identify ATE/ATT/DID. The balance-diagnostics block used to be the
        # only place this surfaced, and it is skipped entirely when X has no
        # covariate columns besides the treatment (audit N-24).
        if not (np.any(D == 1.0) and np.any(D == 0.0)):
            raise ValueError(
                "Both treatment groups must be nonempty for ATE/ATT/DID "
                f"estimation. Got {int(np.sum(D == 1.0))} treated and "
                f"{int(np.sum(D == 0.0))} control unit(s)."
            )

    if isinstance(m, AMEFunctional) and m.coordinate >= X_.shape[1]:
        raise ValueError(
            f"AME coordinate {m.coordinate} is out of range for X with "
            f"{X_.shape[1]} column(s)."
        )

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
        else:
            # Reject anything that is not a generator before it reaches the solver,
            # and refuse branch-wise names: a Riesz representer is signed, so the
            # branch cannot be inferred from the name (a density ratio's can).
            generator = coerce_generator(generator, allow_branchwise_names=False)

        if isinstance(m, (ATTFunctional, DIDFunctional)) and isinstance(
            generator, (UKLGenerator, BPGenerator, BKLGenerator, BoundedBKLGenerator)
        ):
            c_value = float(getattr(generator, "C", 0.0))
            warn = False
            if isinstance(generator, (UKLGenerator, BPGenerator)) and c_value > 0.0:
                warn = True
            if isinstance(generator, (BKLGenerator, BoundedBKLGenerator)):
                warn = True
            if warn:
                warnings.warn(
                    "ATT and DID Riesz representers can have a control-branch magnitude "
                    "below a positive generator shift C. For ATT/DID with KL-type "
                    "branchwise generators, prefer C=0 for UKL/BP, a very small C for "
                    "BKL only when a lower bound is justified, or SquaredGenerator. "
                    "Using an incompatible C can create boundary weights and unstable "
                    "finite-sample estimates.",
                    UserWarning,
                    stacklevel=2,
                )


    # Outcome link inference
    if outcome_link is None:
        in_unit_interval = bool(np.nanmin(y_) >= 0.0 and np.nanmax(y_) <= 1.0)
        outcome_link_ = "logit" if in_unit_interval else "identity"
        if in_unit_interval and not is_binary_y(y_):
            warnings.warn(
                "outcome_link was not specified and Y lies in [0, 1] but is not "
                "binary; inferring outcome_link='logit' (Bernoulli-style outcome "
                "model and TMLE targeting). Pass outcome_link='identity' if Y is "
                "a bounded continuous outcome.",
                UserWarning,
                stacklevel=2,
            )
    else:
        outcome_link_ = str(outcome_link).lower()
        if outcome_link_ not in {"identity", "logit"}:
            raise ValueError("outcome_link must be 'identity' or 'logit'")

    need_outcome = any(e in {"ra", "arw", "tmle"} for e in ests)

    # A logit outcome model can only produce mu in (0, 1); fitting it to an
    # unbounded Y silently breaks RA (and the outcome diagnostics) long before
    # the TMLE branch would reject the same mismatch (audit N-04). The inferred
    # link already satisfies this bound by construction; this guards the
    # explicitly requested one.
    if outcome_link_ == "logit" and need_outcome:
        if not (np.nanmin(y_) >= 0.0 and np.nanmax(y_) <= 1.0):
            raise ValueError(
                "outcome_link='logit' requires Y bounded in [0, 1] "
                "(a Bernoulli-style outcome model cannot track an unbounded "
                "outcome). Use outcome_link='identity' for continuous Y."
            )

    if outcome_models in {None, "auto"}:
        outcome_models_ = "shared" if need_outcome else "none"
    else:
        outcome_models_ = str(outcome_models).lower()

    if outcome_models_ == "none" and need_outcome:
        raise ValueError("RA/ARW/TMLE require an outcome model; set outcome_models!='none'.")

    # ------------------------------------------------------------------
    # Cross-fitting splits
    # ------------------------------------------------------------------
    is_treatment_functional = isinstance(m, (ATEFunctional, ATTFunctional, DIDFunctional))
    if cross_fit:
        # Plain K-fold can hand a rare-treatment fold zero treated units, and
        # the resulting degenerate Riesz fit used to sail through as a
        # "success" (audit EST-07 / K-01). Stratifying on the treatment keeps
        # every fold's group shares as balanced as the counts allow, so this
        # is the default for the treatment-type functionals.
        stratify = is_treatment_functional if stratify_folds is None else bool(stratify_folds)
        if stratify and not is_treatment_functional:
            raise ValueError(
                "stratify_folds=True requires a treatment-type functional "
                "(ATE/ATT/DID); there is no treatment column to stratify on."
            )
        if stratify:
            D_all = X_[:, getattr(m, "treatment_index", 0)]
            splits = list(
                stratified_kfold_splits(D_all, folds=folds, random_state=random_state)
            )
        else:
            splits = list(kfold_splits(n, folds=folds, random_state=random_state))
    else:
        all_idx = np.arange(n)
        splits = [Fold(train=all_idx, test=all_idx)]

    # A training fold without both groups cannot fit a treatment-type Riesz
    # representer: the closed form would return beta = 0 as a "successful" fit
    # and the fold's scores would silently die (audit EST-07 / K-01).
    # Stratified folds avoid this whenever the counts allow -- a group of >= 2
    # units always leaves at least one in every training fold -- so under the
    # default it fires only for a single-unit group. Fail loud, and before any
    # fold is fitted, since the splits are already fixed here.
    if is_treatment_functional:
        t_idx_m = getattr(m, "treatment_index", 0)
        for fold_id_, fold_ in enumerate(splits):
            D_tr_ = X_[fold_.train, t_idx_m]
            n_tr_treated = int(np.sum(D_tr_ == 1.0))
            n_tr_control = int(np.sum(D_tr_ == 0.0))
            if n_tr_treated == 0 or n_tr_control == 0:
                raise ValueError(
                    f"Cross-fitting fold {fold_id_}: the training fold contains "
                    f"{n_tr_treated} treated and {n_tr_control} control "
                    "unit(s). A treatment-type Riesz representer cannot be "
                    "fitted without both groups. With stratified folds (the "
                    "default) this only happens when a group has a single "
                    "unit, so no fold count avoids it: collect more units of "
                    "that group or set cross_fit=False. With "
                    "stratify_folds=False, prefer the stratified default (or "
                    "fewer folds)."
                )

    # Storage for nuisances (cross-fit predictions)
    alpha_obs = np.zeros(n, dtype=float)

    # For GRR-based TMLE (Gaussian): we only need m(alpha) for each observation.
    m_alpha = np.zeros(n, dtype=float)

    # Outcome regression predictions on observed X
    mu_obs: dict[str, NDArray[np.float64]] = {}
    m_mu: dict[str, NDArray[np.float64]] = {}

    # For Bernoulli TMLE with ATE/ATT/DID we need counterfactual mu/alpha.
    cf_cache: dict[str, NDArray[np.float64]] = {}

    # Per-fold Riesz optimizer diagnostics (item F/G/L: failures and clip
    # binding must be visible, not swallowed).
    riesz_fit_stats: dict[str, list] = {
        "success": [],
        "status": [],
        "gradient_norm": [],
        "kkt_residual": [],
        "clip_binding_rate": [],
    }

    # Held-out working-span imbalance (item H): out-of-fold check of the Riesz
    # balancing condition E[alpha*phi_j] = E[m(.,phi_j)] on the fitted span.
    # This is distinct from the raw-covariate SMD balance below. We keep the
    # per-fold summary scalars and the full Delta vectors (the latter are used
    # for the directional bias proxy and then discarded, not returned).
    imbalance_stats: dict[str, list] = {"max": [], "mean": []}
    imbalance_delta: list[NDArray[np.float64]] = []

    # Kernel-health per fold (item B), populated only when the fitted Riesz
    # basis exposes a diagnostics() method (e.g. GaussianRKHSBasis).
    kernel_stats: list[dict[str, float]] = []

    # Outcome coefficient budget (norm) per fold, used for the bias proxy
    # (item I). Keyed by tag.
    outcome_coef_norm_stats: dict[str, list] = {}

    # Inner Riesz-hyperparameter cross-validation (item C). Active only when a
    # grid is supplied; otherwise the fixed riesz_lam path runs (backward compat).
    riesz_cv_config = GRRCVConfig(
        sigma_grid=riesz_sigma_grid,
        lam_grid=riesz_lam_grid,
        n_centers_grid=riesz_n_centers_grid,
        cv_folds=riesz_cv_folds,
        strict_nested=riesz_strict_nested,
        selection_score=riesz_selection_score,
        admissibility_thresholds=riesz_admissibility_thresholds,
        return_path=return_riesz_cv_path,
        random_state=random_state,
    )
    riesz_cv_active = riesz_method_ == "grr" and riesz_cv_config.is_active
    riesz_cv_selected: list[dict] = []
    riesz_cv_paths: list[list[dict]] = []

    # ------------------------------------------------------------------
    # Fit nuisances fold-by-fold
    # ------------------------------------------------------------------
    for fold_id, fold in enumerate(splits):
        train_idx, test_idx = fold.train, fold.test
        X_tr, y_tr = X_[train_idx], y_[train_idx]
        X_te = X_[test_idx]

        # ----- Riesz representer
        if riesz_method_ == "grr":
            if generator is None:
                raise ValueError("generator is required when riesz_method='grr'")

            # Optional inner CV of the Riesz hyper-parameters on this outer
            # training fold only (the eval fold X_te is never passed in).
            lam_fold = riesz_lam
            if riesz_cv_active:
                sel = select_grr_hyperparams(
                    X_train=X_tr,
                    y_train=y_tr,
                    m=m,
                    basis=basis,
                    generator=generator,
                    config=riesz_cv_config,
                    riesz_penalty=riesz_penalty,
                    riesz_lam=riesz_lam,
                    riesz_p_norm=riesz_p_norm,
                    outcome_link=outcome_link_,
                    outcome_penalty=outcome_penalty,
                    outcome_lam=outcome_lam,
                    max_iter=max_iter,
                    tol=tol,
                )
                lam_fold = sel.lam
                overrides: dict[str, object] = {}
                if sel.sigma is not None:
                    overrides["sigma"] = sel.sigma
                if sel.n_centers is not None:
                    overrides["n_centers"] = sel.n_centers
                    # Drop any fixed centers the basis was built with so the outer
                    # refit reselects ``n_centers`` from this outer-training fold.
                    # Otherwise ``copy_with_params`` keeps the original centers and
                    # silently ignores the selected count (the refit feature map
                    # would not match the ``n_centers`` the CV scored).
                    overrides["centers"] = None
                if overrides:
                    basis_r = basis.copy_with_params(**overrides).fit(X_tr, y_tr)
                else:
                    basis_r = basis.copy().fit(X_tr, y_tr)
                riesz_cv_selected.append(
                    {
                        "fold": fold_id,
                        "sigma": sel.sigma,
                        "lam": sel.lam,
                        "n_centers": sel.n_centers,
                        "n_admissible": sel.n_admissible,
                        "n_candidates": sel.n_candidates,
                        "best_score": sel.best_score,
                    }
                )
                if return_riesz_cv_path:
                    riesz_cv_paths.append(sel.path)
            else:
                basis_r = basis.copy().fit(X_tr, y_tr)

            grr = GRRGLM(
                basis=basis_r,
                generator=generator,
                functional=m,
                penalty=riesz_penalty,
                lam=lam_fold,
                p_norm=riesz_p_norm,
            )
            fit_result = grr.fit(
                X_tr, max_iter=max_iter, tol=tol, verbose=verbose, fit_basis=False
            )
            riesz_fit_stats["success"].append(bool(fit_result.success))
            riesz_fit_stats["status"].append(str(getattr(fit_result, "status", "")))
            riesz_fit_stats["gradient_norm"].append(
                float(getattr(fit_result, "gradient_norm", float("nan")))
            )
            riesz_fit_stats["kkt_residual"].append(
                float(getattr(fit_result, "kkt_residual", float("nan")))
            )
            riesz_fit_stats["clip_binding_rate"].append(
                float(getattr(fit_result, "clip_binding_rate", float("nan")))
            )
            _raise_if_fit_failed(result=fit_result, what="Riesz GRR", fold_id=fold_id)

            alpha_te = grr.predict_alpha(X_te)
            alpha_obs[test_idx] = alpha_te

            # ----- Held-out working-span imbalance (item H). On the eval fold
            # I_k the GRR balancing condition E[alpha*phi_j] = E[m(.,phi_j)]
            # should hold for every basis coordinate j. Its out-of-fold
            # violation Delta_j = mean_i(alpha_hat_i * phi_j(X_i) - M_{i,j}) is
            # the natural bias driver for the augmented estimator.
            Phi_te = np.asarray(basis_r(X_te), dtype=float)
            M_te = np.asarray(m.m_basis_matrix(X_te, basis_r), dtype=float)
            delta = np.mean(alpha_te[:, None] * Phi_te - M_te, axis=0)
            abs_delta = np.abs(delta)
            imbalance_stats["max"].append(float(np.max(abs_delta)))
            imbalance_stats["mean"].append(float(np.mean(abs_delta)))
            imbalance_delta.append(np.asarray(delta, dtype=float))

            # ----- Kernel health (item B), when available on the fitted basis.
            kdiag = getattr(basis_r, "diagnostics", None)
            if callable(kdiag):
                kernel_stats.append({k: v for k, v in kdiag(X_tr).items()})

            # TMLE requires applying the functional to the fitted representer.
            if "tmle" in ests:
                m_alpha[test_idx] = m.m_from_function(
                    X_te,
                    predict=grr.predict_alpha,
                    derivative=getattr(grr, "derivative_alpha", None),
                )

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
                alpha1 = cf_cache.setdefault("alpha1", np.zeros(n, dtype=float))
                alpha0 = cf_cache.setdefault("alpha0", np.zeros(n, dtype=float))
                alpha1[test_idx] = grr.predict_alpha(X1)
                alpha0[test_idx] = grr.predict_alpha(X0)

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
                wobj2: LocalPolynomialLSIFWeights = (
                    local_polynomial_nn_lsif_inverse_propensity_weights(
                        Z_tr,
                        D,
                        M,
                        degree=local_poly_degree,
                        standardize=standardize_for_matching,
                        verbose=verbose,
                    )
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
            ob = basis if outcome_basis is None else outcome_basis
            if riesz_method_ == "grr":
                variants = {
                    "shared": basis_r,
                    "separate": ob.copy().fit(X_tr, y_tr),
                }
            else:
                variants = {
                    "shared": basis.copy().fit(X_tr, y_tr),
                    "separate": ob.copy().fit(X_tr, y_tr),
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
            fit_result = out.fit(
                X_tr, y_tr, max_iter=max_iter, tol=tol, verbose=verbose, fit_basis=False
            )
            _raise_if_fit_failed(
                result=fit_result,
                what="Outcome regression",
                fold_id=fold_id,
                tag=tag,
            )

            # Outcome coefficient budget on this fold's working span (item I).
            theta_out = getattr(out, "theta_", None)
            if theta_out is not None:
                outcome_coef_norm_stats.setdefault(tag, []).append(
                    float(np.linalg.norm(np.asarray(theta_out, dtype=float)))
                )

            mu_obs.setdefault(tag, np.zeros(n, dtype=float))[test_idx] = out.predict(X_te)

            # m(gamma_hat)
            m_mu.setdefault(tag, np.zeros(n, dtype=float))[test_idx] = m.m_from_function(
                X_te,
                predict=out.predict,
                derivative=getattr(out, "derivative", None),
            )

            # Cache cf values for Bernoulli TMLE if needed
            if "tmle" in ests and outcome_link_ == "logit" and isinstance(
                m, (ATEFunctional, ATTFunctional, DIDFunctional)
            ):
                t_idx = getattr(m, "treatment_index", 0)
                X1 = X_te.copy()
                X1[:, t_idx] = 1.0
                X0 = X_te.copy()
                X0[:, t_idx] = 0.0
                mu1_cache = cf_cache.setdefault(f"mu1_{tag}", np.zeros(n, dtype=float))
                mu0_cache = cf_cache.setdefault(f"mu0_{tag}", np.zeros(n, dtype=float))
                mu1_cache[test_idx] = out.predict(X1)
                mu0_cache[test_idx] = out.predict(X0)

    # ------------------------------------------------------------------
    # Compute estimators + inference
    # ------------------------------------------------------------------
    estimates: dict[str, SingleEstimate] = {}

    def _pi_correction(theta: float, psi: NDArray[np.float64]) -> NDArray[np.float64]:
        """First-step correction for the estimated pi in ATT/DID functionals.

        With pi estimated by the sample mean of D, the plug-in target
        theta(pi) = E[(D/pi)(gamma1-gamma0)] has d theta/d pi = -theta/pi, so
        the influence function gains -(theta/pi) * (D - pi). Without it the
        reported SE ignores the estimation of pi.
        """

        if isinstance(m, (ATTFunctional, DIDFunctional)) and bool(
            getattr(m, "pi_is_estimated", False)
        ):
            D_loc = X_[:, getattr(m, "treatment_index", 0)].astype(float)
            pi_loc = float(m.pi)
            return psi - (float(theta) / pi_loc) * (D_loc - pi_loc)
        return psi

    def add_est(key: str, name: str, est: float, psi: NDArray[np.float64]) -> None:
        psi = _pi_correction(est, psi)
        se, lo, hi, p = se_ci_pvalue(est, psi, alpha=alpha, null=null)
        estimates[key] = SingleEstimate(
            name=name,
            estimate=float(est),
            se=float(se),
            ci_low=float(lo),
            ci_high=float(hi),
            p_value=float(p),
        )

    # RW always available when we have alpha
    if "rw" in ests:
        theta = float(np.mean(alpha_obs * y_))
        psi = alpha_obs * y_ - theta
        add_est("rw", "RW", theta, psi)

    if need_outcome:
        # Choose the primary outcome model variant
        if outcome_models_ == "shared":
            primary = "shared"
        elif outcome_models_ == "separate":
            primary = "separate"
        else:
            primary = "shared"  # default

        def compute_for_tag(tag: str, suffix: str = "") -> None:
            mu = mu_obs[tag]
            m_mu_tag = m_mu[tag]

            # m(gamma_hat) is NaN when the functional could not be applied to the
            # outcome model (m_from_function raised NotImplementedError above).
            # RA/ARW would then propagate NaN into se_ci_pvalue, which rejects a
            # non-finite estimate with a message that does not name the cause.
            if any(e in ests for e in ("ra", "arw")) and not np.all(np.isfinite(m_mu_tag)):
                raise RuntimeError(
                    "RA/ARW require applying the functional m to the outcome "
                    "regression, and m(gamma_hat) is not finite for this "
                    "functional / outcome-model combination."
                )

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
                    mu_star = sigmoid(_logit(mu) + eps_hat * alpha_obs)

                    if isinstance(m, (ATEFunctional, ATTFunctional, DIDFunctional)):
                        mu1 = cf_cache[f"mu1_{tag}"]
                        mu0 = cf_cache[f"mu0_{tag}"]
                        a1 = cf_cache.get("alpha1")
                        a0 = cf_cache.get("alpha0")
                        if a1 is None or a0 is None:
                            raise RuntimeError(
                                "Missing alpha counterfactual cache for Bernoulli TMLE"
                            )
                        mu1_star = sigmoid(_logit(mu1) + eps_hat * a1)
                        mu0_star = sigmoid(_logit(mu0) + eps_hat * a0)
                        if isinstance(m, ATEFunctional):
                            m_mu_star = mu1_star - mu0_star
                        else:
                            # ATT and DID
                            D = X_[:, getattr(m, "treatment_index", 0)].astype(float)
                            pi = getattr(m, "pi", float(np.mean(D)))
                            m_mu_star = (D / pi) * (mu1_star - mu0_star)
                    elif isinstance(m, AMEFunctional):
                        mu_valid = _open_unit_interval(
                            mu, name="AME Bernoulli outcome prediction"
                        )
                        d_eta = m_mu_tag / (mu_valid * (1.0 - mu_valid))
                        m_mu_star = (
                            mu_star * (1.0 - mu_star) * (d_eta + eps_hat * m_alpha)
                        )
                    else:
                        raise ValueError("Bernoulli TMLE is only implemented for ATE/ATT/DID/AME.")

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
    # Diagnostics: Love plot and balance table
    # ------------------------------------------------------------------
    diagnostics: dict[str, object] = {}

    alpha_abs = np.abs(alpha_obs)
    diagnostics["alpha_abs_mean"] = float(np.mean(alpha_abs))
    diagnostics["alpha_abs_p95"] = float(np.percentile(alpha_abs, 95))
    diagnostics["alpha_abs_max"] = float(np.max(alpha_abs))

    if riesz_fit_stats["success"]:
        diagnostics["optimizer"] = {k: list(v) for k, v in riesz_fit_stats.items()}
        diagnostics["riesz_fit_success_rate"] = float(np.mean(riesz_fit_stats["success"]))
        with warnings.catch_warnings():
            # nanmax of an all-NaN column is a legitimate "not applicable".
            warnings.simplefilter("ignore", category=RuntimeWarning)
            diagnostics["riesz_gradient_norm_max"] = float(
                np.nanmax(riesz_fit_stats["gradient_norm"])
            )
            diagnostics["riesz_kkt_residual_max"] = float(
                np.nanmax(riesz_fit_stats["kkt_residual"])
            )
            binding_max = float(np.nanmax(riesz_fit_stats["clip_binding_rate"]))
        diagnostics["riesz_clip_binding_rate_max"] = binding_max
        if np.isfinite(binding_max) and binding_max > 0.0:
            # Truncated links state representer bounds (they define per-side
            # binding masks); for exact restricted-domain links the same rate
            # instead measures proximity to the dual-domain boundary.
            states_bounds = callable(
                getattr(generator, "lower_binding", None)
            ) and callable(getattr(generator, "upper_binding", None))
            if states_bounds:
                message = (
                    f"The generator's stated representer bound was active for "
                    f"up to {binding_max:.1%} of training observations in at "
                    f"least one fold. The bounds are part of the fitted model; "
                    f"report this binding rate alongside the estimate and "
                    f"check that the bounds suit the application."
                )
            else:
                message = (
                    f"The fitted dual index came within the diagnostic margin "
                    f"of the generator's exact dual-domain boundary for up to "
                    f"{binding_max:.1%} of training observations in at least "
                    f"one fold. The fitted values are exact and nothing was "
                    f"clamped, but weights can be extreme near the boundary."
                )
            warnings.warn(message, UserWarning, stacklevel=2)

    # Inner Riesz CV selections (item C), per outer fold.
    if riesz_cv_selected:
        cv_diag: dict[str, object] = {
            "selected": riesz_cv_selected,
            "strict_nested": True,
        }
        if return_riesz_cv_path and riesz_cv_paths:
            cv_diag["path"] = riesz_cv_paths
        diagnostics["riesz_cv"] = cv_diag
        diagnostics["riesz_cv_selection_score"] = riesz_cv_config.selection_score
        sel_lams = [s["lam"] for s in riesz_cv_selected]
        diagnostics["riesz_cv_lam_median"] = float(np.median(sel_lams))
        sel_sigmas = [s["sigma"] for s in riesz_cv_selected if s["sigma"] is not None]
        if sel_sigmas:
            diagnostics["riesz_cv_sigma_median"] = float(np.median(sel_sigmas))

    # Held-out working-span imbalance (item H). Distinct from the raw-covariate
    # SMD balance below: this checks the GRR balancing condition on the fitted
    # basis span, out of fold.
    if imbalance_stats["max"]:
        diagnostics["imbalance"] = {
            "held_out_working_span_max": list(imbalance_stats["max"]),
            "held_out_working_span_mean": list(imbalance_stats["mean"]),
        }
        diagnostics["held_out_imbalance_max"] = float(np.max(imbalance_stats["max"]))
        diagnostics["held_out_imbalance_mean"] = float(np.mean(imbalance_stats["mean"]))

    # Kernel health (item B), aggregated across folds when available.
    if kernel_stats:
        diagnostics["kernel"] = {"per_fold": kernel_stats}

        def _kcol(key: str) -> NDArray[np.float64]:
            return np.asarray([k.get(key, np.nan) for k in kernel_stats], dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            diagnostics["kernel_median_min"] = float(np.nanmin(_kcol("kernel_median")))
            diagnostics["kernel_feature_variance_min"] = float(
                np.nanmin(_kcol("feature_variance_min"))
            )
            diagnostics["kernel_gram_condition_max"] = float(
                np.nanmax(_kcol("gram_condition_number"))
            )
            diagnostics["kernel_effective_rank_min"] = float(np.nanmin(_kcol("effective_rank")))
        diagnostics["kernel_underfitting_any"] = bool(
            any(bool(k.get("underfitting", False)) for k in kernel_stats)
        )

    # Bias proxy and standardized bias (item I). The empirical second-order
    # term that the balancing condition is meant to kill is
    #   E[alpha_hat * gamma_hat] - E[m(., gamma_hat)],
    # evaluated here directly from the held-out predictions, with no coordinate
    # matching between the Riesz span and the outcome span (pairing the
    # Riesz-span Delta with a separate outcome basis of coincidentally equal
    # column count used to dot unrelated coordinates). The headline b_hat is
    # the unweighted across-fold mean of the per-fold absolute means
    #   (1/K) sum_k |E_{I_k}[alpha_hat*gamma_hat - m(., gamma_hat)]|
    # (the same fold aggregation as before, not a pooled |E_n[...]|), and
    # b_hat_max is the worst fold. The conservative Cauchy-Schwarz bound
    # ||Delta|| * ||theta|| (b_bound) is only a bound when the outcome model is
    # *linear in the Riesz basis coordinates*: the outcome must be fit on the
    # same fitted basis object ("shared") AND use the identity link -- under a
    # logit link gamma_hat = sigmoid(phi^T theta), so theta = 0 gives a
    # constant 0.5 prediction with a generally nonzero second-order term while
    # ||Delta||*||theta|| = 0. Otherwise b_bound is NaN with the reason
    # recorded. Both are diagnostics only and never used for selection.
    if imbalance_stats["max"] and outcome_coef_norm_stats:
        tag_pref = (
            "shared" if "shared" in outcome_coef_norm_stats else next(iter(outcome_coef_norm_stats))
        )
        cnorms = list(outcome_coef_norm_stats[tag_pref])
        same_span = tag_pref == "shared"
        bound_defined = same_span and outcome_link_ == "identity"
        mu_pref = mu_obs[tag_pref]
        m_mu_pref = m_mu[tag_pref]

        b_dir_fold: list[float] = []
        b_bound_fold: list[float] = []
        for k, fold in enumerate(splits):
            te = fold.test
            second_order = alpha_obs[te] * mu_pref[te] - m_mu_pref[te]
            b_dir_fold.append(
                float(abs(float(np.mean(second_order)))) if second_order.size else float("nan")
            )
            if bound_defined and k < len(imbalance_delta) and k < len(cnorms):
                b_bound_fold.append(float(np.linalg.norm(imbalance_delta[k]) * cnorms[k]))
            else:
                b_bound_fold.append(float("nan"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            b_hat = float(np.nanmean(b_dir_fold)) if b_dir_fold else float("nan")
            b_hat_max = float(np.nanmax(b_dir_fold)) if b_dir_fold else float("nan")
            b_bound = float(np.nanmean(b_bound_fold)) if b_bound_fold else float("nan")

        primary = None
        for key in ("arw", "arw (shared)", "rw", "ra", "tmle"):
            if key in estimates:
                primary = estimates[key]
                break
        if primary is None and estimates:
            primary = next(iter(estimates.values()))

        se_primary = float(primary.se) if primary is not None else float("nan")
        v_hat = float(n * se_primary * se_primary) if np.isfinite(se_primary) else float("nan")
        std_bias = (
            float(b_hat / se_primary)
            if np.isfinite(se_primary) and se_primary > 0
            else float("nan")
        )

        diagnostics["bias"] = {
            "b_hat": b_hat,
            "b_hat_max": b_hat_max,
            "b_bound": b_bound,
            "v_hat": v_hat,
            "std_bias": std_bias,
            "outcome_coef_norm_mean": float(np.mean(cnorms)) if cnorms else float("nan"),
            "outcome_tag": tag_pref,
        }
        if not bound_defined:
            cause = (
                "the outcome basis differs from the Riesz basis"
                if not same_span
                else "the logit link makes gamma_hat nonlinear in theta"
            )
            diagnostics["bias"]["b_bound_unavailable_reason"] = (
                "the Cauchy-Schwarz bound ||Delta||*||theta|| needs the outcome "
                f"model to be linear in the Riesz basis coordinates: {cause}"
            )
        diagnostics["bias_proxy"] = b_hat
        diagnostics["std_bias"] = std_bias

    # Outcome-nuisance diagnostics (Step 4). Coverage collapse is driven by the
    # PRODUCT of the Riesz and outcome errors, so the outcome side needs its own
    # visibility: out-of-fold prediction risk (the off-span residual proxy),
    # residual fold statistics, and the working-span coefficient budget. These
    # are additive; point estimates and SEs are unchanged.
    if mu_obs:
        outcome_diag: dict[str, object] = {}
        for tag, mu in mu_obs.items():
            resid = y_ - mu
            if outcome_link_ == "identity":
                cv_risk = float(np.mean(resid * resid))
            else:
                p = _open_unit_interval(mu, name="Outcome prediction")
                cv_risk = float(
                    np.mean(-(y_ * np.log(p) + (1.0 - y_) * np.log1p(-p)))
                )
            fold_means: list[float] = []
            fold_vars: list[float] = []
            for fold in splits:
                r = resid[fold.test]
                fold_means.append(float(np.mean(r)) if r.size else float("nan"))
                fold_vars.append(float(np.var(r, ddof=1)) if r.size > 1 else float("nan"))
            cnorm_list = outcome_coef_norm_stats.get(tag, [])
            outcome_diag[tag] = {
                "cv_risk": cv_risk,
                "residual_mean": float(np.mean(resid)),
                "residual_var": float(np.var(resid, ddof=1)) if resid.size > 1 else float("nan"),
                "residual_fold_mean": fold_means,
                "residual_fold_var": fold_vars,
                "coef_norm_mean": float(np.mean(cnorm_list)) if cnorm_list else float("nan"),
            }
        diagnostics["outcome"] = outcome_diag
        primary_tag = "shared" if "shared" in outcome_diag else next(iter(outcome_diag))
        diagnostics["outcome_cv_risk"] = outcome_diag[primary_tag]["cv_risk"]
        diagnostics["outcome_residual_var"] = outcome_diag[primary_tag]["residual_var"]

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

            target: Literal["ate", "att"] = "ate" if isinstance(m, ATEFunctional) else "att"
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

    return FunctionalEstimate(
        estimand=m.name,
        n=n,
        alpha=alpha,
        null=null,
        estimates=estimates,
        diagnostics=diagnostics,
    )


# ----------------------------------------------------------------------
# Convenience wrappers
# ----------------------------------------------------------------------

def grr_ate(
    *,
    X: ArrayLike,
    Y: ArrayLike,
    treatment_index: int = 0,
    basis: Basis | Callable,
    generator: BregmanGenerator | str | None = None,
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
    generator: BregmanGenerator | str | None = None,
    **kwargs,
) -> FunctionalEstimate:
    """Estimate ATT with the GRR API."""

    t_idx = _validate_treatment_index_arg(treatment_index)
    X_ = as_2d(X)
    _check_treatment_index(X_, t_idx)
    D = X_[:, t_idx]
    pi = float(np.mean(D))
    if pi <= 0 or pi >= 1:
        raise ValueError("ATT requires both treatment groups to be non-empty")
    m = ATTFunctional(treatment_index=t_idx, pi=pi, pi_is_estimated=True)
    return grr_functional(X=X, Y=Y, m=m, basis=basis, generator=generator, **kwargs)


def grr_did(
    *,
    X: ArrayLike,
    Y0: ArrayLike,
    Y1: ArrayLike,
    treatment_index: int = 0,
    basis: Basis | Callable,
    generator: BregmanGenerator | str | None = None,
    **kwargs,
) -> FunctionalEstimate:
    """Panel DID implemented as ATT on ΔY = Y1-Y0."""

    X_ = as_2d(X)
    n = X_.shape[0]
    y0 = as_1d_of_length(Y0, n=n, name="Y0")
    y1 = as_1d_of_length(Y1, n=n, name="Y1")
    dy = y1 - y0

    t_idx = _validate_treatment_index_arg(treatment_index)
    _check_treatment_index(X_, t_idx)
    D = X_[:, t_idx]
    pi = float(np.mean(D))
    if pi <= 0 or pi >= 1:
        raise ValueError("DID requires both treatment groups to be non-empty")
    m = DIDFunctional(treatment_index=t_idx, pi=pi, pi_is_estimated=True)
    return grr_functional(X=X, Y=dy, m=m, basis=basis, generator=generator, **kwargs)


def grr_ame(
    *,
    X: ArrayLike,
    Y: ArrayLike,
    coordinate: int = 0,
    basis: Basis | Callable,
    generator: BregmanGenerator | str | None = None,
    **kwargs,
) -> FunctionalEstimate:
    """Estimate an average marginal effect (average derivative) wrt x_coordinate."""

    m = AMEFunctional(coordinate=coordinate)
    return grr_functional(X=X, Y=Y, m=m, basis=basis, generator=generator, **kwargs)
