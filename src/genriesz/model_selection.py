"""Inner cross-validation for Riesz-side hyper-parameters (design section 3.4).

This module selects the Riesz bandwidth (``sigma``), regularization (``lam``) and
number of kernel centers (``n_centers``) *inside* an outer training fold, so that
the outer evaluation fold is never touched during selection.

Design principles (see ``doc/coverage_failure_improvement_design_revised.md``):

- ``select_grr_hyperparams`` receives the outer *training* sample only. It never
  sees the outer evaluation fold (no leakage of centers, standardization, or
  selection).
- Strict nested CV (``GRRCVConfig.strict_nested=True``, the default): inside each
  *inner* fold every preprocessing step -- standardization, the ``"auto"`` sigma
  median heuristic, kernel-center selection, and the supervised basis fit -- is
  derived from that fold's *inner-training* rows alone. The inner-validation rows
  only ever *evaluate* the already-fitted feature map, so no inner-validation
  observation enters the feature map that scores it. The selected candidate is
  then refit on the whole outer-training fold (bandwidth reported at the
  outer-training median; ``n_centers`` reselected from the outer-training rows).
- ``strict_nested=False`` restores the older *outer-fixed feature map*: centers
  and the sigma median heuristic are computed once on the whole outer-training
  fold and shared across the inner folds. This is cheaper but leaks each inner
  fold's validation rows into the feature map that scores them; it is recorded on
  the result (``GRRCVResult.strict_nested``) and is not the default.
- Selection is two-stage: an *admissibility* screen (optimizer success, effective
  sample size, cap binding, ...) followed by a *criterion* minimization
  (default ``bias_variance``: ``B^2 + V/n + tau_R R + tau_K K``).
- Neither coverage nor any true nuisance/effect is used for selection.
- Generators that modify the estimand (``generator.modifies_estimand``, e.g.
  :class:`~genriesz.generators.BoundedBKLGenerator`) are never admissible
  (design section 9-4). A bound that binds changes the target functional, so
  such candidates are a *target-sensitivity* analysis and must not compete on a
  criterion against candidates that target the original estimand.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import Basis, _median_pairwise_distance
from .functionals import AMEFunctional, LinearFunctional
from .generators import BregmanGenerator
from .glm import GRRGLM, OutcomeGLM
from .utils import Fold, kfold_splits

# Default candidate grids (design section 3.4 / coverage design "Candidate 集合").
DEFAULT_SIGMA_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_LAM_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
DEFAULT_N_CENTERS_BASE: tuple[int, ...] = (80, 120, 240, 400)

SELECTION_SCORES = (
    "bias_variance",
    "bregman_validation",
    "squared_loss_validation",
    "imbalance_validation",
)


def default_admissibility_thresholds() -> dict[str, float | None]:
    """Return the default admissibility thresholds.

    ``None`` means the corresponding screen is *not* enforced by default; the
    revision plan asks for thresholds to be exposed and their sensitivity
    reported rather than baked in as blessed values. The only always-on screens
    are optimizer success and a floor on the effective sample size.
    """

    return {
        "min_ess_ratio": 0.05,
        "max_cap_binding_rate": 0.0,
        # Kernel-health band (scale-free). A collapsed kernel (median ~0) is the
        # small-bandwidth underfitting trap flagged in the design; a saturated
        # kernel (median ~1) is the large-bandwidth trap that yields near-constant
        # features and a deceptively tiny variance. Both must be inadmissible so
        # the bias/variance criterion cannot be gamed by fake precision. Only
        # applied when the basis exposes kernel diagnostics.
        "kernel_median_floor": 1e-3,
        "kernel_median_ceil": 0.8,
        "max_kkt_residual": None,
        "max_abs_alpha": None,
        "max_std_imbalance": None,
    }


@dataclass
class GRRCVConfig:
    """Configuration for inner Riesz-hyperparameter cross-validation.

    Parameters
    ----------
    sigma_grid, lam_grid, n_centers_grid:
        Candidate specs. Each may be ``None`` (do not vary this dimension;
        keep the estimator's fixed value), ``"auto"``, a scalar, or a list.
        ``sigma_grid="auto"`` expands to ``median_pairwise_distance * (0.25, 0.5,
        1, 2, 4)``; ``lam_grid="auto"`` to :data:`DEFAULT_LAM_GRID`;
        ``n_centers_grid="auto"`` to :data:`DEFAULT_N_CENTERS_BASE` (capped at
        ``n``). With ``lam_grid=None`` the single candidate is the caller's
        ``riesz_lam``, so cross-validating ``sigma_grid`` alone leaves the
        penalty at the value the estimator was given.
    cv_folds:
        Number of inner folds.
    strict_nested:
        If True (default), run *strict* nested CV: within each inner fold the
        standardization, the ``"auto"`` sigma median heuristic, the kernel-center
        pool, and the supervised basis fit are all derived from that fold's
        inner-training rows only, so no inner-validation observation enters the
        feature map that scores it (design section 3.4, audit P0-05). If False,
        use the older *outer-fixed feature map*: centers and the median heuristic
        are computed once on the whole outer-training fold and shared across inner
        folds -- cheaper, but the inner-validation rows leak into their own
        scoring feature map. The choice is recorded on
        :class:`GRRCVResult.strict_nested`.
    selection_score:
        One of ``"bias_variance"`` (default), ``"bregman_validation"``,
        ``"squared_loss_validation"``, ``"imbalance_validation"``.
        ``"squared_loss_validation"`` scores every candidate by the held-out
        squared-loss (LSIF) risk of its fitted representer,
        ``1/2 E[alpha_hat^2] - E[m(alpha_hat)]``, regardless of the generator
        used to fit it. This is a generator-agnostic yardstick (minimized at the
        true Riesz representer, uLSIF-style): unlike ``"bregman_validation"``,
        which scores each candidate by its *own* Bregman risk, its value is on the
        same scale for every generator, so paths obtained from separate calls
        with different generators -- e.g. ``SquaredGenerator`` vs. a
        ``BPGenerator`` with varying ``omega``/``C`` -- are directly comparable
        (each call still fits one generator). Caveat: a small LSIF risk does not
        undo an estimand modification. The risk always measures distance to the
        *original* estimand's Riesz representer, so for a ``modifies_estimand``
        generator, or a ``BPGenerator`` whose clip binds, a finite risk is scored
        against a clipped/bounded representer -- check the bound-binding rate and
        admissibility separately. Functionals whose ``m(alpha)`` needs a
        representer derivative (e.g. ``AMEFunctional``) do not support this score
        and raise a clear ``ValueError`` before any candidate is scored.
    admissibility_thresholds:
        Overrides for :func:`default_admissibility_thresholds`.
    tau_R, tau_K:
        Weights of the KKT-residual and kernel-degeneracy penalties in the
        ``bias_variance`` criterion.
    return_path:
        If True, the full candidate path table is returned in the result.
    random_state:
        Seed for the inner splits and the center subsample.
    """

    sigma_grid: object = None
    lam_grid: object = None
    n_centers_grid: object = None
    cv_folds: int = 3
    strict_nested: bool = True
    selection_score: str = "bias_variance"
    admissibility_thresholds: dict[str, float | None] | None = None
    tau_R: float = 1e-2
    tau_K: float = 1e-3
    return_path: bool = False
    random_state: int | None = 0

    def __post_init__(self) -> None:
        if self.selection_score not in SELECTION_SCORES:
            raise ValueError(f"selection_score must be one of {SELECTION_SCORES}")
        if int(self.cv_folds) < 2:
            raise ValueError("cv_folds must be >= 2")

    @property
    def is_active(self) -> bool:
        """Whether any hyper-parameter dimension is being cross-validated."""

        return any(
            g is not None for g in (self.sigma_grid, self.lam_grid, self.n_centers_grid)
        )


@dataclass
class GRRCVResult:
    """Selected Riesz hyper-parameters and the candidate path table.

    ``modifies_estimand`` records that the generator targets a modified estimand
    (design section 9-4). It forces ``n_admissible == 0``: the selection below is
    then a target-sensitivity analysis over the bounded target, not a selection
    over the original one. It is keyword-only so that adding it leaves the
    positional signature and ``__match_args__`` of the previous release intact.

    ``strict_nested`` records whether the inner CV used strict nested
    preprocessing (per-fold centers and median heuristic; the default) or the
    older outer-fixed feature map (audit P0-05). ``fold_provenance`` holds, per
    inner fold, the ``validation_index`` (inner-validation rows), the
    ``preprocess_fit_index`` (rows the standardization / median / center pool were
    fit on) and the ``center_index`` (global rows chosen as kernel centers). Under
    strict nested CV the center and preprocess indices are disjoint from the
    validation index; under the outer-fixed feature map they overlap it (the
    honest record of the leak). Both fields are keyword-only to keep the
    positional signature and ``__match_args__`` intact.
    """

    sigma: float | None
    lam: float
    n_centers: int | None
    selection_score: str
    best_score: float
    n_admissible: int
    n_candidates: int
    path: list[dict] = field(default_factory=list)
    modifies_estimand: bool = field(default=False, kw_only=True)
    strict_nested: bool = field(default=True, kw_only=True)
    fold_provenance: list[dict] = field(default_factory=list, kw_only=True)


def _effective_sample_size(w: NDArray[np.float64]) -> float:
    w = np.abs(np.asarray(w, dtype=float))
    s = float(np.sum(w))
    if s <= 0:
        return 0.0
    return float(s * s / float(np.sum(w * w)))


def _standardized(X: NDArray[np.float64]) -> NDArray[np.float64]:
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std > 0, std, 1.0)
    return (X - mean) / std


def normalize_grid(
    spec: object,
    *,
    kind: str,
    median: float | None = None,
    n: int | None = None,
) -> list[float] | list[int]:
    """Normalize a grid spec (``"auto"`` / scalar / list) into a candidate list."""

    if kind == "sigma":
        if isinstance(spec, str):
            if spec.lower() != "auto":
                raise ValueError("sigma_grid string must be 'auto'")
            if median is None or not np.isfinite(median) or median <= 0:
                raise ValueError("sigma_grid='auto' needs a positive median distance")
            return [float(median) * m for m in DEFAULT_SIGMA_MULTIPLIERS]
        if np.isscalar(spec):
            return [float(spec)]  # type: ignore[arg-type]
        return [float(s) for s in spec]  # type: ignore[union-attr]

    if kind == "lam":
        if spec is None:
            raise ValueError(
                "lam_grid=None means 'do not vary lambda' (keep the estimator's "
                "riesz_lam); the caller must resolve it. Pass 'auto' for the "
                "default grid."
            )
        if isinstance(spec, str):
            if spec.lower() != "auto":
                raise ValueError("lam_grid string must be 'auto'")
            return list(DEFAULT_LAM_GRID)
        if np.isscalar(spec):
            return [float(spec)]  # type: ignore[arg-type]
        return [float(x) for x in spec]  # type: ignore[union-attr]

    if kind == "n_centers":
        if n is None:
            raise ValueError("n_centers grid needs n")
        if isinstance(spec, str):
            if spec.lower() != "auto":
                raise ValueError("n_centers_grid string must be 'auto'")
            return sorted({min(int(c), int(n)) for c in DEFAULT_N_CENTERS_BASE})
        if np.isscalar(spec):
            return [min(int(spec), int(n))]  # type: ignore[arg-type]
        return sorted({min(int(c), int(n)) for c in spec})  # type: ignore[union-attr]

    raise ValueError(f"Unknown grid kind: {kind}")


def _select_center_indices(
    index_pool: NDArray[np.int_], *, n_centers: int, random_state: int | None = 0
) -> NDArray[np.int_]:
    """Pick up to ``n_centers`` global row indices from ``index_pool``.

    ``index_pool`` holds the *global* row indices a fold is allowed to draw
    centers from (its inner-training rows under strict nested CV, or all training
    rows for the outer-fixed feature map). Selecting on the pool -- rather than on
    the full sample -- is what keeps inner-validation rows out of the feature map.
    """

    pool = np.asarray(index_pool, dtype=int)
    n = pool.shape[0]
    m = min(int(n_centers), n)
    rng = np.random.default_rng(random_state)
    return pool[rng.choice(n, size=m, replace=False)]


def select_kernel_centers(
    X_train: NDArray[np.float64], *, n_centers: int, random_state: int | None = 0
) -> NDArray[np.float64]:
    """Select up to ``n_centers`` center rows from the outer training fold."""

    X_train = np.asarray(X_train, dtype=float)
    idx = _select_center_indices(
        np.arange(X_train.shape[0]), n_centers=n_centers, random_state=random_state
    )
    return X_train[idx]


def make_candidate_basis(
    template: Basis, *, sigma: float | None, centers: NDArray[np.float64] | None
) -> Basis:
    """Build an unfitted candidate basis differing only in ``sigma``/``centers``.

    Requires a basis exposing ``copy_with_params`` (e.g. ``GaussianRKHSBasis``)
    when ``sigma`` or ``centers`` are given. For lambda-only CV (both ``None``),
    any basis is cloned with ``copy()``.
    """

    if sigma is None and centers is None:
        return template.copy()
    cwp = getattr(template, "copy_with_params", None)
    if not callable(cwp):
        raise ValueError(
            "sigma/n_centers cross-validation requires a basis with "
            "copy_with_params (e.g. GaussianRKHSBasis). For other bases, "
            "cross-validate lambda only."
        )
    overrides: dict[str, object] = {}
    if sigma is not None:
        overrides["sigma"] = float(sigma)
    if centers is not None:
        overrides["centers"] = centers
    return cwp(**overrides)


def score_grr_candidate(
    *,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    m: LinearFunctional,
    template_basis: Basis,
    generator: BregmanGenerator,
    sigma: float | None | list[float | None],
    lam: float,
    centers: NDArray[np.float64] | None | list[NDArray[np.float64] | None],
    inner_folds: list[Fold],
    riesz_penalty: str | None,
    riesz_p_norm: float | None,
    outcome_link: str,
    outcome_penalty: str | None,
    outcome_lam: float,
    max_iter: int,
    tol: float,
    want_kernel: bool,
    want_squared_loss: bool = False,
) -> dict:
    """Evaluate one candidate over the inner folds and aggregate diagnostics.

    ``sigma`` and ``centers`` may each be a single value (used for every inner
    fold -- the outer-fixed feature map) or a *list aligned with* ``inner_folds``
    (strict nested CV, where the caller resolves a per-fold bandwidth from the
    inner-training median and a per-fold center pool from the inner-training rows).
    The returned ``sigma``/``n_centers`` fields are meaningful only for the single
    value case; for per-fold lists the caller overrides them with the candidate's
    outer-training resolution.

    ``want_squared_loss`` turns on the generator-agnostic squared-loss (LSIF)
    validation risk (only meaningful when it is the selected score). A candidate
    is usable for that score only if the risk is finite on *every* fold; a single
    non-finite fold drops it out of selection (so candidates are never compared on
    a partial-fold average).
    """

    risks: list[float] = []
    imbalances: list[float] = []
    sq_risks: list[float] = []
    variances: list[float] = []
    coef_norms: list[float] = []
    ess_ratios: list[float] = []
    max_alphas: list[float] = []
    kkts: list[float] = []
    bindings: list[float] = []
    eff_ranks: list[float] = []
    kernel_medians: list[float] = []
    all_success = True

    for fi, fold in enumerate(inner_folds):
        itr, iva = fold.train, fold.test
        X_itr, y_itr = X_train[itr], y_train[itr]
        X_iva, y_iva = X_train[iva], y_train[iva]

        # Per-fold bandwidth / centers under strict nested CV (lists aligned with
        # ``inner_folds``); a single value is the outer-fixed feature map.
        sigma_f = sigma[fi] if isinstance(sigma, list) else sigma
        centers_f = centers[fi] if isinstance(centers, list) else centers

        cb = make_candidate_basis(template_basis, sigma=sigma_f, centers=centers_f)
        cb.fit(X_itr, y_itr)

        grr = GRRGLM(
            basis=cb,
            generator=generator,
            functional=m,
            penalty=riesz_penalty,
            lam=lam,
            p_norm=riesz_p_norm,
        )
        fr = grr.fit(X_itr, max_iter=max_iter, tol=tol)
        if not fr.success or grr.beta_ is None:
            all_success = False
            continue

        Phi_iva = np.asarray(cb(X_iva), dtype=float)
        M_iva = np.asarray(m.m_basis_matrix(X_iva, cb), dtype=float)
        beta = grr.beta_
        v_iva = Phi_iva @ beta

        # Unpenalized Bregman-Riesz validation risk.
        try:
            g_star, alpha_iva = generator.conjugate(X_iva, v_iva)
            risks.append(float(np.mean(g_star - (M_iva @ beta))))
        except Exception:
            all_success = False
            continue

        # Generator-agnostic squared-loss (LSIF) validation risk of the fitted
        # representer alpha_hat = generator.inv_grad(phi @ beta):
        #   1/2 E[alpha_hat^2] - E[m(alpha_hat)],
        # minimized at the true Riesz representer. Unlike the Bregman risk above
        # (which uses each candidate's *own* g*), this is a common yardstick that
        # lets candidates fit with different generators be compared directly. Only
        # computed when it is the selected score. Exceptions are NOT swallowed: a
        # functional that cannot express m(alpha) without a representer derivative
        # raises a clear error rather than masquerading as a failed fit.
        if want_squared_loss:
            def _representer(
                XX: NDArray[np.float64],
                _cb: Basis = cb,
                _beta: NDArray[np.float64] = beta,
                _gen: BregmanGenerator = generator,
            ) -> NDArray[np.float64]:
                phi = np.asarray(_cb(XX), dtype=float)
                return np.asarray(_gen.inv_grad(XX, phi @ _beta), dtype=float)

            try:
                m_rep = np.asarray(
                    m.m_from_function(X_iva, predict=_representer, derivative=None),
                    dtype=float,
                )
            except NotImplementedError as exc:
                raise ValueError(
                    "selection_score='squared_loss_validation' requires the "
                    "functional to evaluate m(alpha) from the representer alone "
                    f"(no derivative); {type(m).__name__} does not. Use "
                    "'bregman_validation' or 'bias_variance' instead."
                ) from exc
            sq = 0.5 * float(np.mean(np.square(alpha_iva))) - float(np.mean(m_rep))
            # A non-finite fold makes this candidate incomparable on the LSIF
            # scale; record NaN so strict aggregation drops it from selection.
            sq_risks.append(sq if np.isfinite(sq) else float("nan"))

        # Held-out imbalance on the inner-validation fold.
        delta = np.mean(alpha_iva[:, None] * Phi_iva - M_iva, axis=0)
        imb = float(np.max(np.abs(delta)))
        imbalances.append(imb)

        # Weight-tail diagnostics.
        abs_alpha = np.abs(alpha_iva)
        ess_ratios.append(_effective_sample_size(abs_alpha) / max(len(iva), 1))
        max_alphas.append(float(np.max(abs_alpha)) if abs_alpha.size else 0.0)

        kkts.append(float(fr.kkt_residual))
        binding = float(fr.clip_binding_rate)
        bindings.append(0.0 if not np.isfinite(binding) else binding)

        # Outcome regression on the same span: coefficient budget + score variance.
        out = OutcomeGLM(
            basis=cb, link=outcome_link, penalty=outcome_penalty, lam=outcome_lam
        )
        out_fr = out.fit(X_itr, y_itr, max_iter=max_iter, tol=tol)
        if out.theta_ is not None and out_fr.success:
            coef_norms.append(float(np.linalg.norm(out.theta_)))
            try:
                m_gamma = m.m_from_function(
                    X_iva, predict=out.predict, derivative=getattr(out, "derivative", None)
                )
                gamma_iva = out.predict(X_iva)
                psi = np.asarray(m_gamma, dtype=float) + alpha_iva * (y_iva - gamma_iva)
                variances.append(float(np.var(psi, ddof=1)) if psi.size > 1 else float("nan"))
            except NotImplementedError:
                variances.append(float("nan"))

        if want_kernel:
            kdiag = getattr(cb, "diagnostics", None)
            if callable(kdiag):
                kd = kdiag(X_itr, max_rows=256)
                eff_ranks.append(float(kd.get("effective_rank", np.nan)))
                kernel_medians.append(float(kd.get("kernel_median", np.nan)))
            want_kernel = False  # one representative fold is enough

    def _nanmean(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return float(np.nanmean(xs))

    b_imb = _nanmean(imbalances)
    b_coef = _nanmean(coef_norms)
    b_hat = float(b_imb * b_coef) if np.isfinite(b_imb) and np.isfinite(b_coef) else float("nan")
    v_hat = _nanmean(variances)
    r_hat = _nanmean(kkts)
    eff_rank = _nanmean(eff_ranks)
    k_hat = float(1.0 / eff_rank) if np.isfinite(eff_rank) and eff_rank > 0 else 0.0
    std_imb = (
        float(b_imb / np.sqrt(v_hat)) if np.isfinite(v_hat) and v_hat > 0 else float("nan")
    )
    # Strict aggregation for the LSIF score: valid only if it was scored and
    # finite on *every* fold. A candidate that failed the primary fit on any
    # fold (so its LSIF was never appended) or produced a non-finite value keeps
    # NaN, so it is never recorded -- or selected -- on a partial-fold average.
    sq_val = (
        float(np.mean(sq_risks))
        if len(sq_risks) == len(inner_folds) and bool(np.all(np.isfinite(sq_risks)))
        else float("nan")
    )

    # For per-fold (list) sigma/centers the representative value is ambiguous, so
    # the caller overrides these with the candidate's outer-training resolution.
    if isinstance(sigma, list):
        rep_sigma: float | None = float("nan")
    else:
        rep_sigma = None if sigma is None else float(sigma)
    rep_ncenters = (
        None if isinstance(centers, list) or centers is None else int(centers.shape[0])
    )
    return {
        "sigma": rep_sigma,
        "lam": float(lam),
        "n_centers": rep_ncenters,
        "success": bool(all_success and imbalances),
        "modifies_estimand": bool(getattr(generator, "modifies_estimand", False)),
        "bregman_validation": _nanmean(risks),
        "squared_loss_validation": sq_val,
        "held_out_imbalance": b_imb,
        "std_imbalance": std_imb,
        "b_hat": b_hat,
        "v_hat": v_hat,
        "r_hat": r_hat,
        "k_hat": k_hat,
        "ess_ratio_min": float(np.min(ess_ratios)) if ess_ratios else 0.0,
        "max_abs_alpha": float(np.max(max_alphas)) if max_alphas else float("nan"),
        "cap_binding_rate": float(np.max(bindings)) if bindings else 0.0,
        "effective_rank": eff_rank,
        "kernel_median": _nanmean(kernel_medians),
        "outcome_coef_norm": b_coef,
    }


def _is_admissible(row: dict, thr: dict[str, float | None]) -> bool:
    if not row["success"]:
        return False
    # Design section 9-4: a generator that modifies the estimand is always a
    # target-sensitivity candidate, never an admissible one -- even when its
    # bound happens not to bind on this sample.
    if row.get("modifies_estimand", False):
        return False
    min_ess = thr.get("min_ess_ratio")
    if min_ess is not None and row["ess_ratio_min"] < float(min_ess):
        return False
    max_bind = thr.get("max_cap_binding_rate")
    if max_bind is not None and row["cap_binding_rate"] > float(max_bind):
        return False
    # Kernel-health band (skipped when kernel diagnostics are unavailable, i.e.
    # kernel_median is NaN for non-kernel bases).
    kmed = row["kernel_median"]
    if np.isfinite(kmed):
        floor = thr.get("kernel_median_floor")
        if floor is not None and kmed < float(floor):
            return False
        ceil = thr.get("kernel_median_ceil")
        if ceil is not None and kmed > float(ceil):
            return False
    max_kkt = thr.get("max_kkt_residual")
    if max_kkt is not None and np.isfinite(row["r_hat"]) and row["r_hat"] > float(max_kkt):
        return False
    max_alpha = thr.get("max_abs_alpha")
    if (
        max_alpha is not None
        and np.isfinite(row["max_abs_alpha"])
        and row["max_abs_alpha"] > float(max_alpha)
    ):
        return False
    max_si = thr.get("max_std_imbalance")
    if max_si is not None and np.isfinite(row["std_imbalance"]) and row["std_imbalance"] > float(
        max_si
    ):
        return False
    return True


def _criterion(row: dict, *, score: str, n: int, tau_R: float, tau_K: float) -> float:
    """Selection criterion of one candidate; NaN when a required metric is missing.

    A candidate whose required metrics could not be computed (e.g. its outcome
    fit failed on every inner fold, so ``b_hat``/``v_hat`` are NaN) must not be
    scored as if those pieces were zero -- zero is the *best possible* value, so
    the un-evaluable candidate would beat every honestly-evaluated one. NaN
    criteria are filtered out of both the admissible pool and the fallback pool.
    """

    if score == "bregman_validation":
        return float(row["bregman_validation"])
    if score == "squared_loss_validation":
        return float(row["squared_loss_validation"])
    if score == "imbalance_validation":
        # Requires the score-variance normalization; falling back to the raw
        # imbalance would compare candidates on two different scales.
        return float(row["std_imbalance"])
    # bias_variance (default): B^2 + V/n + tau_R R + tau_K K. B and V are
    # required; R and K are structural extras (K is genuinely absent for
    # non-kernel bases) and only refine the ranking.
    b = row["b_hat"]
    v = row["v_hat"]
    if not (np.isfinite(b) and np.isfinite(v)):
        return float("nan")
    r = row["r_hat"] if np.isfinite(row["r_hat"]) else 0.0
    k = row["k_hat"] if np.isfinite(row["k_hat"]) else 0.0
    return float(b * b + v / max(n, 1) + tau_R * r + tau_K * k)


def _sigma_candidate_specs(
    spec: object, *, global_median: float
) -> list[tuple[str, float | None]]:
    """Sigma candidates as ``(mode, value)`` pairs.

    ``mode`` is ``"none"`` (keep the basis' own bandwidth), ``"abs"`` (an explicit
    value used verbatim in every inner fold) or ``"mult"`` (a multiplier of the
    median heuristic, from the ``"auto"`` grid). Unlike :func:`normalize_grid`,
    which freezes ``"auto"`` into ``median * multiplier`` values up front, the
    multiplier is kept symbolic so strict nested CV can re-anchor it on each inner
    fold's own inner-training median (audit P0-05).
    """

    if spec is None:
        return [("none", None)]
    if isinstance(spec, str):
        if spec.lower() != "auto":
            raise ValueError("sigma_grid string must be 'auto'")
        if not np.isfinite(global_median) or global_median <= 0:
            raise ValueError("sigma_grid='auto' needs a positive median distance")
        return [("mult", float(mult)) for mult in DEFAULT_SIGMA_MULTIPLIERS]
    if np.isscalar(spec):
        return [("abs", float(spec))]  # type: ignore[arg-type]
    return [("abs", float(s)) for s in spec]  # type: ignore[union-attr]


def select_grr_hyperparams(
    *,
    X_train: ArrayLike,
    y_train: ArrayLike,
    m: LinearFunctional,
    basis: Basis,
    generator: BregmanGenerator,
    config: GRRCVConfig,
    riesz_penalty: str | None = "l2",
    riesz_lam: float = 1e-3,
    riesz_p_norm: float | None = None,
    outcome_link: str = "identity",
    outcome_penalty: str | None = "l2",
    outcome_lam: float = 1e-3,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> GRRCVResult:
    """Select Riesz hyper-parameters on the outer *training* sample only.

    The outer evaluation fold must not be passed here. Centers, standardization
    and the inner split are all derived from ``X_train`` alone.

    With ``config.strict_nested=True`` (the default) the inner CV is a *strict*
    nested CV: within each inner fold the standardization, the ``"auto"`` sigma
    median heuristic and the kernel-center pool are all fit on that fold's
    inner-training rows, so an inner-validation observation never enters the
    feature map that scores it (audit P0-05). The returned ``sigma``/``n_centers``
    describe the selected candidate at the *outer-training* resolution -- the
    values it is refit with on the whole ``X_train`` -- and ``fold_provenance``
    records the per-fold indices for leakage tests. ``config.strict_nested=False``
    keeps the older outer-fixed feature map.

    ``riesz_lam`` is the penalty used to fit each scored candidate and, when
    ``config.lam_grid is None``, the sole lambda candidate -- so the returned
    ``lam`` is then ``riesz_lam`` itself.
    """

    if config.selection_score == "squared_loss_validation" and isinstance(m, AMEFunctional):
        raise ValueError(
            "selection_score='squared_loss_validation' is not defined for "
            "AMEFunctional: its m(alpha) needs a derivative of the representer, "
            "which the LSIF risk does not provide. Use 'bregman_validation' or "
            "'bias_variance' for average-derivative functionals."
        )

    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train, dtype=float).reshape(-1)
    n = X_tr.shape[0]

    # Global median (full outer-training). It is the reported bandwidth anchor for
    # "auto" candidates and, under the outer-fixed feature map, the anchor shared
    # across inner folds.
    global_median = _median_pairwise_distance(
        _standardized(X_tr), random_state=config.random_state
    )

    sigma_specs = _sigma_candidate_specs(config.sigma_grid, global_median=global_median)

    lam_list: list[float]
    if config.lam_grid is None:
        lam_list = [float(riesz_lam)]  # keep the estimator's fixed penalty
    else:
        lam_list = [float(x) for x in normalize_grid(config.lam_grid, kind="lam")]

    if config.n_centers_grid is None:
        n_centers_list: list[int | None] = [None]  # keep the basis' own centers
    else:
        n_centers_list = list(normalize_grid(config.n_centers_grid, kind="n_centers", n=n))

    thr = default_admissibility_thresholds()
    if config.admissibility_thresholds:
        thr.update(config.admissibility_thresholds)

    inner_folds = list(kfold_splits(n, folds=config.cv_folds, random_state=config.random_state))
    n_folds = len(inner_folds)
    # Always probe kernel health when the basis supports it: it powers both the
    # degeneracy penalty (K) and the kernel-health admissibility band.
    want_kernel = True

    max_nc = max((c for c in n_centers_list if c is not None), default=None)

    # Per-fold preprocessing anchors and provenance. Strict nested CV keeps the
    # standardization / median heuristic and the kernel-center pool inside each
    # fold's inner-training rows; the outer-fixed feature map fits them once on
    # the whole outer-training fold and shares them (leaking each fold's own
    # validation rows into the map that scores them).
    if config.strict_nested:
        fold_medians = [
            _median_pairwise_distance(
                _standardized(X_tr[fold.train]), random_state=config.random_state
            )
            for fold in inner_folds
        ]
        fold_center_index = [
            _select_center_indices(
                np.asarray(fold.train, dtype=int),
                n_centers=max_nc,
                random_state=config.random_state,
            )
            if max_nc is not None
            else np.asarray([], dtype=int)
            for fold in inner_folds
        ]
        preprocess_fit_index = [np.asarray(fold.train, dtype=int) for fold in inner_folds]
    else:
        fold_medians = [global_median] * n_folds
        global_center_index = (
            _select_center_indices(np.arange(n), n_centers=max_nc, random_state=config.random_state)
            if max_nc is not None
            else np.asarray([], dtype=int)
        )
        fold_center_index = [global_center_index for _ in inner_folds]
        preprocess_fit_index = [np.arange(n, dtype=int) for _ in inner_folds]

    fold_provenance = [
        {
            "validation_index": np.asarray(fold.test, dtype=int),
            "preprocess_fit_index": preprocess_fit_index[fi],
            "center_index": np.asarray(fold_center_index[fi], dtype=int),
        }
        for fi, fold in enumerate(inner_folds)
    ]

    path: list[dict] = []
    for nc in n_centers_list:
        # Per-fold center rows (identical across folds under the outer-fixed map).
        # Under strict nested CV the pool is capped at the inner-training fold size,
        # so a candidate ``nc`` larger than a fold has fewer centers *while scoring*
        # -- an inherent nested-CV limit -- but the reported ``n_centers`` is ``nc``
        # because the selected candidate is refit on the larger outer-training fold.
        if nc is None:
            centers_arg: object = None
        else:
            centers_arg = [X_tr[fold_center_index[fi][:nc]] for fi in range(n_folds)]
        for mode, val in sigma_specs:
            if mode == "none":
                sigma_arg: object = None
                report_sigma: float | None = None
            elif mode == "abs":
                sigma_arg = float(val)  # type: ignore[arg-type]
                report_sigma = float(val)  # type: ignore[arg-type]
            else:  # "mult": re-anchor per fold, report at the outer-training median
                sigma_arg = [float(val) * fold_medians[fi] for fi in range(n_folds)]  # type: ignore[arg-type]
                report_sigma = float(val) * float(global_median)  # type: ignore[arg-type]
            for lam in lam_list:
                row = score_grr_candidate(
                    X_train=X_tr,
                    y_train=y_tr,
                    m=m,
                    template_basis=basis,
                    generator=generator,
                    sigma=sigma_arg,
                    lam=lam,
                    centers=centers_arg,
                    inner_folds=inner_folds,
                    riesz_penalty=riesz_penalty,
                    riesz_p_norm=riesz_p_norm,
                    outcome_link=outcome_link,
                    outcome_penalty=outcome_penalty,
                    outcome_lam=outcome_lam,
                    max_iter=max_iter,
                    tol=tol,
                    want_kernel=want_kernel,
                    want_squared_loss=config.selection_score == "squared_loss_validation",
                )
                # Report the candidate at its outer-training resolution -- the
                # values it is refit with -- not the per-fold ones.
                row["sigma"] = report_sigma
                row["n_centers"] = None if nc is None else int(nc)
                row["admissible"] = _is_admissible(row, thr)
                row["criterion"] = _criterion(
                    row, score=config.selection_score, n=n, tau_R=config.tau_R, tau_K=config.tau_K
                )
                path.append(row)

    modifies_estimand = bool(getattr(generator, "modifies_estimand", False))

    admissible = [r for r in path if r["admissible"] and np.isfinite(r["criterion"])]
    pool = admissible
    if not pool:
        # Fall back before warning: if nothing fitted at all, the failure is the
        # story and a warning about "the selection below" would describe a
        # selection that never happens.
        pool = [r for r in path if r["success"] and np.isfinite(r["criterion"])]
        if not pool:
            raise RuntimeError(
                "No Riesz candidate could be fitted and scored on this training "
                "fold (a candidate whose required selection metrics are missing "
                "-- e.g. its outcome fit failed on every inner fold -- has no "
                "criterion). Check the basis, generator, grids, and the outcome "
                "model, or inspect the CV path (return_riesz_cv_path=True)."
            )
        if modifies_estimand:
            warnings.warn(
                f"Generator {getattr(generator, 'name', type(generator).__name__)} "
                f"modifies the estimand, so none of its candidates enter the "
                f"admissible set (design section 9-4). The hyper-parameters "
                f"selected below tune a target-sensitivity analysis over the "
                f"modified (bounded) estimand -- they are not a selection over "
                f"the original estimand. Report the bound-binding rate alongside "
                f"the estimate.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "No Riesz candidate passed the admissibility screen on this training "
                "fold; falling back to the best-criterion candidate among all fitted "
                "candidates. Inspect the CV path table (return_riesz_cv_path=True).",
                UserWarning,
                stacklevel=2,
            )

    best = min(pool, key=lambda r: r["criterion"])

    return GRRCVResult(
        sigma=best["sigma"],
        lam=best["lam"],
        n_centers=best["n_centers"],
        selection_score=config.selection_score,
        best_score=float(best["criterion"]),
        n_admissible=len(admissible),
        n_candidates=len(path),
        path=path if config.return_path else [],
        modifies_estimand=modifies_estimand,
        strict_nested=bool(config.strict_nested),
        fold_provenance=fold_provenance,
    )
