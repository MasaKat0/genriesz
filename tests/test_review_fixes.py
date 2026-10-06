"""Regression tests for the code review of the strict solvers (blocking items 1-5).

1. Ridge (incl. lam = 0) with an active l1 ball: projected FISTA with the ridge in
   the smooth part, against an independently solved quadratic.
3. Feasibility over every evaluation point (density-ratio numerator rows).
4. Prediction validation (density ratio: underflow to the boundary).
5. Inner CV: solver contract forwarded, explicit "cv_failed" status.
Plus: registered proximal residual, offset_from_alpha derivative, RW_full SE.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from genriesz import (
    GRRGLM,
    ATEFunctional,
    BKLGenerator,
    BPGenerator,
    CallableBasis,
    OutcomeGLM,
    PolynomialBasis,
    SquaredGenerator,
    TreatmentInteractionBasis,
    UKLGenerator,
    fit_density_ratio,
    grr_ate,
    offset_from_alpha,
    rw_full_inference,
)
from genriesz.functionals import LinearFunctional
from genriesz.solvers import STATUSES


def _treated(x):
    return int(x[0] == 1.0)


def _pos(_x):
    return 1


class _FixedTarget(LinearFunctional):
    """mean m(W, phi) = b, evaluating its argument nowhere."""

    def __init__(self, b):
        super().__init__(name="fixed")
        object.__setattr__(self, "b", np.asarray(b, dtype=float))

    def m_basis_matrix(self, X, basis):
        return np.tile(self.b, (len(np.atleast_2d(X)), 1))

    def m_from_predictor(self, X, predict):  # pragma: no cover - unused
        raise NotImplementedError


# Phi' Phi / n = 2 I, so with SQ (g* = v^2/4) the objective is
#   (1 + lam)/2 ||beta||^2 - b' beta  over  ||beta||_1 <= R,
# whose minimizer is the l1-ball projection of b / (1 + lam).
_ISO = CallableBasis(lambda X: np.sqrt(6.0) * np.eye(3)[np.atleast_2d(X)[:, 0].astype(int)])
_X3 = np.arange(3, dtype=float).reshape(-1, 1)
_B = np.array([3.0, 1.0, -0.2])


@pytest.mark.parametrize(
    "penalty, lam, radius, expected, mu",
    [
        # z = b = (3, 1, -0.2), R = 3: tau = 1/2 -> (2.5, 0.5, 0), mu = 1/2.
        (None, 0.0, 3.0, [2.5, 0.5, 0.0], 0.5),
        # z = b / 1.5 = (2, 2/3, -2/15), R = 2: tau = 1/3 -> (5/3, 1/3, 0); the
        # ridge-adjusted gradient (1.5 beta - b) is (-1/2, -1/2, 1/5), so mu = 1/2.
        ("l2", 0.5, 2.0, [5.0 / 3.0, 1.0 / 3.0, 0.0], 0.5),
    ],
    ids=["lam0-ball", "ridge-ball"],
)
def test_ridge_and_active_l1_ball_match_the_independent_solution(
    penalty, lam, radius, expected, mu
):
    model = GRRGLM(
        basis=_ISO, generator=SquaredGenerator(), functional=_FixedTarget(_B),
        penalty=penalty, lam=lam, l1_radius=radius,
    )
    r = model.fit(_X3)
    assert r.status == "ok", r.message
    assert r.solver == "fista"  # the unrestricted Newton solution lies outside the ball
    assert np.max(np.abs(r.beta - np.array(expected))) < 1e-10
    assert r.l1_ball_active
    assert abs(r.ball_multiplier - mu) < 1e-10
    assert r.kkt_residual < 1e-10 and r.prox_residual < 1e-10
    # The reported imbalance excludes the ridge term: Delta = beta - b here.
    assert abs(r.max_abs_imbalance - float(np.max(np.abs(np.array(expected) - _B)))) < 1e-10


def test_ridge_with_an_inactive_ball_keeps_the_newton_solution():
    model = GRRGLM(
        basis=_ISO, generator=SquaredGenerator(), functional=_FixedTarget(_B),
        penalty="l2", lam=0.5, l1_radius=10.0,
    )
    r = model.fit(_X3)
    assert r.status == "ok" and r.solver == "newton"
    assert np.max(np.abs(r.beta - _B / 1.5)) < 1e-12


def test_l1_prox_residual_is_reported_and_used():
    X3 = _X3
    model = GRRGLM(
        basis=_ISO, generator=SquaredGenerator(), functional=_FixedTarget(_B),
        penalty="l1", lam=0.3,
    )
    r = model.fit(X3)
    # Soft-thresholding of b at 0.3 (scaled by the curvature 1): (2.7, 0.7, 0).
    assert np.max(np.abs(r.beta - np.array([2.7, 0.7, 0.0]))) < 1e-10
    assert np.isfinite(r.prox_residual) and r.prox_residual <= 1e-6 * 0.3


def test_status_names_follow_the_registration():
    assert "uncertified_numerical_boundary" in STATUSES
    assert "boundary" not in STATUSES
    assert "cv_failed" in STATUSES


# ---------------------------------------------------------------------------
# 3./4. Density ratio: numerator rows are evaluation points; validated output
# ---------------------------------------------------------------------------
def _linear():
    return CallableBasis(lambda X: np.column_stack([np.ones(len(np.atleast_2d(X))),
                                                    np.atleast_2d(X)]))


def test_density_ratio_numerator_rows_stay_in_the_domain():
    """Review: a BP fit returned "ok" with an invalid numerator prediction."""

    rng = np.random.default_rng(0)
    X_den = rng.uniform(0.0, 1.0, size=(300, 1))
    cand = rng.uniform(0.0, 1.0, size=3000)
    X_num = cand[rng.uniform(0.0, 1.5, size=3000) < 1.5 - cand][:300].reshape(-1, 1)
    gen = BPGenerator(C=0.0, omega=1.0, branch_fn=_pos)
    clean = fit_density_ratio(X_num, X_den, basis=_linear(), generator=gen, penalty=None)
    assert clean.status == "ok" and np.all(clean.domain_mask(X_num))
    # One numerator row at x = 1.5, beyond the point where the decreasing linear
    # ratio leaves the BP range (v < -2): the minimizer over the denominator rows
    # would put it outside, so the fit is blocked and says so.
    X_num2 = np.vstack([X_num, [[1.5]]])
    res = fit_density_ratio(X_num2, X_den, basis=_linear(), generator=gen, penalty=None)
    assert res.status == "domain_prediction"
    assert "evaluation point" in res.fit.message


def test_density_ratio_underflow_is_outside_not_valid():
    rng = np.random.default_rng(1)
    X_den = rng.normal(0.0, 1.0, size=(200, 1))
    X_num = rng.normal(0.5, 1.0, size=(200, 1))
    res = fit_density_ratio(
        X_num, X_den, basis=_linear(), generator=UKLGenerator(C=0.5, branch_fn=_pos),
        penalty=None,
    )
    assert res.status == "ok"
    slope = float(res.beta[1])
    far = np.array([[-800.0 / slope]]) if slope > 0 else np.array([[800.0 / -slope]])
    r, outside, nonfinite = res.classify(far)
    assert outside[0] and not nonfinite[0] and np.isnan(r[0])
    assert not res.domain_mask(far)[0]


# ---------------------------------------------------------------------------
# 5. Inner CV
# ---------------------------------------------------------------------------
def _ate(n=300, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-1.0, 1.0, size=(n, 1))
    D = rng.binomial(1, expit(0.8 * Z[:, 0])).astype(float)
    Y = D + Z[:, 0] + rng.normal(size=n)
    return np.column_stack([D, Z]), Y


def _basis():
    return TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    )


def test_inner_cv_without_a_feasible_candidate_returns_cv_failed():
    """BKL without an offset cannot start anywhere: every candidate fails."""

    X, Y = _ate()
    res = grr_ate(
        X=X, Y=Y, basis=_basis(), generator=BKLGenerator(C=1.0, branch_fn=_treated),
        riesz_lam_grid=[1e-3, 1e-2], outcome_link="identity", folds=2, random_state=0,
        estimators=("arw",),
    )
    assert res.status == "cv_failed"
    assert np.isnan(res.arw.estimate)
    assert res.fold_status[0][1:3] == ("riesz_cv", "cv_failed")


def test_inner_cv_forwards_the_offset_and_alpha_ref():
    X, Y = _ate(seed=1)
    gen = BKLGenerator(C=1.0, branch_fn=_treated)
    alpha_ref = lambda X_: np.where(X_[:, 0] == 1.0, 2.0, -2.0)  # noqa: E731
    kw = dict(X=X, Y=Y, basis=_basis(), generator=gen, riesz_lam_grid=[1e-3, 1e-2],
              outcome_link="identity", folds=2, random_state=0, estimators=("arw",))
    a = grr_ate(riesz_alpha_ref=alpha_ref, **kw)
    b = grr_ate(riesz_offset=offset_from_alpha(gen, alpha_ref), **kw)
    assert a.status == b.status == "ok"
    assert a.arw.estimate == b.arw.estimate
    with pytest.raises(ValueError, match="not both"):
        grr_ate(riesz_alpha_ref=alpha_ref, riesz_offset=0.0, **kw)


def test_inner_cv_generator_grid_needs_alpha_ref_not_a_dual_offset():
    X, Y = _ate(seed=2)
    grid = [UKLGenerator(C=1.0, branch_fn=_treated), BKLGenerator(C=1.0, branch_fn=_treated)]
    alpha_ref = lambda X_: np.where(X_[:, 0] == 1.0, 2.0, -2.0)  # noqa: E731
    kw = dict(X=X, Y=Y, basis=_basis(), generator=grid[0], riesz_generator_grid=grid,
              riesz_selection_score="squared_loss_validation", outcome_link="identity",
              folds=2, random_state=0, estimators=("rw",))
    res = grr_ate(riesz_alpha_ref=alpha_ref, **kw)
    assert res.status == "ok"
    with pytest.raises(ValueError, match="riesz_alpha_ref"):
        grr_ate(riesz_offset=0.0, **kw)


# ---------------------------------------------------------------------------
# offset_from_alpha derivative; RW_full
# ---------------------------------------------------------------------------
def test_offset_from_alpha_attaches_the_chain_rule_derivative():
    gen = UKLGenerator(C=1.0, branch_fn=_pos)

    def alpha_ref(X):
        return 2.0 + X[:, 0] ** 2

    alpha_ref.derivative = lambda X, c: 2.0 * X[:, 0] if c == 0 else np.zeros(len(X))
    off = offset_from_alpha(gen, alpha_ref)
    X = np.array([[0.3], [0.7]])
    h = 1e-6
    fd = (off(X + h) - off(X - h)) / (2 * h)
    assert np.allclose(off.derivative(X, 0), fd, rtol=1e-6)
    assert not hasattr(offset_from_alpha(gen, lambda X_: 2.0 + 0 * X_[:, 0]), "derivative")


def test_rw_full_uses_the_ols_influence_function():
    X, Y = _ate(n=400, seed=3)
    basis = _basis().fit(X)
    grr = GRRGLM(basis=basis, generator=SquaredGenerator(), functional=ATEFunctional(0),
                 penalty=None)
    assert grr.fit(X).status == "ok"
    est = rw_full_inference(grr=grr, X=X, Y=Y)
    alpha = grr.predict_alpha(X)
    ols = OutcomeGLM(basis=basis, link="identity", penalty="l2", lam=0.0)
    ols.fit(X, Y)
    theta = float(np.mean(alpha * Y))
    psi = (ATEFunctional(0).m_from_predictor(X, ols.predict) + alpha * (Y - ols.predict(X))
           - theta)
    assert est.estimate == pytest.approx(theta, rel=1e-12, abs=0)
    assert est.se == pytest.approx(float(np.std(psi, ddof=1) / np.sqrt(len(Y))), rel=1e-10,
                                   abs=0)
    naive = float(np.std(alpha * Y - theta, ddof=1) / np.sqrt(len(Y)))
    assert abs(est.se - naive) > 1e-4


# ---------------------------------------------------------------------------
# Re-review: nonfinite before domain; CV checks counterfactual rows for every score
# ---------------------------------------------------------------------------
def test_nan_dual_coordinate_is_nonfinite_but_underflow_stays_a_domain_failure():
    from genriesz.glm import classify_predictions

    gen = UKLGenerator(C=1.0, branch_fn=_pos)
    X = np.ones((4, 1))
    alpha, outside, nonfinite = classify_predictions(
        gen, X, np.array([np.nan, np.inf, -800.0, 0.0])
    )
    assert list(nonfinite) == [True, True, False, False]
    assert list(outside) == [False, False, True, False]
    assert alpha[3] == 2.0 and np.all(np.isnan(alpha[:3]))


def test_held_out_nan_offset_propagates_as_nonfinite_not_domain_prediction():
    X, Y = _ate(n=300, seed=4)
    ids = np.random.default_rng(0).permutation(np.arange(len(X)) % 2)
    # One extra evaluation-fold row at which the (fixed) offset is NaN.
    X = np.vstack([X, [[1.0, 50.0]]])
    Y = np.concatenate([Y, [0.0]])
    ids = np.concatenate([ids, [0]])

    def offset(X_):
        return np.where(X_[:, 1] > 10.0, np.nan, 0.0)

    res = grr_ate(
        X=X, Y=Y, basis=_basis(), generator=SquaredGenerator(), riesz_offset=offset,
        outcome_link="identity", fold_ids=ids, estimators=("arw",),
    )
    assert res.status == "nonfinite"
    assert np.isnan(res.arw.estimate)
    assert [s for s in res.fold_status if s[2] == "nonfinite"][0][1] == "prediction"


def _cf_trap():
    """Training rows, plus a validation row whose own arm is in the BP domain but
    whose counterfactual arm is not (found by scanning the fitted model)."""

    from genriesz.model_selection import make_candidate_basis

    X, Y = _ate(n=300, seed=5)
    gen = BPGenerator(C=0.0, omega=1.0, branch_fn=_treated)
    cb = make_candidate_basis(_basis(), sigma=None, centers=None).fit(X)
    model = GRRGLM(basis=cb, generator=gen, functional=ATEFunctional(0), lam=1e-2)
    assert model.fit(X).status == "ok"
    zs = np.linspace(-1e3, 1e3, 200001)
    ok1 = model.domain_mask(np.column_stack([np.ones_like(zs), zs]))
    ok0 = model.domain_mask(np.column_stack([np.zeros_like(zs), zs]))
    z = zs[np.flatnonzero(ok1 & ~ok0)[0]]
    return X, Y, gen, np.array([[1.0, z]])


@pytest.mark.parametrize("want_squared_loss", [False, True], ids=["bregman-or-bv", "lsif"])
def test_cv_rejects_folds_with_invalid_counterfactual_validation_rows(want_squared_loss):
    from genriesz.model_selection import score_grr_candidate
    from genriesz.utils import Fold

    X, Y, gen, trap = _cf_trap()
    Xa = np.vstack([X, trap])
    Ya = np.concatenate([Y, [0.0]])
    n = len(X)
    fold = Fold(train=np.arange(n), test=np.arange(n - 50, n + 1))
    kw = dict(
        X_train=Xa, y_train=Ya, m=ATEFunctional(0), template_basis=_basis(), generator=gen,
        sigma=None, lam=1e-2, centers=None, riesz_penalty="l2", riesz_p_norm=None,
        outcome_link="identity", outcome_penalty="l2", outcome_lam=1e-3, max_iter=500,
        tol=1e-8, want_kernel=False, want_squared_loss=want_squared_loss,
    )
    bad = score_grr_candidate(inner_folds=[fold], **kw)
    assert bad["success"] is False
    good = score_grr_candidate(
        inner_folds=[Fold(train=np.arange(n), test=np.arange(n - 50, n))], **kw
    )
    assert good["success"] is True
