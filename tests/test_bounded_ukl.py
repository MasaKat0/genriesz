"""Contract tests for the truncated (bounded) UKL representer model.

The behavioral contract is documented in ``docs/user_guide.rst`` (truncated
representer models). The bounded link saturates at stated representer bounds instead of clipping a
fitted exact model; where a bound binds the estimator targets a modified
estimand, so the class is a target-sensitivity specification and never an
admissible candidate.
"""

from __future__ import annotations

import numpy as np
import pytest

from genriesz.basis import GaussianRKHSBasis, PolynomialBasis, TreatmentInteractionBasis
from genriesz.functionals import ATEFunctional
from genriesz.generators import (
    BoundedBKLGenerator,
    BoundedUKLGenerator,
    DomainError,
    UKLGenerator,
)
from genriesz.glm import GRRGLM
from genriesz.model_selection import GRRCVConfig, select_grr_hyperparams


def _pos_branch(_x):
    return 1


def _treated_branch(x):
    return int(x[0] == 1.0)


def _make_ate(n: int, seed: int):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    e = 1.0 / (1.0 + np.exp(-(0.8 * Z[:, 0] - 0.4 * Z[:, 1])))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = 1.0 + D + Z[:, 0] + 0.5 * Z[:, 1] + rng.normal(scale=0.5, size=n)
    return np.column_stack([D, Z]), Y


# ---------------------------------------------------------------------------
# Entry validation. match= pins the *start* of the entry message so the
# test cannot be satisfied by a later solver error.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_bounded_ukl_rejects_nonfinite_alpha_max(bad):
    with pytest.raises(ValueError, match=r"^alpha_max must be finite"):
        BoundedUKLGenerator(C=1.0, alpha_max=bad, branch_fn=_pos_branch)


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_bounded_bkl_rejects_nonfinite_alpha_max(bad):
    # Audit v3 K-09: a nonfinite bound makes the pre-image interval NaN and
    # the link NaN everywhere, so the constructor must reject it.
    with pytest.raises(ValueError, match=r"^alpha_max must be finite"):
        BoundedBKLGenerator(C=1.0, alpha_max=bad, branch_fn=_pos_branch)


def test_bounded_ukl_requires_ordered_bounds():
    with pytest.raises(ValueError, match=r"^alpha_max must be > "):
        BoundedUKLGenerator(C=1.0, alpha_max=0.5, branch_fn=_pos_branch)
    with pytest.raises(ValueError, match=r"^alpha_min must be > C"):
        BoundedUKLGenerator(C=1.0, alpha_max=20.0, alpha_min=0.9, branch_fn=_pos_branch)
    with pytest.raises(ValueError, match=r"^alpha_max must be > "):
        BoundedUKLGenerator(C=1.0, alpha_max=2.0, alpha_min=2.0, branch_fn=_pos_branch)
    with pytest.raises(ValueError, match=r"^alpha_min must be finite"):
        BoundedUKLGenerator(C=1.0, alpha_max=20.0, alpha_min=np.nan, branch_fn=_pos_branch)


def test_bounded_ukl_requires_branch_fn():
    with pytest.raises(ValueError, match=r"^BoundedUKLGenerator requires branch_fn"):
        BoundedUKLGenerator(C=1.0, alpha_max=20.0)


# ---------------------------------------------------------------------------
# Audit v3 K-20: the truncation is defined in alpha space. The pinned values
# equal the stated bounds exactly -- no dual-space floor that widens in alpha
# space the way the BP omega>1 floor does.
# ---------------------------------------------------------------------------
def test_bounded_ukl_alpha_is_bounded_everywhere():
    amin, amax = 1.5, 20.0
    gen = BoundedUKLGenerator(C=1.0, alpha_max=amax, alpha_min=amin, branch_fn=_pos_branch)
    X = np.ones((9, 2))
    v = np.array([-1e6, -800.0, -5.0, -0.5, 0.0, 1.0, 3.0, 800.0, 1e6])
    alpha = gen.inv_grad(X, v)
    assert np.all(np.isfinite(alpha))
    assert np.all(alpha >= amin - 0.0)
    assert np.all(alpha <= amax + 0.0)
    # Pinned exactly at the stated bounds in the saturated regions.
    assert alpha[0] == amin and alpha[1] == amin
    assert alpha[-1] == amax and alpha[-2] == amax


def test_bounded_ukl_matches_exact_ukl_in_interior():
    exact = UKLGenerator(C=1.0, branch_fn=_pos_branch)
    bounded = BoundedUKLGenerator(C=1.0, alpha_max=1e6, branch_fn=_pos_branch)
    X = np.ones((6, 2))
    v = np.array([-4.0, -1.0, 0.0, 1.0, 2.0, 5.0])  # strictly interior
    assert not np.any(bounded.domain_binding(X, v))
    assert np.allclose(exact.inv_grad(X, v), bounded.inv_grad(X, v), rtol=1e-12, atol=0.0)


def test_bounded_ukl_is_defined_where_exact_ukl_raises():
    # The truncated model covers the extreme dual values: where the exact
    # link overflows or underflows float64, the bounded link pins the
    # representer at the model bound instead of failing -- and instead of any
    # post-hoc clipping of a fitted exact model.
    exact = UKLGenerator(C=1.0, branch_fn=_pos_branch)
    bounded = BoundedUKLGenerator(C=1.0, alpha_max=100.0, alpha_min=1.5, branch_fn=_pos_branch)
    X = np.ones((1, 2))
    for v_extreme, pinned in [(np.array([800.0]), 100.0), (np.array([-800.0]), 1.5)]:
        with pytest.raises(DomainError):
            exact.inv_grad(X, v_extreme)
        assert bounded.inv_grad(X, v_extreme)[0] == pinned


def test_bounded_ukl_dual_domain_mask_accepts_all_finite_v():
    gen = BoundedUKLGenerator(C=1.0, alpha_max=20.0, branch_fn=_pos_branch)
    X = np.ones((4, 2))
    v = np.array([-1e6, 0.0, 1e6, np.nan])
    mask = gen.dual_domain_mask(X, v)
    assert mask.tolist() == [True, True, True, False]


# ---------------------------------------------------------------------------
# Audit GEN-07: binding diagnostics distinguish where the model departs from exact UKL.
# ---------------------------------------------------------------------------
def test_bounded_ukl_reports_binding_exactly():
    amin, amax, C = 1.5, 20.0, 1.0
    gen = BoundedUKLGenerator(C=C, alpha_max=amax, alpha_min=amin, branch_fn=_pos_branch)
    X = np.ones((5, 2))
    z_hi = np.log(amax - C)  # pre-image of alpha_max
    z_lo = np.log(amin - C)  # pre-image of alpha_min
    v = np.array([z_lo - 1.0, z_lo + 0.1, 0.5 * (z_lo + z_hi), z_hi - 0.1, z_hi + 1.0])
    binding = gen.domain_binding(X, v)
    assert binding.tolist() == [True, False, False, False, True]


# ---------------------------------------------------------------------------
# Envelope identity d g*/dv = alpha, including the binding region. Pinned
# alpha is constant in v there, so the identity must hold to finite-difference
# precision on both sides of each clamp.
# ---------------------------------------------------------------------------
def test_bounded_ukl_conjugate_identity_including_binding_region():
    gen = BoundedUKLGenerator(C=1.0, alpha_max=20.0, alpha_min=1.5, branch_fn=_pos_branch)
    X = np.ones((13, 2))
    v = np.linspace(-2.0, 4.0, 13)  # spans lower binding through upper binding
    _, alpha = gen.conjugate(X, v)
    h = 1e-6
    gp, _ = gen.conjugate(X, v + h)
    gm, _ = gen.conjugate(X, v - h)
    fd = (gp - gm) / (2.0 * h)
    rel = np.max(np.abs(fd - alpha) / np.maximum(1.0, np.abs(alpha)))
    assert rel < 1e-5
    assert np.any(gen.domain_binding(X, v))


# ---------------------------------------------------------------------------
# Audit GLM-11: the link is flat where the bound binds, so the alpha
# derivative used by AME-type functionals is exactly zero there. g'' > 0 keeps
# NaN guards silent, so only a direct analytic-vs-FD comparison pins this.
# ---------------------------------------------------------------------------
def test_bounded_ukl_link_is_flat_where_the_bound_binds():
    gen = BoundedUKLGenerator(C=1.0, alpha_max=20.0, alpha_min=1.5, branch_fn=_pos_branch)
    X = np.ones((2, 2))
    z_hi = np.log(20.0 - 1.0)
    z_lo = np.log(1.5 - 1.0)
    v = np.array([z_hi + 0.5, z_lo - 0.5])  # safely inside both binding regions
    h = 1e-4
    fd = (gen.inv_grad(X, v + h) - gen.inv_grad(X, v - h)) / (2.0 * h)
    assert np.all(fd == 0.0)


def test_bounded_bkl_link_is_flat_where_the_bound_binds():
    gen = BoundedBKLGenerator(C=1.0, alpha_max=20.0, branch_fn=_pos_branch)
    X = np.ones((1, 2))
    v = np.array([-1e-3])  # u -> 0 side: the bound binds and alpha == alpha_max
    assert gen.domain_binding(X, v)[0]
    h = 1e-5
    fd = (gen.inv_grad(X, v + h) - gen.inv_grad(X, v - h)) / (2.0 * h)
    assert np.all(fd == 0.0)


# ---------------------------------------------------------------------------
# Cap consistency through the GRR objective, and optimizability.
# Mirrors tests/test_kl_cap_domain.py for BoundedBKL.
# ---------------------------------------------------------------------------
def _fit_ingredients(n: int, seed: int, gen):
    X, _ = _make_ate(n=n, seed=seed)
    m = ATEFunctional(treatment_index=0)
    basis = TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    ).fit(X)
    model = GRRGLM(functional=m, basis=basis, generator=gen, penalty="l2", lam=1e-3)
    return X, m, basis, model


def test_bounded_ukl_grrglm_gradient_matches_finite_difference():
    # alpha_max is tight so the evaluation point genuinely binds (asserted
    # below); the identity must hold through the binding region.
    gen = BoundedUKLGenerator(C=1e-2, alpha_max=1.2, branch_fn=_treated_branch)
    X, m, basis, model = _fit_ingredients(120, 2, gen)
    Phi = np.asarray(basis(X), dtype=float)
    M = np.asarray(m.m_basis_matrix(X, basis), dtype=float)
    p = Phi.shape[1]

    def fun(beta):
        g_star, _ = gen.conjugate(X, Phi @ beta)
        return float(np.mean(g_star - (M @ beta))) + model.penalty.value(beta)

    def jac(beta):
        _, alpha = gen.conjugate(X, Phi @ beta)
        return (alpha[:, None] * Phi - M).mean(axis=0) + model.penalty.grad(beta)

    rng = np.random.default_rng(3)
    beta = 0.1 * rng.normal(size=p)
    assert np.any(gen.domain_binding(X, Phi @ beta))
    analytic = jac(beta)
    h = 1e-6
    fd = np.empty(p)
    for j in range(p):
        e = np.zeros(p)
        e[j] = h
        fd[j] = (fun(beta + e) - fun(beta - e)) / (2.0 * h)
    rel = np.max(np.abs(fd - analytic) / np.maximum(1.0, np.abs(analytic)))
    assert rel < 1e-5


def test_bounded_ukl_is_optimizable_and_bounded_from_cold_start():
    amax = 40.0
    gen = BoundedUKLGenerator(C=1e-2, alpha_max=amax, branch_fn=_treated_branch)
    X, _, _, model = _fit_ingredients(200, 4, gen)
    res = model.fit(X)
    assert res.success
    assert res.status == "converged"
    alpha = model.predict_alpha(X)
    assert np.max(np.abs(alpha)) <= amax + 1e-9


# ---------------------------------------------------------------------------
# The truncation is part of the model. For the ATE arms of UKL,
# alpha = 1/e (treated) and 1/(1-e) (control), so a symmetric propensity
# window [e_min, e_max] is the magnitude interval [1/e_max, 1/e_min].
# ---------------------------------------------------------------------------
def test_propensity_bounds_construct_the_stated_alpha_interval():
    gen = BoundedUKLGenerator.from_propensity_bounds(0.01, 0.99, branch_fn=_pos_branch)
    assert gen.alpha_min == pytest.approx(1.0 / 0.99)
    assert gen.alpha_max == pytest.approx(100.0)
    assert gen.modifies_estimand is True


@pytest.mark.parametrize(
    "e_min, e_max, prefix",
    [
        (0.0, 0.99, r"^e_min must be"),
        (0.01, 1.0, r"^e_max must be"),
        (0.5, 0.5, r"^e_min must be"),
        (0.6, 0.4, r"^e_min must be"),
    ],
)
def test_propensity_bounds_are_validated_at_entry(e_min, e_max, prefix):
    with pytest.raises(ValueError, match=prefix):
        BoundedUKLGenerator.from_propensity_bounds(e_min, e_max, branch_fn=_pos_branch)


def test_implied_propensity_stays_within_the_stated_interval():
    gen = BoundedUKLGenerator.from_propensity_bounds(0.01, 0.99, branch_fn=_pos_branch)
    X = np.ones((41, 2))
    v = np.linspace(-30.0, 30.0, 41)
    alpha = gen.inv_grad(X, v)
    implied_e = 1.0 / np.abs(alpha)
    assert np.all(implied_e >= 0.01 - 1e-12)
    assert np.all(implied_e <= 0.99 + 1e-12)


def test_exact_ukl_still_refuses_out_of_range_values():
    # Introducing the bounded model must not soften the exact model.
    exact = UKLGenerator(C=1.0, branch_fn=_pos_branch)
    with pytest.raises(DomainError):
        exact.inv_grad(np.ones((1, 2)), np.array([800.0]))


# ---------------------------------------------------------------------------
# Section 9-4 seam in model selection, mirrored from
# tests/test_generators_compat.py. The flag, not the binding rate, excludes
# the candidate; and the flag must not mask quality failures.
# ---------------------------------------------------------------------------
def _select(gen, *, n: int = 250, seed: int = 10, admissibility_thresholds=None):
    X, Y = _make_ate(n=n, seed=seed)
    return select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(treatment_index=0),
        basis=GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0),
        generator=gen,
        config=GRRCVConfig(
            lam_grid=[1e-2, 1e-1],
            cv_folds=2,
            return_path=True,
            random_state=0,
            admissibility_thresholds=admissibility_thresholds,
        ),
        outcome_link="identity",
    )


def test_bounded_ukl_sets_the_estimand_flag():
    gen = BoundedUKLGenerator(C=1e-2, alpha_max=30.0, branch_fn=_treated_branch)
    assert gen.modifies_estimand is True
    assert UKLGenerator(C=1e-2, branch_fn=_treated_branch).modifies_estimand is False


def test_bounded_ukl_is_excluded_from_the_admissible_set():
    gen = BoundedUKLGenerator(C=1e-2, alpha_max=30.0, branch_fn=_treated_branch)
    with pytest.raises(RuntimeError, match="modifying the estimand"):
        _select(gen)


def test_ukl_exclusion_holds_even_when_the_bound_never_binds():
    # A loose bound never binds, yet the candidate is still target-sensitivity
    # (flag screen, not binding-rate screen). The quality floor is disabled
    # here so the flag alone is pinned; the no-bypass rule is pinned below.
    gen = BoundedUKLGenerator(C=1e-2, alpha_max=1e6, branch_fn=_treated_branch)
    with pytest.raises(RuntimeError, match="modifying the estimand"):
        _select(gen, admissibility_thresholds={"min_ess_ratio": None})


def test_estimand_flag_does_not_bypass_the_quality_checks_for_ukl():
    # Audit CV-11: with an effectively-unclipped bound the fit concentrates
    # weight (max alpha large, small ESS ratio). With the ESS floor active the
    # candidate must not surface through the sensitivity path as if the flag
    # were its only violation. The screen's single source of truth is pinned
    # directly by test_quality_checks_are_not_removed_from_the_screen.
    gen = BoundedUKLGenerator(C=1e-2, alpha_max=1e6, branch_fn=_treated_branch)
    with pytest.raises(RuntimeError):
        _select(gen, admissibility_thresholds={"min_ess_ratio": 0.5})


def test_exact_ukl_selection_never_carries_the_estimand_flag():
    # Selecting among exact UKL candidates must never return a result
    # whose path or flag claims a modified estimand (no silent substitution).
    res = _select(UKLGenerator(C=1e-2, branch_fn=_treated_branch))
    assert res.modifies_estimand is False
    assert all(not r["modifies_estimand"] for r in res.path)


# ---------------------------------------------------------------------------
# GLM-11 (audit v3): the AME-path derivative must be exactly zero where the
# bounded link pins alpha. The 1/g'' path returns a finite wrong value there
# and g'' > 0 keeps every NaN guard silent, so only a direct comparison of the
# AME-path derivative pins this contract.
# ---------------------------------------------------------------------------
def _flat_derivative_case(gen, X):
    basis = PolynomialBasis(degree=1, include_bias=True).fit(X)
    model = GRRGLM(
        functional=ATEFunctional(treatment_index=0),
        basis=basis,
        generator=gen,
        penalty="l2",
        lam=0.0,
    )
    model.beta_ = np.array([0.0, 1.0])  # v(x) = x, so dv/dx = 1 everywhere
    v = np.asarray(basis(X), dtype=float) @ model.beta_
    binding = gen.domain_binding(X, v)
    out = model.derivative_alpha(X, 0)
    return binding, out


def test_bounded_ukl_derivative_alpha_is_zero_where_the_bound_binds():
    gen = BoundedUKLGenerator(
        C=1.0, alpha_max=5.0, alpha_min=1.5, branch_fn=_pos_branch
    )
    X = np.array([[-3.0], [0.5], [3.0]])
    binding, out = _flat_derivative_case(gen, X)
    assert binding.tolist() == [True, False, True]
    assert np.all(out[binding] == 0.0)
    assert np.all(np.isfinite(out[~binding]))
    assert np.all(out[~binding] != 0.0)


def test_bounded_bkl_derivative_alpha_is_zero_where_the_bound_binds():
    gen = BoundedBKLGenerator(C=1.0, alpha_max=5.0, branch_fn=_pos_branch)
    X = np.array([[-3.0], [-1.5], [0.5]])
    binding, out = _flat_derivative_case(gen, X)
    assert binding.tolist() == [False, False, True]
    assert np.all(out[binding] == 0.0)
    assert np.all(np.isfinite(out[~binding]))
    assert np.all(out[~binding] != 0.0)


# ---------------------------------------------------------------------------
# Propensity-window mapping, entry finiteness, per-side binding diagnostics,
# and the quality-check screen itself.
# ---------------------------------------------------------------------------
def _neg_branch(_x):
    return 0


def test_propensity_bounds_reject_an_asymmetric_window():
    # The treated arm has |alpha| = 1/e and the control arm 1/(1-e): one
    # magnitude interval covers both arms only when e_max = 1 - e_min. An
    # asymmetric window must state alpha_min and alpha_max directly.
    with pytest.raises(ValueError, match=r"^e_max must equal 1 - e_min"):
        BoundedUKLGenerator.from_propensity_bounds(0.1, 0.8, branch_fn=_pos_branch)


def test_implied_propensity_holds_on_the_control_branch():
    gen = BoundedUKLGenerator.from_propensity_bounds(0.01, 0.99, branch_fn=_neg_branch)
    X = np.ones((41, 2))
    v = np.linspace(-30.0, 30.0, 41)
    alpha = gen.inv_grad(X, v)
    assert np.all(alpha < 0.0)
    implied_e = 1.0 - 1.0 / np.abs(alpha)
    assert np.all(implied_e >= 0.01 - 1e-12)
    assert np.all(implied_e <= 0.99 + 1e-12)


@pytest.mark.parametrize(
    "make",
    [
        lambda: BoundedUKLGenerator(C=np.nan, alpha_max=20.0, branch_fn=_pos_branch),
        lambda: BoundedBKLGenerator(C=np.nan, alpha_max=20.0, branch_fn=_pos_branch),
    ],
    ids=["bounded_ukl", "bounded_bkl"],
)
def test_bounded_generators_reject_a_nonfinite_shift(make):
    with pytest.raises(ValueError, match=r"^C must be finite"):
        make()


def test_binding_diagnostics_report_each_side():
    amin, amax, C = 1.5, 20.0, 1.0
    gen = BoundedUKLGenerator(C=C, alpha_max=amax, alpha_min=amin, branch_fn=_pos_branch)
    X = np.ones((5, 2))
    z_hi = np.log(amax - C)
    z_lo = np.log(amin - C)
    v = np.array([z_lo - 1.0, z_lo - 0.5, 0.5 * (z_lo + z_hi), z_hi + 1.0, 0.0])
    assert gen.lower_binding(X, v).tolist() == [True, True, False, False, False]
    assert gen.upper_binding(X, v).tolist() == [False, False, False, True, False]
    diag = gen.binding_diagnostics(X, v)
    assert diag == {
        "alpha_lower_bound": amin,
        "alpha_upper_bound": amax,
        "n_lower_binding": 2,
        "n_upper_binding": 1,
    }


def test_fit_result_records_per_side_binding_rates():
    gen = BoundedUKLGenerator(C=1e-2, alpha_max=1.2, branch_fn=_treated_branch)
    X, _, _, model = _fit_ingredients(200, 4, gen)
    res = model.fit(X)
    assert res.success
    assert np.isfinite(res.binding_rate_lower)
    assert np.isfinite(res.binding_rate_upper)
    assert res.clip_binding_rate == pytest.approx(
        res.binding_rate_lower + res.binding_rate_upper
    )


def test_quality_checks_are_not_removed_from_the_screen():
    # The screen is a single source of truth: a bounded candidate that also
    # violates the ESS floor must list BOTH reasons. A selection-level test
    # cannot see this (any all-inadmissible outcome raises the same error),
    # so the predicate is pinned directly.
    from genriesz.model_selection import _violated_thresholds

    row = {
        "success": True,
        "modifies_estimand": True,
        "ess_ratio_min": 0.01,
        "cap_binding_rate": 0.0,
        "kernel_median": 1.0,
        "r_hat": 0.0,
        "max_abs_alpha": 10.0,
        "std_imbalance": 0.1,
    }
    violations = _violated_thresholds(row, {"min_ess_ratio": 0.5})
    assert "modifies_estimand" in violations
    assert "min_ess_ratio" in violations


def test_bounded_bkl_binding_diagnostics_report_each_side():
    gen = BoundedBKLGenerator(C=1.0, alpha_max=5.0, branch_fn=_pos_branch)
    X = np.ones((4, 2))
    u_min = gen._u_min
    u_floor = gen._u_floor
    v = np.array([u_floor - 1.0, 0.5 * (u_floor + u_min), u_min + 0.1, u_min + 0.2])
    assert gen.lower_binding(X, v).tolist() == [True, False, False, False]
    assert gen.upper_binding(X, v).tolist() == [False, False, True, True]
    diag = gen.binding_diagnostics(X, v)
    assert diag == {
        "alpha_lower_bound": gen.alpha_floor,
        "alpha_upper_bound": 5.0,
        "n_lower_binding": 1,
        "n_upper_binding": 2,
    }


def test_bounded_bkl_fit_result_records_per_side_binding_rates():
    gen = BoundedBKLGenerator(C=1e-2, alpha_max=5.0, branch_fn=_treated_branch)
    X, _, _, model = _fit_ingredients(200, 4, gen)
    res = model.fit(X)
    assert res.success
    assert np.isfinite(res.binding_rate_lower)
    assert np.isfinite(res.binding_rate_upper)
    assert res.clip_binding_rate == pytest.approx(
        res.binding_rate_lower + res.binding_rate_upper
    )


def test_fit_result_keeps_its_positional_field_order():
    from genriesz.glm import FitResult

    assert FitResult.__match_args__ == (
        "beta",
        "success",
        "message",
        "n_iter",
        "status",
        "objective_value",
        "gradient_norm",
        "kkt_residual",
        "clip_binding_rate",
        "fit_time",
        "binding_rate_lower",
        "binding_rate_upper",
    )
