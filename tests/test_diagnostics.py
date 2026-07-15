"""Tests for the Step-1 (result-preserving) diagnostics.

Covers:
- GaussianRKHSBasis.diagnostics() kernel health (item B),
- held-out working-span imbalance and bias proxy in grr_functional (items H, I),
- the utils helpers bias_proxy / coverage_decomposition.

These diagnostics are additive: they must not change point estimates or SEs.
"""

from __future__ import annotations

import numpy as np
import pytest

from genriesz import (
    ATEFunctional,
    GaussianRKHSBasis,
    PolynomialBasis,
    SquaredGenerator,
    bias_proxy,
    coverage_decomposition,
    grr_ate,
    grr_functional,
    oracle_decomposition,
)


def _make_synthetic_ate(n: int = 400, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    logits = 0.6 * Z[:, 0] - 0.4 * Z[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n).astype(float)
    tau = 1.0
    Y = Z[:, 0] + 0.5 * Z[:, 1] + tau * D + rng.normal(scale=0.5, size=n)
    X = np.column_stack([D, Z])
    return X, Y, tau


def _poly_phi(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        d = X[0]
        z = X[1:]
        return np.concatenate([[1.0], [d], z, d * z])
    d = X[:, [0]]
    z = X[:, 1:]
    return np.concatenate([np.ones((len(X), 1)), d, z, d * z], axis=1)


# ---------------------------------------------------------------------------
# Kernel health (item B)
# ---------------------------------------------------------------------------

def test_rkhs_diagnostics_requires_fit():
    basis = GaussianRKHSBasis(n_centers=20, sigma=1.0, random_state=0)
    with pytest.raises(RuntimeError, match="fit"):
        basis.diagnostics(np.zeros((5, 3)))


def test_rkhs_kernel_health_flags_bandwidth_extremes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))

    # A moderate bandwidth near the median pairwise distance is healthy.
    mid = GaussianRKHSBasis(n_centers=80, sigma=1.0, random_state=0).fit(X)
    d_mid = mid.diagnostics(X)
    median_dist = d_mid["median_pairwise_distance"]
    assert np.isfinite(median_dist) and median_dist > 0
    # Kernel activations are neither collapsed nor saturated.
    assert 1e-3 < d_mid["kernel_median"] < 1.0
    assert d_mid["underfitting"] is False

    # Tiny bandwidth: every off-diagonal kernel value collapses to ~0 -> each
    # point only sees itself. This must be flagged as underfitting.
    small = GaussianRKHSBasis(n_centers=80, sigma=0.02, random_state=0).fit(X)
    d_small = small.diagnostics(X)
    assert d_small["kernel_median"] < 1e-3
    assert d_small["underfitting"] is True

    # Huge bandwidth: every kernel value saturates to ~1 -> features are nearly
    # constant, so their variance collapses (no signal).
    big = GaussianRKHSBasis(n_centers=80, sigma=50.0, random_state=0).fit(X)
    d_big = big.diagnostics(X)
    assert d_big["kernel_median"] > 0.99
    assert d_big["feature_variance_median"] < d_mid["feature_variance_median"]
    assert d_big["underfitting"] is False


# ---------------------------------------------------------------------------
# Held-out imbalance + bias proxy in grr_functional (items H, I)
# ---------------------------------------------------------------------------

def test_grr_reports_held_out_imbalance_and_bias_diagnostics():
    X, Y, _ = _make_synthetic_ate(n=400, d=3, seed=1)
    basis = GaussianRKHSBasis(n_centers=60, sigma=1.0, random_state=0)
    res = grr_ate(
        X=X,
        Y=Y,
        basis=basis,
        generator=SquaredGenerator(),
        riesz_lam=1e-2,
        folds=5,
        random_state=0,
    )
    d = res.diagnostics

    # Held-out working-span imbalance (item H): one entry per fold, all finite.
    assert "imbalance" in d
    per_fold = d["imbalance"]["held_out_working_span_max"]
    assert len(per_fold) == 5
    assert all(np.isfinite(v) and v >= 0 for v in per_fold)
    assert np.isfinite(d["held_out_imbalance_max"])
    assert d["held_out_imbalance_max"] == pytest.approx(max(per_fold))

    # Kernel health aggregated across folds (item B).
    assert "kernel" in d and len(d["kernel"]["per_fold"]) == 5
    for key in (
        "kernel_median_min",
        "kernel_feature_variance_min",
        "kernel_gram_condition_max",
        "kernel_effective_rank_min",
    ):
        assert np.isfinite(d[key])

    # Bias proxy (item I): directional b_hat, conservative bound, standardized.
    b = d["bias"]
    assert np.isfinite(b["b_hat"]) and b["b_hat"] >= 0
    assert np.isfinite(b["b_bound"]) and b["b_bound"] >= b["b_hat"]
    assert b["outcome_tag"] == "shared"
    # std_bias = b_hat / se of the primary (ARW) estimate.
    assert b["std_bias"] == pytest.approx(b["b_hat"] / res.arw.se)


def test_bias_diagnostics_with_separate_outcome_basis_depend_on_predictions_only():
    """P0-04: the directional term must not pair coordinates across bases.

    With ``outcome_models='separate'`` and an outcome basis of coincidentally
    equal column count, the Riesz-span ``Delta`` used to be dotted with the
    separate-basis ``theta``. Rescaling the outcome basis columns leaves the
    predictions (hence the true second-order term) unchanged while rescaling
    ``theta`` inversely -- so any coordinate-pairing implementation moves with
    the scale, and the prediction-based one does not. The cross-basis
    Cauchy-Schwarz bound has no meaning at all, so it must be NaN.
    """

    X, Y, _ = _make_synthetic_ate(n=300, d=3, seed=4)

    def feats(Z):
        Z = np.asarray(Z, dtype=float)
        return np.column_stack([np.ones(len(Z)), Z])

    def feats_scaled(Z):
        return 1000.0 * feats(Z)

    common = dict(
        X=X,
        Y=Y,
        m=ATEFunctional(treatment_index=0),
        basis=PolynomialBasis(degree=1, include_bias=True),  # same column count as feats
        generator=SquaredGenerator(),
        riesz_lam=1e-3,
        outcome_models="separate",
        outcome_lam=1e-10,
        estimators=("arw",),
        folds=3,
        random_state=0,
    )
    b_plain = grr_functional(outcome_basis=feats, **common).diagnostics["bias"]
    b_scaled = grr_functional(outcome_basis=feats_scaled, **common).diagnostics["bias"]

    assert b_plain["outcome_tag"] == "separate"
    assert np.isfinite(b_plain["b_hat"]) and b_plain["b_hat"] >= 0
    assert b_plain["b_hat"] == pytest.approx(b_scaled["b_hat"], rel=1e-6)
    assert np.isnan(b_plain["b_bound"])
    assert "b_bound_unavailable_reason" in b_plain


def test_bias_bound_is_nan_for_logit_link_even_on_the_shared_basis():
    """||Delta||*||theta|| bounds Delta^T theta, and under a logit link
    gamma_hat = sigmoid(phi^T theta) is not linear in theta: theta = 0 gives
    the constant prediction 0.5 with a generally nonzero second-order term
    while the would-be bound is 0. So b_bound must be NaN for logit."""

    rng = np.random.default_rng(5)
    n = 300
    Z = rng.normal(size=(n, 2))
    D = (rng.uniform(size=n) < 0.5).astype(float)
    X = np.column_stack([D, Z])
    Yb = (Z[:, 0] + 0.5 * D + rng.normal(size=n) > 0).astype(float)

    res = grr_ate(
        X=X,
        Y=Yb,
        basis=PolynomialBasis(degree=1, include_bias=True),
        generator=SquaredGenerator(),
        riesz_lam=1e-3,
        outcome_link="logit",
        estimators=("arw",),
        folds=3,
        random_state=0,
    )
    b = res.diagnostics["bias"]
    assert b["outcome_tag"] == "shared"
    assert np.isfinite(b["b_hat"])
    assert np.isnan(b["b_bound"])
    assert "logit" in b["b_bound_unavailable_reason"]


def test_tiny_bandwidth_underfitting_is_visible():
    """A too-small bandwidth gives tiny 'balanced' weights but a wrong answer.

    The imbalance alone would look fine; the kernel-health underfitting flag is
    what exposes the failure (revision plan section 5.2).
    """

    X, Y, tau = _make_synthetic_ate(n=400, d=3, seed=2)
    basis = GaussianRKHSBasis(n_centers=60, sigma=0.05, random_state=0)
    res = grr_ate(
        X=X,
        Y=Y,
        basis=basis,
        generator=SquaredGenerator(),
        riesz_lam=1e-2,
        folds=5,
        random_state=0,
    )
    d = res.diagnostics
    # Severe underfitting: the estimate collapses far from the truth ...
    assert abs(res.arw.estimate - tau) > 0.5
    # ... yet the raw imbalance is small (the trap), while kernel health flags it.
    assert d["kernel_underfitting_any"] is True
    assert d["kernel_median_min"] < 1e-3


def test_diagnostics_do_not_change_estimates_or_se():
    """The imbalance/kernel/bias diagnostics are additive and deterministic.

    Two identical fits produce identical point estimates and SEs, and the
    diagnostics never touch the alpha weights used for inference.
    """

    X, Y, _ = _make_synthetic_ate(n=300, d=2, seed=3)
    kwargs = dict(
        X=X,
        Y=Y,
        m=ATEFunctional(treatment_index=0),
        basis=_poly_phi,
        generator=SquaredGenerator(C=0.0).as_generator(),
        cross_fit=True,
        folds=3,
        random_state=0,
        estimators=("ra", "rw", "arw"),
        riesz_lam=1e-3,
    )
    r1 = grr_functional(**kwargs)
    r2 = grr_functional(**kwargs)
    for k in ("ra", "rw", "arw"):
        assert r1.estimates[k].estimate == pytest.approx(r2.estimates[k].estimate, abs=0.0)
        assert r1.estimates[k].se == pytest.approx(r2.estimates[k].se, abs=0.0)

    # A non-kernel basis still yields imbalance + bias, but no kernel health.
    assert "imbalance" in r1.diagnostics
    assert "bias" in r1.diagnostics
    assert "kernel" not in r1.diagnostics


# ---------------------------------------------------------------------------
# utils helpers
# ---------------------------------------------------------------------------

def test_bias_proxy_product_and_nonfinite():
    assert bias_proxy(0.5, 2.0) == pytest.approx(1.0)
    assert bias_proxy(-0.5, 2.0) == pytest.approx(1.0)  # magnitudes
    assert np.isnan(bias_proxy(np.inf, 1.0))


def test_outcome_nuisance_diagnostics_reported():
    X, Y, _ = _make_synthetic_ate(n=400, d=3, seed=6)
    res = grr_ate(
        X=X,
        Y=Y,
        basis=GaussianRKHSBasis(n_centers=60, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        riesz_lam=1e-2,
        folds=5,
        random_state=0,
        outcome_models="shared",
    )
    d = res.diagnostics
    assert "outcome" in d and "shared" in d["outcome"]
    od = d["outcome"]["shared"]
    assert np.isfinite(od["cv_risk"]) and od["cv_risk"] > 0
    assert np.isfinite(od["residual_var"]) and od["residual_var"] > 0
    assert len(od["residual_fold_mean"]) == 5 and len(od["residual_fold_var"]) == 5
    # Out-of-fold residuals are approximately mean-zero.
    assert abs(od["residual_mean"]) < 0.1
    assert d["outcome_cv_risk"] == pytest.approx(od["cv_risk"])


def test_oracle_decomposition_isolates_nuisance_errors():
    rng = np.random.default_rng(0)
    n = 500
    y = rng.normal(size=n)
    alpha0 = rng.normal(size=n)
    gamma0 = rng.normal(size=n)
    # Perfect nuisances: RMSE 0, product drift 0, all oracle estimators equal.
    m_g0 = rng.normal(size=n)
    d = oracle_decomposition(
        y=y, alpha_hat=alpha0, alpha0=alpha0, gamma_hat=gamma0, gamma0=gamma0,
        m_gamma_hat=m_g0, m_gamma0=m_g0,
    )
    assert d["alpha_rmse"] == pytest.approx(0.0)
    assert d["gamma_rmse"] == pytest.approx(0.0)
    assert d["product_drift"] == pytest.approx(0.0)
    assert d["theta_true_alpha"] == pytest.approx(d["theta_true_both"])
    assert d["theta_true_gamma"] == pytest.approx(d["theta_true_both"])

    # Biased Riesz estimate -> nonzero alpha_rmse and product drift.
    alpha_hat = alpha0 + 0.5 * gamma0  # error correlated with gamma error direction
    gamma_hat = gamma0 + 0.3 * rng.normal(size=n)
    d2 = oracle_decomposition(
        y=y, alpha_hat=alpha_hat, alpha0=alpha0, gamma_hat=gamma_hat, gamma0=gamma0,
        m_gamma_hat=m_g0, m_gamma0=m_g0,
    )
    assert d2["alpha_rmse"] > 0 and d2["gamma_rmse"] > 0

    with pytest.raises(ValueError, match="same length"):
        oracle_decomposition(
            y=y, alpha_hat=alpha0[:-1], alpha0=alpha0, gamma_hat=gamma0, gamma0=gamma0,
            m_gamma_hat=m_g0, m_gamma0=m_g0,
        )


def test_coverage_decomposition_pieces():
    d = coverage_decomposition(estimate=1.0, se=0.1, n=100, b_hat=0.05, truth=1.05)
    assert d["v_hat"] == pytest.approx(100 * 0.1 * 0.1)
    assert d["std_bias"] == pytest.approx(0.05 / 0.1)
    assert d["bias"] == pytest.approx(1.0 - 1.05)
    # |bias| = 0.05 <= 1.96 * 0.1, so the nominal Wald interval covers.
    assert d["covered"] == 1.0

    # Without a truth, no oracle fields are added.
    d2 = coverage_decomposition(estimate=1.0, se=0.1, n=100, b_hat=0.05)
    assert "bias" not in d2 and "covered" not in d2


def test_coverage_decomposition_uses_the_requested_significance_level():
    # |bias| = 0.18 with se = 0.1: covered at the 95% level (z=1.96) but not at
    # the 90% level (z=1.645). A fixed 97.5% quantile would call both covered.
    kw = dict(estimate=1.0, se=0.1, n=100, b_hat=0.05, truth=1.18)
    d95 = coverage_decomposition(**kw)  # default significance_level=0.05
    d90 = coverage_decomposition(**kw, significance_level=0.10)
    assert d95["covered"] == 1.0
    assert d90["covered"] == 0.0
    assert d95["significance_level"] == pytest.approx(0.05)
    assert d90["confidence_level"] == pytest.approx(0.90)

    with pytest.raises(ValueError, match="significance_level"):
        coverage_decomposition(**kw, significance_level=1.5)

    # A degenerate interval cannot be judged either way.
    d0 = coverage_decomposition(estimate=1.0, se=0.0, n=100, b_hat=0.05, truth=1.0)
    assert np.isnan(d0["covered"])


def test_oracle_decomposition_rejects_broadcastable_m_gamma():
    # A length-1 m_gamma array would silently broadcast in the oracle one-step
    # means and return a wrong decomposition.
    n = 5
    ones = np.ones(n)
    with pytest.raises(ValueError, match="same length"):
        oracle_decomposition(
            y=ones, alpha_hat=ones, alpha0=ones, gamma_hat=ones, gamma0=ones,
            m_gamma_hat=np.array([9.0]), m_gamma0=ones,
        )
    with pytest.raises(ValueError, match="same length"):
        oracle_decomposition(
            y=ones, alpha_hat=ones, alpha0=ones, gamma_hat=ones, gamma0=ones,
            m_gamma_hat=ones, m_gamma0=np.array([1.0]),
        )
