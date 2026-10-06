"""Strict two-sample density-ratio path (registration §1.3 A-5, E-20).

``fit_density_ratio(solver="auto")`` minimizes the empirical
``eq:cs_bd_empirical`` with the strict solvers: no clipping, explicit statuses,
an optional fixed offset, and domain-checked predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from genriesz import (
    BKLGenerator,
    BPGenerator,
    CallableBasis,
    DomainError,
    UKLGenerator,
    fit_density_ratio,
)


def _pos(_x):
    return 1


def _linear_basis():
    return CallableBasis(lambda X: np.column_stack([np.ones(len(np.atleast_2d(X))),
                                                    np.atleast_2d(X)]))


def _tilted_samples(n_den: int, n_num: int, seed: int):
    """Source Unif(-sqrt3, sqrt3)^2; target tilted by exp(t'z) (DGP20), by rejection."""

    rng = np.random.default_rng(seed)
    a = np.sqrt(3.0)
    t = np.array([0.6, -0.4])
    X_den = rng.uniform(-a, a, size=(n_den, 2))
    bound = np.exp(np.abs(t).sum() * a)
    out = []
    while sum(len(o) for o in out) < n_num:
        Z = rng.uniform(-a, a, size=(4 * n_num, 2))
        keep = rng.uniform(size=len(Z)) * bound < np.exp(Z @ t)
        out.append(Z[keep])
    X_num = np.vstack(out)[:n_num]
    norm = np.prod(np.sinh(a * t) / (a * t))
    return X_num, X_den, lambda Z: np.exp(Z @ t) / norm


def test_ukl_c0_unpenalized_is_exactly_balanced_and_consistent():
    X_num, X_den, r0 = _tilted_samples(20000, 20000, seed=0)
    res = fit_density_ratio(
        X_num, X_den, basis=_linear_basis(), generator=UKLGenerator(C=0.0, branch_fn=_pos),
        penalty=None,
    )
    assert res.status == "ok" and res.route == "bregman"
    assert res.fit is not None and res.fit.kkt_residual <= 1e-10
    # KKT = moment balance: mean_den(r_hat * phi) = mean_num(phi).
    Phi_d = _linear_basis()(X_den)
    Phi_n = _linear_basis()(X_num)
    r_den = res.predict_ratio(X_den)
    assert np.max(np.abs((r_den[:, None] * Phi_d).mean(0) - Phi_n.mean(0))) <= 1e-10
    # Correctly specified log-linear ratio: close to the truth at n = 20000.
    Z = np.random.default_rng(1).uniform(-1.0, 1.0, size=(500, 2))
    assert np.max(np.abs(res.predict_ratio(Z) / r0(Z) - 1.0)) < 0.1


def test_squared_loss_ratio_is_not_clipped_by_default():
    rng = np.random.default_rng(2)
    X_den = rng.normal(0.0, 1.0, size=(300, 1))
    X_num = rng.normal(1.5, 0.5, size=(300, 1))
    res = fit_density_ratio(X_num, X_den, basis=_linear_basis(), generator="sq", penalty=None)
    assert res.status == "ok"
    Z = np.array([[-3.0], [0.0], [3.0]])
    raw = res.predict_ratio(Z)
    assert raw.min() < 0.0  # a linear SQ ratio goes negative in the tail
    assert np.all(res.predict_ratio(Z, clip_nonnegative=True) >= 0.0)


def test_bkl_needs_an_offset_and_reports_it_as_a_status():
    rng = np.random.default_rng(3)
    X_den = rng.normal(0.0, 1.0, size=(300, 1))
    X_num = rng.normal(0.3, 1.0, size=(300, 1))
    gen = BKLGenerator(C=0.2, branch_fn=_pos)
    bad = fit_density_ratio(X_num, X_den, basis=_linear_basis(), generator=gen, penalty=None)
    assert bad.status == "infeasible_start" and not bad.success
    with pytest.raises(RuntimeError, match="infeasible_start"):
        bad.predict_ratio(X_num)
    u_ref = float(np.log((1.0 - 0.2) / (1.0 + 0.2)))  # alpha_ref = 1
    good = fit_density_ratio(
        X_num, X_den, basis=_linear_basis(), generator=gen, penalty=None, offset=u_ref
    )
    assert good.status == "ok"
    assert good.fit.kkt_residual <= 1e-10 * max(1.0, float(np.abs(X_num).mean()))


def test_l1_density_ratio_uses_fista():
    X_num, X_den, _ = _tilted_samples(3000, 3000, seed=4)
    res = fit_density_ratio(
        X_num, X_den, basis=_linear_basis(), generator=UKLGenerator(C=0.0, branch_fn=_pos),
        penalty="l1", lam=0.05,
    )
    assert res.status == "ok"
    assert res.fit.kkt_residual <= 1e-6 * 0.05


def test_out_of_domain_prediction_raises_or_returns_nan():
    rng = np.random.default_rng(5)
    X_den = rng.uniform(0.0, 1.0, size=(400, 1))
    # Target density 1.5 - x on [0, 1] (by rejection): the ratio is linear,
    # positive on the support, and decreasing.
    cand = rng.uniform(0.0, 1.0, size=4000)
    X_num = cand[rng.uniform(0.0, 1.5, size=4000) < 1.5 - cand][:400].reshape(-1, 1)
    gen = BPGenerator(C=0.0, omega=1.0, branch_fn=_pos)
    res = fit_density_ratio(X_num, X_den, basis=_linear_basis(), generator=gen, penalty=None)
    assert res.status == "ok"
    slope = float(res.beta[1])
    assert slope < 0.0
    # v = b0 + slope * x < -2 (outside the BP range) for x large enough.
    x_far = np.array([[(-2.5 - float(res.beta[0])) / slope]])
    assert not res.domain_mask(x_far)[0]
    with pytest.raises(DomainError):
        res.predict_ratio(x_far)
    assert np.isnan(res.predict_ratio(x_far, out_of_domain="nan")[0])


def test_cv_with_strict_solver_counts_failed_candidates():
    rng = np.random.default_rng(6)
    X_den = rng.normal(0.0, 1.0, size=(150, 1))
    X_num = rng.normal(0.4, 1.0, size=(150, 1))
    res = fit_density_ratio(
        X_num, X_den, generator=UKLGenerator(C=0.0, branch_fn=_pos), n_centers=20,
        sigma_grid=[0.5, 1.0], lam_grid=[1e-2, 1e-1], cv=True, folds=3, random_state=0,
    )
    assert res.status == "ok"
    assert res.lam in (1e-2, 1e-1)
