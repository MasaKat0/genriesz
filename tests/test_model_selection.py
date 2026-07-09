"""Tests for inner Riesz-hyperparameter cross-validation (Step 3b, item C).

Covers the no-leakage guarantees, the two-stage selection (admissibility screen
+ criterion), grid normalization, and backward compatibility of ``grr_functional``
when no grid is supplied.
"""

from __future__ import annotations

import numpy as np
import pytest

from genriesz import (
    ATEFunctional,
    GaussianRKHSBasis,
    GRRCVConfig,
    PolynomialBasis,
    SquaredGenerator,
    grr_ate,
    select_grr_hyperparams,
)
from genriesz.model_selection import (
    make_candidate_basis,
    normalize_grid,
    select_kernel_centers,
)


def _make_ate(n: int = 400, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    ps = 1.0 / (1.0 + np.exp(-(0.6 * Z[:, 0] - 0.4 * Z[:, 1])))
    D = (rng.uniform(size=n) < ps).astype(float)
    X = np.column_stack([D, Z])
    Y = D + Z[:, 0] + 0.5 * Z[:, 1] + rng.normal(scale=0.5, size=n)
    return X, Y, 1.0


# ---------------------------------------------------------------------------
# Grid normalization
# ---------------------------------------------------------------------------

def test_normalize_grid_variants():
    assert normalize_grid([0.1, 0.2], kind="sigma") == [0.1, 0.2]
    assert normalize_grid(0.5, kind="sigma") == [0.5]
    auto = normalize_grid("auto", kind="sigma", median=2.0)
    assert auto == [2.0 * m for m in (0.25, 0.5, 1.0, 2.0, 4.0)]

    assert normalize_grid(None, kind="lam") == list(
        (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
    )
    assert normalize_grid(1e-2, kind="lam") == [1e-2]

    # n_centers is capped at n and de-duplicated.
    assert normalize_grid("auto", kind="n_centers", n=100) == [80, 100]
    assert normalize_grid([50, 500], kind="n_centers", n=200) == [50, 200]


def test_normalize_grid_auto_sigma_requires_median():
    with pytest.raises(ValueError, match="positive median"):
        normalize_grid("auto", kind="sigma", median=0.0)


# ---------------------------------------------------------------------------
# No leakage: centers within training, selection is a pure function of X_train
# ---------------------------------------------------------------------------

def test_select_kernel_centers_are_within_training():
    X, _, _ = _make_ate(n=200)
    centers = select_kernel_centers(X, n_centers=30, random_state=0)
    assert centers.shape == (30, X.shape[1])
    # Every returned center is an actual training row.
    for c in centers:
        assert np.any(np.all(np.isclose(X, c), axis=1))


def test_selection_is_deterministic_function_of_training():
    X, Y, _ = _make_ate(n=300, seed=1)
    cfg = GRRCVConfig(sigma_grid="auto", lam_grid=[1e-2, 1e-1], random_state=0)
    common = dict(
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        config=cfg,
        outcome_link="identity",
    )
    r1 = select_grr_hyperparams(X_train=X, y_train=Y, **common)
    r2 = select_grr_hyperparams(X_train=X, y_train=Y, **common)
    assert (r1.sigma, r1.lam, r1.n_centers) == (r2.sigma, r2.lam, r2.n_centers)


# ---------------------------------------------------------------------------
# Two-stage selection: kernel-health screen excludes degenerate bandwidths
# ---------------------------------------------------------------------------

def test_kernel_health_screen_excludes_degenerate_bandwidths():
    X, Y, _ = _make_ate(n=400, seed=2)
    cfg = GRRCVConfig(
        sigma_grid="auto",  # median * (0.25, 0.5, 1, 2, 4): includes both traps
        lam_grid=[1e-2, 1e-1],
        n_centers_grid=[60],
        return_path=True,
        random_state=0,
    )
    res = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=60, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        config=cfg,
        outcome_link="identity",
    )
    by_sigma = {round(r["sigma"], 3): r for r in res.path}
    sigmas = sorted(by_sigma)
    # Smallest bandwidth collapses the kernel (median ~ 0): inadmissible.
    assert by_sigma[sigmas[0]]["kernel_median"] < 1e-3
    assert all(not by_sigma[sigmas[0]]["admissible"] for _ in [0])
    # Largest bandwidth saturates the kernel (median -> 1): inadmissible.
    assert by_sigma[sigmas[-1]]["kernel_median"] > 0.8
    assert not by_sigma[sigmas[-1]]["admissible"]
    # At least one healthy candidate survives, and the selected sigma is not a
    # degenerate extreme.
    assert res.n_admissible >= 1
    assert sigmas[0] < res.sigma < sigmas[-1]


def test_make_candidate_basis_requires_rkhs_for_sigma():
    poly = PolynomialBasis(degree=2)
    with pytest.raises(ValueError, match="copy_with_params"):
        make_candidate_basis(poly, sigma=1.0, centers=None)
    # Lambda-only CV (no sigma/centers) works for any basis via copy().
    clone = make_candidate_basis(poly, sigma=None, centers=None)
    assert isinstance(clone, PolynomialBasis)


# ---------------------------------------------------------------------------
# grr_functional integration
# ---------------------------------------------------------------------------

def test_grr_functional_without_grids_is_unchanged():
    X, Y, _ = _make_ate(n=300, seed=3)
    kw = dict(
        X=X,
        Y=Y,
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        riesz_lam=1e-2,
        folds=4,
        random_state=0,
    )
    r1 = grr_ate(**kw)
    r2 = grr_ate(**kw)
    assert r1.arw.estimate == pytest.approx(r2.arw.estimate, abs=0.0)
    # No CV requested -> no CV diagnostics.
    assert "riesz_cv" not in r1.diagnostics


def test_grr_functional_cv_reports_selection_and_beats_fixed_trap():
    X, Y, tau = _make_ate(n=400, seed=4)
    fixed = grr_ate(
        X=X,
        Y=Y,
        basis=GaussianRKHSBasis(n_centers=60, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        riesz_lam=1e-2,
        folds=4,
        random_state=0,
    )
    tuned = grr_ate(
        X=X,
        Y=Y,
        basis=GaussianRKHSBasis(n_centers=60, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        folds=4,
        random_state=0,
        riesz_sigma_grid="auto",
        riesz_lam_grid=[1e-2, 1e-1],
        riesz_n_centers_grid=[60, 120],
    )
    cv = tuned.diagnostics["riesz_cv"]
    assert len(cv["selected"]) == 4
    for s in cv["selected"]:
        assert s["sigma"] is not None and s["sigma"] > 0
        assert s["n_admissible"] >= 1
    assert tuned.diagnostics["riesz_cv_selection_score"] == "bias_variance"
    # The tuned estimate is at least as close to the truth as the fixed baseline,
    # and its SE is not the fake-tiny value of a saturated kernel.
    assert abs(tuned.arw.estimate - tau) <= abs(fixed.arw.estimate - tau) + 0.05
    assert tuned.arw.se > 1e-2


def test_grr_functional_cv_lambda_only_with_any_basis():
    X, Y, _ = _make_ate(n=300, seed=5)

    def phi(A):
        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            d, z = A[0], A[1:]
            return np.concatenate([[1.0], [d], z, d * z])
        d, z = A[:, [0]], A[:, 1:]
        return np.concatenate([np.ones((len(A), 1)), d, z, d * z], axis=1)

    res = grr_ate(
        X=X,
        Y=Y,
        basis=phi,
        generator=SquaredGenerator(C=0.0).as_generator(),
        folds=3,
        random_state=0,
        riesz_lam_grid=[1e-3, 1e-2, 1e-1],
    )
    sel = res.diagnostics["riesz_cv"]["selected"]
    assert len(sel) == 3
    for s in sel:
        assert s["sigma"] is None  # no bandwidth CV for a callable basis
        assert s["lam"] in (1e-3, 1e-2, 1e-1)
