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
    BPGenerator,
    GaussianRKHSBasis,
    GRRCVConfig,
    PolynomialBasis,
    SquaredGenerator,
    grr_ate,
    select_grr_hyperparams,
)
from genriesz.functionals import AMEFunctional, ATTFunctional
from genriesz.glm import GRRGLM
from genriesz.model_selection import (
    make_candidate_basis,
    normalize_grid,
    score_grr_candidate,
    select_kernel_centers,
)
from genriesz.utils import Fold


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


# ---------------------------------------------------------------------------
# squared_loss_validation: a generator-agnostic (uLSIF-style) selection score
# ---------------------------------------------------------------------------

def test_squared_loss_validation_selects_and_is_consistent():
    X, Y, _ = _make_ate(n=350, seed=6)
    cfg = GRRCVConfig(
        sigma_grid="auto",
        lam_grid=[1e-2, 1e-1],
        selection_score="squared_loss_validation",
        return_path=True,
        random_state=0,
    )
    common = dict(
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        config=cfg,
        outcome_link="identity",
    )
    res = select_grr_hyperparams(X_train=X, y_train=Y, **common)

    assert res.selection_score == "squared_loss_validation"
    assert np.isfinite(res.best_score)
    assert res.sigma is not None and res.sigma > 0.0

    # Deterministic function of the training sample.
    res2 = select_grr_hyperparams(X_train=X, y_train=Y, **common)
    assert (res.sigma, res.lam, res.n_centers) == (res2.sigma, res2.lam, res2.n_centers)

    # Every fitted candidate carries a finite LSIF score, and the criterion used
    # for selection is exactly that score.
    fitted = [r for r in res.path if r["success"]]
    assert fitted
    for r in fitted:
        assert np.isfinite(r["squared_loss_validation"])
        assert r["criterion"] == pytest.approx(r["squared_loss_validation"])

    # The winner minimizes the score over the pool that was actually used
    # (admissible first, else all fitted candidates).
    adm = [r["criterion"] for r in res.path if r["admissible"] and np.isfinite(r["criterion"])]
    if adm:
        assert res.best_score == pytest.approx(min(adm))
    else:
        fit_c = [r["criterion"] for r in res.path if r["success"] and np.isfinite(r["criterion"])]
        assert res.best_score == pytest.approx(min(fit_c))


def test_squared_loss_validation_scores_non_squared_generator():
    # The LSIF risk is built from the fitted representer, not the generator's own
    # Bregman conjugate, so a candidate fit with a *non-squared* generator (here
    # BP) is still scored on the common yardstick.
    X, Y, _ = _make_ate(n=400, seed=7)

    def branch(x):  # treated branch where D == 1 (first column of X)
        return 1 if float(x[0]) > 0.5 else 0

    cfg = GRRCVConfig(
        lam_grid=[1e-2, 1e-1],
        selection_score="squared_loss_validation",
        return_path=True,
        random_state=0,
    )
    res = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=BPGenerator(C=1.0, omega=0.5, branch_fn=branch),
        config=cfg,
        outcome_link="identity",
    )

    assert res.selection_score == "squared_loss_validation"
    assert np.isfinite(res.best_score)
    fitted = [r for r in res.path if r["success"]]
    assert fitted  # at least one BP candidate fit and received a finite LSIF score
    assert all(np.isfinite(r["squared_loss_validation"]) for r in fitted)

    # Deterministic function of the training sample for a non-squared generator.
    res2 = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=BPGenerator(C=1.0, omega=0.5, branch_fn=branch),
        config=cfg,
        outcome_link="identity",
    )
    assert (res.sigma, res.lam) == (res2.sigma, res2.lam)
    assert res.best_score == pytest.approx(res2.best_score)


def test_squared_loss_validation_matches_hand_computed_lsif_risk():
    # Directly validate the LSIF formula (and alpha_iva == inv_grad) against an
    # independent reconstruction of the fit on a single fold.
    X, Y, _ = _make_ate(n=240, seed=11)
    n = X.shape[0]
    idx = np.arange(n)
    fold = Fold(train=idx[: n // 2], test=idx[n // 2 :])

    m = ATEFunctional(0)
    gen = SquaredGenerator()
    template = GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0)
    kw = dict(riesz_penalty="l2", lam=1e-2)

    row = score_grr_candidate(
        X_train=X,
        y_train=Y,
        m=m,
        template_basis=template,
        generator=gen,
        sigma=None,
        centers=None,
        inner_folds=[fold],
        riesz_p_norm=None,
        outcome_link="identity",
        outcome_penalty="l2",
        outcome_lam=1e-3,
        max_iter=500,
        tol=1e-8,
        want_kernel=False,
        want_squared_loss=True,
        **kw,
    )
    assert row["success"]

    # Independent reconstruction of the same single-fold fit.
    cb = make_candidate_basis(template, sigma=None, centers=None)
    cb.fit(X[fold.train], Y[fold.train])
    grr = GRRGLM(basis=cb, generator=gen, functional=m, penalty="l2", lam=1e-2)
    fr = grr.fit(X[fold.train], max_iter=500, tol=1e-8)
    assert fr.success and grr.beta_ is not None
    beta = grr.beta_

    X_te = X[fold.test]
    v = np.asarray(cb(X_te), dtype=float) @ beta
    _, alpha = gen.conjugate(X_te, v)
    # The representer identity: conjugate's maximizer equals inv_grad(v).
    assert np.allclose(alpha, gen.inv_grad(X_te, v))

    def rep(A):
        return gen.inv_grad(A, np.asarray(cb(A), dtype=float) @ beta)

    m_rep = np.asarray(m.m_from_function(X_te, predict=rep, derivative=None), dtype=float)
    expected = 0.5 * float(np.mean(alpha**2)) - float(np.mean(m_rep))
    assert row["squared_loss_validation"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_squared_loss_validation_surfaces_functional_incompatibility():
    # A functional whose m(alpha) needs a derivative but is *not* AME slips past
    # the isinstance guard, so the scorer must raise a clear ValueError rather
    # than swallow it into a misleading "all candidates failed to fit".
    class _DerivOnlyFunctional(ATEFunctional):
        def m_from_function(self, X, *, predict, derivative=None):
            if derivative is None:
                raise NotImplementedError("needs a derivative")
            return super().m_from_function(X, predict=predict, derivative=derivative)

    X, Y, _ = _make_ate(n=200, seed=12)
    cfg = GRRCVConfig(
        lam_grid=[1e-2],
        selection_score="squared_loss_validation",
        random_state=0,
    )
    with pytest.raises(ValueError, match="representer alone"):
        select_grr_hyperparams(
            X_train=X,
            y_train=Y,
            m=_DerivOnlyFunctional(0),
            basis=GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0),
            generator=SquaredGenerator(),
            config=cfg,
            outcome_link="identity",
        )


def test_squared_loss_validation_does_not_swallow_valueerror():
    # A genuine ValueError from the functional (not a NotImplementedError about a
    # missing derivative) must propagate unchanged, never be turned into NaN.
    class _RaisesValueError(ATEFunctional):
        def m_from_function(self, X, *, predict, derivative=None):
            raise ValueError("boom-from-functional")

    X, Y, _ = _make_ate(n=200, seed=14)
    cfg = GRRCVConfig(
        lam_grid=[1e-2],
        selection_score="squared_loss_validation",
        random_state=0,
    )
    with pytest.raises(ValueError, match="boom-from-functional"):
        select_grr_hyperparams(
            X_train=X,
            y_train=Y,
            m=_RaisesValueError(0),
            basis=GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0),
            generator=SquaredGenerator(),
            config=cfg,
            outcome_link="identity",
        )


def test_squared_loss_score_is_nan_when_a_fold_fails_to_fit():
    # When the primary Riesz fit fails on a fold, that fold is never scored, so
    # the LSIF aggregate must be NaN (strict: len(sq_risks) == len(folds)), not a
    # partial-fold average. Here max_iter=0 on the L-BFGS (l1) path fails both
    # folds, and the candidate is both success=False and NaN-scored.
    X, Y, _ = _make_ate(n=240, seed=13)
    n = X.shape[0]
    idx = np.arange(n)
    folds = [
        Fold(train=idx[: n // 2], test=idx[n // 2 :]),
        Fold(train=idx[n // 2 :], test=idx[: n // 2]),
    ]
    row = score_grr_candidate(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(0),
        template_basis=GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0),
        generator=SquaredGenerator(C=0.0).as_generator(),
        sigma=None,
        lam=1e-3,
        centers=None,
        inner_folds=folds,
        riesz_penalty="l1",  # numeric path; SQ + l2 is closed form and cannot fail
        riesz_p_norm=None,
        outcome_link="identity",
        outcome_penalty="l2",
        outcome_lam=1e-3,
        max_iter=0,  # zero iterations -> the fit fails on every fold
        tol=1e-8,
        want_kernel=False,
        want_squared_loss=True,
    )
    assert row["success"] is False
    assert np.isnan(row["squared_loss_validation"])


def test_incompatible_functional_selects_fine_under_other_scores():
    # want_squared_loss=False must skip the LSIF path entirely, so an AME
    # functional (whose m(alpha) needs a derivative) still selects under a
    # different score without raising the squared-loss incompatibility error.
    # A derivative-capable basis (Polynomial) is required for AME at all.
    X, Y, _ = _make_ate(n=300, seed=15)
    cfg = GRRCVConfig(
        lam_grid=[1e-2, 1e-1],
        selection_score="bregman_validation",
        random_state=0,
    )
    res = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=AMEFunctional(coordinate=1),
        basis=PolynomialBasis(degree=2),
        generator=SquaredGenerator(),
        config=cfg,
        outcome_link="identity",
    )
    assert res.selection_score == "bregman_validation"
    assert np.isfinite(res.best_score)


def test_squared_loss_validation_works_for_att():
    X, Y, _ = _make_ate(n=350, seed=9)
    pi = float(np.mean(X[:, 0]))
    cfg = GRRCVConfig(
        lam_grid=[1e-2, 1e-1],
        selection_score="squared_loss_validation",
        random_state=0,
    )
    res = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATTFunctional(treatment_index=0, pi=pi, pi_is_estimated=True),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        config=cfg,
        outcome_link="identity",
    )
    assert np.isfinite(res.best_score)
    assert res.lam in (1e-2, 1e-1)


def test_squared_loss_validation_rejects_ame():
    X, Y, _ = _make_ate(n=200, seed=8)
    cfg = GRRCVConfig(
        lam_grid=[1e-2],
        selection_score="squared_loss_validation",
        random_state=0,
    )
    with pytest.raises(ValueError, match="not defined for"):
        select_grr_hyperparams(
            X_train=X,
            y_train=Y,
            m=AMEFunctional(coordinate=1),
            basis=GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0),
            generator=SquaredGenerator(),
            config=cfg,
            outcome_link="identity",
        )
