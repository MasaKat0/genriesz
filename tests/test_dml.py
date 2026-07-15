import numpy as np
import pytest

from genriesz import ATEFunctional, SquaredGenerator, grr_functional
from genriesz.utils import kfold_splits


def _make_synthetic_ate(n: int = 300, d: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    logits = 0.7 * Z[:, 0] - 0.3 * Z[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n)
    tau = 1.0
    mu0 = Z[:, 0] + 0.25 * Z[:, 1] ** 2
    Y = mu0 + tau * D + rng.normal(scale=1.0, size=n)
    X = np.concatenate([D.reshape(-1, 1), Z], axis=1)
    return X, Y, tau


def phi(X: np.ndarray) -> np.ndarray:
    """Simple ATE-style interaction features.

    X = [D, Z...]
    Phi = [1, D, Z, D*Z]
    """

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        d = X[0]
        z = X[1:]
        return np.concatenate([[1.0], [d], z, d * z])
    d = X[:, [0]]
    z = X[:, 1:]
    return np.concatenate([np.ones((len(X), 1)), d, z, d * z], axis=1)


def test_grr_functional_ra_rw_arw_tmle_run():
    X, Y, _ = _make_synthetic_ate(n=200, d=2, seed=0)
    gen = SquaredGenerator(C=0.0).as_generator()

    res = grr_functional(
        X=X,
        Y=Y,
        m=ATEFunctional(treatment_index=0),
        basis=phi,
        generator=gen,
        cross_fit=True,
        folds=3,
        random_state=0,
        estimators=("ra", "rw", "arw", "tmle"),
        outcome_models="shared",
        riesz_penalty="l2",
        riesz_lam=1e-3,
        max_iter=200,
        tol=1e-9,
    )

    # Keys are canonicalized to 'ra'/'rw'/'arw'/'tmle'.
    for k in ["rw", "ra", "arw", "tmle"]:
        assert k in res.estimates
        assert np.isfinite(res.estimates[k].estimate)
        assert np.isfinite(res.estimates[k].se)

    s = res.summary_text()
    assert isinstance(s, str)
    assert "rw" in s.lower()


def test_grr_functional_raises_when_riesz_optimization_fails():
    X, Y, _ = _make_synthetic_ate(n=120, d=2, seed=2)
    gen = SquaredGenerator(C=0.0).as_generator()

    # riesz_penalty='l1' forces the numeric (L-BFGS) path: the SQ + l2 case is
    # solved in closed form and cannot fail via max_iter.
    with pytest.raises(RuntimeError, match="Riesz GRR optimization failed"):
        grr_functional(
            X=X,
            Y=Y,
            m=ATEFunctional(treatment_index=0),
            basis=phi,
            generator=gen,
            cross_fit=False,
            estimators=("rw",),
            riesz_penalty="l1",
            riesz_lam=1e-3,
            max_iter=0,
        )


def test_kfold_splits_rejects_more_folds_than_observations():
    with pytest.raises(ValueError, match="folds must be <= n"):
        list(kfold_splits(3, folds=4, random_state=0))


def test_att_pi_correction_affects_se_but_not_estimate():
    from genriesz import ATTFunctional, grr_att

    X, Y, _ = _make_synthetic_ate(n=400, d=2, seed=7)
    D = X[:, 0]
    pi = float(np.mean(D))
    common = dict(
        X=X,
        Y=Y,
        basis=phi,
        generator=SquaredGenerator(C=0.0).as_generator(),
        cross_fit=True,
        folds=2,
        random_state=0,
        estimators=("arw",),
        riesz_lam=1e-3,
    )

    res_est = grr_functional(
        m=ATTFunctional(treatment_index=0, pi=pi, pi_is_estimated=True), **common
    )
    res_known = grr_functional(
        m=ATTFunctional(treatment_index=0, pi=pi, pi_is_estimated=False), **common
    )

    # The point estimate is identical (the correction term has mean zero when
    # pi is the sample mean of D); only the SE changes.
    assert res_est.arw.estimate == pytest.approx(res_known.arw.estimate, rel=0, abs=1e-12)
    assert res_est.arw.se != pytest.approx(res_known.arw.se, rel=1e-6)

    # grr_att marks pi as estimated and must match the corrected variant.
    res_wrapper = grr_att(
        X=X,
        Y=Y,
        treatment_index=0,
        basis=phi,
        generator=SquaredGenerator(C=0.0).as_generator(),
        cross_fit=True,
        folds=2,
        random_state=0,
        estimators=("arw",),
        riesz_lam=1e-3,
    )
    assert res_wrapper.arw.se == pytest.approx(res_est.arw.se, rel=1e-12)


def test_outcome_link_inference_warns_for_continuous_unit_interval_y():
    rng = np.random.default_rng(11)
    X, _, _ = _make_synthetic_ate(n=150, d=2, seed=11)
    Y = rng.uniform(0.05, 0.95, size=len(X))  # continuous, not binary

    with pytest.warns(UserWarning, match="outcome_link"):
        grr_functional(
            X=X,
            Y=Y,
            m=ATEFunctional(treatment_index=0),
            basis=phi,
            generator=SquaredGenerator(C=0.0).as_generator(),
            cross_fit=True,
            folds=2,
            random_state=0,
            estimators=("arw",),
        )


def _make_rare_treatment(n: int = 40, n_treated: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    D = np.zeros(n)
    D[:n_treated] = 1.0
    rng.shuffle(D)
    Y = 2.0 * D + Z[:, 0] + 0.1 * rng.normal(size=n)
    return np.column_stack([D, Z]), Y


def test_cross_fitting_is_stratified_for_treatment_functionals():
    """3 treated / 40 units / 5 folds (audit EST-07 / K-01).

    Plain K-fold can put every treated unit into a single test fold; the
    training folds then hold no treated unit, the ATT M-matrix is identically
    zero, and the run used to *succeed* with alpha == 0 out of fold and a
    deceptively tight CI. Stratified folds keep >= 2 treated units in every
    training fold, so the same call must now go through cleanly.
    """
    from genriesz import grr_att

    X, Y = _make_rare_treatment()
    res = grr_att(
        X=X,
        Y=Y,
        treatment_index=0,
        basis=phi,
        generator=SquaredGenerator(C=0.0).as_generator(),
        cross_fit=True,
        folds=5,
        random_state=0,
        estimators=("arw",),
        riesz_lam=1e-3,
    )
    assert np.isfinite(res.arw.estimate)
    assert np.isfinite(res.arw.se) and res.arw.se > 0


def test_training_fold_without_both_groups_fails_loudly():
    # A single treated unit cannot be in every training fold: whichever test
    # fold receives it leaves its training fold all-control, and no split can
    # avoid that. The failure must be an explicit error before any fold is
    # fitted, not a silent degenerate fit.
    from genriesz import grr_att

    X, Y = _make_rare_treatment(n_treated=1)
    with pytest.raises(ValueError, match="training fold contains"):
        grr_att(
            X=X,
            Y=Y,
            treatment_index=0,
            basis=phi,
            generator=SquaredGenerator(C=0.0).as_generator(),
            cross_fit=True,
            folds=5,
            random_state=0,
            estimators=("arw",),
        )


def test_stratify_folds_true_requires_a_treatment_functional():
    from genriesz import AMEFunctional

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    Y = X[:, 0] + rng.normal(size=60)
    with pytest.raises(ValueError, match="stratify_folds"):
        grr_functional(
            X=X,
            Y=Y,
            m=AMEFunctional(coordinate=0),
            basis=lambda A: np.c_[np.ones(len(A)), np.asarray(A, dtype=float)],
            generator=SquaredGenerator(C=0.0).as_generator(),
            estimators=("rw",),
            cross_fit=True,
            folds=3,
            stratify_folds=True,
        )


def test_explicit_logit_link_rejects_unbounded_y_before_fitting():
    # The logit outcome model can only produce mu in (0, 1). With an unbounded
    # Y it used to run anyway for RA/RW/ARW -- only the TMLE branch checked --
    # and RA came back wildly wrong with a tiny SE (audit N-04).
    X, Y, _ = _make_synthetic_ate(n=120, d=2, seed=3)
    with pytest.raises(ValueError, match=r"bounded in \[0, 1\]"):
        grr_functional(
            X=X,
            Y=10.0 * Y,
            m=ATEFunctional(treatment_index=0),
            basis=phi,
            generator=SquaredGenerator(C=0.0).as_generator(),
            outcome_link="logit",
            estimators=("ra",),
            cross_fit=False,
        )


def test_single_group_sample_fails_at_the_entry_check():
    # With a treatment-only X (no covariate columns) the balance-diagnostics
    # block is skipped entirely, so an all-treated sample used to return
    # without any error at all (audit N-24).
    n = 30
    X = np.ones((n, 1))
    Y = np.ones(n)
    with pytest.raises(ValueError, match="Both treatment groups"):
        grr_functional(
            X=X,
            Y=Y,
            m=ATEFunctional(treatment_index=0),
            basis=phi,
            generator=SquaredGenerator(C=0.0).as_generator(),
            estimators=("rw",),
            cross_fit=False,
        )


def test_non_integral_folds_are_rejected():
    X, Y, _ = _make_synthetic_ate(n=60, d=2, seed=5)
    with pytest.raises(ValueError, match="folds must be an integer"):
        grr_functional(
            X=X,
            Y=Y,
            m=ATEFunctional(treatment_index=0),
            basis=phi,
            generator=SquaredGenerator(C=0.0).as_generator(),
            estimators=("rw",),
            cross_fit=True,
            folds=2.7,
        )
