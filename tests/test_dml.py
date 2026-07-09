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
