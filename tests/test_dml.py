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

    with pytest.raises(RuntimeError, match="Riesz GRR optimization failed"):
        grr_functional(
            X=X,
            Y=Y,
            m=ATEFunctional(treatment_index=0),
            basis=phi,
            generator=gen,
            cross_fit=False,
            estimators=("rw",),
            max_iter=0,
        )


def test_kfold_splits_rejects_more_folds_than_observations():
    with pytest.raises(ValueError, match="folds must be <= n"):
        list(kfold_splits(3, folds=4, random_state=0))
