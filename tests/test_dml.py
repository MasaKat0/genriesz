import numpy as np

from genriesz import ATEFunctional, SquaredGenerator, grr_functional


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
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        d = X[0]
        z = X[1:]
        return np.concatenate([[1.0], [d], z, d * z])
    d = X[:, [0]]
    z = X[:, 1:]
    return np.concatenate([np.ones((len(X), 1)), d, z, d * z], axis=1)


def test_grr_functional_dm_ipw_aipw_run():
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
        estimators=("dm", "ipw", "aipw"),
        outcome_models="shared",
        riesz_penalty="l2",
        riesz_lam=1e-3,
        max_iter=200,
        tol=1e-9,
    )

    assert "ipw" in res.estimates
    assert "dm_shared" in res.estimates
    assert "aipw_shared" in res.estimates

    for k in ["ipw", "dm_shared", "aipw_shared"]:
        assert np.isfinite(res.estimates[k].theta)
        assert np.isfinite(res.estimates[k].stderr)

    s = res.summary_text()
    assert isinstance(s, str)
    assert "ipw" in s.lower()
