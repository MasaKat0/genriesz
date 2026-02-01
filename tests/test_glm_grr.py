import numpy as np

from genriesz import GRR, SquaredGenerator, UKLGenerator


def test_squared_link_inverse():
    gen = SquaredGenerator(C=0.5).as_generator()
    X = np.zeros((5, 2))
    alpha = np.linspace(-2, 2, 5)
    v = gen.evaluate_grad(X, alpha)
    alpha2 = gen.evaluate_inv_grad(X, v)
    assert np.allclose(alpha, alpha2)


def _make_synthetic_ate(n: int = 400, d: int = 2, seed: int = 0):
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


def m_ate(x: np.ndarray, gamma):
    z = x[1:]
    x1 = np.concatenate([[1.0], z])
    x0 = np.concatenate([[0.0], z])
    return float(gamma(x1) - gamma(x0))


def phi(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        d = X[0]
        z = X[1:]
        return np.concatenate([[1.0], [d], z, d * z])
    d = X[:, [0]]
    z = X[:, 1:]
    return np.concatenate([np.ones((len(X), 1)), d, z, d * z], axis=1)


def test_grr_runs_sq_and_ukl():
    X, Y, tau = _make_synthetic_ate(n=300, d=2, seed=0)

    sq = SquaredGenerator(C=0.0).as_generator()
    est_sq = GRR(basis=phi, m=m_ate, generator=sq, penalty="l2", lam=1e-3)
    est_sq.fit(X, max_iter=200, tol=1e-9)
    ate_sq = est_sq.estimate_linear_functional(Y, X)
    assert np.isfinite(ate_sq)

    ukl = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1)).as_generator()
    est_ukl = GRR(basis=phi, m=m_ate, generator=ukl, penalty="l2", lam=1e-3)
    est_ukl.fit(X, max_iter=200, tol=1e-9)
    ate_ukl = est_ukl.estimate_linear_functional(Y, X)
    assert np.isfinite(ate_ukl)

    # In small samples we only test that we are in the right ballpark.
    assert abs(ate_ukl - tau) < 1.0


def test_grr_lp_penalty_runs():
    X, Y, _ = _make_synthetic_ate(n=250, d=2, seed=1)
    gen = SquaredGenerator(C=0.0).as_generator()

    est = GRR(basis=phi, m=m_ate, generator=gen, penalty="l1.5", lam=1e-3)
    est.fit(X, max_iter=200, tol=1e-9)
    ate = est.estimate_linear_functional(Y, X)
    assert np.isfinite(ate)
