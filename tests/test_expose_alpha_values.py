import numpy as np

from genriesz import ATEFunctional, ATTFunctional, SquaredGenerator, grr_functional


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
    return X, Y


def phi(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        d = X[0]
        z = X[1:]
        return np.concatenate([[1.0], [d], z, d * z])
    d = X[:, [0]]
    z = X[:, 1:]
    return np.concatenate([np.ones((len(X), 1)), d, z, d * z], axis=1)


def _fit(*, expose: bool, cross_fit: bool = True, lam: float = 1e-3, m=None):
    X, Y = _make_synthetic_ate(n=240, d=2, seed=3)
    return grr_functional(
        X=X,
        Y=Y,
        m=m if m is not None else ATEFunctional(treatment_index=0),
        basis=phi,
        generator=SquaredGenerator(C=0.0).as_generator(),
        cross_fit=cross_fit,
        folds=4,
        random_state=0,
        riesz_lam=lam,
        outcome_models="none",
        estimators=("rw",),
        expose_alpha_values=expose,
    )


def test_flag_off_keeps_diagnostics_unchanged():
    res = _fit(expose=False)
    assert "alpha_values" not in res.diagnostics
    assert "m_alpha_values" not in res.diagnostics


def test_flag_on_exposes_aligned_out_of_fold_values():
    res_off = _fit(expose=False)
    res_on = _fit(expose=True)

    alpha_values = res_on.diagnostics["alpha_values"]
    m_alpha_values = res_on.diagnostics["m_alpha_values"]
    n = res_on.n
    assert alpha_values.shape == (n,)
    assert m_alpha_values.shape == (n,)
    assert np.all(np.isfinite(alpha_values))
    assert np.all(np.isfinite(m_alpha_values))

    # The exposed vector must be the same object the scalar diagnostics
    # summarize, and exposing it must not perturb anything else.
    assert np.isclose(np.abs(alpha_values).max(), res_on.diagnostics["alpha_abs_max"])
    assert np.isclose(np.abs(alpha_values).mean(), res_on.diagnostics["alpha_abs_mean"])
    for key, val in res_off.diagnostics.items():
        if isinstance(val, float):
            assert np.isclose(val, res_on.diagnostics[key], equal_nan=True), key
    assert {e.estimate for e in res_off.estimates.values()} == {
        e.estimate for e in res_on.estimates.values()
    }


def test_sq_self_balance_identity_in_sample():
    # Without cross-fitting and with a vanishing ridge, the SQ first-order
    # condition gives mean(alpha_hat * phi_j) = mean(m(W, phi_j)) for every j,
    # hence mean(alpha_hat^2) = mean(m(W, alpha_hat)) because alpha_hat lies in
    # the span of phi. This pins down the semantics of both exposed vectors.
    res = _fit(expose=True, cross_fit=False, lam=1e-10)
    alpha_values = res.diagnostics["alpha_values"]
    m_alpha_values = res.diagnostics["m_alpha_values"]
    assert np.isclose(
        np.mean(alpha_values**2), np.mean(m_alpha_values), rtol=1e-6, atol=1e-8
    )


def test_flag_on_works_for_att():
    X, _ = _make_synthetic_ate(n=240, d=2, seed=3)
    pi = float(np.mean(X[:, 0]))
    res = _fit(expose=True, m=ATTFunctional(treatment_index=0, pi=pi, pi_is_estimated=True))
    alpha_values = res.diagnostics["alpha_values"]
    m_alpha_values = res.diagnostics["m_alpha_values"]
    assert alpha_values.shape == (res.n,)
    assert m_alpha_values.shape == (res.n,)
    assert np.all(np.isfinite(alpha_values))
    assert np.all(np.isfinite(m_alpha_values))
