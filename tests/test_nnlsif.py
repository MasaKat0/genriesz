import numpy as np

from genriesz import (
    local_polynomial_nn_lsif_inverse_propensity_weights,
    nn_matching_inverse_propensity_weights,
)


def test_nn_matching_inverse_propensity_weights_simple():
    # Two treated, two controls, 1D covariate. No distance ties.
    Z = np.array([[0.0], [10.0], [2.0], [8.0]], dtype=float)
    D = np.array([1, 1, 0, 0], dtype=int)

    out = nn_matching_inverse_propensity_weights(X=Z, D=D, M=1, standardize=False)
    w = out.w

    assert w.shape == (4,)
    assert np.all(np.isfinite(w))

    # Matching weights are >= 1 for both groups in this construction.
    assert np.min(w) >= 1.0


def test_local_polynomial_nn_lsif_inverse_propensity_weights_runs():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    D = (rng.random(n) < 0.5).astype(int)

    out = local_polynomial_nn_lsif_inverse_propensity_weights(
        X=X,
        D=D,
        M=10,
        degree=1,
        kernel="ball",
        standardize=True,
    )

    assert out.w.shape == (n,)
    assert np.all(np.isfinite(out.w))


def test_local_polynomial_nn_lsif_inverse_propensity_weight_scale():
    rng = np.random.default_rng(1)
    n = 600
    X = rng.normal(size=(n, 2))
    D = rng.binomial(1, 0.5, size=n).astype(int)

    out = local_polynomial_nn_lsif_inverse_propensity_weights(
        X=X,
        D=D,
        M=40,
        degree=1,
        kernel="ball",
        standardize=True,
        clip_min=None,
        ridge=1e-8,
    )

    pi1 = float(np.mean(D))
    pi0 = 1.0 - pi1
    assert np.isclose(float(np.mean(out.w1)), 1.0 / pi1, rtol=0.35)
    assert np.isclose(float(np.mean(out.w0)), 1.0 / pi0, rtol=0.35)
