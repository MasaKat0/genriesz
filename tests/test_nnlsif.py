import numpy as np


def test_nn_matching_weights_simple():
    from genriesz.nnlsif import nn_matching_weights

    # Two treated, two controls, 1D covariate. No distance ties.
    # X = [D, Z]
    X = np.array(
        [
            [1, 0.0],
            [1, 10.0],
            [0, 2.0],
            [0, 8.0],
        ],
        dtype=float,
    )

    w = nn_matching_weights(X=X, treatment_index=0, M=1, standardize=False)

    # Basic sanity
    assert w.shape == (4,)
    assert np.all(np.isfinite(w))

    # Matching weights are >= 1 for both groups in this construction.
    assert np.min(w) >= 1.0


def test_local_polynomial_nnlsif_weights_runs():
    from genriesz.nnlsif import local_polynomial_nnlsif_weights

    rng = np.random.default_rng(0)
    n = 200
    Z = rng.normal(size=(n, 2))
    D = (rng.random(n) < 0.5).astype(int)

    X = np.column_stack([D, Z])

    w = local_polynomial_nnlsif_weights(
        X=X,
        treatment_index=0,
        M=10,
        degree=1,
        kernel="knn_ball",
        standardize=True,
    )

    assert w.shape == (n,)
    assert np.all(np.isfinite(w))


def test_local_polynomial_catchment_degree0_matches_matching():
    """Catchment+degree0 should reproduce matching weights (by construction)."""

    from genriesz.nnlsif import local_polynomial_nnlsif_weights, nn_matching_weights

    rng = np.random.default_rng(1)
    n = 300
    Z = rng.normal(size=(n, 3))
    D = (rng.random(n) < 0.4).astype(int)
    X = np.column_stack([D, Z])

    w_match = nn_matching_weights(X=X, treatment_index=0, M=5, standardize=True)
    w_lp0 = local_polynomial_nnlsif_weights(
        X=X,
        treatment_index=0,
        M=5,
        degree=0,
        kernel="catchment",
        standardize=True,
    )

    # Exact equality is not guaranteed due to floating point and standardization,
    # but they should be extremely close.
    np.testing.assert_allclose(w_lp0, w_match, atol=1e-10, rtol=1e-10)
