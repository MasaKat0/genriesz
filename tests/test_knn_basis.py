import numpy as np


def test_knn_catchment_basis_shapes_and_row_sums():
    from genriesz import KNNCatchmentBasis

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(20, 3))
    queries = rng.normal(size=(7, 3))

    k = 3
    basis = KNNCatchmentBasis(n_neighbors=k, include_bias=False).fit(centers)
    Phi = basis(queries)

    assert Phi.shape == (len(queries), len(centers))
    # Each row should have exactly k ones (up to ties, but in continuous data ties are unlikely).
    row_sums = Phi.sum(axis=1)
    assert np.all(row_sums == k)

    # Single-row call returns 1D array.
    v = basis(queries[0])
    assert v.ndim == 1
    assert v.shape[0] == len(centers)


def test_knn_catchment_basis_include_bias():
    from genriesz import KNNCatchmentBasis

    rng = np.random.default_rng(1)
    centers = rng.normal(size=(10, 2))
    queries = rng.normal(size=(4, 2))

    basis = KNNCatchmentBasis(n_neighbors=1, include_bias=True).fit(centers)
    Phi = basis(queries)
    assert Phi.shape == (len(queries), len(centers) + 1)
    assert np.all(Phi[:, -1] == 1.0)