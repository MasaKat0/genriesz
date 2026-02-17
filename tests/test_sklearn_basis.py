import numpy as np

import pytest


def test_random_forest_leaf_basis_shapes():
    pytest.importorskip("sklearn")

    from sklearn.ensemble import RandomForestClassifier

    from genriesz.sklearn_basis import RandomForestLeafBasis

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=0)
    basis = RandomForestLeafBasis(rf, include_bias=True).fit(X, y)

    Z = basis(X)
    assert Z.ndim == 2
    assert Z.shape[0] == len(X)
    assert Z.shape[1] == basis.n_output_

    z1 = basis(X[0])
    assert z1.ndim == 1
    assert z1.shape[0] == basis.n_output_
