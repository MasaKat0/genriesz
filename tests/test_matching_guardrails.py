import numpy as np
import pytest

from genriesz import PolynomialBasis, SquaredGenerator, grr_ate, grr_att, grr_did


def _toy_data(seed: int = 0, n: int = 200):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-Z[:, 0]))
    D = rng.binomial(1, e, size=n).astype(float)
    # Simple outcome with a constant effect
    Y0 = Z[:, 0] + rng.normal(scale=0.5, size=n)
    Y1 = Y0 + 1.0 + rng.normal(scale=0.2, size=n)
    Y = Y0 + D * 1.0 + rng.normal(scale=0.2, size=n)
    X = np.column_stack([D, Z])
    return X, Y, Y0, Y1


def test_matching_riesz_method_is_ate_only_and_no_tmle():
    X, Y, Y0, Y1 = _toy_data()
    basis = PolynomialBasis(degree=1, include_bias=True)

    # ATE should run with matching weights, but only without cross-fitting.
    res = grr_ate(
        X=X,
        Y=Y,
        basis=basis,
        generator=SquaredGenerator(C=0.0).as_generator(),  # ignored by matching methods
        riesz_method="nn_matching",
        cross_fit=False,
        estimators=("rw",),
    )
    assert "rw" in res.estimates

    # TMLE is not supported for matching-based representers.
    with pytest.raises(ValueError):
        grr_ate(
            X=X,
            Y=Y,
            basis=basis,
            generator=SquaredGenerator(C=0.0).as_generator(),
            riesz_method="nn_matching",
            cross_fit=False,
            estimators=("tmle",),
        )

    # ATT/DID should raise (guard rails against silent misuse).
    with pytest.raises(ValueError):
        grr_att(
            X=X,
            Y=Y,
            basis=basis,
            generator=SquaredGenerator(C=0.0).as_generator(),
            riesz_method="nn_matching",
            cross_fit=False,
            estimators=("rw",),
        )

    with pytest.raises(ValueError):
        grr_did(
            X=X,
            Y0=Y0,
            Y1=Y1,
            basis=basis,
            generator=SquaredGenerator(C=0.0).as_generator(),
            riesz_method="nn_matching",
            cross_fit=False,
            estimators=("rw",),
        )


def test_treatment_functionals_reject_nonbinary_treatment_early():
    X, Y, _, _ = _toy_data()
    X = X.copy()
    X[:, 0] = np.linspace(0.1, 0.9, len(X))

    with pytest.raises(ValueError, match="binary"):
        grr_ate(
            X=X,
            Y=Y,
            basis=PolynomialBasis(degree=1, include_bias=True),
            generator=SquaredGenerator(C=0.0).as_generator(),
            cross_fit=False,
            estimators=("rw",),
        )
