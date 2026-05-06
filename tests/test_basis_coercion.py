import numpy as np

from genriesz import PolynomialBasis, SquaredGenerator, TreatmentInteractionBasis, grr_ame, grr_ate


def _make_synthetic_ate(n: int = 200, d_z: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d_z))
    logits = 0.6 * Z[:, 0] - 0.4 * Z[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n)
    tau = 1.0
    mu0 = 0.5 * Z[:, 0] + 0.25 * Z[:, 1] ** 2
    Y = mu0 + tau * D + rng.normal(scale=1.0, size=n)
    X = np.column_stack([D, Z])
    return X, Y


def _make_synthetic_ame(n: int = 200, d: int = 2, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # Smooth outcome; AME wrt coordinate 0 has a well-defined derivative.
    Y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(scale=0.2, size=n)
    return X, Y


def test_grr_ate_accepts_unfitted_basis_objects():
    X, Y = _make_synthetic_ate(n=150, d_z=2, seed=0)

    psi = PolynomialBasis(degree=2, include_bias=True)  # not fit()
    phi = TreatmentInteractionBasis(base_basis=psi, treatment_index=0)  # not fit()

    gen = SquaredGenerator(C=0.0).as_generator()

    res = grr_ate(
        X=X,
        Y=Y,
        basis=phi,
        generator=gen,
        cross_fit=True,
        folds=3,
        random_state=0,
        estimators=("ra", "rw", "arw"),
        outcome_models="shared",
        riesz_penalty="l2",
        riesz_lam=1e-3,
        max_iter=150,
        tol=1e-9,
    )

    assert "rw" in res.estimates
    assert "ra" in res.estimates
    assert "arw" in res.estimates
    assert np.isfinite(res.estimates["rw"].estimate)


def test_grr_ame_accepts_unfitted_polynomial_basis():
    X, Y = _make_synthetic_ame(n=150, d=2, seed=1)

    basis = PolynomialBasis(degree=3, include_bias=True)  # not fit()
    gen = SquaredGenerator(C=0.0).as_generator()

    res = grr_ame(
        X=X,
        Y=Y,
        coordinate=0,
        basis=basis,
        generator=gen,
        cross_fit=True,
        folds=3,
        random_state=0,
        estimators=("ra", "rw", "arw", "tmle"),
        outcome_models="shared",
        riesz_penalty="l2",
        riesz_lam=1e-3,
        max_iter=150,
        tol=1e-9,
    )

    for key in ["rw", "ra", "arw", "tmle"]:
        assert key in res.estimates
        assert np.isfinite(res.estimates[key].estimate)


def test_polynomial_basis_feature_count_and_unique_powers():
    import math

    X = np.zeros((2, 3))
    for degree in range(4):
        basis = PolynomialBasis(degree=degree, include_bias=True).fit(X)
        expected = math.comb(X.shape[1] + degree, degree)
        assert basis.n_features == expected
        assert len({tuple(row) for row in basis._powers}) == expected


class _TreatmentTargetBasis:
    def __init__(self):
        self.y_seen = None
        self._n_features = 1

    def fit(self, X, y=None):
        self.y_seen = None if y is None else np.asarray(y).copy()
        return self

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        return np.ones((X.shape[0], 1), dtype=float)

    @property
    def n_features(self):
        return self._n_features

    def copy(self):
        return _TreatmentTargetBasis()


def test_treatment_interaction_basis_uses_treatment_for_base_basis_fit():
    X, _ = _make_synthetic_ate(n=20, d_z=2, seed=3)
    base = _TreatmentTargetBasis()
    basis = TreatmentInteractionBasis(base_basis=base, treatment_index=0).fit(X)
    assert np.array_equal(basis.base_basis.y_seen, X[:, 0])
