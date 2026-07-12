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


def test_grr_ame_binary_outcome_keeps_logit_tmle():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(180, 2))
    p = 1.0 / (1.0 + np.exp(-(0.7 * X[:, 0] - 0.4 * X[:, 1])))
    Y = rng.binomial(1, p, size=len(X)).astype(float)

    res = grr_ame(
        X=X,
        Y=Y,
        coordinate=0,
        basis=PolynomialBasis(degree=2, include_bias=True),
        generator=SquaredGenerator(C=0.0).as_generator(),
        cross_fit=True,
        folds=3,
        random_state=0,
        estimators=("tmle",),
        outcome_models="shared",
        riesz_penalty="l2",
        riesz_lam=1e-3,
        max_iter=150,
        tol=1e-9,
    )

    assert "tmle" in res.estimates
    assert np.isfinite(res.estimates["tmle"].estimate)
    assert np.isfinite(res.estimates["tmle"].se)


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


def test_unfitted_data_dependent_bases_raise_instead_of_leaking():
    import pytest

    from genriesz import (
        GaussianRKHSBasis,
        RBFNystromBasis,
        RBFRandomFourierBasis,
    )

    rng = np.random.default_rng(5)
    X_eval = rng.normal(size=(10, 2))

    for basis in [
        GaussianRKHSBasis(n_centers=5, random_state=0),
        RBFRandomFourierBasis(n_features=8, random_state=0),
        RBFNystromBasis(n_centers=5, random_state=0),
        TreatmentInteractionBasis(
            base_basis=PolynomialBasis(degree=1), treatment_index=0
        ),
    ]:
        with pytest.raises(RuntimeError, match="fit"):
            basis(X_eval)


def test_kernel_bases_reject_a_column_count_that_differs_from_fit():
    """A (n, 1) input against a d-column fit must raise, not broadcast.

    ``(X - mean) / std`` broadcasts a single-column X against a d-entry mean, so
    these bases used to return a (n, d) feature matrix built from nonsense
    instead of failing. PolynomialBasis already checked the column count; the
    kernel bases now match it.
    """
    import pytest

    from genriesz import (
        GaussianRKHSBasis,
        KNNCatchmentBasis,
        RBFNystromBasis,
        RBFRandomFourierBasis,
    )

    rng = np.random.default_rng(11)
    X_fit = rng.normal(size=(30, 3))
    X_bad = rng.normal(size=(10, 1))  # broadcasts against a 3-column fit
    X_wide = rng.normal(size=(10, 4))  # never broadcast, but must raise the same way

    for basis in [
        GaussianRKHSBasis(n_centers=8, sigma=1.0, random_state=0).fit(X_fit),
        RBFRandomFourierBasis(n_features=8, sigma=1.0, random_state=0).fit(X_fit),
        RBFNystromBasis(n_centers=8, sigma=1.0, random_state=0).fit(X_fit),
        KNNCatchmentBasis(n_neighbors=2).fit(X_fit),
    ]:
        for X_wrong in (X_bad, X_wide):
            with pytest.raises(ValueError, match="column"):
                basis(X_wrong)

    rff = RBFRandomFourierBasis(n_features=8, sigma=1.0, random_state=0).fit(X_fit)
    with pytest.raises(ValueError, match="column"):
        rff.derivative(X_bad, 0)

    rkhs = GaussianRKHSBasis(n_centers=8, sigma=1.0, random_state=0).fit(X_fit)
    with pytest.raises(ValueError, match="column"):
        rkhs.diagnostics(X_bad)

    # The fitted column count still evaluates normally.
    assert rkhs(rng.normal(size=(5, 3))).shape == (5, rkhs.n_features)


def test_unfitted_polynomial_basis_raises_by_default():
    """Default PolynomialBasis refuses an unfitted call/derivative.

    PolynomialBasis does not leak (``fit`` reads only the column count), but it
    still raises by default so that every basis honours the same contract.
    """
    import pytest

    rng = np.random.default_rng(6)
    X_eval = rng.normal(size=(10, 2))

    basis = PolynomialBasis(degree=2, include_bias=True)  # not fit(), auto_fit=False
    with pytest.raises(RuntimeError, match="fit"):
        basis(X_eval)
    with pytest.raises(RuntimeError, match="fit"):
        basis.derivative(X_eval, 0)


def test_polynomial_basis_auto_fit_opt_in_matches_explicit_fit():
    """auto_fit=True restores the old fit-on-the-fly behaviour, leak-free."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 3))

    explicit = PolynomialBasis(degree=2, include_bias=True).fit(X)
    on_the_fly = PolynomialBasis(degree=2, include_bias=True, auto_fit=True)

    # No prior fit(): the first call fits from X, and the derivative also works.
    np.testing.assert_allclose(on_the_fly(X), explicit(X))
    np.testing.assert_allclose(
        on_the_fly.derivative(X, 0), explicit.derivative(X, 0)
    )

    # derivative-first path: a fresh, unfitted auto_fit basis must also fit when
    # derivative() is the very first call (its own guard triggers the fit).
    deriv_first = PolynomialBasis(degree=2, include_bias=True, auto_fit=True)
    np.testing.assert_allclose(
        deriv_first.derivative(X, 0), explicit.derivative(X, 0)
    )
