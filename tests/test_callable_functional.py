import numpy as np

from genriesz import ATEFunctional, CallableFunctional, PolynomialBasis


def test_callable_functional_m_basis_matrix_matches_ate_closed_form():
    """The per-coordinate construction matches the vectorized ATE functional.

    ATEFunctional builds M by ``basis(X1) - basis(X0)`` directly; writing the
    same functional as a black-box callable and letting CallableFunctional probe
    each basis coordinate must give the identical matrix.
    """
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(25, 2))
    D = (rng.random(25) > 0.5).astype(float)
    X = np.column_stack([D, Z])
    basis = PolynomialBasis(degree=2).fit(X)

    def m(x_row, gamma):
        x1 = x_row.copy()
        x1[0] = 1.0
        x0 = x_row.copy()
        x0[0] = 0.0
        return gamma(x1) - gamma(x0)

    M_callable = CallableFunctional(m).m_basis_matrix(X, basis)
    M_ate = ATEFunctional(treatment_index=0).m_basis_matrix(X, basis)

    np.testing.assert_allclose(M_callable, M_ate)


def test_callable_functional_per_row_cache_does_not_leak_across_rows():
    """The per-row basis cache must give each row its own basis vector.

    With ``m(x, gamma) = gamma(x)``, M[i, :] must equal ``basis(X_i)``; a cache
    that leaked between rows would return a stale vector for later rows.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3))
    basis = PolynomialBasis(degree=1).fit(X)

    def m(x_row, gamma):
        return gamma(x_row)

    M = CallableFunctional(m).m_basis_matrix(X, basis)
    np.testing.assert_allclose(M, basis(X))
