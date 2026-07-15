"""Input contracts of the built-in functionals and the estimation entry point.

Covers the shape contract of ``m_from_predictor`` (a ``(n, 1)`` prediction used
to broadcast against the ``(n,)`` treatment vector and silently produce an
``(n, n)`` matrix for ATT/DID), the ``treatment_index`` / ``pi`` argument
validation, and the entry-point checks of ``grr_functional``.
"""

from __future__ import annotations

import numpy as np
import pytest

from genriesz import PolynomialBasis, grr_ate
from genriesz.functionals import ATEFunctional, ATTFunctional, DIDFunctional


def _make_x(n: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    D = rng.integers(0, 2, size=n).astype(float)
    Z = rng.normal(size=n)
    return np.column_stack([D, Z])


# ---------------------------------------------------------------------------
# m_from_predictor shape contract
# ---------------------------------------------------------------------------


def test_att_m_from_predictor_accepts_column_predictions():
    """A (n, 1) prediction (sklearn-style) must give the same (n,) result as (n,)."""

    X = _make_x()
    m = ATTFunctional(treatment_index=0, pi=0.5)

    def predict_flat(Z: np.ndarray) -> np.ndarray:
        return 2.0 * Z[:, 1]

    def predict_column(Z: np.ndarray) -> np.ndarray:
        return (2.0 * Z[:, 1]).reshape(-1, 1)

    out_flat = m.m_from_predictor(X, predict_flat)
    out_col = m.m_from_predictor(X, predict_column)

    assert out_flat.shape == (X.shape[0],)
    assert out_col.shape == (X.shape[0],)  # used to broadcast to (n, n)
    np.testing.assert_allclose(out_col, out_flat)


def test_ate_m_from_predictor_accepts_column_predictions():
    X = _make_x()
    m = ATEFunctional(treatment_index=0)

    def predict_column(Z: np.ndarray) -> np.ndarray:
        return (Z[:, 0] + Z[:, 1]).reshape(-1, 1)

    out = m.m_from_predictor(X, predict_column)
    assert out.shape == (X.shape[0],)
    np.testing.assert_allclose(out, 1.0)  # gamma = D + Z, so gamma(1,.) - gamma(0,.) = 1


@pytest.mark.parametrize(
    "bad_shape",
    [
        lambda mu: mu.reshape(1, -1),  # (1, n): transposed
        lambda mu: np.column_stack([mu, mu]),  # (n, 2): multi-column
        lambda mu: mu[:-1],  # (n-1,): wrong length
        lambda mu: np.array(float(mu[0])),  # scalar
    ],
)
def test_m_from_predictor_rejects_ambiguous_prediction_shapes(bad_shape):
    X = _make_x()
    for m in (ATEFunctional(treatment_index=0), ATTFunctional(treatment_index=0, pi=0.5)):
        with pytest.raises(ValueError, match="must return an array of shape"):
            m.m_from_predictor(X, lambda Z: bad_shape(Z[:, 1] * 1.0))


# ---------------------------------------------------------------------------
# treatment_index / pi validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [ATEFunctional, ATTFunctional, DIDFunctional])
def test_negative_treatment_index_is_rejected(cls):
    kwargs = {} if cls is ATEFunctional else {"pi": 0.5}
    with pytest.raises(ValueError, match="non-negative"):
        cls(treatment_index=-1, **kwargs)


def test_out_of_range_treatment_index_is_a_clear_error():
    X = _make_x()  # 2 columns
    m = ATEFunctional(treatment_index=5)
    with pytest.raises(ValueError, match="out of range"):
        m.m_from_predictor(X, lambda Z: Z[:, 1] * 1.0)
    m_att = ATTFunctional(treatment_index=5, pi=0.5)
    with pytest.raises(ValueError, match="out of range"):
        m_att.m_from_predictor(X, lambda Z: Z[:, 1] * 1.0)


@pytest.mark.parametrize("pi", [0.0, -0.5, 1.2, float("nan"), float("inf")])
def test_att_rejects_pi_outside_the_probability_range(pi):
    with pytest.raises(ValueError, match="pi"):
        ATTFunctional(treatment_index=0, pi=pi)


def test_att_estimated_pi_of_one_is_rejected():
    # pi = 1 as a *known* value is a (degenerate but explicit) choice; as a
    # sample mean it means "no controls", which cannot identify the ATT.
    ATTFunctional(treatment_index=0, pi=1.0, pi_is_estimated=False)
    with pytest.raises(ValueError, match="no control observations"):
        ATTFunctional(treatment_index=0, pi=1.0, pi_is_estimated=True)


# ---------------------------------------------------------------------------
# grr_functional entry-point validation
# ---------------------------------------------------------------------------


def _small_ate_data(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    D = rng.integers(0, 2, size=n).astype(float)
    Z = rng.normal(size=n)
    X = np.column_stack([D, Z])
    Y = D + Z + rng.normal(size=n)
    return X, Y


def test_empty_estimators_tuple_is_rejected():
    X, Y = _small_ate_data()
    with pytest.raises(ValueError, match="at least one of"):
        grr_ate(X=X, Y=Y, basis=PolynomialBasis(degree=2), generator="sq", estimators=())


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_invalid_significance_level_fails_before_fitting(alpha):
    X, Y = _small_ate_data()
    with pytest.raises(ValueError, match=r"alpha \(significance level\)"):
        grr_ate(X=X, Y=Y, basis=PolynomialBasis(degree=2), generator="sq", alpha=alpha)


def test_non_finite_null_fails_before_fitting():
    X, Y = _small_ate_data()
    with pytest.raises(ValueError, match="null must be finite"):
        grr_ate(X=X, Y=Y, basis=PolynomialBasis(degree=2), generator="sq", null=float("nan"))
