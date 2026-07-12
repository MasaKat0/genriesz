"""Tests for the helpers consolidated into ``genriesz.utils`` (design item X).

Two things are pinned here: the behaviour of each shared helper, and the absence
of the per-module private copies that used to shadow them.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest

import genriesz
from genriesz.utils import (
    as_1d_of_length,
    as_2d,
    kfold_splits,
    sigmoid,
    solve_stationarity,
    standardize_columns,
)

# Modules that used to carry private copies of the shared helpers, mapped to the
# names they must no longer define. This is a name check, not a clone detector:
# it catches a revert or a re-import under the old name, but a duplicate spelled
# `_ensure_2d` would slip past it.
_CONSOLIDATED = {
    "genriesz.glm": ("_as_2d", "_as_1d", "_sigmoid"),
    "genriesz.functionals": ("_as_2d",),
    "genriesz.estimation": ("_as_2d", "_as_1d", "_expit"),
    "genriesz.density_ratio": ("_as_2d", "_sigmoid"),
    "genriesz.matching": ("_as_2d",),
    "genriesz.generators": ("_as_2d", "_as_1d"),
    "genriesz.basis": ("_as_2d",),
}


@pytest.mark.parametrize("module_name,names", sorted(_CONSOLIDATED.items()))
def test_modules_do_not_redefine_the_old_helper_names(module_name, names):
    module = importlib.import_module(module_name)
    for name in names:
        assert not hasattr(module, name), (
            f"{module_name} defines {name}; it should use the shared helper in genriesz.utils"
        )


# ---------------------------------------------------------------------------
# as_2d / as_1d_of_length
# ---------------------------------------------------------------------------


def test_as_2d_accepts_2d_and_casts_to_float():
    out = as_2d([[1, 2], [3, 4]])
    assert out.dtype == np.float64
    assert out.shape == (2, 2)


@pytest.mark.parametrize("bad", [np.zeros(3), np.zeros((2, 2, 2))])
def test_as_2d_rejects_wrong_ndim_and_names_the_argument(bad):
    with pytest.raises(ValueError, match=r"centers must be a 2D array"):
        as_2d(bad, name="centers")


def test_as_2d_defaults_the_name_to_X():
    with pytest.raises(ValueError, match=r"^X must be a 2D array"):
        as_2d(np.zeros(3))


def test_as_1d_of_length_flattens_and_checks_length():
    out = as_1d_of_length([[1.0], [2.0], [3.0]], n=3, name="v")
    assert out.shape == (3,)

    with pytest.raises(ValueError, match=r"alpha must have length 3"):
        as_1d_of_length([1.0, 2.0], n=3, name="alpha")


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------


def test_sigmoid_matches_the_naive_formula_in_the_safe_range():
    z = np.linspace(-30.0, 30.0, 401)
    naive = 1.0 / (1.0 + np.exp(-z))
    np.testing.assert_allclose(sigmoid(z), naive, rtol=0, atol=1e-15)


def test_sigmoid_is_stable_where_the_naive_formula_overflows():
    z = np.array([-800.0, -50.0, 0.0, 50.0, 800.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow/underflow becomes a failure
        out = sigmoid(z)

    assert np.all(np.isfinite(out))
    assert out[0] == 0.0
    assert out[2] == 0.5
    assert out[-1] == 1.0

    # The branch-free formula really does overflow on this input, so the test
    # above is not vacuous.
    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError):
            1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# standardize_columns
# ---------------------------------------------------------------------------


def test_standardize_columns_centers_and_scales():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[3.0, -2.0], scale=[5.0, 0.5], size=(200, 2))
    Xs, mean, std = standardize_columns(X)

    np.testing.assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Xs.std(axis=0, ddof=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(mean, X.mean(axis=0))
    np.testing.assert_allclose(std, X.std(axis=0, ddof=0))
    # The returned mean/std reproduce the transform on fresh data.
    np.testing.assert_allclose((X - mean) / std, Xs)


def test_standardize_columns_leaves_a_constant_column_at_zero():
    X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
    Xs, _, std = standardize_columns(X)

    assert std[0] == 1.0  # zero spread is not divided by
    np.testing.assert_allclose(Xs[:, 0], 0.0)
    assert np.all(np.isfinite(Xs))


def test_standardize_columns_disabled_is_the_identity():
    X = np.array([[1.0, 5.0], [3.0, 6.0]])
    Xs, mean, std = standardize_columns(X, enabled=False)

    np.testing.assert_array_equal(Xs, X)
    np.testing.assert_array_equal(mean, np.zeros(2))
    np.testing.assert_array_equal(std, np.ones(2))


def test_standardize_columns_propagates_nan_rather_than_masking_it():
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 7.0]])
    Xs, _, _ = standardize_columns(X)

    assert np.all(np.isnan(Xs[:, 0]))
    assert np.all(np.isfinite(Xs[:, 1]))


# ---------------------------------------------------------------------------
# crossfit_splits delegates to kfold_splits
# ---------------------------------------------------------------------------


def _crossfit_splits_before_consolidation(n, *, n_folds, seed):
    """The implementation `crossfit_splits` had before it delegated to `kfold_splits`.

    Comparing against this -- rather than against the delegate -- is what makes the
    test independent of `kfold_splits`'s defaults and internals.
    """

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    splits = []
    for fold in np.array_split(indices, n_folds):
        test = np.asarray(fold, dtype=int)
        splits.append((np.setdiff1d(indices, test, assume_unique=True), test))
    return splits


@pytest.mark.parametrize("n,folds,seed", [(17, 4, 7), (10, 3, 0), (101, 5, 42), (4, 2, 1)])
def test_crossfit_splits_still_produces_the_pre_consolidation_splits(n, folds, seed):
    smr = genriesz.load_scorematchingriesz()
    got = smr.crossfit_splits(n, n_folds=folds, seed=seed)
    want = _crossfit_splits_before_consolidation(n, n_folds=folds, seed=seed)

    assert len(got) == len(want)
    for (train_got, test_got), (train_want, test_want) in zip(got, want, strict=True):
        np.testing.assert_array_equal(train_got, train_want)
        np.testing.assert_array_equal(test_got, test_want)
        assert train_got.dtype == train_want.dtype
        assert test_got.dtype == test_want.dtype


@pytest.mark.parametrize("n,folds,seed", [(17, 4, 7), (10, 3, 0)])
def test_crossfit_splits_delegates_to_kfold_splits(n, folds, seed):
    smr = genriesz.load_scorematchingriesz()
    got = smr.crossfit_splits(n, n_folds=folds, seed=seed)
    folds_iter = kfold_splits(n, folds=folds, random_state=seed, shuffle=True)
    want = [(f.train, f.test) for f in folds_iter]

    for (train_got, test_got), (train_want, test_want) in zip(got, want, strict=True):
        np.testing.assert_array_equal(train_got, train_want)
        np.testing.assert_array_equal(test_got, test_want)


@pytest.mark.parametrize(
    "n,folds,message",
    [
        (1, 2, "n must be at least 2."),
        (10, 1, "n_folds must be at least 2."),
        (3, 4, "n_folds must be at most n."),
    ],
)
def test_crossfit_splits_errors_name_the_argument_the_caller_passed(n, folds, message):
    # kfold_splits speaks of `folds`; this public wrapper takes `n_folds`, so it
    # validates its own bounds rather than letting the delegate's message leak.
    smr = genriesz.load_scorematchingriesz()
    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        smr.crossfit_splits(n, n_folds=folds)


# ---------------------------------------------------------------------------
# solve_stationarity: a singular quadratic either has minimizers or none at all.
# ---------------------------------------------------------------------------

def test_solve_stationarity_uses_the_direct_solve_when_nonsingular():
    A = np.array([[2.0, 0.0], [0.0, 4.0]])
    b = np.array([2.0, 4.0])
    np.testing.assert_allclose(solve_stationarity(A, b), [1.0, 1.0])


def test_solve_stationarity_returns_the_minimum_norm_solution_when_solvable():
    # A is singular but b is in its range: the minimizers form a line, and the
    # minimum-norm one is a genuine solution of A beta = b.
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    b = np.array([2.0, 2.0])
    beta = solve_stationarity(A, b)
    np.testing.assert_allclose(A @ beta, b, atol=1e-12)
    np.testing.assert_allclose(beta, [1.0, 1.0])  # min-norm among beta0+beta1=2


def test_solve_stationarity_raises_when_b_leaves_the_range_of_A():
    # No stationary point: the quadratic runs to -inf along the null space of A.
    # lstsq would return the finite, non-solving vector [2, 0] instead.
    A = np.array([[0.5, 0.0], [0.0, 0.0]])
    b = np.array([1.0, 1.0])
    with pytest.raises(np.linalg.LinAlgError, match="unbounded below"):
        solve_stationarity(A, b)
