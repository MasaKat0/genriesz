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
    _null_space_resolution,
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


def test_solve_stationarity_catches_the_case_where_numpy_solve_does_not_raise():
    """A numerically singular Gram matrix that ``np.linalg.solve`` still accepts.

    With ``p > n`` the Gram matrix is rank-deficient in exact arithmetic, but
    rounding can leave its smallest eigenvalue at ~1e-16 rather than 0, so LAPACK
    reports no error and returns a ~1e16-norm vector that solves a *perturbed*
    system. Checking the residual only on the ``lstsq`` branch missed this
    entirely, so the guard must not be keyed on whether ``solve`` raised.
    """
    Phi = np.array(
        [
            [-0.98912135, -0.36778665, 1.28792526],
            [0.19397442, 0.92023090, 0.57710379],
        ]
    )
    A = 0.5 * Phi.T @ Phi / 2
    null_dir = np.linalg.svd(Phi, full_matrices=True)[2][-1]

    # np.linalg.solve really does succeed here -- the test is not vacuous.
    huge = np.linalg.solve(A, Phi.mean(axis=0) + null_dir)
    assert np.linalg.norm(huge) > 1e15

    with pytest.raises(np.linalg.LinAlgError, match="unbounded below"):
        solve_stationarity(A, Phi.mean(axis=0) + null_dir)

    # The same matrix with a right-hand side inside its range still solves.
    b_in_range = A @ np.array([1.0, 2.0, -1.0])
    beta = solve_stationarity(A, b_in_range)
    np.testing.assert_allclose(A @ beta, b_in_range, atol=1e-10)


def test_solve_stationarity_criteria_are_scale_invariant():
    # A null-space component of b is a geometric fact, so shrinking b must not
    # let it slip under an absolute tolerance, and rescaling A must not matter.
    # The extreme scales matter: ||b|| itself underflows to 0 below ~1e-154 and
    # overflows to inf above ~1e154, and a criterion keyed on it then compares
    # 0 > 0 or inf > inf -- both false -- and accepts a system with no solution.
    A = np.array([[0.5, 0.0], [0.0, 0.0]])
    b = np.array([1.0, 1.0])
    for a_scale in (1e-6, 1.0, 1e6):
        for b_scale in (1e-300, 1e-9, 1.0, 1e9, 1e300):
            with pytest.raises(np.linalg.LinAlgError):
                solve_stationarity(a_scale * A, b_scale * b)


@pytest.mark.parametrize("a_scale", [1e-150, 1.0, 1e150])
@pytest.mark.parametrize("b_scale", [1e-150, 1.0, 1e150])
def test_solve_stationarity_is_linear_across_extreme_scales(a_scale, b_scale):
    # Rescaling both sides to unit sup-norm must be a pure reparametrization:
    # scaling A or b just scales the solution, all the way out to where the raw
    # norms would have underflowed or overflowed.
    rng = np.random.default_rng(4)
    Phi = rng.normal(size=(200, 10))
    A = 0.5 * (Phi.T @ Phi) / 200 + 1e-3 * np.eye(10)
    reference = solve_stationarity(A, np.ones(10))

    beta = solve_stationarity(A * a_scale, np.ones(10) * b_scale)
    assert np.all(np.isfinite(beta))
    np.testing.assert_allclose(beta / (b_scale / a_scale), reference, rtol=1e-12)


def test_solve_stationarity_refuses_a_solution_float64_cannot_represent():
    # lambda_max ~ 1e-320 against a unit b: the exact solution is ~1e320. The
    # eigen-path division used to overflow and return [inf, nan] as a success.
    with pytest.raises(np.linalg.LinAlgError, match="overflows float64"):
        solve_stationarity(np.diag([1e-320, 0.0]), np.array([1.0, 0.0]))


def test_solve_stationarity_rejects_an_indefinite_matrix():
    # A negative eigenvalue means the quadratic is unbounded below along it, even
    # though the system A beta = b is perfectly solvable.
    with pytest.raises(np.linalg.LinAlgError, match="not positive semi-definite"):
        solve_stationarity(np.diag([1.0, -1.0]), np.array([1.0, 0.0]))


def test_solve_stationarity_symmetrizes_a_matrix_within_the_tolerance():
    """Tolerating BLAS rounding is not the same as solving one system.

    LAPACK reads one triangle and eigh the other, so an A that is asymmetric --
    even by less than the validation tolerance -- has them solve different
    systems. This one is asymmetric by ~5e-11 (inside the tolerance) and singular
    when read as stored; the answer must be a solution of the symmetrized matrix,
    which is the system both routines then see.
    """
    A = np.array([[1.0, 1.0], [1.00000000005, 1.00000000005]])
    A_sym = 0.5 * A + 0.5 * A.T
    b = np.linalg.eigh(A_sym)[1][:, -1]  # in the range of the symmetrized matrix

    beta = solve_stationarity(A, b)
    resid = np.linalg.norm(A_sym @ beta - b) / np.linalg.norm(b)
    assert resid < 1e-12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rcond": -1.0},
        {"rcond": np.nan},
        {"rcond": 1.0},
        {"rtol": 1.0},  # rtol >= 1 would accept a b that is purely null-space
        {"rtol": 1e300},
        {"rtol": np.nan},  # NaN compares false against everything
        {"rtol": 0.0},  # promises exactness float64 cannot deliver
    ],
)
def test_solve_stationarity_rejects_tolerances_that_would_disable_the_check(kwargs):
    with pytest.raises(ValueError, match=r"must be finite and in"):
        solve_stationarity(np.diag([1.0, 0.0]), np.array([0.0, 1.0]), **kwargs)


def test_solve_stationarity_rejects_a_non_symmetric_or_non_finite_system():
    # LAPACK reads one triangle and eigh the other, so a non-symmetric A would
    # have them solve two different systems. This one is singular with b outside
    # its range, and used to come back "solved" from the upper triangle alone.
    with pytest.raises(ValueError, match="symmetric"):
        solve_stationarity(np.array([[1.0, 0.5], [2.0, 1.0]]), np.array([0.0, 1.0]))

    with pytest.raises(ValueError, match="finite"):
        solve_stationarity(np.array([[1.0, np.nan], [np.nan, 1.0]]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        solve_stationarity(np.eye(2), np.array([np.nan, 1.0]))

    with pytest.raises(ValueError, match="square"):
        solve_stationarity(np.ones((2, 3)), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="1D of length"):
        solve_stationarity(np.eye(2), np.array([1.0, 2.0, 3.0]))


def test_solve_stationarity_rejects_a_tiny_but_real_null_component():
    # Unbounded below along e2, just slowly: the objective still has no minimum.
    A = np.diag([1.0, 0.0])
    b = np.array([1.0, 5e-9])
    with pytest.raises(np.linalg.LinAlgError, match="unbounded below"):
        solve_stationarity(A, b)


def test_solve_stationarity_treats_a_numerically_zero_eigenvalue_as_null():
    # An exact solution exists on paper (beta = [0, 2e8, 0]), but only by
    # dividing by an eigenvalue of 1e-16. Below the numerical rank threshold that
    # direction is null, and an unpenalized fit must not return that vector.
    A = np.diag([1.0, 1e-16, 0.0])
    b = np.array([0.0, 2e-8, 0.0])
    with pytest.raises(np.linalg.LinAlgError):
        solve_stationarity(A, b)

    # A ridge lifts the direction above the threshold and the fit succeeds.
    beta = solve_stationarity(A + 1e-3 * np.eye(3), b)
    assert np.all(np.isfinite(beta))


def test_solve_stationarity_rejects_a_matrix_whose_cholesky_pivots_look_healthy():
    """Cholesky pivots do not bound the eigenvalue ratio.

    ``A = L L' / (M^2 + 1)`` with ``L = [[1, 0], [M, 1]]`` has a pivot ratio of
    essentially 1 -- so a pivot-based gate waves it through -- yet its eigenvalue
    ratio is ~1e-10, i.e. singular to working precision. The relative residual of
    the resulting 1e10-norm vector is ~2e-9, small enough to slip past a residual
    check too. Only a real condition estimate (or the eigendecomposition itself)
    catches this.
    """
    scale = 320.0
    L = np.array([[1.0, 0.0], [scale, 1.0]])
    A = (L @ L.T) / (scale**2 + 1.0)
    evals, evecs = np.linalg.eigh(A)

    pivots = np.diag(np.linalg.cholesky(A))
    assert (pivots.min() / pivots.max()) ** 2 > 0.99  # pivots claim it is healthy
    assert evals[0] / evals[-1] < 1e-10  # while it is numerically rank-deficient

    with pytest.raises(np.linalg.LinAlgError, match="unbounded below"):
        solve_stationarity(A, evecs[:, 0])


def test_solve_stationarity_keeps_the_fast_path_for_a_penalized_gram_matrix(monkeypatch):
    # The Cholesky/pocon gate must not send an ordinary ridge-penalized system to
    # the eigendecomposition: that path is ~10x slower and runs on every fit.
    # Breaking eigh is what makes this an assertion about the path taken, not
    # just about the answer.
    rng = np.random.default_rng(3)
    Phi = rng.normal(size=(500, 60))
    A = 0.5 * (Phi.T @ Phi) / 500 + 1e-3 * np.eye(60)
    b = rng.normal(size=60)

    def no_eigh(*args, **kwargs):
        raise AssertionError("the fast path should not have fallen back to eigh")

    monkeypatch.setattr(np.linalg, "eigh", no_eigh)

    beta = solve_stationarity(A, b)
    np.testing.assert_allclose(A @ beta, b, atol=1e-10)


def test_solve_stationarity_fast_path_follows_the_rcond_it_was_given():
    """The gate moves with ``rcond``; it is not a fixed constant.

    ``A = diag(1, 1e-6)`` is well within the default numerical rank, so the fast
    path solves it. Widening the numerical null space to ``rcond=1e-5`` puts the
    1e-6 direction inside it, and ``b`` lies entirely along that direction -- a
    fixed gate would have let the fast path return ``[0, 1e6]`` anyway.
    """
    A = np.diag([1.0, 1e-6])
    b = np.array([0.0, 1.0])

    np.testing.assert_allclose(solve_stationarity(A, b), [0.0, 1e6], rtol=1e-9)

    with pytest.raises(np.linalg.LinAlgError, match="unbounded below"):
        solve_stationarity(A, b, rcond=1e-5)


@pytest.mark.parametrize(
    "A,b,solvable",
    [
        (np.array([[2.0]]), np.array([4.0]), True),  # p = 1
        (np.zeros((2, 2)), np.zeros(2), True),  # A = 0, b = 0 -> beta = 0
        (np.zeros((2, 2)), np.array([1.0, 0.0]), False),  # A = 0, b != 0
    ],
)
def test_solve_stationarity_edge_shapes(A, b, solvable):
    if solvable:
        beta = solve_stationarity(A, b)
        np.testing.assert_allclose(A @ beta, b, atol=1e-12)
    else:
        with pytest.raises(np.linalg.LinAlgError):
            solve_stationarity(A, b)


def test_solve_stationarity_checks_definiteness_even_when_b_is_zero():
    # beta = 0 is a stationary point of an indefinite quadratic, but the quadratic
    # is still unbounded below along the negative eigenvector. Returning zeros for
    # b = 0 used to short-circuit the definiteness check entirely.
    with pytest.raises(np.linalg.LinAlgError, match="not positive semi-definite"):
        solve_stationarity(np.diag([1.0, -1.0]), np.zeros(2))
    with pytest.raises(np.linalg.LinAlgError, match="not positive semi-definite"):
        solve_stationarity(np.array([[-1.0]]), np.zeros(1))

    # A PSD matrix with b = 0 still returns the zero solution.
    np.testing.assert_allclose(solve_stationarity(np.diag([1.0, 0.0]), np.zeros(2)), [0.0, 0.0])


def test_solve_stationarity_reports_an_ambiguous_rank_instead_of_guessing():
    """A retained eigenvalue sitting on the rank threshold cannot be trusted.

    An eigenvector is determined to ~eps/gap, so an eigenvalue ~1e-10 above its
    neighbours carries ~1e-6 of eigenvector error -- enough for leakage from the
    retained subspace to cancel a real null-space component of b, and hand back a
    finite vector whose residual is ~3e-6 while every tolerance says it solved.
    The split has to be declared undecidable rather than guessed.
    """
    Q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    A = Q @ np.diag([1.0, 1.01e-10, 0.0]) @ Q.T
    A = 0.5 * (A + A.T)
    b = 0.76 * Q[:, 0] - 0.61 * Q[:, 1] + 2.965e-6 * Q[:, 2]  # real null component

    with pytest.raises(np.linalg.LinAlgError, match="numerical rank of A is ambiguous"):
        solve_stationarity(A, b)


def test_solve_stationarity_certifies_a_null_component_the_decomposition_resolves():
    """The leak that hides a null-space component is measured, not assumed.

    Both matrices here have spectrum (1, 1e-8, 0), so the worst-case eigenvector
    error ~eps/lambda_min_kept is ~2e-7 for either -- and a floor set from that
    worst case waves through the 1e-7 null-space component of b in both, reporting
    a stationary point for an objective that has none.

    But the two decompositions are not equally accurate. eigh recovers a diagonal
    matrix exactly: its null basis has zero residual, it resolves the component
    perfectly, and the absent solution has to be reported. Only when the
    eigenvectors really are uncertain -- the generic rotation, whose measured leak
    is ~2e-7 -- is the component genuinely indistinguishable from decomposition
    noise, and accepting it means returning the exact solution of a system that far
    away from the one asked about.
    """
    exact = np.diag([1.0, 1e-8, 0.0])
    with pytest.raises(np.linalg.LinAlgError, match="unbounded below"):
        solve_stationarity(exact, np.array([1.0, 0.0, 1e-7]))

    Q, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(3, 3)))
    rotated = Q @ np.diag([1.0, 1e-8, 0.0]) @ Q.T
    rotated = 0.5 * (rotated + rotated.T)
    beta = solve_stationarity(rotated, Q[:, 0] + 1e-7 * Q[:, 2])
    assert np.all(np.isfinite(beta))

    # A component the rotated decomposition *can* resolve is still rejected: the
    # leak buys tolerance, it does not disable the test.
    with pytest.raises(np.linalg.LinAlgError, match="unbounded below"):
        solve_stationarity(rotated, Q[:, 0] + 1e-4 * Q[:, 2])


def test_solve_stationarity_does_not_believe_an_unmeasurably_small_residual():
    """The leak is a bound on the eigenvector error, not a reading of it.

    ``A = T diag(1, 2^-21) T'`` is built from dyadic entries, so ``A @ q`` is
    *exactly* zero for the integer null vector ``q`` -- and the residual of the
    computed null basis, evaluated in the same arithmetic, cancels to ~1e-21 while
    the basis is actually off by an angle of ~1e-10. Reading that residual as the
    eigenvector error understates it by nine orders of magnitude, which lands the
    split wrong in both directions at once: a b outside the range of A is waved
    through with a guarantee it does not meet, and a b that lies exactly *in* the
    range is failed as unbounded below. Bounding the rounding of the residual's own
    evaluation -- the dot products behind ``A @ V`` are only good to ``n eps (|A|
    |V|)`` -- is what keeps the leak an upper bound.
    """
    T = np.array([[-16.0, 12.0], [9.0, -16.0], [-10.0, 13.0]])
    A = T @ np.diag([1.0, 2.0**-21]) @ T.T
    q = np.array([-43.0, 88.0, 148.0])
    assert np.all(A @ q == 0.0)  # exactly, in float64

    # b in range(A) = range(T): consistent, and must not be called unbounded below.
    beta = solve_stationarity(A, T[:, 1])
    assert np.all(np.isfinite(beta))

    # b outside range(A): the null component (~1.2e-10 of ||b||) is real but far
    # below the leak the decomposition can honestly claim (~7e-8), so it is
    # accepted -- and the promise made about it, that some b within 2*leak of this
    # one is consistent, does hold. What must not happen is the old claim of a
    # 1e-13 leak, which this b violates by two orders of magnitude.
    b = np.array([0.5962246454995345, 0.7541753704477557, -0.2752011677099944])
    null_component = abs(q @ b) / (np.linalg.norm(q) * np.linalg.norm(b))
    evals, evecs = np.linalg.eigh(A / np.abs(A).max())
    keep = evals > 1e-10 * evals[-1]
    leak = _null_space_resolution(A / np.abs(A).max(), evals, evecs, keep, float(evals[keep].min()))

    assert null_component <= 2 * leak  # the guarantee the returned beta comes with
    assert np.all(np.isfinite(solve_stationarity(A, b)))


def test_solve_stationarity_refuses_a_solution_that_underflows():
    # scale = b_scale / a_max underflows to 0, so the only representable beta is
    # zero -- which does not solve the system (residual is 100% of b).
    tiny = np.nextafter(0.0, 1.0)  # 5e-324
    with pytest.raises(np.linalg.LinAlgError, match="underflows float64"):
        solve_stationarity(np.array([[1e308]]), np.array([tiny]))


def test_solve_stationarity_solves_a_subnormal_system():
    # Normalizing before symmetrizing matters: 0.5 * A on a subnormal A collapses
    # it to the zero matrix, and a solvable system would be called unbounded.
    tiny = np.nextafter(0.0, 1.0)
    np.testing.assert_allclose(
        solve_stationarity(np.array([[tiny]]), np.array([tiny])), [1.0]
    )
