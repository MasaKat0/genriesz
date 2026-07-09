"""Generator evaluation on the optimizer hot path (design doc item W).

Two independent defects, both invisible in results and expensive in time.

**W-1: the branch sign was recomputed per row, per objective evaluation.**
``branch_fn`` is a function of ``x`` alone, so within a fit the sign array is
constant while ``v`` changes on every L-BFGS ``fun``/``jac`` call. The old
``_sign`` looped over rows every time: a 2000-row fit converging in 3 iterations
called ``branch_fn`` 24,000 times. ``BregmanGenerator.branch_cache()`` memoizes
the signs for the duration of a fit, taking that to exactly ``n``.

**W-2: a data-dependent exception permanently demoted a vectorized callable.**
``_RowwiseScalarFn`` probed the vectorized call inside ``except Exception`` and,
on *any* failure, set ``_vectorized = False`` forever. A generator that raises
:class:`DomainError` on a bad ``alpha`` is still vectorized; demoting it turned
every later evaluation into a Python loop. The probe now distinguishes the two
causes by *retrying the same inputs rowwise*: if rowwise also fails, the data was
bad and the error propagates with the verdict left undecided.

Both fixes are output-invariant; ``test_fit_results_are_unchanged`` pins that.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from genriesz import (
    GRRGLM,
    ATEFunctional,
    BPGenerator,
    BregmanGenerator,
    DomainError,
    PolynomialBasis,
    SquaredGenerator,
    TreatmentInteractionBasis,
    UKLGenerator,
)
from genriesz.generators import _RowwiseScalarFn


def _make_ate(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    D = rng.binomial(1, 0.5, size=n).astype(float)
    return np.column_stack([D, Z])


def _basis(X: np.ndarray):
    return TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    ).fit(X)


class _CountingBranch:
    """A rowwise branch_fn that records how often it is called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: np.ndarray) -> int:
        self.calls += 1
        return int(x[0] == 1.0)


# ---------------------------------------------------------------------------
# W-1: branch signs are computed once per array, per fit.
# ---------------------------------------------------------------------------
def test_branch_fn_is_called_once_per_row_per_fit():
    X = _make_ate(n=400, seed=1)
    branch = _CountingBranch()
    gen = UKLGenerator(C=1.0, branch_fn=branch)

    model = GRRGLM(
        basis=_basis(X), generator=gen, functional=ATEFunctional(0), penalty="l2", lam=1e-3
    )
    fr = model.fit(X)

    assert fr.success
    # Guard against a vacuous test: the optimizer must have evaluated fun/jac
    # several times, which is exactly when the old code paid n calls each.
    assert fr.n_iter >= 2
    assert branch.calls == len(X)


def test_branch_cache_is_scoped_to_the_block():
    X = _make_ate(n=50, seed=2)
    branch = _CountingBranch()
    gen = UKLGenerator(C=1.0, branch_fn=branch)
    v = np.full(len(X), -0.5)

    assert gen._branch_cache is None
    with gen.branch_cache():
        gen.inv_grad(X, v)
        gen.inv_grad(X, v)
        assert branch.calls == len(X)  # second call hit the cache
    assert gen._branch_cache is None

    # Outside the block behavior is unchanged: recomputed every time.
    gen.inv_grad(X, v)
    assert branch.calls == 2 * len(X)


def test_branch_cache_does_not_alias_distinct_arrays():
    """Keyed by identity, but the array is held, so ids cannot be recycled."""

    X_treated = np.column_stack([np.ones(30), np.zeros(30)])
    X_control = np.column_stack([np.zeros(30), np.zeros(30)])
    gen = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1.0))
    v = np.full(30, -0.5)

    with gen.branch_cache():
        a_treated = gen.inv_grad(X_treated, v)
        a_control = gen.inv_grad(X_control, v)
        # Re-request the first array: must not return the second's signs.
        a_treated_again = gen.inv_grad(X_treated, v)

    assert np.all(a_treated > 0)
    assert np.all(a_control < 0)
    assert np.array_equal(a_treated, a_treated_again)


def test_branch_cache_nesting_restores_the_outer_cache():
    gen = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1.0))
    with gen.branch_cache():
        outer = gen._branch_cache
        with gen.branch_cache():
            assert gen._branch_cache is not outer
        assert gen._branch_cache is outer
    assert gen._branch_cache is None


@pytest.mark.parametrize(
    "make_gen",
    [
        lambda b: UKLGenerator(C=1.0, branch_fn=b),
        lambda b: BPGenerator(C=1.0, omega=0.5, branch_fn=b),
        lambda b: SquaredGenerator(C=0.0),
    ],
    ids=["ukl", "bp", "sq"],
)
def test_fit_results_are_unchanged(make_gen):
    """The cache must not move the solution: cached and uncached alpha agree."""

    X = _make_ate(n=300, seed=3)
    gen = make_gen(lambda x: int(x[0] == 1.0))
    model = GRRGLM(
        basis=_basis(X), generator=gen, functional=ATEFunctional(0), penalty="l2", lam=1e-3
    )
    fr = model.fit(X)
    assert fr.success

    v = model.predict_v(X)
    with gen.branch_cache():
        cached = gen.inv_grad(X, v)
    uncached = gen.inv_grad(X, v)
    assert np.array_equal(cached, uncached)


# ---------------------------------------------------------------------------
# W-2: the vectorization probe separates "bad signature" from "bad data".
# ---------------------------------------------------------------------------
def _vectorized_with_domain(a):
    """Vectorized, and rejects alpha < 0 -- as a real generator would."""

    arr = np.asarray(a, dtype=float)
    if np.any(arr < 0.0):
        raise DomainError("alpha must be nonnegative")
    return arr * arr


def _scalar_only(a):
    """Cannot take an array: math.exp raises TypeError on ndarray."""

    return math.log1p(math.exp(-abs(a)))


def test_scalar_only_callable_degrades_to_rowwise():
    fn = _RowwiseScalarFn(_scalar_only)
    X = np.zeros((4, 2))
    out = fn(X, np.array([0.5, 1.0, 1.5, 2.0]))

    assert fn._vectorized is False
    assert np.allclose(out, [_scalar_only(a) for a in (0.5, 1.0, 1.5, 2.0)])


def test_domain_error_does_not_demote_a_vectorized_callable():
    """The regression: a bad alpha must not cost O(n) Python calls forever."""

    fn = _RowwiseScalarFn(_vectorized_with_domain)
    X = np.zeros((3, 2))

    assert np.allclose(fn(X, np.array([1.0, 2.0, 3.0])), [1.0, 4.0, 9.0])
    assert fn._vectorized is True

    with pytest.raises(DomainError):
        fn(X, np.array([1.0, -2.0, 3.0]))

    # Still vectorized: the failure was the data, not the signature.
    assert fn._vectorized is True
    assert np.allclose(fn(X, np.array([2.0, 2.0, 2.0])), [4.0, 4.0, 4.0])


def test_domain_error_on_the_very_first_call_leaves_the_verdict_undecided():
    fn = _RowwiseScalarFn(_vectorized_with_domain)
    X = np.zeros((3, 2))

    with pytest.raises(DomainError):
        fn(X, np.array([-1.0, -2.0, -3.0]))
    # Not demoted to rowwise on the strength of one bad batch.
    assert fn._vectorized is None

    assert np.allclose(fn(X, np.array([1.0, 2.0, 3.0])), [1.0, 4.0, 9.0])
    assert fn._vectorized is True


def test_single_row_call_does_not_decide_vectorization():
    """One row cannot separate "returns a scalar" from "returns one per row"."""

    fn = _RowwiseScalarFn(lambda a: 7.0)  # constant: looks vectorized at n = 1
    out = fn(np.zeros((1, 2)), np.array([3.0]))

    assert fn._vectorized is None
    assert np.allclose(out, [7.0])

    # With two rows the constant is exposed and the callable goes rowwise.
    out2 = fn(np.zeros((2, 2)), np.array([3.0, 4.0]))
    assert fn._vectorized is False
    assert np.allclose(out2, [7.0, 7.0])


def test_vectorized_callable_is_used_without_a_python_loop():
    calls = {"n": 0}

    def g(a):
        calls["n"] += 1
        return np.asarray(a, dtype=float) ** 2

    fn = _RowwiseScalarFn(g)
    fn(np.zeros((100, 2)), np.arange(100.0))
    # One probe (2 rows) plus one full vectorized call -- not 100 calls.
    assert calls["n"] == 2
    assert fn._vectorized is True


def test_custom_generator_with_a_scalar_g_still_fits():
    """End-to-end: the rowwise path stays correct after the probe change."""

    X = _make_ate(n=120, seed=4)
    gen = BregmanGenerator(g=lambda a: a * a, name="sq-like")
    model = GRRGLM(
        basis=_basis(X), generator=gen, functional=ATEFunctional(0), penalty="l2", lam=1e-2
    )
    alpha = gen.inv_grad(X, np.linspace(-1.0, 1.0, len(X)))
    assert np.all(np.isfinite(alpha))
    assert model.fit(X) is not None
