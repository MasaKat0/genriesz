"""Generator evaluation on the optimizer hot path (design doc item W).

Two independent defects, both invisible in results and expensive in time.

**W-1: the branch sign was recomputed per row, per objective evaluation.**
``branch_fn`` is a function of ``x`` alone, so within a fit the sign array is
constant while ``v`` changes on every L-BFGS ``fun``/``jac`` call. The old
``_sign`` looped over rows every time: a 2000-row fit converging in 3 iterations
called ``branch_fn`` 24,000 times. ``BregmanGenerator.branch_cache()`` memoizes
the signs for the duration of a fit, taking that to exactly ``n``.

**W-2: custom generator callables have an explicit scalar contract.**
``_RowwiseScalarFn`` accepts ``f(alpha)`` or ``f(x, alpha)`` and evaluates the
callable once per observation. It does not probe a vectorized call and does not
change execution paths after an exception.

Both fixes are output-invariant;
``test_fit_results_are_bit_identical_with_and_without_the_cache`` pins that by
running two real fits and comparing beta, the KKT residual and the binding rate.
"""

from __future__ import annotations

import contextlib
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
    PUGenerator,
    SquaredGenerator,
    TreatmentInteractionBasis,
    UKLGenerator,
)
from genriesz.generators import _BRANCH_CACHE_MAX_ENTRIES, _RowwiseScalarFn


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


def test_branch_cache_is_bounded_for_callers_that_never_reuse_an_array():
    """A fresh array per call must cost speed, not memory."""

    gen = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1.0))
    v = np.full(20, -0.5)

    with gen.branch_cache():
        for _ in range(5 * _BRANCH_CACHE_MAX_ENTRIES):
            # int dtype forces np.asarray(..., dtype=float) to copy, so every
            # call reaches _sign with a brand-new array object.
            gen.inv_grad(np.ones((20, 2), dtype=np.int64), v)
            assert len(gen._branch_cache) <= _BRANCH_CACHE_MAX_ENTRIES


def test_solver_keeps_a_single_cache_entry_even_for_non_float_input():
    """GRRGLM normalizes X once, so the bound is never approached in a fit."""

    X = _make_ate(n=80, seed=5).astype(np.float32)
    gen = UKLGenerator(C=1.0, branch_fn=_CountingBranch())
    sizes: list[int] = []
    original = gen._branch_signs

    def spy(arr):
        sizes.append(len(gen._branch_cache or {}))
        return original(arr)

    gen._branch_signs = spy  # type: ignore[method-assign]
    fr = GRRGLM(
        basis=_basis(X), generator=gen, functional=ATEFunctional(0), penalty="l2", lam=1e-3
    ).fit(X)

    assert fr.success
    assert sizes == [0]  # computed exactly once, on an empty cache


def test_branch_cache_nesting_restores_the_outer_cache():
    gen = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1.0))
    with gen.branch_cache():
        outer = gen._branch_cache
        with gen.branch_cache():
            assert gen._branch_cache is not outer
        assert gen._branch_cache is outer
    assert gen._branch_cache is None


def test_squared_generator_never_reaches_the_cache():
    """SQ + L2 is solved in closed form, before the cache is entered.

    It is therefore excluded from the parametrization below, where it would be a
    vacuous case rather than a passing one.
    """

    X = _make_ate(n=120, seed=6)
    gen = SquaredGenerator(C=0.0)

    def boom():
        raise AssertionError("branch_cache must not be entered on the closed-form path")

    gen.branch_cache = boom  # type: ignore[method-assign]
    fr = GRRGLM(
        basis=_basis(X), generator=gen, functional=ATEFunctional(0), penalty="l2", lam=1e-3
    ).fit(X)

    assert fr.success
    assert fr.status == "closed_form"


@pytest.mark.parametrize(
    "make_gen",
    [
        lambda b: UKLGenerator(C=1.0, branch_fn=b),
        lambda b: BPGenerator(C=1.0, omega=0.5, branch_fn=b),
        lambda b: PUGenerator(C=1.0, branch_fn=b),
    ],
    ids=["ukl", "bp", "pu"],
)
def test_fit_results_are_bit_identical_with_and_without_the_cache(make_gen, monkeypatch):
    """The whole point of item W: the optimizer must land in the same place.

    Compare two real fits -- one with ``branch_cache`` active, one with it
    neutralized -- on beta, the KKT residual and the binding rate.
    """

    X = _make_ate(n=300, seed=3)

    def run(disable_cache: bool):
        branch = _CountingBranch()
        gen = make_gen(branch)
        if disable_cache:
            monkeypatch.setattr(gen, "branch_cache", contextlib.nullcontext)
        model = GRRGLM(
            basis=_basis(X), generator=gen, functional=ATEFunctional(0), penalty="l2", lam=1e-3
        )
        fr = model.fit(X)
        assert fr.success
        assert fr.status == "converged"  # the iterative path, not closed form
        return model.beta_, fr.kkt_residual, fr.clip_binding_rate, branch.calls

    beta_c, kkt_c, bind_c, calls_c = run(disable_cache=False)
    beta_u, kkt_u, bind_u, calls_u = run(disable_cache=True)

    # The disabled run really is uncached, so the comparison below has teeth.
    assert calls_c == len(X)
    assert calls_u > calls_c

    assert np.array_equal(beta_c, beta_u)
    assert kkt_c == kkt_u
    assert bind_c == bind_u


# ---------------------------------------------------------------------------
# W-2: custom generator callables are evaluated row by row without probing.
# ---------------------------------------------------------------------------
def _scalar_with_domain(a):
    if a < 0.0:
        raise DomainError("alpha must be nonnegative")
    return a * a


def _scalar_only(a):
    return math.log1p(math.exp(-abs(a)))


def test_scalar_callable_is_evaluated_rowwise():
    calls = {"n": 0}

    def scalar(a):
        calls["n"] += 1
        return a * a

    fn = _RowwiseScalarFn(scalar)
    X = np.zeros((4, 2))
    out = fn(X, np.array([0.5, 1.0, 1.5, 2.0]))

    assert fn._vectorized is False
    assert calls["n"] == 4
    assert np.allclose(out, [0.25, 1.0, 2.25, 4.0])


def test_domain_error_propagates_without_changing_the_execution_path():
    fn = _RowwiseScalarFn(_scalar_with_domain)
    X = np.zeros((3, 2))

    with pytest.raises(DomainError):
        fn(X, np.array([1.0, -2.0, 3.0]))
    assert fn._vectorized is False
    assert np.allclose(fn(X, np.array([2.0, 2.0, 2.0])), [4.0, 4.0, 4.0])


def test_vectorized_only_callable_is_outside_the_scalar_contract():
    fn = _RowwiseScalarFn(lambda Xv, av: Xv[:, 0] + av)
    with pytest.raises(IndexError):
        fn(np.ones((2, 2)), np.array([2.0, 3.0]))
    assert fn._vectorized is False


def test_single_row_and_empty_inputs_keep_the_rowwise_contract():
    fn = _RowwiseScalarFn(lambda a: 7.0)
    assert np.allclose(fn(np.zeros((1, 2)), np.array([3.0])), [7.0])
    assert fn(np.zeros((0, 2)), np.zeros(0)).shape == (0,)
    assert fn._vectorized is False


def test_custom_generator_with_explicit_derivatives_matches_closed_form():
    X = _make_ate(n=120, seed=4)
    v = np.linspace(-1.0, 1.0, len(X))

    generator = BregmanGenerator(
        g=lambda a: math.pow(a, 2.0),
        grad=lambda a: 2.0 * a,
        inv_grad=lambda value: 0.5 * value,
        grad2=lambda _a: 2.0,
        name="scalar-g",
    )
    alpha = generator.inv_grad(X, v)

    assert generator._g._vectorized is False
    assert np.allclose(alpha, v / 2.0, atol=0.0)
    assert np.allclose(generator.grad(X, alpha), 2.0 * alpha, atol=0.0)
    assert np.allclose(generator.grad2(X, alpha), 2.0, atol=0.0)
