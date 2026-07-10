"""Regression tests for item X-2: the two coercion paths must agree.

``estimation`` and ``density_ratio`` used to carry private, non-identical
copies of ``_coerce_basis``, and only ``density_ratio`` coerced generators.
Two user-visible bugs followed:

1. ``fit_density_ratio(basis=<stateful basis defined outside genriesz>)``
   wrapped the object in ``CallableBasis``, whose ``fit`` infers ``n_features``
   by *calling* the wrapped object instead of delegating to its ``fit``. The
   user's ``fit`` never ran and the basis raised.
2. ``grr_functional`` never checked the generator's type, so a string or any
   other object travelled into the L-BFGS objective and surfaced as an
   ``AttributeError`` from an error-formatting path inside ``glm``.
"""

import numpy as np
import pytest

import genriesz
from genriesz.basis import BaseBasis, CallableBasis, PolynomialBasis, coerce_basis
from genriesz.generators import (
    BKLGenerator,
    BPGenerator,
    BregmanGenerator,
    PUGenerator,
    SquaredGenerator,
    UKLGenerator,
    coerce_generator,
)


class StatefulDuckBasis:
    """A Basis that satisfies the protocol without inheriting BaseBasis."""

    def __init__(self):
        self._mean = None
        self.fit_calls = 0

    def fit(self, X, y=None):
        self._mean = np.asarray(X, dtype=float).mean(axis=0)
        self.fit_calls += 1
        return self

    def copy(self):
        new = StatefulDuckBasis()
        new._mean = None if self._mean is None else self._mean.copy()
        return new

    def __call__(self, X):
        if self._mean is None:
            raise RuntimeError("StatefulDuckBasis used before fit()")
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X - self._mean])

    @property
    def n_features(self):
        if self._mean is None:
            raise RuntimeError("StatefulDuckBasis used before fit()")
        return 1 + len(self._mean)


def _two_samples(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(60, 2)), rng.normal(loc=0.4, size=(80, 2))


def _ate_sample(n: int = 200, seed: int = 3):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    D = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.6 * Z[:, 0])), size=n).astype(float)
    Y = 0.5 * Z[:, 0] + 1.0 * D + rng.normal(scale=0.5, size=n)
    return np.column_stack([D, Z]), Y


# ---------------------------------------------------------------- coerce_basis


def test_coerce_basis_passes_a_duck_typed_stateful_basis_through_unwrapped():
    duck = StatefulDuckBasis()
    assert coerce_basis(duck) is duck


def test_coerce_basis_wraps_a_plain_callable():
    out = coerce_basis(lambda X: np.asarray(X, dtype=float))
    assert isinstance(out, CallableBasis)


def test_coerce_basis_keeps_built_in_bases():
    psi = PolynomialBasis(degree=2)
    assert coerce_basis(psi) is psi
    assert isinstance(psi, BaseBasis)


def test_coerce_basis_rejects_a_non_callable():
    with pytest.raises(TypeError, match="callable"):
        coerce_basis(object())


def test_coerce_basis_does_not_touch_n_features_on_an_unfitted_basis():
    """PolynomialBasis.n_features raises before fit; coercion must not probe it."""

    psi = PolynomialBasis(degree=2)
    with pytest.raises(RuntimeError, match="must be fit"):
        _ = psi.n_features
    coerce_basis(psi)  # must not raise


def test_wrapping_a_stateful_basis_would_skip_its_fit():
    """Pin the mechanism that made the old density_ratio path fail."""

    duck = StatefulDuckBasis()
    wrapped = CallableBasis(duck)
    with pytest.raises(RuntimeError, match="before fit"):
        wrapped.fit(np.zeros((4, 2)))
    assert duck.fit_calls == 0


# ------------------------------------------------- basis parity across modules


def test_fit_density_ratio_fits_a_duck_typed_stateful_basis():
    Xn, Xd = _two_samples()
    result = genriesz.fit_density_ratio(Xn, Xd, basis=StatefulDuckBasis(), lam=0.1)
    assert np.shape(result.beta) == (3,)
    assert np.all(np.isfinite(result.predict_ratio(Xd[:10])))


def test_grr_ate_fits_the_same_duck_typed_stateful_basis():
    X, Y = _ate_sample()
    result = genriesz.grr_ate(
        X=X, Y=Y, basis=StatefulDuckBasis(), generator=SquaredGenerator(C=0.0)
    )
    assert result.estimand == "ATE"


def test_fit_density_ratio_does_not_mutate_the_caller_s_basis():
    """density_ratio fits a copy, so the caller's object stays unfitted."""

    Xn, Xd = _two_samples()
    duck = StatefulDuckBasis()
    genriesz.fit_density_ratio(Xn, Xd, basis=duck, lam=0.1)
    assert duck.fit_calls == 0
    with pytest.raises(RuntimeError, match="before fit"):
        duck(Xn)


def test_both_modules_delegate_to_the_shared_coerce_basis():
    """A name check, not a clone detector: a re-implementation under another
    name would still pass. It guards against the private copies coming back."""

    import genriesz.density_ratio as dr
    import genriesz.estimation as est

    assert not hasattr(dr, "_coerce_basis")
    assert not hasattr(est, "_coerce_basis")
    assert est.coerce_basis is coerce_basis
    assert dr.coerce_basis is coerce_basis


# ------------------------------------------------------------ coerce_generator


def test_coerce_generator_returns_an_instance_unchanged():
    gen = SquaredGenerator(C=0.0)
    assert coerce_generator(gen) is gen


def test_coerce_generator_applies_branch_fn_to_named_branchwise_generators():
    def pos(_x):
        return 1

    for name, cls in [
        ("ukl", UKLGenerator),
        ("bkl", BKLGenerator),
        ("bp", BPGenerator),
        ("pu", PUGenerator),
    ]:
        gen = coerce_generator(name, branch_fn=pos)
        assert isinstance(gen, cls)
        assert gen.branch_fn is pos


def test_coerce_generator_rejects_branchwise_names_when_disallowed():
    for name in ["ukl", "bkl", "bp", "power", "pu"]:
        with pytest.raises(ValueError, match="branch_fn"):
            coerce_generator(name, allow_branchwise_names=False)


def test_coerce_generator_allows_squared_names_when_branchwise_disallowed():
    for name in ["sq", "squared", "lsif"]:
        gen = coerce_generator(name, allow_branchwise_names=False)
        assert isinstance(gen, SquaredGenerator)


def test_coerce_generator_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown generator name"):
        coerce_generator("nope")


def test_coerce_generator_rejects_a_non_generator():
    with pytest.raises(TypeError, match="BregmanGenerator"):
        coerce_generator(42)


# --------------------------------------------- generator parity across modules


def test_grr_functional_rejects_a_non_generator_before_the_solver():
    """On main this reached scipy and raised AttributeError from glm's error path."""

    X, Y = _ate_sample()
    with pytest.raises(TypeError, match="BregmanGenerator"):
        genriesz.grr_ate(X=X, Y=Y, basis=PolynomialBasis(degree=1), generator=42)


def test_grr_functional_rejects_branchwise_generator_names():
    """A Riesz representer is negative on the controls, so 'bkl' cannot pick a branch."""

    X, Y = _ate_sample()
    with pytest.raises(ValueError, match="branch_fn"):
        genriesz.grr_ate(X=X, Y=Y, basis=PolynomialBasis(degree=1), generator="bkl")


def test_grr_functional_accepts_the_squared_name_and_matches_the_instance():
    X, Y = _ate_sample()
    kw = dict(X=X, Y=Y, basis=PolynomialBasis(degree=2))
    by_name = genriesz.grr_ate(generator="sq", **kw)
    by_instance = genriesz.grr_ate(generator=SquaredGenerator(C=0.0), **kw)
    for key in by_instance.estimates:
        assert by_name.estimates[key].estimate == pytest.approx(
            by_instance.estimates[key].estimate
        )


def test_fit_density_ratio_still_accepts_branchwise_names():
    """A density ratio is nonnegative, so the positive branch is always right."""

    Xn, Xd = _two_samples()
    for name in ["sq", "ukl", "bkl", "bp", "pu"]:
        result = genriesz.fit_density_ratio(Xn, Xd, generator=name, lam=0.1)
        assert isinstance(result.generator, BregmanGenerator)


def test_fit_density_ratio_named_generator_uses_the_positive_branch():
    Xn, Xd = _two_samples()
    result = genriesz.fit_density_ratio(Xn, Xd, generator="ukl", lam=0.1)
    assert result.generator.branch_fn(np.array([0.0, 0.0])) == 1


def test_fit_density_ratio_rejects_an_unknown_generator_name():
    Xn, Xd = _two_samples()
    with pytest.raises(ValueError, match="Unknown generator name"):
        genriesz.fit_density_ratio(Xn, Xd, generator="nope")


def test_fit_density_ratio_rejects_a_non_generator():
    Xn, Xd = _two_samples()
    with pytest.raises(TypeError, match="BregmanGenerator"):
        genriesz.fit_density_ratio(Xn, Xd, generator=42)
