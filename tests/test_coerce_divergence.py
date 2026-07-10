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
from genriesz.basis import (
    BaseBasis,
    CallableBasis,
    PolynomialBasis,
    _instances_define_getattr,
    coerce_basis,
)
from genriesz.generators import (
    BKLGenerator,
    BPGenerator,
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


class _MetadataCarryingCallable:
    """A plain feature map whose ``fit`` and ``copy`` are data, not methods.

    It is not a Basis. ``hasattr`` cannot tell it apart from one, so a
    ``hasattr``-only predicate routes it past :class:`CallableBasis` and then
    dies on ``basis.copy()``.
    """

    fit = "fitted at import time"
    copy = "deep"

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])


class _NonDeepcopyableCallable:
    """A feature map that refuses to be deep-copied.

    genriesz refits a copy of the basis per cross-fitting fold, so a basis it
    cannot copy cannot be isolated. Both entry points must say so.
    """

    def __deepcopy__(self, memo):
        raise TypeError("cannot deepcopy this feature map")

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])


class _PropertyFitCallable:
    """A plain feature map whose ``fit`` is a property that raises when read.

    Deciding whether this is a Basis must not read the property.
    """

    @property
    def fit(self):
        raise RuntimeError("fit getter touched")

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])


class _ProxyBasis:
    """A Basis that exposes ``fit`` and ``copy`` only through ``__getattr__``."""

    def __init__(self):
        object.__setattr__(self, "_inner", StatefulDuckBasis())

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, X):
        return self._inner(X)


class _BadGetattrCallable:
    """``__getattr__`` raises ValueError for a missing name, breaking the data model.

    The dynamic fallback in ``_lookup_method`` cannot ask such an object a
    question safely, so the error propagates. ``hasattr`` on main behaved the
    same way, which is why main's ``grr_ate`` already raised.
    """

    def __getattr__(self, name):
        raise ValueError(f"no attribute {name}")

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])


class _StatefulCallable:
    """A feature map that caches the first sample it sees.

    It breaks the ``basis(X) -> Phi`` purity contract. Copying isolates the
    folds anyway; sharing it would leak one fold's training mean into another.
    """

    def __init__(self):
        self.seen = None

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        if self.seen is None:
            self.seen = X.mean(axis=0)
        return np.column_stack([np.ones(len(X)), X - self.seen])


class _ShadowedCopyBasis(BaseBasis):
    """A protocol violation: a data attribute named ``copy`` hides the method."""

    def __init__(self):
        self.copy = True  # e.g. a scikit-learn style ``copy=True`` parameter

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])


class _UncopyableBasis(BaseBasis):
    """A stateful Basis that refuses to be copied. Neither path can honour it."""

    def __init__(self):
        self.fitted_on = None

    def __deepcopy__(self, memo):
        raise TypeError("cannot deepcopy this basis")

    def fit(self, X, y=None):
        self.fitted_on = len(np.asarray(X))
        return self

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])


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


def test_coerce_basis_wraps_a_callable_carrying_non_callable_fit_and_copy():
    """``hasattr`` is too weak: only a *callable* fit and copy make a Basis."""

    out = coerce_basis(_MetadataCarryingCallable())
    assert isinstance(out, CallableBasis)
    assert callable(out.copy)


def test_fit_density_ratio_accepts_a_callable_carrying_non_callable_metadata():
    Xn, Xd = _two_samples()
    result = genriesz.fit_density_ratio(Xn, Xd, basis=_MetadataCarryingCallable(), lam=0.1)
    assert np.shape(result.beta) == (3,)


def test_grr_ate_accepts_a_callable_carrying_non_callable_metadata():
    X, Y = _ate_sample()
    result = genriesz.grr_ate(
        X=X, Y=Y, basis=_MetadataCarryingCallable(), generator=SquaredGenerator(C=0.0)
    )
    assert result.estimand == "ATE"


def test_coercing_a_copyable_basis_yields_something_copy_and_fit_can_be_called_on():
    """density_ratio calls .copy().fit(...) unconditionally, so exercise both.

    Checking ``callable(obj.copy)`` alone would not support removing the old
    try/except: a callable ``copy`` can still raise when called. This covers the
    copyable inputs only; the bases that deliberately refuse to be copied are
    pinned separately below.
    """

    X = np.linspace(-1.0, 1.0, 12).reshape(6, 2)
    for spec in [
        PolynomialBasis(degree=2),
        StatefulDuckBasis(),
        _MetadataCarryingCallable(),
        _PropertyFitCallable(),
        _ProxyBasis(),
        lambda Z: np.column_stack([np.ones(len(Z)), Z]),
    ]:
        fitted = coerce_basis(spec).copy().fit(X)
        assert np.asarray(fitted(X)).shape[0] == X.shape[0]


def test_callable_basis_copy_isolates_the_wrapped_feature_map():
    """Cross-fitting refits a copy per fold, so no two copies may share state."""

    func = _StatefulCallable()
    wrapper = CallableBasis(func)
    fold_a, fold_b = wrapper.copy(), wrapper.copy()

    # Distinct from the original and, crucially, from each other.
    assert fold_a.func is not func
    assert fold_b.func is not func
    assert fold_a.func is not fold_b.func

    fold_a.func(np.zeros((3, 2)))
    fold_b.func(np.ones((3, 2)))
    assert func.seen is None
    assert not np.array_equal(fold_a.func.seen, fold_b.func.seen)


def test_coerce_basis_rejects_a_base_basis_that_shadows_copy():
    """A legible error, not 'NoneType' object is not callable from deep inside."""

    with pytest.raises(TypeError, match="shadows the Basis method 'copy'"):
        coerce_basis(_ShadowedCopyBasis())


def test_coerce_basis_does_not_run_a_fit_property_while_deciding():
    """The n_features hazard applies to 'fit' and 'copy' too, if a caller

    defines them as properties. Coercion must inspect statically."""

    out = coerce_basis(_PropertyFitCallable())
    assert isinstance(out, CallableBasis)


def test_both_paths_accept_a_feature_map_whose_fit_is_a_raising_property():
    """On main, density_ratio wrapped it and worked while grr_ate read the property."""

    Xn, Xd = _two_samples()
    X, Y = _ate_sample()

    result = genriesz.fit_density_ratio(Xn, Xd, basis=_PropertyFitCallable(), lam=0.1)
    assert np.shape(result.beta) == (3,)

    estimate = genriesz.grr_ate(
        X=X, Y=Y, basis=_PropertyFitCallable(), generator=SquaredGenerator(C=0.0)
    )
    assert estimate.estimand == "ATE"


def test_coerce_basis_still_recognises_a_basis_behind_getattr():
    """An object that defines __getattr__ opted into dynamic lookup, so use it."""

    proxy = _ProxyBasis()
    assert coerce_basis(proxy) is proxy


def test_coerce_basis_propagates_a_getattr_that_breaks_the_data_model():
    """Deliberate. __getattr__ must raise AttributeError for a missing name.

    main's grr_ate raised here too, for the same reason: hasattr only swallows
    AttributeError. Only main's fit_density_ratio 'worked', by never asking.
    """

    with pytest.raises(ValueError, match="no attribute fit"):
        coerce_basis(_BadGetattrCallable())


def test_coerce_basis_handles_slots_staticmethod_and_classmethod_bases():
    """getattr_static returns the raw descriptor, so unwrap the two that need it."""

    class SlotsDuck:
        __slots__ = ("_seen",)

        def __init__(self):
            self._seen = None

        def fit(self, X, y=None):
            self._seen = np.asarray(X, dtype=float).mean(axis=0)
            return self

        def copy(self):
            return SlotsDuck()

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    class StaticDuck:
        @staticmethod
        def fit(X, y=None):
            return StaticDuck()

        @staticmethod
        def copy():
            return StaticDuck()

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    class ClassDuck:
        @classmethod
        def fit(cls, X, y=None):
            return cls()

        @classmethod
        def copy(cls):
            return cls()

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    for duck in [SlotsDuck(), StaticDuck(), ClassDuck()]:
        assert coerce_basis(duck) is duck


def test_coerce_basis_recognises_methods_stored_in_slots():
    """getattr_static returns the slot descriptor, not the callable inside it.

    Reading a slot runs no user code, so it is safe to resolve. Failing to do so
    wrapped the basis and skipped its fit -- the very bug X-2 removes.
    """

    class SlotMethodBasis:
        __slots__ = ("fit", "copy", "_seen")

        def __init__(self):
            self._seen = None
            self.fit = self._fit
            self.copy = lambda: SlotMethodBasis()

        def _fit(self, X, y=None):
            self._seen = np.asarray(X, dtype=float).mean(axis=0)
            return self

        def __call__(self, X):
            if self._seen is None:
                raise RuntimeError("SlotMethodBasis used before fit()")
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X - self._seen])

    basis = SlotMethodBasis()
    assert coerce_basis(basis) is basis

    X, Y = _ate_sample()
    result = genriesz.grr_ate(
        X=X, Y=Y, basis=SlotMethodBasis(), generator=SquaredGenerator(C=0.0)
    )
    assert result.estimand == "ATE"

    Xn, Xd = _two_samples()
    ratio = genriesz.fit_density_ratio(Xn, Xd, basis=SlotMethodBasis(), lam=0.1)
    assert np.shape(ratio.beta) == (3,)


def test_coerce_basis_wraps_a_partial_and_a_ufunc():
    import functools

    def phi(X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])

    assert isinstance(coerce_basis(functools.partial(phi)), CallableBasis)
    assert isinstance(coerce_basis(np.exp), CallableBasis)


def test_coerce_basis_sees_an_inherited_getattr():
    class _Base:
        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_inner"), name)

    class _Derived(_Base):
        def __init__(self):
            object.__setattr__(self, "_inner", StatefulDuckBasis())

        def __call__(self, X):
            return object.__getattribute__(self, "_inner")(X)

    proxy = _Derived()
    assert coerce_basis(proxy) is proxy


def test_a_metaclass_getattr_does_not_make_instances_look_dynamic():
    """__getattr__ on the metaclass governs the class object, not its instances.

    ``hasattr(type(obj), '__getattr__')`` finds the metaclass's and answers the
    wrong question. Today both spellings end up wrapping such an object anyway,
    so this pins the predicate rather than an end-to-end behaviour.
    """

    class _Meta(type):
        def __getattr__(cls, name):
            raise RuntimeError("metaclass __getattr__ was consulted")

    class _Map(metaclass=_Meta):
        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    assert hasattr(type(_Map()), "__getattr__")  # the wrong question says yes
    assert not _instances_define_getattr(type(_Map()))  # the right one says no
    assert isinstance(coerce_basis(_Map()), CallableBasis)


@pytest.mark.parametrize("callable_descriptor", [False, True])
def test_coerce_basis_does_not_run_a_custom_descriptor_named_fit(callable_descriptor):
    """inspect.isroutine is duck-typed and answers True for any non-data descriptor.

    Resolving one would run the caller's ``__get__``, which is what main's
    grr_ate did. Only the descriptor types that merely fetch are resolved.

    An unresolved descriptor must be reported as *absent*, not returned. A
    descriptor may define ``__call__`` as well as ``__get__``, and returning it
    would pass the callable test, mistake the feature map for a Basis, and run
    the ``__get__`` anyway on the first ``basis.copy()``.
    """

    class _RaisingDescriptor:
        def __get__(self, obj, objtype=None):
            raise RuntimeError("custom __get__ ran")

    class _CallableRaisingDescriptor(_RaisingDescriptor):
        def __call__(self, *args, **kwargs):
            return None

    descriptor = _CallableRaisingDescriptor if callable_descriptor else _RaisingDescriptor

    class _Map:
        fit = descriptor()
        copy = descriptor()

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    assert isinstance(coerce_basis(_Map()), CallableBasis)

    Xn, Xd = _two_samples()
    X, Y = _ate_sample()
    assert np.shape(genriesz.fit_density_ratio(Xn, Xd, basis=_Map(), lam=0.1).beta) == (3,)
    estimate = genriesz.grr_ate(X=X, Y=Y, basis=_Map(), generator=SquaredGenerator(C=0.0))
    assert estimate.estimand == "ATE"


def test_coerce_basis_does_not_trust_a_subclass_of_an_inert_wrapper():
    """The inert types are matched exactly: a subclass may override __get__.

    ``staticmethod`` and ``partialmethod`` are ordinary classes. A subclass can
    replace ``__get__``, or intercept the read of the wrapped object that
    ``_is_inert`` performs, and either would run the caller's code.
    """

    import functools

    def _plain_fit(self, X=None, y=None):
        return self

    class _EvilStaticMethod(staticmethod):
        def __get__(self, obj, objtype=None):
            raise RuntimeError("staticmethod.__get__ ran")

    class _EvilPartialMethod(functools.partialmethod):
        def __getattribute__(self, name):
            if name == "func":
                raise RuntimeError("partialmethod.func was read")
            return super().__getattribute__(name)

    class _StaticMap:
        fit = _EvilStaticMethod(_plain_fit)
        copy = _EvilStaticMethod(_plain_fit)

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    class _PartialMap:
        fit = _EvilPartialMethod(_plain_fit)
        copy = _EvilPartialMethod(_plain_fit)

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    Xn, Xd = _two_samples()
    for factory in (_StaticMap, _PartialMap):
        assert isinstance(coerce_basis(factory()), CallableBasis)
        ratio = genriesz.fit_density_ratio(Xn, Xd, basis=factory(), lam=0.1)
        assert np.shape(ratio.beta) == (3,)


def test_a_classmethod_is_inert_only_as_far_as_what_it_wraps():
    """classmethod.__get__ delegates to the wrapped descriptor on Python 3.10-3.12.

    The delegation (chained classmethod descriptors) was removed in 3.13, so a
    check written against 3.13 alone would let the caller's ``__get__`` run on
    every other supported version. Decide by type, not by what today's
    interpreter happens to do.

    ``staticmethod`` is the contrast: it hands back what it wraps without
    binding it, on every version, so it is inert whatever it holds.
    """

    from genriesz.basis import _is_inert

    class _RaisingDescriptor:
        def __get__(self, obj, objtype=None):
            raise RuntimeError("custom __get__ ran")

    def _plain_fit(cls, X=None, y=None):
        return cls

    assert _is_inert(classmethod(_plain_fit))
    assert not _is_inert(classmethod(_RaisingDescriptor()))
    assert _is_inert(staticmethod(_RaisingDescriptor()))
    assert staticmethod(_RaisingDescriptor()).__get__(object(), object) is not None

    class _Map:
        fit = classmethod(_RaisingDescriptor())
        copy = classmethod(_RaisingDescriptor())

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    assert isinstance(coerce_basis(_Map()), CallableBasis)

    Xn, Xd = _two_samples()
    assert np.shape(genriesz.fit_density_ratio(Xn, Xd, basis=_Map(), lam=0.1).beta) == (3,)


def test_a_setter_only_descriptor_does_not_outrank_the_instance_dict():
    """A data descriptor needs a ``__get__`` to answer a *read*.

    ``__set__`` alone does not shadow the instance dict. Treating it as though it
    did reports the class attribute as the method, and the object is called a
    Basis on the strength of a value attribute access would never return.
    """

    class _SetterOnly:
        def __set__(self, obj, value):
            pass

        def __call__(self, *args, **kwargs):
            return None

    class _Map:
        fit = _SetterOnly()
        copy = _SetterOnly()

        def __init__(self):
            object.__getattribute__(self, "__dict__").update(fit=None, copy=None)

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    assert _Map().copy is None  # what attribute access actually returns
    assert isinstance(coerce_basis(_Map()), CallableBasis)

    Xn, Xd = _two_samples()
    assert np.shape(genriesz.fit_density_ratio(Xn, Xd, basis=_Map(), lam=0.1).beta) == (3,)


def test_a_class_object_is_a_feature_map_not_a_basis():
    """The Basis protocol lives on instances. A class passed as a basis is a callable."""

    class _Trap:
        def __get__(self, obj, objtype=None):
            raise RuntimeError("custom __get__ ran")

        def __call__(self, *args, **kwargs):
            return None

    class _ClassFeature:
        fit = _Trap()
        copy = _Trap()

        def __new__(cls, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    assert isinstance(coerce_basis(_ClassFeature), CallableBasis)

    Xn, Xd = _two_samples()
    ratio = genriesz.fit_density_ratio(Xn, Xd, basis=_ClassFeature, lam=0.1)
    assert np.shape(ratio.beta) == (3,)


def test_an_object_that_writes_getattribute_is_not_certified_as_a_basis():
    """A static answer says nothing about what ``basis.fit`` will return."""

    class _Map:
        def fit(self, X, y=None):
            return self

        def copy(self):
            return _Map()

        def __getattribute__(self, name):
            if name in ("fit", "copy"):
                raise RuntimeError(f"blocked {name}")
            return object.__getattribute__(self, name)

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    assert isinstance(coerce_basis(_Map()), CallableBasis)

    Xn, Xd = _two_samples()
    assert np.shape(genriesz.fit_density_ratio(Xn, Xd, basis=_Map(), lam=0.1).beta) == (3,)


def test_a_builtin_getattribute_does_not_disqualify_a_basis():
    """dict, list and the rest install their own ``__getattribute__`` in C.

    Only a ``__getattribute__`` the caller wrote makes the static answer a lie.
    """

    class _DictBasis(dict):  # dict.__dict__ carries a __getattribute__ slot wrapper
        def fit(self, X, y=None):
            return self

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    basis = _DictBasis()
    assert coerce_basis(basis) is basis


def test_a_staticmethod_is_inert_whatever_it_wraps():
    """staticmethod.__get__ returns the wrapped object; it never binds it.

    Unwrapping it as though it delegated refuses a basis main accepted.
    """

    class _CallableTrap:
        def __get__(self, obj, objtype=None):
            raise RuntimeError("__get__ must not run")

        def __call__(self, *args, **kwargs):
            return _StaticBasis()

    class _StaticBasis(BaseBasis):
        fit = staticmethod(_CallableTrap())
        copy = staticmethod(_CallableTrap())

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    basis = _StaticBasis()
    assert coerce_basis(basis) is basis

    Xn, Xd = _two_samples()
    assert np.shape(genriesz.fit_density_ratio(Xn, Xd, basis=_StaticBasis(), lam=0.1).beta) == (3,)


def test_an_unset_slot_falls_through_to_getattr():
    """Attribute access answers an unset slot by consulting ``__getattr__`` next."""

    class _DynamicSlot(BaseBasis):
        __slots__ = ("copy",)

        def __getattr__(self, name):
            if name == "copy":
                return lambda: _DynamicSlot()
            raise AttributeError(name)

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    basis = _DynamicSlot()
    assert callable(basis.copy)
    assert coerce_basis(basis) is basis

    Xn, Xd = _two_samples()
    assert np.shape(genriesz.fit_density_ratio(Xn, Xd, basis=_DynamicSlot(), lam=0.1).beta) == (3,)


def test_a_descriptor_lifted_from_an_unrelated_class_is_not_bound():
    """A slot accessor carries the class it was defined on, and refuses others."""

    class _Donor:
        __slots__ = ("fit",)

    class _DonorDict:
        pass

    class _ForeignSlotMap:
        fit = _Donor.__dict__["fit"]

        def copy(self):
            return _ForeignSlotMap()

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    class _ForeignDictMap:
        __dict__ = _DonorDict.__dict__["__dict__"]

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    Xn, Xd = _two_samples()
    for factory in (_ForeignSlotMap, _ForeignDictMap):
        assert isinstance(coerce_basis(factory()), CallableBasis)
        assert np.shape(genriesz.fit_density_ratio(Xn, Xd, basis=factory(), lam=0.1).beta) == (3,)


def test_the_shadowed_copy_error_does_not_consult_a_hostile_metaclass():
    """Naming the offending type must not read ``__name__`` through the metaclass."""

    class _NameBlockingMeta(type):
        def __getattribute__(cls, name):
            if name == "__name__":
                raise RuntimeError("metaclass __name__ ran")
            return type.__getattribute__(cls, name)

    class _BadCopy(BaseBasis, metaclass=_NameBlockingMeta):
        copy = None

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    with pytest.raises(TypeError, match="shadows the Basis method 'copy'"):
        coerce_basis(_BadCopy())


def test_a_data_descriptor_outranks_the_instance_dict():
    """Walking the MRO by hand must honour the precedence attribute access uses.

    A ``property`` named ``fit`` wins over ``obj.__dict__['fit']``. Reading the
    instance dict first would report the shadowed value as the method, call the
    object a Basis, and then run the property's getter on the first ``fit()``.
    """

    class _Map:
        def __init__(self):
            object.__setattr__(self, "_seen", None)
            self.__dict__["fit"] = lambda X, y=None: None
            self.__dict__["copy"] = lambda: None

        @property
        def fit(self):
            raise RuntimeError("property getter ran")

        @property
        def copy(self):
            raise RuntimeError("property getter ran")

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    assert isinstance(coerce_basis(_Map()), CallableBasis)

    Xn, Xd = _two_samples()
    assert np.shape(genriesz.fit_density_ratio(Xn, Xd, basis=_Map(), lam=0.1).beta) == (3,)


def test_a_shadowed_instance_dict_is_not_read():
    """``__dict__`` itself may be a property the caller wrote."""

    class _Map:
        @property
        def __dict__(self):
            raise RuntimeError("__dict__ getter ran")

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    assert isinstance(coerce_basis(_Map()), CallableBasis)

    Xn, Xd = _two_samples()
    assert np.shape(genriesz.fit_density_ratio(Xn, Xd, basis=_Map(), lam=0.1).beta) == (3,)


def test_coerce_basis_does_not_consult_a_hostile_metaclass():
    """Classifying an object must not read its class through the metaclass.

    ``inspect.getattr_static`` reads ``entry.__dict__`` for each class in the
    MRO, which goes through the metaclass's ``__getattribute__``. Reading
    ``cls.__mro__`` does too. Both run caller code for a plain feature map that
    main wrapped without complaint.
    """

    class _MetaBlockingDict(type):
        def __getattribute__(cls, name):
            if name == "__dict__":
                raise RuntimeError("metaclass __dict__ was read")
            return type.__getattribute__(cls, name)

    class _MetaBlockingMro(type):
        def __getattribute__(cls, name):
            if name == "__mro__":
                raise RuntimeError("metaclass __mro__ was read")
            return type.__getattribute__(cls, name)

    class _DictMap(metaclass=_MetaBlockingDict):
        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    class _MroMap(metaclass=_MetaBlockingMro):
        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    # Classification is safe for both. Only _MroMap survives the per-fold
    # deepcopy; _DictMap is a basis that cannot be copied, and both entry points
    # now refuse it alike (see the strictening recorded in the design note).
    assert isinstance(coerce_basis(_DictMap()), CallableBasis)
    assert isinstance(coerce_basis(_MroMap()), CallableBasis)

    Xn, Xd = _two_samples()
    ratio = genriesz.fit_density_ratio(Xn, Xd, basis=_MroMap(), lam=0.1)
    assert np.shape(ratio.beta) == (3,)


def test_coerce_basis_recognises_a_partialmethod_and_a_c_implemented_method():
    """Both are ordinary ways to spell a method, and neither __get__ runs user code."""

    import functools

    class _PartialMethodBasis:
        def __init__(self):
            self._seen = None

        def _fit(self, X, y=None, *, scale=1.0):
            self._seen = scale * np.asarray(X, dtype=float).mean(axis=0)
            return self

        fit = functools.partialmethod(_fit, scale=1.0)

        def copy(self):
            return _PartialMethodBasis()

        def __call__(self, X):
            if self._seen is None:
                raise RuntimeError("used before fit()")
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X - self._seen])

    basis = _PartialMethodBasis()
    assert coerce_basis(basis) is basis

    X, Y = _ate_sample()
    estimate = genriesz.grr_ate(
        X=X, Y=Y, basis=_PartialMethodBasis(), generator=SquaredGenerator(C=0.0)
    )
    assert estimate.estimand == "ATE"

    Xn, Xd = _two_samples()
    ratio = genriesz.fit_density_ratio(Xn, Xd, basis=_PartialMethodBasis(), lam=0.1)
    assert np.shape(ratio.beta) == (3,)

    class _CMethodBasis(dict):  # dict.copy is a C method_descriptor
        def fit(self, X, y=None):
            return self

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    c_basis = _CMethodBasis()
    assert coerce_basis(c_basis) is c_basis


def test_a_partialmethod_is_inert_only_as_far_as_what_it_wraps():
    """partialmethod.__get__ delegates, so it cannot be trusted unconditionally."""

    import functools

    class _RaisingDescriptor:
        def __get__(self, obj, objtype=None):
            raise RuntimeError("custom __get__ ran")

    class _Map:
        fit = functools.partialmethod(_RaisingDescriptor())
        copy = functools.partialmethod(_RaisingDescriptor())

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([np.ones(len(X)), X])

    assert isinstance(coerce_basis(_Map()), CallableBasis)

    Xn, Xd = _two_samples()
    assert np.shape(genriesz.fit_density_ratio(Xn, Xd, basis=_Map(), lam=0.1).beta) == (3,)


def test_is_inert_terminates_on_a_self_referential_partialmethod():
    """pm.func = pm would recurse without end. Unresolved is the safe answer."""

    import functools

    from genriesz.basis import _is_inert

    def _f(self, X=None, y=None):
        return self

    pm = functools.partialmethod(_f)
    assert _is_inert(pm)

    pm.func = pm
    assert not _is_inert(pm)


def test_instances_define_getattr_sees_the_class_and_its_bases():
    class _Base:
        def __getattr__(self, name):
            raise AttributeError(name)

    class _Derived(_Base):
        pass

    assert _instances_define_getattr(_Base)
    assert _instances_define_getattr(_Derived)
    assert not _instances_define_getattr(int)


def test_both_paths_accept_a_proxy_basis():
    """On main, grr_ate accepted it while density_ratio wrapped it and raised."""

    Xn, Xd = _two_samples()
    X, Y = _ate_sample()

    result = genriesz.fit_density_ratio(Xn, Xd, basis=_ProxyBasis(), lam=0.1)
    assert np.shape(result.beta) == (3,)

    estimate = genriesz.grr_ate(
        X=X, Y=Y, basis=_ProxyBasis(), generator=SquaredGenerator(C=0.0)
    )
    assert estimate.estimand == "ATE"


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


def test_a_non_deepcopyable_callable_fails_the_same_way_on_both_paths():
    """main let fit_density_ratio through by fitting in place; grr_ate already raised."""

    Xn, Xd = _two_samples()
    X, Y = _ate_sample()

    with pytest.raises(TypeError, match="cannot deepcopy"):
        genriesz.fit_density_ratio(Xn, Xd, basis=_NonDeepcopyableCallable(), lam=0.1)

    with pytest.raises(TypeError, match="cannot deepcopy"):
        genriesz.grr_ate(
            X=X, Y=Y, basis=_NonDeepcopyableCallable(), generator=SquaredGenerator(C=0.0)
        )


def test_a_stateful_callable_basis_does_not_leak_across_cross_fitting_folds():
    """The reason CallableBasis.copy must deepcopy rather than share.

    Each fold refits ``basis.copy()``. If the wrapped callable were shared, the
    first fold's training mean would define the features of every later fold.
    """

    X, Y = _ate_sample()
    func = _StatefulCallable()
    genriesz.grr_ate(X=X, Y=Y, basis=func, generator=SquaredGenerator(C=0.0))
    assert func.seen is None  # the caller's object was never fitted


def test_each_fold_gets_its_own_copy_of_a_stateful_callable():
    """Pin what the leak test above can only imply: the folds see different data.

    Every copy records the sample it was fitted on. Cross-fitting must produce
    more than one distinct record, which sharing the callable could not.
    """

    X, Y = _ate_sample()
    seen: list[np.ndarray] = []

    class _Recording(_StatefulCallable):
        def __call__(self, Z):
            first = self.seen is None
            out = super().__call__(Z)
            if first:
                seen.append(np.asarray(self.seen, dtype=float).copy())
            return out

    genriesz.grr_ate(X=X, Y=Y, basis=_Recording(), generator=SquaredGenerator(C=0.0))

    assert len(seen) >= 2
    distinct = {tuple(np.round(row, 12)) for row in seen}
    assert len(distinct) >= 2


def test_fit_density_ratio_reports_a_shadowed_copy_instead_of_fitting_in_place():
    """main swallowed the failure and fitted the caller's basis in place."""

    Xn, Xd = _two_samples()
    with pytest.raises(TypeError, match="shadows the Basis method 'copy'"):
        genriesz.fit_density_ratio(Xn, Xd, basis=_ShadowedCopyBasis(), lam=0.1)


def test_an_uncopyable_basis_fails_the_same_way_on_both_paths():
    """The heart of item X-2.

    On main, a Basis that refuses to be copied made ``fit_density_ratio``
    silently succeed -- fitting the caller's object in place, against the
    contract -- while ``grr_ate`` raised. Both must now raise.
    """

    Xn, Xd = _two_samples()
    X, Y = _ate_sample()

    with pytest.raises(TypeError, match="cannot deepcopy"):
        genriesz.fit_density_ratio(Xn, Xd, basis=_UncopyableBasis(), lam=0.1)

    with pytest.raises(TypeError, match="cannot deepcopy"):
        genriesz.grr_ate(
            X=X, Y=Y, basis=_UncopyableBasis(), generator=SquaredGenerator(C=0.0)
        )


def test_an_uncopyable_basis_is_not_fitted_in_place():
    Xn, Xd = _two_samples()
    basis = _UncopyableBasis()
    with pytest.raises(TypeError, match="cannot deepcopy"):
        genriesz.fit_density_ratio(Xn, Xd, basis=basis, lam=0.1)
    assert basis.fitted_on is None


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


def test_coerce_generator_reproduces_the_name_to_generator_map():
    """Pin the class, C, omega and branch for every name, including the aliases."""

    def pos(_x):
        return 1

    expected = {
        "ukl": (UKLGenerator, 0.0, None),
        "bkl": (BKLGenerator, 1.0, None),
        "bp": (BPGenerator, 0.0, 0.5),
        "power": (BPGenerator, 0.0, 0.5),
        "pu": (PUGenerator, 1.0, None),
    }
    for name, (cls, c_value, omega) in expected.items():
        gen = coerce_generator(name, branch_fn=pos)
        assert isinstance(gen, cls)
        assert float(gen.C) == pytest.approx(c_value)
        assert gen.branch_fn is pos
        if omega is not None:
            assert float(gen.omega) == pytest.approx(omega)

    for name in ["sq", "squared", "lsif"]:
        gen = coerce_generator(name)
        assert isinstance(gen, SquaredGenerator)
        assert float(gen.C) == pytest.approx(0.0)


def test_coerce_generator_ignores_surrounding_whitespace_and_case():
    assert isinstance(coerce_generator("  SQ  "), SquaredGenerator)


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
    expected = {
        "sq": SquaredGenerator,
        "ukl": UKLGenerator,
        "bkl": BKLGenerator,
        "bp": BPGenerator,
        "power": BPGenerator,
        "pu": PUGenerator,
    }
    for name, cls in expected.items():
        result = genriesz.fit_density_ratio(Xn, Xd, generator=name, lam=0.1)
        assert isinstance(result.generator, cls)


def test_grr_functional_still_accepts_a_raw_g_without_a_generator():
    """The new validation must not fire on the generator=None, g=... path."""

    X, Y = _ate_sample()
    result = genriesz.grr_ate(
        X=X,
        Y=Y,
        basis=PolynomialBasis(degree=2),
        g=lambda a: 0.5 * a**2,
        grad_g=lambda a: a,
        inv_grad_g=lambda v: v,
        grad2_g=lambda a: np.ones_like(a),
    )
    assert isinstance(result.estimates["rw"].estimate, float)


def test_grr_functional_still_rejects_both_generator_and_g():
    X, Y = _ate_sample()
    with pytest.raises(ValueError, match="not both"):
        genriesz.grr_ate(
            X=X,
            Y=Y,
            basis=PolynomialBasis(degree=1),
            generator=SquaredGenerator(C=0.0),
            g=lambda a: a,
        )


def test_grr_functional_still_requires_a_generator_or_g():
    X, Y = _ate_sample()
    with pytest.raises(ValueError, match="must provide generator or g"):
        genriesz.grr_ate(X=X, Y=Y, basis=PolynomialBasis(degree=1))


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
