"""Basis and feature-map utilities.

In *genriesz*, Riesz representers and nuisance regressions are typically fit
in a (possibly high-dimensional) linear model on top of a **basis** / feature
map ``phi(x)``.

The API is intentionally lightweight:

- ``basis.fit(X, y=None)`` (optional)
- ``basis(X) -> (n, p)`` feature matrix
- ``basis.derivative(X, coordinate) -> (n, p)`` (optional; required for AME)

All docstrings and comments are in English as requested.
"""

from __future__ import annotations

import copy
import functools
import types
import warnings
from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import standardize_columns


def _as_2d_allow_1d(X: ArrayLike, *, name: str = "X") -> tuple[NDArray[np.float64], bool]:
    """Return (X2d, is_single).

    This helper lets bases support both batch inputs (n, d) and single-row
    inputs (d,). For single-row inputs, we reshape to (1, d).
    """

    X_ = np.asarray(X, dtype=float)
    if X_.ndim == 1:
        return X_.reshape(1, -1), True
    if X_.ndim == 2:
        return X_, False
    raise ValueError(f"{name} must be 1D or 2D. Got shape {X_.shape}.")


class Basis(Protocol):
    """Protocol for basis objects."""

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Basis:
        ...

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        ...

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        ...

    @property
    def n_features(self) -> int:
        ...

    def copy(self) -> Basis:
        ...


class BaseBasis:
    """Convenience base class implementing ``copy`` and a no-op ``fit``."""

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> BaseBasis:
        return self

    def copy(self):
        # Cross-fitting refits a copy per fold, so the copy must not share state
        # with the original or with the other folds. Sharing anything mutable
        # here would let one fold's training data reach another fold's features.
        return copy.deepcopy(self)

    @property
    def n_features(self) -> int:
        raise NotImplementedError

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement derivative().")


class CallableBasis(BaseBasis):
    """Wrap a Python callable as a basis.

    This is useful when you want to define a custom feature map
    without creating a full class.

    Parameters
    ----------
    func:
        A callable returning a feature matrix. It may accept either a 2D array
        ``X`` or a 1D array ``x``.
    derivative:
        Optional callable implementing the derivative feature map required by
        AME-type functionals.

    Notes
    -----
    We infer ``n_features`` during ``fit`` (or the first call).
    """

    def __init__(
        self,
        func: Callable[[ArrayLike], ArrayLike],
        *,
        derivative: Callable[[ArrayLike, int], ArrayLike] | None = None,
    ):
        self.func = func
        self._derivative = derivative
        self._n_features: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> CallableBasis:
        Phi = np.asarray(self.__call__(X), dtype=float)
        if Phi.ndim == 1:
            self._n_features = int(Phi.shape[0])
        elif Phi.ndim == 2:
            self._n_features = int(Phi.shape[1])
        else:
            raise ValueError("Callable basis must return 1D or 2D array")
        return self

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            raise RuntimeError("CallableBasis must be fit() before use.")
        return int(self._n_features)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        out = self.func(X2[0] if single else X2)
        Phi = np.asarray(out, dtype=float)

        if single:
            if Phi.ndim == 2 and Phi.shape[0] == 1:
                Phi = Phi[0]
            if Phi.ndim != 1:
                raise ValueError(
                    "CallableBasis(func) returned an invalid shape for single-row input"
                )
            if self._n_features is None:
                self._n_features = int(Phi.shape[0])
            return Phi

        if Phi.ndim == 1:
            # If the callable returns (p,) for batch input, treat it as one row.
            Phi = Phi.reshape(1, -1)
        if Phi.ndim != 2 or Phi.shape[0] != X2.shape[0]:
            raise ValueError("CallableBasis(func) must return an array of shape (n, p)")
        if self._n_features is None:
            self._n_features = int(Phi.shape[1])
        return Phi

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        if self._derivative is None:
            return super().derivative(X, coordinate)
        out = self._derivative(X, int(coordinate))
        return np.asarray(out, dtype=float)


# A class's ``__mro__``, ``__dict__`` and ``__name__`` read through the
# *metaclass*, whose ``__getattribute__`` the caller may have written. Invoke
# type's own accessors instead: they read the C-level slots and run nothing.
_type_mro = type.__dict__["__mro__"].__get__
_type_dict = type.__dict__["__dict__"].__get__
_type_name = type.__dict__["__name__"].__get__


# A distinct absence marker: ``None`` is a legitimate attribute value.
_MISSING = object()


def _classmethod_delegates() -> bool:
    """Whether ``classmethod.__get__`` calls the ``__get__`` of what it wraps.

    Chained classmethod descriptors were deprecated in 3.11 and removed in 3.13.
    Ask the interpreter rather than the version: the answer is what decides
    whether wrapping a descriptor in ``classmethod`` can run the caller's code.
    """

    class _Probe:
        ran = False

        def __get__(self, obj, objtype=None):
            _Probe.ran = True

    class _Holder:
        attr = classmethod(_Probe())

    bound = _Holder().attr  # noqa: F841 - the point is the binding, not the value
    return _Probe.ran


_CLASSMETHOD_DELEGATES = _classmethod_delegates()


# Descriptor types whose ``__get__`` is implemented in C and merely fetches:
# plain functions and the C-level routines, the slot accessors that ``__slots__``
# installs, a bound method (which returns itself, and became a descriptor in
# 3.13), and ``staticmethod``, which hands back what it wraps without binding it.
# Membership is tested by *exact* type, never with ``isinstance``: a subclass may
# override ``__get__`` with anything at all.
#
# Spelled out rather than tested with ``inspect.isroutine``, which is duck-typed:
# it answers True for any non-data descriptor, whose ``__get__`` is arbitrary.
_INERT_DESCRIPTORS = frozenset(
    {
        types.FunctionType,
        types.BuiltinFunctionType,
        types.MethodType,
        types.MethodDescriptorType,
        types.WrapperDescriptorType,
        types.MemberDescriptorType,
        types.ClassMethodDescriptorType,
        staticmethod,
    }
    | (set() if _CLASSMETHOD_DELEGATES else {classmethod})
)


# Wrappers whose ``__get__`` delegates to the object they wrap, and so are inert
# only as far as that object is. ``partialmethod`` delegates on every version;
# ``classmethod`` only where the interpreter still chains descriptors, and it is
# inert outright everywhere else. ``staticmethod`` never delegates -- it returns
# what it wraps untouched. Keyed by exact type, for the reason given above.
_DELEGATING_WRAPPERS = {functools.partialmethod: lambda attr: attr.func}
if _CLASSMETHOD_DELEGATES:
    _DELEGATING_WRAPPERS[classmethod] = lambda attr: attr.__func__


# Descriptors that carry the class they were defined on and refuse to bind to
# anything else. A plain function's ``__objclass__``, if the caller sets one, is
# metadata and constrains nothing, so only these types are asked.
_BINDING_CONSTRAINED = frozenset(
    {
        types.MemberDescriptorType,
        types.GetSetDescriptorType,
        types.MethodDescriptorType,
        types.WrapperDescriptorType,
        types.ClassMethodDescriptorType,
    }
)


def _dict_lookup(namespace, name: str) -> object:
    """Find a string key without letting the caller's keys answer the question.

    ``name in namespace`` hashes ``name`` and compares it against every colliding
    entry, and a namespace -- a class body, or an instance dict written through
    ``obj.__dict__`` -- may hold a key whose ``__hash__`` collides and whose
    ``__eq__`` is the caller's code. Iterating compares nothing.
    """

    for key, value in namespace.items():
        if type(key) is str and key == name:
            return value
    return _MISSING


def _class_lookup(cls: type, name: str) -> object:
    """Find ``name`` in ``cls``'s MRO without consulting the metaclass."""

    for klass in _type_mro(cls):
        found = _dict_lookup(_type_dict(klass), name)
        if found is not _MISSING:
            return found
    return _MISSING


def _defines(cls: type, name: str) -> bool:
    return _class_lookup(cls, name) is not _MISSING


def _is_subclass(cls: type, parent: type) -> bool:
    """``issubclass`` without the ``__class__`` and ``__subclasscheck__`` hooks.

    ``isinstance(obj, C)`` reads ``obj.__class__`` when the type check misses,
    and that is the caller's ``__getattribute__``. Deciding what an object *is*
    must not ask the object.
    """

    return any(klass is parent for klass in _type_mro(cls))


def _is_inert(attr: object) -> bool:
    """Whether binding ``attr`` would run code the caller wrote.

    An object that is not a descriptor at all binds to itself, so it is inert.
    A descriptor is inert only if its ``__get__`` is one of the C fetchers, or
    if it is a wrapper (see ``_DELEGATING_WRAPPERS``) around something inert:
    ``partialmethod(f)`` is safe for a function ``f`` and unsafe for a descriptor
    the caller defined, because its ``__get__`` delegates to the wrapped one's.

    A wrapper reached twice is a cycle -- ``pm.func = pm`` -- and is reported as
    not inert, which leaves it unresolved. A deep but finite nest is walked to
    the end, because binding it is what the interpreter would do.
    """

    seen: list[object] = []
    while True:
        cls = type(attr)
        unwrap = _DELEGATING_WRAPPERS.get(cls)
        if unwrap is None:
            return cls in _INERT_DESCRIPTORS or not _defines(cls, "__get__")
        if any(attr is wrapper for wrapper in seen):
            return False
        seen.append(attr)
        attr = unwrap(attr)
        if _overrides_attribute_access(type(attr)):
            # ``partialmethod.__get__`` delegates by *reading* ``func.__get__``,
            # so the wrapped object's own attribute access runs before any of the
            # reasoning below applies.
            return False


def _is_data_descriptor(attr: object) -> bool:
    """Whether ``attr`` outranks the instance dict when the attribute is *read*.

    A ``__set__`` alone does not do it. Reading consults the instance dict first
    unless the class attribute can answer the read, which means it needs a
    ``__get__`` as well.
    """

    cls = type(attr)
    if not _defines(cls, "__get__"):
        return False
    return _defines(cls, "__set__") or _defines(cls, "__delete__")


def _descriptor_binds_to(descriptor: object, cls: type) -> bool:
    """Whether ``descriptor.__get__`` would accept an instance of ``cls``.

    A slot accessor or a C routine carries the class it was defined on. Lifted
    onto an unrelated class it still looks like the right type, but binding it
    raises. Only those types are asked: a caller may set ``__objclass__`` on a
    plain function as metadata, where it constrains nothing, and reading it off
    an arbitrary object would run that object's ``__getattribute__``.
    """

    if type(descriptor) not in _BINDING_CONSTRAINED:
        return True
    owner = descriptor.__objclass__  # type: ignore[attr-defined]
    return any(klass is owner for klass in _type_mro(cls))


def _overrides_attribute_access(cls: type) -> bool:
    """Whether ``cls`` decides for itself what attribute access returns.

    A static answer then says nothing about what ``obj.fit`` will hand back, so
    the object cannot be certified as a Basis on the strength of one.

    Only the first ``__getattribute__`` in the MRO is consulted, since that is
    the one attribute access uses. ``object``, ``dict`` and the other built-ins
    each install their own as a C slot wrapper bound to themselves; anything else
    -- a function, ``None``, or a slot wrapper lifted off an unrelated class --
    is not the standard machinery and is not trusted.
    """

    for klass in _type_mro(cls):
        getattribute = _dict_lookup(_type_dict(klass), "__getattribute__")
        if getattribute is _MISSING:
            continue
        if type(getattribute) is not types.WrapperDescriptorType:
            return True
        return not _descriptor_binds_to(getattribute, cls)
    return False


def _instance_dict(obj: object) -> dict | None:
    """``obj``'s own attribute dict, or None if it has none we can read safely."""

    descriptor = _class_lookup(type(obj), "__dict__")
    if type(descriptor) is not types.GetSetDescriptorType:
        return None  # ``__slots__``, or a ``__dict__`` the caller has shadowed
    if not _descriptor_binds_to(descriptor, type(obj)):
        return None  # a ``__dict__`` accessor lifted from an unrelated class
    return descriptor.__get__(obj)


def _instances_define_getattr(cls: type) -> bool:
    """Whether ``cls``'s *instances* resolve missing attributes dynamically.

    Scanning the MRO dicts runs no code and asks the right question. Reading
    ``cls.__getattr__`` would instead find one defined on the metaclass, which
    governs attribute access on the class object, not on its instances.
    """

    return any(
        _dict_lookup(_type_dict(klass), "__getattr__") is not _MISSING
        for klass in _type_mro(cls)
    )


def _static_lookup(obj: object, name: str) -> tuple[object, bool]:
    """Find ``name`` as attribute access would, but run no code at all.

    Returns the raw attribute and whether it came from the class, since only a
    class attribute goes through the descriptor protocol. Data descriptors take
    precedence over the instance dict, as the data model prescribes.
    """

    from_class = _class_lookup(type(obj), name)
    if from_class is not _MISSING and _is_data_descriptor(from_class):
        return from_class, True
    namespace = _instance_dict(obj)
    if namespace is not None:
        from_instance = _dict_lookup(namespace, name)
        if from_instance is not _MISSING:
            return from_instance, False
    return from_class, True


def _lookup_method(obj: object, name: str) -> object | None:
    """Find a method without running code that deciding a type should not run.

    Reading the attribute normally would execute any descriptor behind it, and
    also the caller's ``__getattribute__``. This module already refuses to probe
    ``n_features`` for that reason, and ``fit`` and ``copy`` are no different: a
    caller's feature map may define either as a ``@property`` whose getter raises.
    So the name is looked up statically, walking the MRO dicts directly.

    A static lookup finds the descriptor rather than the value, which is wrong
    for the ones that merely fetch: a method must come back bound, and a value
    stored in a ``__slots__`` member must come back as the value. Bind those, and
    only those -- see :func:`_is_inert`. A ``@property``, or any descriptor the
    caller wrote, stays unresolved and is reported as absent, so an object whose
    ``fit`` is one is a feature map rather than a Basis. Note that such a
    descriptor may itself be callable: returning it would be enough to mistake
    the feature map for a Basis, and its ``__get__`` would then run after all.

    An object that defines ``__getattr__`` has asked for dynamic resolution, and
    a Basis may legitimately be a proxy, so a name the static lookup missed is
    fetched normally. Its ``__getattr__`` must raise ``AttributeError`` for a
    missing name, as the data model requires; anything else propagates.

    A class object is never a Basis: the protocol lives on instances, and a class
    passed here is a feature map that builds its features in ``__new__``. Saying
    so early also avoids the separate lookup rules attribute access on a class
    obeys.
    """

    if _is_subclass(type(obj), type):
        return None
    attr, from_class = _static_lookup(obj, name)
    if attr is _MISSING:
        return _dynamic_lookup(obj, name)
    if not from_class:
        return attr  # an instance-dict value is returned unbound, as attribute access does
    if not _is_inert(attr) or not _descriptor_binds_to(attr, type(obj)):
        return None
    if not _defines(type(attr), "__get__"):
        return attr  # not a descriptor: it binds to itself
    if type(attr) is types.MemberDescriptorType:
        # Reading an unset slot raises AttributeError, the data model's way of
        # saying the attribute is absent. That is the one meaning it can carry,
        # and attribute access answers it by consulting ``__getattr__`` next.
        try:
            return _bind(attr, obj)
        except AttributeError:
            return _dynamic_lookup(obj, name)
    return _bind(attr, obj)


def _bind(descriptor: object, obj: object) -> object:
    """Invoke the descriptor protocol the way the interpreter invokes it.

    ``descriptor.__get__`` is an attribute lookup *on the descriptor*, and an
    instance dict entry named ``__get__`` shadows the slot for every type whose
    own ``__get__`` is a non-data descriptor -- which is all of the inert ones.
    The protocol uses the type's slot, and so must this.
    """

    return type(descriptor).__get__(descriptor, obj, type(obj))  # type: ignore[attr-defined]


def _dynamic_lookup(obj: object, name: str) -> object | None:
    """Ask an object that opted into dynamic attributes, and only such an object."""

    if _instances_define_getattr(type(obj)):
        return getattr(obj, name, None)
    return None


def coerce_basis(basis: Basis | Callable) -> Basis:
    """Coerce a basis specification into a :class:`Basis`.

    The public API documents that users may pass either a Basis instance or a
    plain callable ``basis(X) -> Phi``. Only the latter is wrapped in
    :class:`CallableBasis`.

    A stateful basis defined outside this package satisfies :class:`Basis`
    without inheriting from :class:`BaseBasis`, so it is recognised by duck
    typing. Wrapping such an object would be wrong: ``CallableBasis.fit``
    infers ``n_features`` by *calling* the wrapped object rather than by
    delegating to its ``fit``, so the user's ``fit`` would never run.

    Every returned object has a callable ``copy``: a :class:`BaseBasis` is
    checked here, a duck-typed Basis by the predicate below, and
    :class:`CallableBasis` supplies its own. Callers may therefore call
    ``.copy()`` without guarding. Whether that call *succeeds* is the basis's
    own responsibility, and a failure is reported rather than swallowed.

    Notes
    -----
    Deciding what ``basis`` *is* must not run ``basis``'s code where that can be
    avoided. Do not probe ``basis.n_features``: many bases (e.g.
    :class:`PolynomialBasis` and :class:`TreatmentInteractionBasis`) expose it as
    a property that only works after ``fit()``, and ``hasattr(obj, 'n_features')``
    would trigger it. The same hazard applies to ``fit`` and ``copy``, which a
    caller's feature map may define as properties, so both go through
    :func:`_lookup_method`. The one place code does run is that helper's
    ``__getattr__`` fallback, for objects that opted into dynamic attributes.

    Define ``fit`` and ``copy`` as methods. A property named ``fit`` is read as
    a property, not as a method, and the object is treated as a feature map.
    """

    # CallableBasis is a BaseBasis, so this covers every built-in basis.
    if _is_subclass(type(basis), BaseBasis):
        if not callable(_lookup_method(basis, "copy")):
            shadow, _ = _static_lookup(basis, "copy")
            raise TypeError(
                f"{_type_name(type(basis))} shadows the Basis method 'copy' with a "
                f"{_type_name(type(shadow))}. genriesz fits a copy of the basis, so "
                "'copy' must be a method returning a new basis. Rename the attribute."
            )
        return basis

    # A user-defined Basis: ``copy`` and ``fit`` are part of the protocol, and
    # both must be *callable*. A plain feature map carrying unrelated ``fit`` or
    # ``copy`` attributes is not a Basis, and belongs in the CallableBasis branch
    # below; ``hasattr`` alone would misroute it and then fail on ``basis.copy()``.
    #
    # An object that writes its own ``__getattribute__`` is not certified here.
    # What the static lookup found is not what ``basis.fit`` will return, so the
    # evidence for calling it a Basis does not exist. It is a feature map.
    if (
        not _overrides_attribute_access(type(basis))
        and callable(_lookup_method(basis, "fit"))
        and callable(_lookup_method(basis, "copy"))
        and callable(basis)
    ):
        return basis  # type: ignore[return-value]

    # Otherwise, interpret it as a raw callable feature map.
    if callable(basis):
        return CallableBasis(basis)

    raise TypeError("basis must be a Basis instance or a callable basis(X)->Phi")


_UNFITTED_POLYNOMIAL_MESSAGE = (
    "PolynomialBasis must be fit() before use. Pass auto_fit=True to fit on the "
    "fly; that is leak-free here because PolynomialBasis learns only the monomial "
    "layout from the column count, not any data-dependent statistic."
)


class PolynomialBasis(BaseBasis):
    """Full polynomial features up to a given total degree.

    This class intentionally implements polynomial features without relying on
    scikit-learn so that the *core* package can depend only on NumPy and SciPy.
    The enumeration of monomials is deterministic and ordered by total degree,
    then lexicographic within each total degree.

    Derivatives are implemented analytically via the monomial exponent table.

    Calling the basis before :meth:`fit` raises ``RuntimeError``, as every basis
    but :class:`CallableBasis` does. Pass ``auto_fit=True`` to instead fit on the
    first call; this is leak-free for polynomials, whose ``fit`` reads only the
    number of input columns, not any data-dependent statistic.
    """

    def __init__(
        self, degree: int = 2, *, include_bias: bool = True, auto_fit: bool = False
    ):
        if int(degree) < 0:
            raise ValueError("degree must be >= 0")
        self.degree = int(degree)
        self.include_bias = bool(include_bias)
        # When False (default), calling before ``fit`` raises, matching every
        # other basis. When True, an unfitted call fits on the fly. That is safe
        # here -- unlike the kernel bases, ``fit`` learns only the monomial
        # layout from the column count, not any data-dependent statistic -- so
        # it leaks nothing, but it stays opt-in for a uniform contract.
        self.auto_fit = bool(auto_fit)

        self._powers: NDArray[np.int64] | None = None
        self._n_input_features: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> PolynomialBasis:
        X2, _ = _as_2d_allow_1d(X)

        d = int(X2.shape[1])
        self._n_input_features = d

        # Enumerate multi-indices u in Z^d_{>=0} with |u|<=degree.
        # Ordering: total degree, then lexicographic.
        powers: list[list[int]] = []

        def rec(pos: int, remaining: int, cur: list[int]) -> None:
            if pos == d - 1:
                cur[pos] = remaining
                powers.append(cur.copy())
                return
            for e in range(remaining + 1):
                cur[pos] = e
                rec(pos + 1, remaining - e, cur)

        cur = [0] * d
        for total in range(self.degree + 1):
            rec(0, total, cur)

        P = np.asarray(powers, dtype=int)
        if not self.include_bias:
            # Drop the intercept term u=(0,...,0), which is first by construction.
            P = P[1:, :]
        self._powers = P
        return self

    @property
    def n_features(self) -> int:
        if self._powers is None:
            raise RuntimeError("PolynomialBasis must be fit() before use.")
        return int(self._powers.shape[0])

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        if self._powers is None or self._n_input_features is None:
            if not self.auto_fit:
                raise RuntimeError(_UNFITTED_POLYNOMIAL_MESSAGE)
            self.fit(X2)
        if self._n_input_features is None or int(X2.shape[1]) != int(self._n_input_features):
            raise ValueError(
                "PolynomialBasis received X with a different number of columns than at fit time"
            )

        powers = self._powers
        n, d = X2.shape
        p = int(powers.shape[0])

        Phi = np.ones((n, p), dtype=float)
        for k in range(d):
            pk = powers[:, k]
            if np.all(pk == 0):
                continue
            Phi *= np.power(X2[:, [k]], pk.reshape(1, -1))

        return Phi[0] if single else Phi

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        """Derivative of the feature map wrt ``X[:, coordinate]``."""

        X2, single = _as_2d_allow_1d(X)
        if self._powers is None or self._n_input_features is None:
            if not self.auto_fit:
                raise RuntimeError(_UNFITTED_POLYNOMIAL_MESSAGE)
            self.fit(X2)
        if self._n_input_features is None or int(X2.shape[1]) != int(self._n_input_features):
            raise ValueError(
                "PolynomialBasis received X with a different number of columns than at fit time"
            )

        n, d = X2.shape
        coordinate = int(coordinate)
        if coordinate < 0 or coordinate >= d:
            raise ValueError(f"coordinate must be in [0, {d-1}]. Got {coordinate}.")

        powers = self._powers  # (p, d)
        p = powers.shape[0]
        pk = powers[:, coordinate].astype(int)  # (p,)

        Phi = np.asarray(self.__call__(X2), dtype=float)  # (n, p)

        # Default formula: d/dx_k prod x^p = p_k * prod x^p / x_k.
        xk = X2[:, coordinate].reshape(n, 1)
        xk_safe = np.where(xk != 0.0, xk, 1.0)

        der = Phi * pk.reshape(1, p) / xk_safe

        # Fix special case: p_k == 1 and x_k == 0. Derivative is the product of other factors.
        mask_feat = pk == 1
        if np.any(mask_feat):
            mask_obs = (xk.reshape(-1) == 0.0)
            if np.any(mask_obs):
                other_powers = powers[mask_feat].copy()
                other_powers[:, coordinate] = 0
                rest = np.ones((mask_obs.sum(), other_powers.shape[0]), dtype=float)
                X_sub = X2[mask_obs]
                for j in range(d):
                    pj = other_powers[:, j]
                    if np.all(pj == 0):
                        continue
                    rest *= np.power(X_sub[:, [j]], pj.reshape(1, -1))
                der[np.ix_(mask_obs, np.where(mask_feat)[0])] = rest

        der[:, pk == 0] = 0.0

        return der[0] if single else der


class TreatmentInteractionBasis(BaseBasis):
    """Interaction basis for binary-treatment functionals.

    Given a base basis on covariates ``Z`` (excluding the treatment), this basis
    maps ``X = [D, Z]`` to

        phi(X) = [ D * psi(Z) , (1 - D) * psi(Z) ].

    This is a convenient default for ATE/ATT/DID-style functionals.

    Notes
    -----
    ``fit(X, y)`` intentionally ignores the ``y`` argument passed by callers and
    fits the base basis with the treatment indicator as the supervision target,
    i.e. ``base_basis.fit(Z, y=D)``. Supervised base bases (e.g. forest-leaf
    encodings) therefore learn propensity-style splits, not outcome-style
    splits. Pass a pre-fit base basis if you need different supervision.
    """

    def __init__(self, *, base_basis: BaseBasis, treatment_index: int = 0):
        self.base_basis = base_basis
        self.treatment_index = int(treatment_index)
        self._base_dim: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> TreatmentInteractionBasis:
        X2, _ = _as_2d_allow_1d(X)
        if self.treatment_index < 0 or self.treatment_index >= X2.shape[1]:
            raise ValueError("treatment_index is out of bounds")
        D = X2[:, self.treatment_index]
        Z = np.delete(X2, self.treatment_index, axis=1)
        self.base_basis.fit(Z, y=D)
        self._base_dim = int(self.base_basis.n_features)
        return self

    @property
    def n_features(self) -> int:
        if self._base_dim is None:
            raise RuntimeError("TreatmentInteractionBasis must be fit() before use.")
        return 2 * int(self._base_dim)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        if self._base_dim is None:
            raise RuntimeError(
                "TreatmentInteractionBasis must be fit() on training data before use. "
                "Silently fitting on evaluation data would leak information."
            )

        D = X2[:, self.treatment_index].reshape(-1, 1)
        if not np.all(np.isin(np.unique(D), [0.0, 1.0])):
            raise ValueError("Treatment column must be binary (0/1).")

        Z = np.delete(X2, self.treatment_index, axis=1)
        Psi = np.asarray(self.base_basis(Z), dtype=float)
        out = np.concatenate([D * Psi, (1.0 - D) * Psi], axis=1)
        return out[0] if single else out

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        if self._base_dim is None:
            raise RuntimeError(
                "TreatmentInteractionBasis must be fit() on training data before use. "
                "Silently fitting on evaluation data would leak information."
            )

        coordinate = int(coordinate)
        if coordinate == self.treatment_index:
            raise ValueError(
                "Derivative w.r.t. the treatment indicator is not supported (binary variable)."
            )

        d = X2.shape[1]
        if coordinate < 0 or coordinate >= d:
            raise ValueError(f"coordinate must be in [0, {d-1}]. Got {coordinate}.")

        # Map full coordinate index -> Z coordinate index
        z_coord = coordinate - 1 if coordinate > self.treatment_index else coordinate

        D = X2[:, self.treatment_index].reshape(-1, 1)
        Z = np.delete(X2, self.treatment_index, axis=1)
        dPsi = self.base_basis.derivative(Z, z_coord)

        out = np.concatenate([D * dPsi, (1.0 - D) * dPsi], axis=1)
        return out[0] if single else out


class RBFRandomFourierBasis(BaseBasis):
    """RBF random Fourier features (Rahimi-Recht) with optional standardization."""

    def __init__(
        self,
        *,
        n_features: int = 500,
        sigma: float | str = 1.0,
        include_bias: bool = True,
        standardize: bool = True,
        random_state: int | None = None,
    ):
        if int(n_features) <= 0:
            raise ValueError("n_features must be positive")
        self.n_features_rff = int(n_features)
        # ``sigma`` may be a positive float or "auto" (median heuristic at fit).
        self.sigma = _validate_sigma_input(sigma)
        self.include_bias = bool(include_bias)
        self.standardize = bool(standardize)
        self.random_state = random_state

        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None
        self._W: NDArray[np.float64] | None = None
        self._b: NDArray[np.float64] | None = None
        self._sigma_resolved: float | None = None

    @property
    def sigma_(self) -> float:
        """The resolved bandwidth used by the kernel (after ``fit``)."""

        if self._sigma_resolved is None:
            raise RuntimeError(
                "RBFRandomFourierBasis must be fit() before sigma_ is available."
            )
        return float(self._sigma_resolved)

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> RBFRandomFourierBasis:
        X2, _ = _as_2d_allow_1d(X)
        n, d = X2.shape

        Xs, mean, std = standardize_columns(X2, enabled=self.standardize)
        sigma = _resolve_sigma(self.sigma, Xs, random_state=self.random_state)

        rng = np.random.default_rng(self.random_state)
        W = rng.normal(loc=0.0, scale=1.0 / sigma, size=(d, self.n_features_rff))
        b = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features_rff)

        self._mean = mean.astype(float)
        self._std = std.astype(float)
        self._W = W.astype(float)
        self._b = b.astype(float)
        self._sigma_resolved = float(sigma)
        return self

    @property
    def n_features(self) -> int:
        if self._W is None:
            raise RuntimeError("RBFRandomFourierBasis must be fit() before use.")
        return int(self.n_features_rff + (1 if self.include_bias else 0))

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        if self._W is None or self._b is None or self._mean is None or self._std is None:
            raise RuntimeError(
                "RBFRandomFourierBasis must be fit() on training data before use. "
                "Silently fitting on evaluation data would leak its standardization."
            )

        Z = (X2 - self._mean) / self._std
        proj = Z @ self._W + self._b
        feats = np.sqrt(2.0 / self.n_features_rff) * np.cos(proj)
        if self.include_bias:
            feats = np.column_stack([np.ones(len(X2), dtype=float), feats])
        feats = feats.astype(float)
        return feats[0] if single else feats

    def derivative(self, X: ArrayLike, coordinate: int) -> NDArray[np.float64]:
        """Derivative of the feature map wrt ``X[:, coordinate]``.

        d/dx_k [ sqrt(2/D) cos(W[:,j]^T (x-mu)/std + b_j) ]
            = -sqrt(2/D) * W[k,j]/std[k] * sin(W[:,j]^T (x-mu)/std + b_j)
        """
        X2, single = _as_2d_allow_1d(X)
        if self._W is None or self._b is None or self._mean is None or self._std is None:
            raise RuntimeError(
                "RBFRandomFourierBasis must be fit() on training data before use. "
                "Silently fitting on evaluation data would leak its standardization."
            )

        n, d = X2.shape
        coordinate = int(coordinate)
        if coordinate < 0 or coordinate >= d:
            raise ValueError(f"coordinate must be in [0, {d - 1}]. Got {coordinate}.")

        Z = (X2 - self._mean) / self._std
        proj = Z @ self._W + self._b  # (n, D)
        scale = self._W[coordinate, :] / self._std[coordinate]  # (D,)
        dfeats = np.sqrt(2.0 / self.n_features_rff) * (-np.sin(proj)) * scale  # (n, D)
        if self.include_bias:
            dfeats = np.column_stack([np.zeros(n, dtype=float), dfeats])
        dfeats = dfeats.astype(float)
        return dfeats[0] if single else dfeats


def _rbf_kernel(
    X: NDArray[np.float64],
    C: NDArray[np.float64],
    *,
    sigma: float,
) -> NDArray[np.float64]:
    """Compute the Gaussian (RBF) kernel matrix K(X, C)."""

    if float(sigma) <= 0:
        raise ValueError("sigma must be positive")
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)

    x2 = np.sum(X * X, axis=1).reshape(-1, 1)
    c2 = np.sum(C * C, axis=1).reshape(1, -1)
    dist2 = x2 + c2 - 2.0 * (X @ C.T)
    dist2 = np.maximum(dist2, 0.0)
    return np.exp(-dist2 / (2.0 * sigma * sigma))


def _median_pairwise_distance(
    Xs: NDArray[np.float64], *, max_rows: int = 512, random_state: int | None = 0
) -> float:
    """Median Euclidean distance between distinct rows of ``Xs``.

    Subsamples to at most ``max_rows`` rows because the computation is
    ``O(rows^2)``. Returns NaN when fewer than two rows are available.
    """

    Xs = np.asarray(Xs, dtype=float)
    n = Xs.shape[0]
    if n > int(max_rows):
        rng = np.random.default_rng(random_state)
        Xs = Xs[rng.choice(n, size=int(max_rows), replace=False)]
    x2 = np.sum(Xs * Xs, axis=1).reshape(-1, 1)
    dist2 = np.maximum(x2 + x2.reshape(1, -1) - 2.0 * (Xs @ Xs.T), 0.0)
    iu = np.triu_indices(Xs.shape[0], k=1)
    if iu[0].size == 0:
        return float("nan")
    return float(np.median(np.sqrt(dist2[iu])))


def _validate_sigma_input(sigma: float | str) -> float | str:
    """Validate a bandwidth argument that may be a positive float or ``"auto"``."""

    if isinstance(sigma, str):
        s = sigma.lower()
        if s != "auto":
            raise ValueError("sigma must be a positive float or 'auto'")
        return "auto"
    if float(sigma) <= 0:
        raise ValueError("sigma must be positive")
    return float(sigma)


def _resolve_sigma(
    sigma: float | str, Xs: NDArray[np.float64], *, random_state: int | None = 0
) -> float:
    """Resolve ``sigma`` to a positive float, applying the median heuristic.

    ``"auto"`` maps to the median pairwise distance of the (already standardized)
    training rows ``Xs``. If that heuristic is degenerate (e.g. duplicate rows),
    it falls back to ``1.0`` with a warning so the kernel stays well defined.
    """

    if not isinstance(sigma, str):
        return float(sigma)
    med = _median_pairwise_distance(Xs, random_state=random_state)
    if not np.isfinite(med) or med <= 0:
        warnings.warn(
            "sigma='auto' median heuristic produced a non-positive bandwidth "
            "(degenerate training rows); falling back to sigma=1.0.",
            UserWarning,
            stacklevel=3,
        )
        return 1.0
    return float(med)


class GaussianRKHSBasis(BaseBasis):
    """Gaussian-kernel RKHS basis using kernel evaluations at fitted center points.

    The feature map is

        phi_j(x) = K(x, c_j),

    where {c_j} are fitted centers (one basis function per center).

    This is a convenient default for RKHS regression in moderate dimensions.
    """

    def __init__(
        self,
        *,
        n_centers: int = 300,
        sigma: float | str = 1.0,
        include_bias: bool = True,
        standardize: bool = True,
        random_state: int | None = None,
        centers: ArrayLike | None = None,
    ):
        if int(n_centers) <= 0:
            raise ValueError("n_centers must be positive")
        self.n_centers = int(n_centers)
        # ``sigma`` may be a positive float or the string "auto" (median
        # heuristic, resolved on the training sample at fit time).
        self.sigma = _validate_sigma_input(sigma)
        self.include_bias = bool(include_bias)
        self.standardize = bool(standardize)
        self.random_state = random_state
        self._centers_input = centers

        self._centers: NDArray[np.float64] | None = None
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None
        self._sigma_resolved: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> GaussianRKHSBasis:
        X2, _ = _as_2d_allow_1d(X)
        n, d = X2.shape

        Xs, mean, std = standardize_columns(X2, enabled=self.standardize)

        if self._centers_input is not None:
            C = np.asarray(self._centers_input, dtype=float)
            if C.ndim != 2 or C.shape[1] != d:
                raise ValueError("centers must be 2D with the same number of columns as X")
            Cs = (C - mean) / std
        else:
            m = min(self.n_centers, n)
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n, size=m, replace=False)
            Cs = Xs[idx]

        self._mean = mean.astype(float)
        self._std = std.astype(float)
        self._centers = Cs.astype(float)
        # Resolve the bandwidth from this (training) fold only.
        self._sigma_resolved = _resolve_sigma(self.sigma, Xs, random_state=self.random_state)
        return self

    @property
    def sigma_(self) -> float:
        """The resolved bandwidth used by the kernel (after ``fit``)."""

        if self._sigma_resolved is None:
            raise RuntimeError("GaussianRKHSBasis must be fit() before sigma_ is available.")
        return float(self._sigma_resolved)

    def copy_with_params(self, **overrides: object) -> GaussianRKHSBasis:
        """Return a fresh, unfitted basis with selected constructor overrides.

        Used to build cross-validation candidates that differ only in the given
        parameters (e.g. ``sigma`` or ``n_centers``). The center-selection seed
        is pinned so candidates share the same center subsample unless
        ``random_state``/``n_centers``/``centers`` is itself overridden.
        """

        params: dict[str, object] = {
            "n_centers": self.n_centers,
            "sigma": self.sigma,
            "include_bias": self.include_bias,
            "standardize": self.standardize,
            "random_state": self.random_state,
            "centers": self._centers_input,
        }
        params.update(overrides)
        if params.get("random_state") is None:
            params["random_state"] = 0
        return GaussianRKHSBasis(**params)  # type: ignore[arg-type]

    @property
    def centers(self) -> NDArray[np.float64]:
        if self._centers is None:
            raise RuntimeError("GaussianRKHSBasis must be fit() before use.")
        return self._centers

    @property
    def n_features(self) -> int:
        if self._centers is None:
            raise RuntimeError("GaussianRKHSBasis must be fit() before use.")
        m = int(self._centers.shape[0])
        return m + (1 if self.include_bias else 0)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        if self._centers is None or self._mean is None or self._std is None:
            raise RuntimeError(
                "GaussianRKHSBasis must be fit() on training data before use. "
                "Silently fitting on evaluation data would leak centers and "
                "standardization."
            )

        Xs = (X2 - self._mean) / self._std
        K = _rbf_kernel(Xs, self._centers, sigma=self.sigma_)
        if self.include_bias:
            K = np.column_stack([np.ones(len(X2), dtype=float), K])
        K = K.astype(float)
        return K[0] if single else K

    def diagnostics(
        self,
        X: ArrayLike,
        *,
        ridge: float = 1e-8,
        max_rows: int = 512,
        random_state: int | None = 0,
    ) -> dict[str, float]:
        """Kernel-health summary of the fitted RBF feature map on ``X``.

        Reports whether the bandwidth ``sigma`` is in a usable range. A tiny
        ``sigma`` collapses every off-diagonal kernel value to ~0 (underfitting:
        each point only sees itself); a huge ``sigma`` makes all features nearly
        constant (features carry no signal). The returned scalars feed the
        kernel-health column of the coverage-failure tables.

        Parameters
        ----------
        X:
            Data on which to probe the kernel (typically the training fold).
        ridge:
            Ridge added to the Gram matrix before its condition number.
        max_rows:
            Subsample size used for the pairwise-distance / Gram statistics
            (kept modest because those are ``O(rows^2)`` / ``O(rows^3)``).
        random_state:
            Seed for the subsample (does not affect the fitted centers).

        Notes
        -----
        This is a read-only probe: it does not refit centers or standardization.
        """

        if self._centers is None or self._mean is None or self._std is None:
            raise RuntimeError("GaussianRKHSBasis must be fit() before diagnostics().")

        X2, _ = _as_2d_allow_1d(X)
        n = X2.shape[0]
        if n > int(max_rows):
            rng = np.random.default_rng(random_state)
            sub = rng.choice(n, size=int(max_rows), replace=False)
            X2 = X2[sub]

        Xs = (X2 - self._mean) / self._std

        # Median pairwise distance among the (standardized) probe rows: the scale
        # the median heuristic would target.
        med_pairwise = _median_pairwise_distance(Xs, max_rows=max_rows, random_state=random_state)

        # RBF activations (exclude the constant bias column).
        K = _rbf_kernel(Xs, self._centers, sigma=self.sigma_)
        row_l2 = np.sqrt(np.sum(K * K, axis=1))
        feat_var = np.var(K, axis=0)

        Phi = K
        if self.include_bias:
            Phi = np.column_stack([np.ones(Phi.shape[0], dtype=float), Phi])
        gram = (Phi.T @ Phi) / Phi.shape[0]
        gram = gram + float(ridge) * np.eye(gram.shape[0])
        evals = np.linalg.eigvalsh(gram)
        evals = np.clip(evals, 0.0, None)
        cond = float(evals[-1] / evals[0]) if evals[0] > 0 else float("inf")
        total = float(np.sum(evals))
        if total > 0:
            pk = evals / total
            pk = pk[pk > 0]
            eff_rank = float(np.exp(-np.sum(pk * np.log(pk))))
        else:
            eff_rank = float("nan")

        kernel_median = float(np.median(K))
        return {
            "sigma": float(self.sigma_),
            "median_pairwise_distance": med_pairwise,
            "kernel_median": kernel_median,
            "kernel_p05": float(np.percentile(K, 5)),
            "kernel_p95": float(np.percentile(K, 95)),
            "row_l2_mean": float(np.mean(row_l2)),
            "row_l2_min": float(np.min(row_l2)),
            "feature_variance_median": float(np.median(feat_var)),
            "feature_variance_min": float(np.min(feat_var)),
            "gram_condition_number": cond,
            "effective_rank": eff_rank,
            "underfitting": bool(kernel_median < 1e-3),
        }


class RBFNystromBasis(BaseBasis):
    """Nyström feature map for the RBF kernel.

    This class computes a Nyström approximation to the implicit RBF-RKHS feature
    map. Compared to :class:`GaussianRKHSBasis`, Nyström features apply an
    additional whitening transform based on the kernel matrix among centers.

    The feature map is

        Phi(x) = K(x, C) (K(C, C) + jitter I)^{-1/2}.

    Notes
    -----
    - This is an O(m^3) preprocessing step in the number of centers ``m``.
    - For large ``m``, consider :class:`RBFRandomFourierBasis`.
    """

    def __init__(
        self,
        *,
        n_centers: int = 300,
        sigma: float | str = 1.0,
        include_bias: bool = True,
        standardize: bool = True,
        random_state: int | None = None,
        centers: ArrayLike | None = None,
        jitter: float = 1e-8,
    ):
        if int(n_centers) <= 0:
            raise ValueError("n_centers must be positive")
        if float(jitter) <= 0:
            raise ValueError("jitter must be positive")

        self.n_centers = int(n_centers)
        # ``sigma`` may be a positive float or "auto" (median heuristic at fit).
        self.sigma = _validate_sigma_input(sigma)
        self.include_bias = bool(include_bias)
        self.standardize = bool(standardize)
        self.random_state = random_state
        self._centers_input = centers
        self.jitter = float(jitter)

        self._centers: NDArray[np.float64] | None = None
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None
        self._inv_sqrt: NDArray[np.float64] | None = None
        self._sigma_resolved: float | None = None

    @property
    def sigma_(self) -> float:
        """The resolved bandwidth used by the kernel (after ``fit``)."""

        if self._sigma_resolved is None:
            raise RuntimeError("RBFNystromBasis must be fit() before sigma_ is available.")
        return float(self._sigma_resolved)

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> RBFNystromBasis:
        X2, _ = _as_2d_allow_1d(X)
        n, d = X2.shape

        Xs, mean, std = standardize_columns(X2, enabled=self.standardize)
        self._sigma_resolved = _resolve_sigma(self.sigma, Xs, random_state=self.random_state)

        if self._centers_input is not None:
            C = np.asarray(self._centers_input, dtype=float)
            if C.ndim != 2 or C.shape[1] != d:
                raise ValueError("centers must be 2D with the same number of columns as X")
            Cs = (C - mean) / std
        else:
            m = min(self.n_centers, n)
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n, size=m, replace=False)
            Cs = Xs[idx]

        Kmm = _rbf_kernel(Cs, Cs, sigma=self.sigma_)
        Kmm = Kmm + self.jitter * np.eye(Kmm.shape[0])

        # Symmetric eigendecomposition
        evals, evecs = np.linalg.eigh(Kmm)
        evals = np.maximum(evals, self.jitter)
        inv_sqrt = evecs @ (np.diag(1.0 / np.sqrt(evals))) @ evecs.T

        self._mean = mean.astype(float)
        self._std = std.astype(float)
        self._centers = Cs.astype(float)
        self._inv_sqrt = inv_sqrt.astype(float)
        return self

    @property
    def centers(self) -> NDArray[np.float64]:
        if self._centers is None:
            raise RuntimeError("RBFNystromBasis must be fit() before use.")
        return self._centers

    @property
    def n_features(self) -> int:
        if self._centers is None:
            raise RuntimeError("RBFNystromBasis must be fit() before use.")
        m = int(self._centers.shape[0])
        return m + (1 if self.include_bias else 0)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        X2, single = _as_2d_allow_1d(X)
        if (
            self._centers is None
            or self._mean is None
            or self._std is None
            or self._inv_sqrt is None
        ):
            raise RuntimeError(
                "RBFNystromBasis must be fit() on training data before use. "
                "Silently fitting on evaluation data would leak centers and "
                "standardization."
            )

        Xs = (X2 - self._mean) / self._std
        Knm = _rbf_kernel(Xs, self._centers, sigma=self.sigma_)
        Phi = Knm @ self._inv_sqrt
        if self.include_bias:
            Phi = np.column_stack([np.ones(len(X2), dtype=float), Phi])
        Phi = Phi.astype(float)
        return Phi[0] if single else Phi


class KNNCatchmentBasis(BaseBasis):
    """kNN nearest-neighbor indicator basis.

    After fitting on a set of *centers*, evaluating on query points returns a
    (dense) indicator matrix whose entry (i, j) is 1 if center j is among the
    k nearest neighbors of query i, and 0 otherwise.

    Notes
    -----
    - This is mainly intended for small-to-medium center sets used in notebooks.
    - For large-scale matching, prefer the dedicated NN/LSIF utilities.
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 1,
        include_bias: bool = False,
        standardize: bool = True,
        random_state: int | None = None,
    ):
        if int(n_neighbors) <= 0:
            raise ValueError("n_neighbors must be positive")
        self.n_neighbors = int(n_neighbors)
        self.include_bias = bool(include_bias)
        self.standardize = bool(standardize)
        self.random_state = random_state

        self._centers: NDArray[np.float64] | None = None
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None
        self._nn = None

    def fit(self, centers: ArrayLike, y: ArrayLike | None = None) -> KNNCatchmentBasis:
        from scipy.spatial import cKDTree

        C = np.asarray(centers, dtype=float)
        if C.ndim != 2:
            raise ValueError(f"centers must be 2D. Got shape {C.shape}.")
        if C.shape[0] == 0:
            raise ValueError("centers must contain at least one row.")
        if self.n_neighbors > C.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} exceeds the number of centers={C.shape[0]}."
            )

        C_std, mean, std = standardize_columns(C, enabled=self.standardize)

        tree = cKDTree(C_std)

        self._centers = C
        self._mean = mean
        self._std = std
        self._nn = tree
        return self

    @property
    def n_features(self) -> int:
        if self._centers is None:
            raise RuntimeError("KNNCatchmentBasis must be fit() before use.")
        m = int(self._centers.shape[0])
        return m + (1 if self.include_bias else 0)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        if self._nn is None or self._centers is None or self._mean is None or self._std is None:
            raise RuntimeError("KNNCatchmentBasis must be fit() before use.")

        Q2, single = _as_2d_allow_1d(X)

        Q_std = (Q2 - self._mean) / self._std

        # cKDTree.query returns shape (n,) when k=1 and (n,k) when k>1.
        _, ind = self._nn.query(Q_std, k=self.n_neighbors)
        ind = np.asarray(ind)
        if ind.ndim == 1:
            ind = ind.reshape(-1, 1)

        n = len(Q2)
        m = int(self._centers.shape[0])
        Phi = np.zeros((n, m), dtype=float)

        for i in range(n):
            Phi[i, ind[i]] = 1.0

        if self.include_bias:
            Phi = np.column_stack([np.ones(n, dtype=float), Phi])

        return Phi[0] if single else Phi
