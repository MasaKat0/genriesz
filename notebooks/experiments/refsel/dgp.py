"""Data-generating processes for the reference-based selection experiment.

The designs follow ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md`` section 6.
Both designs keep ``theta_0 = E[Y(1) - Y(0)] = 1`` for every value of the hidden
misspecification scale, because the hidden direction enters the untreated
regression function and the treatment index but never the conditional treatment
effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

Design = Literal["low", "high"]

#: Standard deviation of the outcome noise. Known in simulation, so the audit in
#: :mod:`refsel.audit` can evaluate the conditional score variance analytically.
NOISE_SD = 1.0

#: True average treatment effect in every design.
THETA0 = 1.0


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a reproducible seed from a base seed and scenario identifiers.

    The digest does not depend on worker count or job completion order, so the
    same scenario and repetition always receive the same random numbers.
    """

    payload = "|".join(str(x) for x in (base_seed, *parts)).encode("utf-8")
    digest = blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


@dataclass(frozen=True)
class GeneratedData:
    """One simulated sample.

    ``y`` is ``None`` for integration samples, which the analytic audit
    evaluates without drawing outcome noise.
    """

    X: FloatArray
    gamma0: FloatArray
    alpha0: FloatArray
    tau: FloatArray
    propensity: FloatArray
    y: FloatArray | None = None

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    def outcomes(self) -> FloatArray:
        if self.y is None:
            raise RuntimeError("This sample was generated without outcomes.")
        return self.y


def covariate_dimension(design: Design) -> int:
    return 5 if design == "low" else 50


def _covariance_matrix(d: int, rho: float = 0.5) -> FloatArray:
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def hidden_direction(Z: FloatArray) -> FloatArray:
    """Return the direction that no candidate dictionary can represent.

    ``cos(2 pi Z_1)`` is nearly orthogonal to the span of the rich dictionary
    under a standard normal first coordinate. The projection residual is checked
    in ``tests/test_reference_selection_experiment.py``.
    """

    return np.cos(2.0 * np.pi * np.asarray(Z, dtype=float)[:, 0])


def _treatment_index(Z: FloatArray, design: Design, hidden_scale: float) -> FloatArray:
    if design == "low":
        base = (
            0.6 * Z[:, 0]
            - 0.4 * Z[:, 1]
            + 0.5 * (Z[:, 2] ** 2 - 1.0)
            + 0.3 * np.sin(Z[:, 3])
        )
    else:
        base = (
            0.5 * Z[:, 0]
            - 0.4 * Z[:, 1]
            + 0.3 * Z[:, 2] * Z[:, 3]
            + 0.4 * np.sin(Z[:, 4])
            + 0.2 * (Z[:, 5] ** 2 - 1.0)
        )
    if hidden_scale == 0.0:
        return base
    return base + hidden_scale * hidden_direction(Z)


def _mu0(Z: FloatArray, design: Design, hidden_scale: float) -> FloatArray:
    base = (
        1.0
        + Z[:, 0]
        + 0.5 * (Z[:, 1] ** 2 - 1.0)
        + 0.5 * np.sin(Z[:, 2])
        + 0.25 * Z[:, 3] * Z[:, 4]
    )
    if design == "high":
        base = base + 0.25 * np.abs(Z[:, 5])
    if hidden_scale == 0.0:
        return base
    return base + hidden_scale * hidden_direction(Z)


def _tau(Z: FloatArray, design: Design) -> FloatArray:
    """Conditional treatment effect. Never touched by the hidden direction.

    This keeps ``theta_0 = E[tau(Z)] = 1`` for every ``hidden_scale``.
    """

    if design == "low":
        return 1.0 + 0.5 * Z[:, 0] - 0.25 * Z[:, 1]
    return 1.0 + 0.4 * Z[:, 0] - 0.2 * Z[:, 1] + 0.2 * np.sin(Z[:, 2])


def propensity(
    Z: FloatArray,
    *,
    design: Design,
    overlap_scale: float,
    hidden_scale: float,
) -> FloatArray:
    index = np.clip(overlap_scale * _treatment_index(Z, design, hidden_scale), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-index))


def true_representer(
    X: FloatArray, *, design: Design, overlap_scale: float, hidden_scale: float
) -> FloatArray:
    X = np.asarray(X, dtype=float)
    D = X[:, 0]
    e = propensity(X[:, 1:], design=design, overlap_scale=overlap_scale, hidden_scale=hidden_scale)
    return D / e - (1.0 - D) / (1.0 - e)


def true_outcome(X: FloatArray, *, design: Design, hidden_scale: float) -> FloatArray:
    X = np.asarray(X, dtype=float)
    D = X[:, 0]
    Z = X[:, 1:]
    return _mu0(Z, design, hidden_scale) + D * _tau(Z, design)


def generate_data(
    *,
    n: int,
    design: Design,
    overlap_scale: float,
    hidden_scale: float,
    seed: int,
    with_outcome: bool = True,
) -> GeneratedData:
    """Draw one sample from a design.

    Set ``with_outcome=False`` for integration samples. The analytic audit needs
    ``alpha0`` and ``gamma0`` but not realized outcomes, so skipping the noise
    draw saves both time and memory.
    """

    rng = np.random.default_rng(seed)
    d = covariate_dimension(design)
    if design == "low":
        Z = rng.normal(size=(n, d))
    else:
        Z = rng.multivariate_normal(np.zeros(d), _covariance_matrix(d), size=n)
    e = propensity(Z, design=design, overlap_scale=overlap_scale, hidden_scale=hidden_scale)
    D = rng.binomial(1, e, size=n).astype(float)
    tau = _tau(Z, design)
    gamma0 = _mu0(Z, design, hidden_scale) + D * tau
    alpha0 = D / e - (1.0 - D) / (1.0 - e)
    X = np.column_stack((D, Z))
    y = gamma0 + NOISE_SD * rng.normal(size=n) if with_outcome else None
    return GeneratedData(X=X, gamma0=gamma0, alpha0=alpha0, tau=tau, propensity=e, y=y)


@dataclass(frozen=True)
class FoldRoles:
    """Index sets for one rotation of the three sample roles."""

    training: IntArray
    diagnostic: IntArray
    evaluation: IntArray


def make_fold_roles(n: int, n_folds: int, seed: int) -> tuple[FoldRoles, ...]:
    """Create rotating training, diagnostic, and evaluation roles.

    Fold ``k`` is the evaluation sample and fold ``k + 1 mod n_folds`` is the
    diagnostic sample; the remaining folds train.
    """

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = tuple(np.asarray(x, dtype=np.int64) for x in np.array_split(perm, n_folds))
    roles: list[FoldRoles] = []
    all_idx = np.arange(n, dtype=np.int64)
    for k in range(n_folds):
        evaluation = folds[k]
        diagnostic = folds[(k + 1) % n_folds]
        excluded = np.zeros(n, dtype=bool)
        excluded[evaluation] = True
        excluded[diagnostic] = True
        training = all_idx[~excluded]
        roles.append(FoldRoles(training=training, diagnostic=diagnostic, evaluation=evaluation))
    return tuple(roles)
