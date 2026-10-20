"""Histogram density-ratio shape, scalar GRR fits and ridge score moments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def basis(u, z):
    """The six orthonormal quadratic regressors on the unit square."""
    u, z = np.broadcast_arrays(np.asarray(u, float), np.asarray(z, float))
    if u.ndim != 1 or not np.isfinite(u).all() or not np.isfinite(z).all():
        raise ValueError("covariates must be finite one-dimensional arrays")
    if np.any((u <= 0) | (u >= 1) | (z <= 0) | (z >= 1)):
        raise ValueError("covariates must lie strictly inside the unit square")
    x, w = 2 * u - 1, 2 * z - 1
    return np.column_stack(
        (
            np.ones(len(u)),
            np.sqrt(3) * x,
            np.sqrt(3) * w,
            3 * x * w,
            np.sqrt(5) * (6 * u**2 - 6 * u + 1),
            np.sqrt(5) * (6 * z**2 - 6 * z + 1),
        )
    )


def fit_shape(source_counts, target_counts):
    """Minimize the marginal UKL criterion with one added count in each cell."""
    source, target = np.asarray(source_counts), np.asarray(target_counts)
    if (
        source.ndim != 1
        or source.shape != target.shape
        or not len(source)
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or np.any(source < 0)
        or np.any(target < 0)
        or np.any(source != np.floor(source))
        or np.any(target != np.floor(target))
        or source.sum() <= 0
        or source.sum() != target.sum()
    ):
        raise ValueError("equal positive pilot sizes and nonnegative integer counts are required")
    q = (target + 1) / (source + 1)
    shape = q / q.mean()
    gradient = (source + 1) - (target + 1) / q
    return shape, {
        "pilot_stationarity": float(np.max(np.abs(gradient))),
        "pilot_stationarity_scaled": float(np.max(np.abs(gradient)) / (1 + np.max(source + 1))),
        "shape_integral_error": float(abs(shape.mean() - 1)),
        "shape_min": float(shape.min()),
        "shape_max": float(shape.max()),
    }


def fit_scales(shape, source_v, source_v2, target_v, size):
    """Fit SQ and UKL scalar GRR objectives from their complete sample moments."""
    shape = np.asarray(shape, float)
    arrays = tuple(np.asarray(x, float) for x in (source_v, source_v2, target_v))
    if (
        size <= 0
        or shape.ndim != 1
        or any(x.shape != shape.shape for x in arrays)
        or np.any(shape <= 0)
        or not np.isfinite(shape).all()
        or any(not np.isfinite(x).all() or np.any(x < 0) for x in arrays)
    ):
        raise ValueError("positive finite shape and valid scale moments are required")
    c = float(arrays[0] @ shape / size)
    a = float(arrays[1] @ shape**2 / size)
    b = float(arrays[2] @ shape / size)
    if not (a > 0 and b > 0 and c > 0):
        raise ValueError("scale moments must be strictly positive")
    sq, ukl = b / a, 1 / c
    if not np.isfinite([sq, ukl]).all() or min(sq, ukl) <= 0:
        raise ArithmeticError("the scalar GRR solution is not finite and interior")
    return {"SQ": sq, "UKL": ukl}, {
        "scale_source_v": c,
        "scale_source_v2": a,
        "scale_target_v": b,
        "SQ_stationarity_scaled": abs(sq * a - b) / (1 + abs(b)),
        "UKL_stationarity_scaled": abs(ukl * c - 1),
    }


def fit_ridge(gram, xy, size, penalty):
    """Fit every regression coefficient, leaving the intercept unpenalized."""
    gram, xy = np.asarray(gram, float) / size, np.asarray(xy, float) / size
    if (
        size <= 0
        or penalty <= 0
        or gram.shape != (6, 6)
        or xy.shape != (6,)
        or not np.isfinite(gram).all()
        or not np.isfinite(xy).all()
    ):
        raise ValueError("finite six-regressor moments and positive size and penalty are required")
    matrix = gram + np.diag([0, *([penalty] * 5)])
    coefficient = np.linalg.solve(matrix, xy)
    residual = float(np.max(np.abs(matrix @ coefficient - xy)) / (1 + np.max(np.abs(xy))))
    if not np.isfinite(coefficient).all():
        raise ArithmeticError("ridge coefficients are not finite")
    return coefficient, {
        "normal_equation_scaled": residual,
        "ridge_min_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
    }


@dataclass
class SourceMoments:
    """All source observations enter both ridge and evaluation score moments."""

    h: int
    b: float
    n: int = 0
    pilot_n: int = 0
    scale_n: int = 0
    gram: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    xy: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def __post_init__(self):
        if self.h < 1 or not 0 < self.b < 0.5:
            raise ValueError("positive histogram size and exponent below one half are required")
        self.pilot_counts = np.zeros(self.h, dtype=np.int64)
        self.cell_counts = np.zeros(self.h, dtype=np.int64)
        self.v_augmented = np.zeros((self.h, 7))
        self.v2_augmented = np.zeros((self.h, 7, 7))
        self.scale_v = np.zeros(self.h)
        self.scale_v2 = np.zeros(self.h)

    def add(self, u, z, y, *, pilot):
        phi = basis(u, z)
        y = np.asarray(y, float)
        if y.shape != (len(phi),) or not np.isfinite(y).all():
            raise ValueError("outcomes must be finite and match the covariates")
        cells = (np.asarray(z) * self.h).astype(int)
        v = np.asarray(u) ** (-self.b)
        augmented = np.column_stack((phi, y))
        self.n += len(y)
        self.gram += phi.T @ phi
        self.xy += phi.T @ y
        counts = np.bincount(cells, minlength=self.h)
        self.cell_counts += counts
        if pilot:
            self.pilot_n += len(y)
            self.pilot_counts += counts
        else:
            self.scale_n += len(y)
            self.scale_v += np.bincount(cells, weights=v, minlength=self.h)
            self.scale_v2 += np.bincount(cells, weights=v**2, minlength=self.h)
        weighted = augmented * v[:, None]
        for i in range(7):
            self.v_augmented[:, i] += np.bincount(cells, weights=weighted[:, i], minlength=self.h)
            for j in range(i, 7):
                term = np.bincount(cells, weights=weighted[:, i] * weighted[:, j], minlength=self.h)
                self.v2_augmented[:, i, j] += term
                if i != j:
                    self.v2_augmented[:, j, i] += term

    def score(self, coefficient, shape, scale):
        residual_coefficient = np.r_[-np.asarray(coefficient), 1.0]
        first = float(
            scale * (np.asarray(shape) @ self.v_augmented @ residual_coefficient) / self.n
        )
        second = float(
            scale**2
            * np.einsum(
                "h,hij,i,j->",
                np.asarray(shape) ** 2,
                self.v2_augmented,
                residual_coefficient,
                residual_coefficient,
            )
            / self.n
        )
        if second < 0 or not np.isfinite([first, second]).all():
            raise ArithmeticError("invalid source score moments")
        return first, second


@dataclass
class TargetMoments:
    """Target regression moments and pilot and scale sample moments."""

    h: int
    b: float
    n: int = 0
    pilot_n: int = 0
    scale_n: int = 0
    phi: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gram: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))

    def __post_init__(self):
        if self.h < 1 or not 0 < self.b < 0.5:
            raise ValueError("positive histogram size and exponent below one half are required")
        self.pilot_counts = np.zeros(self.h, dtype=np.int64)
        self.cell_counts = np.zeros(self.h, dtype=np.int64)
        self.scale_v = np.zeros(self.h)

    def add(self, u, z, *, pilot):
        phi = basis(u, z)
        cells = (np.asarray(z) * self.h).astype(int)
        counts = np.bincount(cells, minlength=self.h)
        self.n += len(phi)
        self.phi += phi.sum(axis=0)
        self.gram += phi.T @ phi
        self.cell_counts += counts
        if pilot:
            self.pilot_n += len(phi)
            self.pilot_counts += counts
        else:
            self.scale_n += len(phi)
            self.scale_v += np.bincount(cells, weights=np.asarray(u) ** (-self.b), minlength=self.h)

    def score(self, coefficient, center):
        coefficient = np.asarray(coefficient)
        first = float(self.phi @ coefficient / self.n)
        second = float(
            coefficient @ self.gram @ coefficient / self.n - 2 * center * first + center**2
        )
        if second < 0 or not np.isfinite([first, second]).all():
            raise ArithmeticError("invalid target score moments")
        return first, second
