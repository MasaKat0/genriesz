"""Nonlinear partition GRR fitting and joint continuous-covariate regressions."""

from __future__ import annotations

import heapq

import numpy as np

from .att_histogram import (
    GRADIENT_TOL,
    MAX_NODES,
    OBJECTIVE_TOL,
    _OddsProfile,
    _quadratic_min,
    _root_candidate,
)
from .att_histogram import (
    _fit_info as _base_fit_info,
)

DEFAULT_BOUNDS = ((-3.0, 2.0), (0.05, 0.49))


def _vector(value, name, *, nonempty=False):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not np.isfinite(value).all() or (nonempty and not len(value)):
        raise ValueError(f"{name} must be a finite {'nonempty ' if nonempty else ''}vector")
    return value


def _u_vector(value, name):
    value = _vector(value, name)
    if np.any((value <= 0) | (value > 1)):
        raise ValueError(f"{name} must belong to (0,1]")
    return value


def _z_ids(value, n_cells):
    value = _vector(value, "Z")
    if not isinstance(n_cells, (int, np.integer)) or n_cells < 1:
        raise ValueError("n_cells must be a positive integer")
    if np.any((value < 0) | (value >= 1)):
        raise ValueError("Z coordinates must belong to [0,1)")
    return np.floor(n_cells * value).astype(int)


class _CellProfile(_OddsProfile):
    """Cell-scaled objective, with source and target sums divided by M/J."""

    def __init__(self, source_u, target_u, loss, normalizer, penalty=0.0,
                 bounds=DEFAULT_BOUNDS, source_size=None):
        source_u = _u_vector(source_u, "source_u")
        target_u = _u_vector(target_u, "target_u")
        self.arm = {"SQ": "SQ", "UKL": "UKL0"}.get(loss)
        self.bounds = np.asarray(bounds, dtype=float)
        if (
            self.arm is None or not np.isfinite(normalizer) or normalizer <= 0
            or not np.isfinite(penalty) or penalty < 0
            or self.bounds.shape != (2, 2) or not np.isfinite(self.bounds).all()
            or np.any(self.bounds[:, 0] >= self.bounds[:, 1])
            or not 0 <= self.bounds[1, 0] < self.bounds[1, 1] < 0.5
        ):
            raise ValueError(
                "require SQ or UKL, positive normalizer, nonnegative penalty, and valid bounds"
            )
        self.vc, self.vt = -np.log(source_u), -np.log(target_u)
        self.V = np.concatenate((self.vc, self.vt))
        self.n = float(normalizer)
        self.p = len(target_u) / self.n
        self.td = float(self.vt.sum() / self.n)
        self.penalty = float(penalty)
        self.source_size = float(normalizer if source_size is None else source_size)
        if not np.isfinite(self.source_size) or self.source_size <= 0:
            raise ValueError("source_size must be positive and finite")
        self.clo, self.chi = np.exp(self.bounds[0])
        self._moment_cache = {}
        self._profile_cache = {}

    def at(self, b):
        b = float(b)
        if b in self._profile_cache:
            return self._profile_cache[b]
        m1, d1, _, m2, d2, _ = self.moments(b)
        if self.arm == "SQ":
            c = float(np.clip(m1 / m2, self.clo, self.chi)) if m2 > 0 else (
                self.chi if m1 > 0 else self.clo
            )
            value = c * c * m2 - 2 * c * m1
            slope = c * c * d2 - 2 * c * d1
        else:
            c = float(np.clip(self.p / m2, self.clo, self.chi)) if m2 > 0 else (
                self.chi if self.p > 0 else self.clo
            )
            value = c * m2 - self.p * np.log(c) - self.td * b
            slope = c * d2 - self.td
        a = self.bounds[0, 0] if c == self.clo else (
            self.bounds[0, 1] if c == self.chi else np.log(c)
        )
        result = (float(value + self.penalty * b * b / 2),
                  float(slope + self.penalty * b), float(a))
        self._profile_cache[b] = result
        return result

    def objective(self, coefficient):
        value, gradient = super().objective(coefficient)
        b = coefficient[1]
        return value + self.penalty * b * b / 2, gradient + [0.0, self.penalty * b]


def _sq_bound(profile, lo, hi):
    """Analytic lower bound evaluated in double precision, without interval arithmetic."""
    mid = (lo + hi) / 2
    points = (lo, mid, hi)
    moment = [profile.moments(x) for x in points]
    ml, _, mu = moment
    penalty = profile.penalty
    if not len(profile.vc):
        chord = (mu[0] - ml[0]) / (hi - lo)
        target_slope = 2 * profile.chi * chord
        if penalty == 0 or target_slope >= penalty * hi:
            x = hi
        elif target_slope <= penalty * lo:
            x = lo
        else:
            x = target_slope / penalty
        return float(-2 * profile.chi * (ml[0] + chord * (x - lo)) + penalty * x * x / 2)
    clo = np.clip(ml[0] / mu[3], profile.clo, profile.chi)
    chi = np.clip(mu[0] / ml[3], profile.clo, profile.chi)
    dlow = _quadratic_min(ml[4], mu[1], clo, chi) + penalty * lo
    dhigh = max(ml[1] * -2 * c + mu[4] * c * c for c in (clo, chi)) + penalty * hi
    margin = 64 * np.finfo(float).eps * (1 + abs(dlow) + abs(dhigh))
    if dlow > margin:
        return profile.at(lo)[0]
    if dhigh < -margin:
        return profile.at(hi)[0]
    starts = np.array([m[3] + m[4] * (lo - x) for m, x in zip(moment, points, strict=True)])
    slopes = np.array([m[4] for m in moment])
    knots = [lo, hi]
    for i in range(3):
        for j in range(i):
            if slopes[i] != slopes[j]:
                x = lo + (starts[j] - starts[i]) / (slopes[i] - slopes[j])
                if lo < x < hi:
                    knots.append(float(x))
    raw_bounds = []
    for x in knots:
        upper_m1 = ml[0] + (x - lo) * (mu[0] - ml[0]) / (hi - lo)
        lower_m2 = float(np.max(starts + slopes * (x - lo)))
        raw_bounds.append(_quadratic_min(lower_m2, upper_m1, clo, chi))
    # On each segment the unpenalized envelope is concave; each penalty tangent is affine.
    lower = max(
        min(value + penalty * t * x - penalty * t * t / 2
            for value, x in zip(raw_bounds, knots, strict=True))
        for t in points
    )
    hlow = _quadratic_min(ml[5], mu[2], clo, chi)
    cross = max(abs(2 * clo * ml[4] - 2 * mu[1]), abs(2 * chi * mu[4] - 2 * ml[1]))
    curvature = hlow - cross**2 / (2 * ml[3]) + penalty
    if curvature > 0:
        for x in points:
            value, derivative, _ = profile.at(x)
            y = np.clip(x - derivative / curvature, lo, hi)
            lower = max(lower, value + derivative * (y - x) + curvature * (y - x)**2 / 2)
    return float(lower)


def _fit_info(profile, coefficient, lower, nodes, forced_status=None):
    info = _base_fit_info(profile, coefficient, lower, nodes)
    objective_tol = min(OBJECTIVE_TOL, profile.source_size**-2)
    gradient_tol = min(GRADIENT_TOL, profile.source_size**-1)
    if forced_status is not None:
        info["status"] = forced_status
    elif info["status"] not in ("nonfinite", "numerical_bound_failure"):
        info["status"] = (
            "global_gap_not_met" if info["gap"] > objective_tol * (1 + abs(info["objective"]))
            else "stationarity_not_met" if info["stationarity_scaled"] > gradient_tol else "ok"
        )
    info.update(objective_tolerance=objective_tol, gradient_tolerance=gradient_tol,
                source_size=profile.source_size)
    return info


def _candidate_key(profile, b):
    value, _, a = profile.at(b)
    info = _base_fit_info(profile, np.array([a, b]), value, 0)
    tolerance = min(GRADIENT_TOL, profile.source_size**-1)
    stationary = np.isfinite(info["stationarity_scaled"]) and (
        info["stationarity_scaled"] <= tolerance
    )
    return not stationary, value, b, a


def _solve_profile(profile):
    lo, hi = map(float, profile.bounds[1])
    if not len(profile.vt):
        coefficient = np.array([profile.bounds[0, 0], lo])
        value, _ = profile.objective(coefficient)
        return coefficient, _fit_info(profile, coefficient, value, 0)
    candidates = [lo, hi, (lo + hi) / 2]
    root = _root_candidate(profile, lo, hi)
    if root is not None:
        candidates.append(root)
    best = min(candidates, key=lambda b: _candidate_key(profile, b))
    if profile.arm == "UKL0":
        lower = max(
            profile.at(b)[0] + profile.at(b)[1] * ((lo if profile.at(b)[1] >= 0 else hi) - b)
            for b in candidates
        )
        coefficient = np.array([profile.at(best)[2], best])
        return coefficient, _fit_info(profile, coefficient, lower, len(candidates))
    heap = [(_sq_bound(profile, lo, hi), lo, hi)]
    terminal_lower = np.inf
    nodes = 1
    forced = None
    while heap:
        lower = min(terminal_lower, heap[0][0])
        coefficient = np.array([profile.at(best)[2], best])
        info = _fit_info(profile, coefficient, lower, nodes)
        if info["status"] in ("ok", "nonfinite", "numerical_bound_failure"):
            return coefficient, info
        if nodes >= MAX_NODES:
            forced = "global_gap_not_met"
            break
        bound, left, right = heapq.heappop(heap)
        if bound >= profile.at(best)[0]:
            terminal_lower = min(terminal_lower, bound)
            continue
        mid = (left + right) / 2
        if not left < mid < right:
            terminal_lower = min(terminal_lower, bound)
            forced = "global_gap_not_met"
            break
        for a, b in ((left, mid), (mid, right)):
            tests = [a, (a + b) / 2, b]
            root = _root_candidate(profile, a, b)
            if root is not None:
                tests.append(root)
            best = min([best, *tests], key=lambda x: _candidate_key(profile, x))
            heapq.heappush(heap, (_sq_bound(profile, a, b), a, b))
            nodes += 1
    lower = min(terminal_lower, heap[0][0] if heap else np.inf)
    coefficient = np.array([profile.at(best)[2], best])
    return coefficient, _fit_info(profile, coefficient, lower, nodes, forced)


def _fit_cell(source_u, target_u, loss, normalizer, penalty, bounds, source_size=None):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            profile = _CellProfile(
                source_u, target_u, loss, normalizer, penalty, bounds, source_size,
            )
            return _solve_profile(profile)
    except (FloatingPointError, OverflowError) as error:
        return np.full(2, np.nan), {"status": "nonfinite", "message": str(error)}


def _ratio_inputs(source_u, source_z, target_u, target_z, n_cells):
    source_u = _u_vector(source_u, "source_u")
    target_u = _u_vector(target_u, "target_u")
    source_ids, target_ids = _z_ids(source_z, n_cells), _z_ids(target_z, n_cells)
    if not len(source_u) or len(source_u) != len(target_u):
        raise ValueError("source and target fitting samples must have the same positive size")
    if source_ids.shape != source_u.shape or target_ids.shape != target_u.shape:
        raise ValueError("U and Z lengths must agree within each sample")
    return source_u, source_ids, target_u, target_ids


def partition_ratio_objective(coefficient, source_u, source_z, target_u, target_z,
                              loss, penalty=0.0, bounds=DEFAULT_BOUNDS):
    """Return the complete empirical GRR objective and its J by 2 gradient."""
    coefficient = np.asarray(coefficient, dtype=float)
    if coefficient.ndim != 2 or coefficient.shape[1] != 2 or not np.isfinite(coefficient).all():
        raise ValueError("coefficient must be a finite J by 2 matrix")
    j = len(coefficient)
    su, si, tu, ti = _ratio_inputs(source_u, source_z, target_u, target_z, j)
    value, gradient = 0.0, np.zeros_like(coefficient)
    for cell in range(j):
        profile = _CellProfile(
            su[si == cell], tu[ti == cell], loss, len(su) / j, penalty, bounds, len(su),
        )
        cell_value, cell_gradient = profile.objective(coefficient[cell])
        value += cell_value / j
        gradient[cell] = cell_gradient / j
    return float(value), gradient


def fit_partition_ratio(source_u, source_z, target_u, target_z, n_cells, loss,
                        penalty=0.0, bounds=DEFAULT_BOUNDS):
    """Fit every cell's intercept and exponent using the same compact box problem."""
    su, si, tu, ti = _ratio_inputs(source_u, source_z, target_u, target_z, n_cells)
    coefficient = np.full((n_cells, 2), np.nan)
    cells = []
    for cell in range(n_cells):
        coefficient[cell], info = _fit_cell(
            su[si == cell], tu[ti == cell], loss, len(su) / n_cells, penalty, bounds, len(su),
        )
        cells.append({"cell": cell, "source_count": int(np.sum(si == cell)),
                      "target_count": int(np.sum(ti == cell)), **info})
        if info["status"] != "ok":
            return coefficient, {"status": info["status"], "failed_cell": cell, "cells": cells}
    value = float(np.mean([item["objective"] for item in cells]))
    lower = float(np.mean([item["lower_bound"] for item in cells]))
    return coefficient, {
        "status": "ok", "coefficient": coefficient.tolist(), "cells": cells,
        "objective": value, "lower_bound": lower, "upper_bound": value, "gap": value - lower,
        "stationarity_scaled_max": float(max(item["stationarity_scaled"] for item in cells)),
        "boundary": any(item["boundary"] for item in cells),
        "boundary_cells": int(sum(item["boundary"] for item in cells)),
        "nodes": int(sum(item["nodes"] for item in cells)),
        "objective_tolerance": cells[0]["objective_tolerance"],
        "gradient_tolerance": cells[0]["gradient_tolerance"], "source_size": len(su),
        "normalizer": len(su) / n_cells, "penalty": float(penalty),
    }


def predict_partition_ratio(coefficient, U, Z):
    coefficient = np.asarray(coefficient, dtype=float)
    if coefficient.ndim != 2 or coefficient.shape[1] != 2 or not np.isfinite(coefficient).all():
        raise ValueError("coefficient must be a finite J by 2 matrix")
    U = _u_vector(U, "U")
    ids = _z_ids(Z, len(coefficient))
    if ids.shape != U.shape:
        raise ValueError("U and Z lengths must agree")
    return np.exp(coefficient[ids, 0] - coefficient[ids, 1] * np.log(U))


def _tensor_grid(T, n_bins):
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[1] < 1 or not np.isfinite(T).all():
        raise ValueError("T must be a finite matrix with at least one coordinate")
    if np.any((T < 0) | (T >= 1)):
        raise ValueError("T coordinates must belong to [0,1)")
    bins = np.asarray(n_bins)
    if bins.ndim == 0:
        bins = np.repeat(bins, T.shape[1])
    if bins.shape != (T.shape[1],) or not np.issubdtype(bins.dtype, np.integer) or np.any(bins < 1):
        raise ValueError("n_bins must contain positive integers, one per coordinate")
    bins = tuple(map(int, bins))
    ids = np.ravel_multi_index(np.floor(T * bins).astype(int).T, bins)
    return T, bins, ids


def tensor_cell_ids(T, n_bins):
    """Index an isotropic or anisotropic tensor histogram on [0,1)."""
    return _tensor_grid(T, n_bins)[2]


def _regression_inputs(T, Z, Y=None):
    T = np.asarray(T, dtype=float)
    Z = _vector(Z, "Z")
    if T.ndim != 2 or not np.isfinite(T).all() or len(T) != len(Z):
        raise ValueError("T must be a finite matrix with one row per Z")
    if Y is None:
        return T, Z
    Y = _vector(Y, "Y", nonempty=True)
    if len(Y) != len(T):
        raise ValueError("T, Z and Y lengths must agree")
    return T, Z, Y


def fit_tensor_histogram_ols(T, Z, Y, n_bins):
    """Jointly fit tensor-cell indicators and continuous Z by minimum-norm OLS."""
    T, Z, Y = _regression_inputs(T, Z, Y)
    _, bins, ids = _tensor_grid(T, n_bins)
    n_cells = int(np.prod(bins))
    counts = np.bincount(ids, minlength=n_cells)
    occupied = counts > 0
    zmean, ymean = np.zeros(n_cells), np.zeros(n_cells)
    zmean[occupied] = np.bincount(ids, weights=Z, minlength=n_cells)[occupied] / counts[occupied]
    ymean[occupied] = np.bincount(ids, weights=Y, minlength=n_cells)[occupied] / counts[occupied]
    zmin, zmax = np.full(n_cells, np.inf), np.full(n_cells, -np.inf)
    np.minimum.at(zmin, ids, Z)
    np.maximum.at(zmax, ids, Z)
    has_residual = bool(np.any(zmax[occupied] > zmin[occupied]))
    if has_residual:
        residual_z = Z - zmean[ids]
        slope = float(residual_z @ (Y - ymean[ids]) / (residual_z @ residual_z))
    else:
        slope = float(zmean @ ymean / (1 + zmean @ zmean))
    coefficient = ymean - slope * zmean
    if not np.isfinite(slope) or not np.isfinite(coefficient).all():
        raise FloatingPointError("joint histogram OLS produced nonfinite coefficients")
    return {
        "slope": slope, "coefficients": coefficient, "counts": counts, "bins": bins,
        "rank": int(occupied.sum() + has_residual), "empty_cells": int((~occupied).sum()),
        "status": "ok",
    }


def predict_tensor_histogram_ols(fit, T, Z):
    T, Z = _regression_inputs(T, Z)
    _, _, ids = _tensor_grid(T, fit["bins"])
    return fit["slope"] * Z + np.asarray(fit["coefficients"])[ids]


def fit_global_ols(T, Z, Y):
    """Fit an intercept, every T coordinate and continuous Z by minimum-norm OLS."""
    T, Z, Y = _regression_inputs(T, Z, Y)
    design = np.column_stack((np.ones(len(T)), T, Z))
    coefficient, _, rank, singular = np.linalg.lstsq(design, Y, rcond=None)
    return {"coefficients": coefficient, "rank": int(rank), "singular_values": singular,
            "dimension": T.shape[1], "status": "ok"}


def predict_global_ols(fit, T, Z):
    T, Z = _regression_inputs(T, Z)
    if T.shape[1] != fit["dimension"]:
        raise ValueError("T has an incompatible number of coordinates")
    return np.column_stack((np.ones(len(T)), T, Z)) @ fit["coefficients"]
