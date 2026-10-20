"""Histogram outcome regression and bounded odds fitting for the ATT."""

from __future__ import annotations

import heapq

import numpy as np
from scipy import optimize

OBJECTIVE_TOL = 1e-12
GRADIENT_TOL = 1e-8
MAX_NODES = 100000


def cell_ids(T, J):
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[1] < 1 or not isinstance(J, (int, np.integer)) or J < 1:
        raise ValueError("T must be a matrix and J a positive integer")
    if not np.isfinite(T).all() or np.any((T < 0) | (T >= 1)):
        raise ValueError("grid coordinates must belong to [0,1)")
    return np.ravel_multi_index(np.floor(J * T).astype(int).T, (J,) * T.shape[1])


def _binary(D, n):
    D = np.asarray(D, dtype=float)
    if D.shape != (n,) or not np.isfinite(D).all() or np.any((D != 0) & (D != 1)):
        raise ValueError("D must be a finite binary vector")
    return D


def fit_cell_ols(T, D, Y, J):
    """Fit D and all cell indicators, choosing minimum Euclidean norm."""
    T = np.asarray(T, dtype=float)
    ids = cell_ids(T, J)
    n = len(ids)
    D = _binary(D, n)
    Y = np.asarray(Y, dtype=float)
    if n == 0 or Y.shape != (n,) or not np.isfinite(Y).all():
        raise ValueError("Y must be a nonempty finite vector")
    p = J ** T.shape[1]
    counts = np.bincount(ids, minlength=p)
    treated = np.bincount(ids, weights=D, minlength=p)
    ysum = np.bincount(ids, weights=Y, minlength=p)
    occupied = counts > 0
    dmean, ymean = np.zeros(p), np.zeros(p)
    dmean[occupied] = treated[occupied] / counts[occupied]
    ymean[occupied] = ysum[occupied] / counts[occupied]
    mixed = (treated > 0) & (treated < counts)
    if mixed.any():
        denominator = np.sum(treated[mixed] * (counts[mixed] - treated[mixed]) / counts[mixed])
        tau = float(np.sum((D - dmean[ids]) * (Y - ymean[ids])) / denominator)
    else:
        tau = float(dmean @ ymean / (1.0 + dmean @ dmean))
    coefficients = ymean - tau * dmean
    return {
        "tau": tau,
        "coefficients": coefficients,
        "counts": counts,
        "rank": int(occupied.sum() + mixed.any()),
        "empty_cells": int((~occupied).sum()),
    }


def predict_cell_ols(fit, T, D, J):
    ids = cell_ids(T, J)
    return fit["tau"] * _binary(D, len(ids)) + np.asarray(fit["coefficients"])[ids]


def _quadratic_min(A, B, lo, hi):
    """Minimum of A*c*c-2*B*c on a positive interval, A>=0."""
    c = np.clip(B / A, lo, hi) if A > 0 else (hi if B > 0 else lo)
    return float(A * c * c - 2 * B * c)


class _OddsProfile:
    def __init__(self, U, D, arm, bounds):
        U = np.asarray(U, dtype=float)
        if U.ndim != 1 or len(U) == 0 or not np.isfinite(U).all() or np.any((U <= 0) | (U > 1)):
            raise ValueError("U must be a nonempty finite vector in (0,1]")
        self.D = _binary(D, len(U))
        self.V = -np.log(U)
        self.n = len(U)
        self.arm = arm
        self.bounds = np.asarray(bounds, dtype=float)
        if (
            arm not in ("SQ", "UKL0") or self.bounds.shape != (2, 2)
            or not np.isfinite(self.bounds).all()
            or np.any(self.bounds[:, 0] >= self.bounds[:, 1]) or self.bounds[1, 0] < 0
        ):
            raise ValueError("require SQ or UKL0 and two ordered finite coefficient bounds")
        self.clo, self.chi = np.exp(self.bounds[0])
        self.vt, self.vc = self.V[self.D == 1], self.V[self.D == 0]
        self.p = float(self.D.mean())
        self.td = float(self.D @ self.V / self.n)
        self._moment_cache = {}
        self._profile_cache = {}

    def moments(self, b):
        if b in self._moment_cache:
            return self._moment_cache[b]
        t = np.exp(b * self.vt)
        factor = 2 if self.arm == "SQ" else 1
        c = np.exp(factor * b * self.vc)
        value = np.array([
            t.sum(), t @ self.vt, t @ (self.vt**2),
            c.sum(), factor * (c @ self.vc), factor**2 * (c @ (self.vc**2)),
        ]) / self.n
        self._moment_cache[b] = value
        return value

    def at(self, b):
        if b in self._profile_cache:
            return self._profile_cache[b]
        m1, _, _, m2, d2, _ = self.moments(float(b))
        if self.arm == "SQ":
            c = float(np.clip(m1 / m2, self.clo, self.chi))
            value = c * c * m2 - 2 * c * m1
            slope = c * c * d2 - 2 * c * self.moments(float(b))[1]
        else:
            c = float(np.clip(self.p / m2, self.clo, self.chi))
            value = c * m2 - self.p * np.log(c) - self.td * b
            slope = c * d2 - self.td
        a = self.bounds[0, 0] if c == self.clo else (
            self.bounds[0, 1] if c == self.chi else np.log(c)
        )
        result = float(value), float(slope), float(a)
        self._profile_cache[b] = result
        return result

    def objective(self, coefficient):
        a, b = coefficient
        m1, d1, _, m2, d2, _ = self.moments(float(b))
        c = np.exp(a)
        if self.arm == "SQ":
            return float(c * c * m2 - 2 * c * m1), np.array([
                2 * (c * c * m2 - c * m1), c * c * d2 - 2 * c * d1,
            ])
        return float(c * m2 - self.p * a - self.td * b), np.array([
            c * m2 - self.p, c * d2 - self.td,
        ])


def odds_objective(coefficient, U, D, arm):
    """The unpenalized sample odds objective and its two derivatives."""
    return _OddsProfile(U, D, arm, ((-100.0, 100.0), (0.0, 100.0))).objective(coefficient)


def _projected(gradient, coefficient, bounds):
    projected = np.asarray(gradient).copy()
    active = []
    for j, (lo, hi) in enumerate(bounds):
        lower, upper = coefficient[j] == lo, coefficient[j] == hi
        if lower:
            projected[j] = min(0.0, projected[j])
        if upper:
            projected[j] = max(0.0, projected[j])
        active.append({"lower": bool(lower), "upper": bool(upper)})
    return projected, active


def _root_candidate(profile, lo, hi):
    if profile.at(lo)[1] < 0 < profile.at(hi)[1]:
        return float(optimize.brentq(
            lambda b: profile.at(float(b))[1], lo, hi,
            xtol=np.nextafter(0.0, 1.0), rtol=4 * np.finfo(float).eps,
        ))
    return None


def _sq_bound(profile, lo, hi):
    """Analytic profile lower bound, evaluated in ordinary double precision."""
    mid = (lo + hi) / 2
    points = (lo, mid, hi)
    moment = [profile.moments(x) for x in points]
    ml, _, mu = moment
    clo = np.clip(ml[0] / mu[3], profile.clo, profile.chi)
    chi = np.clip(mu[0] / ml[3], profile.clo, profile.chi)
    dlow = _quadratic_min(ml[4], mu[1], clo, chi)
    dhigh = max(ml[1] * -2 * c + mu[4] * c * c for c in (clo, chi))
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
    bounds = []
    for x in knots:
        upper_m1 = ml[0] + (x - lo) * (mu[0] - ml[0]) / (hi - lo)
        lower_m2 = float(np.max(starts + slopes * (x - lo)))
        bounds.append(_quadratic_min(lower_m2, upper_m1, clo, chi))
    lower = min(bounds)
    hlow = _quadratic_min(ml[5], mu[2], clo, chi)
    cross = max(abs(2 * clo * ml[4] - 2 * mu[1]), abs(2 * chi * mu[4] - 2 * ml[1]))
    curvature = hlow - cross**2 / (2 * ml[3])
    if curvature > 0:
        for x in points:
            value, derivative, _ = profile.at(x)
            y = np.clip(x - derivative / curvature, lo, hi)
            lower = max(lower, value + derivative * (y - x) + curvature * (y - x)**2 / 2)
    return float(lower)


def _fit_info(profile, coefficient, lower, nodes, forced_status=None):
    value, gradient = profile.objective(coefficient)
    projected, active = _projected(gradient, coefficient, profile.bounds)
    gap = float(value - lower)
    scale = 1 + abs(value)
    roundoff = 64 * np.finfo(float).eps * scale
    finite = np.isfinite([*coefficient, value, lower, *gradient, *projected]).all()
    status = forced_status or (
        "nonfinite" if not finite else
        "numerical_bound_failure" if gap < -roundoff else
        "global_gap_not_met" if gap > OBJECTIVE_TOL * scale else
        "stationarity_not_met" if np.max(np.abs(projected)) > GRADIENT_TOL * scale else "ok"
    )
    return {
        "status": status, "coefficient": np.asarray(coefficient).tolist(),
        "objective": value, "gradient": gradient.tolist(),
        "projected_gradient": projected.tolist(),
        "stationarity_scaled": float(np.max(np.abs(projected)) / scale),
        "lower_bound": float(lower), "upper_bound": value, "gap": gap,
        "negative_gap": bool(gap < 0), "bound_roundoff_tolerance": roundoff,
        "boundary": any(item["lower"] or item["upper"] for item in active),
        "active_bounds": active, "nodes": int(nodes),
        "objective_tolerance": OBJECTIVE_TOL, "gradient_tolerance": GRADIENT_TOL,
    }


def _fit_att_odds(U, D, arm, bounds):
    profile = _OddsProfile(U, D, arm, bounds)
    alo, ahi = profile.bounds[0]
    lo, hi = profile.bounds[1]
    if not len(profile.vt) or not len(profile.vc) or np.all(profile.V == profile.V[0]):
        if not len(profile.vt):
            coefficient = np.array([alo, lo])
        elif not len(profile.vc):
            coefficient = np.array([ahi, hi if np.any(profile.V > 0) else lo])
        else:
            v = profile.V[0]
            z = np.clip(np.log(profile.p / (1 - profile.p)), alo + lo * v, ahi + hi * v)
            b = max(lo, (z - ahi) / v) if v > 0 else lo
            coefficient = np.array([np.clip(z - b * v, alo, ahi), b])
        value, _ = profile.objective(coefficient)
        return coefficient, _fit_info(profile, coefficient, value, 0)
    candidates = [float(lo), float(hi), float((lo + hi) / 2)]
    root = _root_candidate(profile, lo, hi)
    if root is not None:
        candidates.append(root)
    best = min(candidates, key=lambda b: (profile.at(b)[0], b, profile.at(b)[2]))
    if arm == "UKL0":
        lower = max(
            profile.at(b)[0] + profile.at(b)[1] * ((lo if profile.at(b)[1] >= 0 else hi) - b)
            for b in candidates
        )
        coefficient = np.array([profile.at(best)[2], best])
        return coefficient, _fit_info(profile, coefficient, lower, len(candidates))
    heap = [(_sq_bound(profile, lo, hi), float(lo), float(hi))]
    terminal_lower = np.inf
    nodes = 1
    forced = None
    while heap:
        lower = min(terminal_lower, heap[0][0])
        coefficient = np.array([profile.at(best)[2], best])
        info = _fit_info(profile, coefficient, lower, nodes)
        if info["status"] == "ok" or info["status"] in ("nonfinite", "numerical_bound_failure"):
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
            best = min([best, *tests], key=lambda x: (profile.at(x)[0], x, profile.at(x)[2]))
            child_bound = _sq_bound(profile, a, b)
            heapq.heappush(heap, (child_bound, a, b))
            nodes += 1
    lower = min(terminal_lower, heap[0][0] if heap else np.inf)
    coefficient = np.array([profile.at(best)[2], best])
    return coefficient, _fit_info(profile, coefficient, lower, nodes, forced)


def fit_att_odds(U, D, arm, bounds):
    """Numerically minimize the full coefficient box using analytic bounds."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _fit_att_odds(U, D, arm, bounds)
    except (FloatingPointError, OverflowError) as error:
        return np.full(2, np.nan), {
            "status": "nonfinite", "message": str(error), "coefficient": [np.nan, np.nan],
            "boundary": False, "gradient": [np.nan, np.nan],
            "projected_gradient": [np.nan, np.nan], "gap": np.nan,
        }


def att_score_record(D, Y, odds, g, g1, g0):
    """Use a common empirical treated share and the recentered ATT score."""
    Y, odds, g, g1, g0 = (np.asarray(v, dtype=float) for v in (Y, odds, g, g1, g0))
    D = _binary(D, len(Y))
    if len(D) == 0 or any(v.shape != D.shape for v in (Y, odds, g, g1, g0)):
        raise ValueError("score inputs must be nonempty vectors of equal length")
    pi = float(D.mean())
    if pi == 0:
        return {"estimate": 0.0, "variance": 0.0, "se": 0.0, "status": "no_treated"}
    alpha = (D - (1 - D) * odds) / pi
    uncentered = D * (g1 - g0) / pi + alpha * (Y - g)
    theta = float(uncentered.mean())
    influence = uncentered - D * theta / pi
    if not np.isfinite(influence).all():
        return {"estimate": np.nan, "variance": np.nan, "se": np.nan, "status": "nonfinite"}
    with np.errstate(over="ignore", invalid="ignore"):
        variance = float(np.mean(influence**2))
    if not np.isfinite(variance):
        return {"estimate": np.nan, "variance": np.nan, "se": np.nan, "status": "nonfinite"}
    scaled_alpha = alpha / np.max(np.abs(alpha))
    top = float(np.max(np.abs(influence)))
    scaled_score = influence / top if top > 0 else np.zeros_like(influence)
    score_mass = float(scaled_score @ scaled_score)
    return {
        "estimate": theta, "variance": variance, "se": float(np.sqrt(variance / len(D))),
        "status": "ok", "max_abs_alpha": float(np.max(np.abs(alpha))),
        "ess": float(np.sum(np.abs(scaled_alpha))**2 / (scaled_alpha @ scaled_alpha)),
        "score_concentration": float(np.max(scaled_score**2) / score_mass)
        if score_mass > 0 else 0.0,
    }
