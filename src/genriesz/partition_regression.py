"""Count-smoothed density ratios and local cubic series regression."""

from __future__ import annotations

import numpy as np


def local_basis(u, cells):
    """Return cell indices and the four Lebesgue-normalized local polynomials."""
    u = np.asarray(u, dtype=float)
    if (
        u.ndim != 1
        or not len(u)
        or not np.isfinite(u).all()
        or np.any((u < 0) | (u > 1))
        or not isinstance(cells, (int, np.integer))
        or isinstance(cells, bool)
        or cells < 1
    ):
        raise ValueError("a finite vector in [0,1] and a positive cell count are required")
    index = (u * cells).astype(int)
    index[u == 1] = cells - 1
    x = 2 * (cells * u - index) - 1
    values = np.sqrt(cells) * np.stack(
        (
            np.ones_like(x),
            np.sqrt(3) * x,
            np.sqrt(5) * (3 * x**2 - 1) / 2,
            np.sqrt(7) * (5 * x**3 - 3 * x) / 2,
        ),
        axis=1,
    )
    return index, values


def regression_moments(u, y, cells):
    """Exact block sufficient statistics; no population law is used."""
    index, values = local_basis(u, cells)
    y = np.asarray(y, dtype=float)
    if y.shape != index.shape or not np.isfinite(y).all():
        raise ValueError("finite outcomes matching the covariates are required")
    gram = np.empty((cells, 4, 4))
    for j in range(4):
        for k in range(j + 1):
            gram[:, j, k] = gram[:, k, j] = np.bincount(
                index, weights=values[:, j] * values[:, k], minlength=cells
            )
    xy = np.stack(
        [np.bincount(index, weights=values[:, j] * y, minlength=cells) for j in range(4)],
        axis=1,
    )
    return dict(gram=gram, xy=xy, yy=float(y @ y), size=len(y))


def _validate_moments(moments):
    gram, xy = np.asarray(moments["gram"]), np.asarray(moments["xy"])
    size, yy = moments["size"], moments["yy"]
    if (
        gram.ndim != 3
        or gram.shape[1:] != (4, 4)
        or xy.shape != gram.shape[:2]
        or not len(gram)
        or not isinstance(size, (int, np.integer))
        or isinstance(size, bool)
        or size < 1
        or not np.isfinite(gram).all()
        or not np.isfinite(xy).all()
        or not np.isfinite(yy)
        or yy < 0
        or not np.allclose(gram, gram.swapaxes(1, 2), rtol=0, atol=1e-12)
    ):
        raise ValueError("finite symmetric local sufficient statistics are required")
    if np.linalg.eigvalsh(gram).min() < -1e-10 * (1 + np.max(np.abs(gram))):
        raise ValueError("the local Gram matrices must be positive semidefinite")
    return gram / size, xy / size


def fit_regression(moments, penalty):
    """Minimum-norm normal-equation solution, including rank-deficient samples."""
    gram, xy = _validate_moments(moments)
    if not np.isscalar(penalty) or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("a finite nonnegative penalty is required")
    matrix = gram + penalty * np.diag([0, 1, 1, 1])
    solutions = [np.linalg.lstsq(a, b, rcond=None) for a, b in zip(matrix, xy, strict=True)]
    coefficient = np.stack([solution[0] for solution in solutions])
    residual = float(np.max(np.abs(np.einsum("hjk,hk->hj", matrix, coefficient) - xy)))
    residual /= 1 + float(np.max(np.abs(xy)))
    if not np.isfinite(coefficient).all() or residual > 1e-12:
        raise ArithmeticError("the local regression normal equations failed")
    return coefficient, dict(
        normal_equation_scaled=residual,
        gram_min_eigenvalue=float(np.linalg.eigvalsh(gram).min()),
        normal_min_eigenvalue=float(np.linalg.eigvalsh(matrix).min()),
        normal_rank=sum(solution[2] for solution in solutions),
    )


def predict(u, coefficient):
    coefficient = np.asarray(coefficient, dtype=float)
    if coefficient.ndim != 2 or coefficient.shape[1] != 4 or not np.isfinite(coefficient).all():
        raise ValueError("finite local cubic coefficients are required")
    index, values = local_basis(u, len(coefficient))
    return np.einsum("ij,ij->i", values, coefficient[index])


def select_penalty(folds, penalties):
    """Twofold squared prediction error, with ties resolved by the smaller penalty."""
    penalties = np.asarray(penalties, dtype=float)
    if (
        len(folds) != 2
        or penalties.ndim != 1
        or len(penalties) < 2
        or penalties[0] != 0
        or not np.isfinite(penalties).all()
        or np.any(np.diff(penalties) <= 0)
    ):
        raise ValueError("two folds and strictly increasing penalties including zero are required")
    for fold in folds:
        _validate_moments(fold)
    if folds[0]["gram"].shape != folds[1]["gram"].shape:
        raise ValueError("the same regression basis must be used in both folds")
    total = sum(fold["size"] for fold in folds)
    scores = []
    for penalty in penalties:
        score = 0.0
        for validation in range(2):
            coef, _ = fit_regression(folds[1 - validation], penalty)
            moments = folds[validation]
            score += (
                moments["yy"]
                - 2 * np.sum(coef * moments["xy"])
                + np.einsum("hj,hjk,hk->", coef, moments["gram"], coef)
            )
        scores.append(score / total)
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or np.min(scores) < 0:
        raise ArithmeticError("the squared validation error is invalid")
    return float(penalties[np.argmin(scores)]), scores


def fit_ratio(source, target, cells):
    """The common positive SQ and UKL minimizer with one added count per cell."""
    source_index, _ = local_basis(source, cells)
    target_index, _ = local_basis(target, cells)
    source_count = np.bincount(source_index, minlength=cells)
    target_count = np.bincount(target_index, minlength=cells)
    s, t = (source_count + 1) / len(source), (target_count + 1) / len(target)
    ratio = t / s
    sq = float(np.max(np.abs(s * ratio - t)) / (1 + np.max(t)))
    ukl = float(np.max(np.abs(s - t / ratio)) / (1 + np.max(s)))
    if not np.isfinite(ratio).all() or np.min(ratio) <= 0 or max(sq, ukl) > 1e-12:
        raise ArithmeticError("the positive density-ratio first-order equations failed")
    return ratio, dict(
        SQ_stationarity=sq,
        UKL_stationarity=ukl,
        ratio_min=float(ratio.min()),
        ratio_max=float(ratio.max()),
        source_empty=int(np.sum(source_count == 0)),
        target_empty=int(np.sum(target_count == 0)),
    )


def predict_ratio(u, ratio):
    ratio = np.asarray(ratio, dtype=float)
    if ratio.ndim != 1 or not len(ratio) or not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError("positive finite density-ratio coefficients are required")
    index, _ = local_basis(u, len(ratio))
    return ratio[index]
