"""Utility functions for GRR experiment notebooks.

The notebooks are designed for two modes:

- FAST_MODE=True: small grids that execute quickly and display all tables/figures.
- FAST_MODE=False: publication-scale grids. Increase repetitions and grids as needed.

The module intentionally lives under notebooks/experiments so it does not modify
or extend the genriesz package API.
"""
from __future__ import annotations

import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit


def find_project_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Find the repository root containing src/genriesz."""
    p = Path.cwd() if start is None else Path(start)
    p = p.resolve()
    for cand in [p, *p.parents]:
        if (cand / "src" / "genriesz").exists():
            return cand
    raise RuntimeError("Could not find repository root containing src/genriesz")


def setup_paths() -> Path:
    """Make the local src directory importable from a notebook."""
    root = find_project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


ROOT = setup_paths()

from genriesz import (  # noqa: E402
    ATEFunctional,
    ATTFunctional,
    BKLGenerator,
    BPGenerator,
    CallableBasis,
    GRRGLM,
    GaussianRKHSBasis,
    PolynomialBasis,
    RBFRandomFourierBasis,
    RBFNystromBasis,
    SquaredGenerator,
    TreatmentInteractionBasis,
    UKLGenerator,
    grr_ate,
    grr_att,
)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def display_mode_banner(fast_mode: bool, full_hint: str = "set FAST_MODE=False for publication grids") -> str:
    return f"Mode: {'FAST' if fast_mode else 'FULL'} ({full_hint})."


def result_to_frame(res: Any, *, true_theta: float | None = None, label: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, est in res.estimates.items():
        row = {
            "label": label,
            "estimator": key,
            "estimate": est.estimate,
            "se": est.se,
            "ci_low": est.ci_low,
            "ci_high": est.ci_high,
            "p_value": est.p_value,
        }
        if true_theta is not None:
            row["true_theta"] = float(true_theta)
            row["bias"] = float(est.estimate - true_theta)
            row["squared_error"] = float((est.estimate - true_theta) ** 2)
        for k, v in res.diagnostics.items():
            if k == "love_plot":
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_mc(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Monte Carlo summary from long estimate-level rows."""
    agg: dict[str, Any] = {
        "estimate": ["mean", "std"],
        "se": "mean",
    }
    if "bias" in df.columns:
        agg["bias"] = "mean"
    if "squared_error" in df.columns:
        agg["squared_error"] = lambda x: float(np.sqrt(np.mean(x)))
    for col in ["alpha_abs_p95", "alpha_abs_max", "max_abs_smd_weighted", "ess_treated", "ess_control"]:
        if col in df.columns:
            agg[col] = "mean"
    out = df.groupby(group_cols, dropna=False).agg(agg)
    out.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") for col in out.columns.to_flat_index()]
    out = out.reset_index()
    if "squared_error_<lambda>" in out.columns:
        out = out.rename(columns={"squared_error_<lambda>": "rmse"})
    # approximate Wald coverage when true_theta is available
    if "true_theta" in df.columns:
        cov = df.assign(covered=(df["ci_low"] <= df["true_theta"]) & (df["true_theta"] <= df["ci_high"]))
        cov = cov.groupby(group_cols, dropna=False)["covered"].mean().rename("coverage").reset_index()
        out = out.merge(cov, on=group_cols, how="left")
    return out


def safe_display_frame(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return a rounded, reasonably short frame for notebook display."""
    out = df.copy()
    for c in out.select_dtypes(include=["float", "float64", "float32"]).columns:
        out[c] = out[c].astype(float).round(4)
    return out.head(n)


def plot_metric_by_lambda(df: pd.DataFrame, *, metric: str, title: str, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for label, g in df.groupby("loss"):
        g = g.sort_values("lambda")
        ax.plot(g["lambda"], g[metric], marker="o", label=str(label))
    ax.set_xscale("log")
    ax.set_xlabel("Riesz regularization lambda")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_bar_table(df: pd.DataFrame, *, x: str, y: str, title: str, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[x].astype(str), df[y].astype(float))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    return ax


# ---------------------------------------------------------------------------
# Basis and generator helpers
# ---------------------------------------------------------------------------


class SelectedColumnsBasis:
    """Simple basis that selects columns from a matrix, optionally adding an intercept."""

    def __init__(self, columns: Iterable[int], *, include_bias: bool = True):
        self.columns = list(int(c) for c in columns)
        self.include_bias = bool(include_bias)
        self._n_features: int | None = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if self.columns and max(self.columns) >= X.shape[1]:
            raise ValueError("Selected column index is out of bounds")
        self._n_features = len(self.columns) + (1 if self.include_bias else 0)
        return self

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        parts = []
        if self.include_bias:
            parts.append(np.ones((X.shape[0], 1)))
        if self.columns:
            parts.append(X[:, self.columns])
        if not parts:
            out = np.empty((X.shape[0], 0))
        else:
            out = np.column_stack(parts)
        self._n_features = out.shape[1]
        return out[0] if single else out

    @property
    def n_features(self):
        if self._n_features is None:
            return len(self.columns) + (1 if self.include_bias else 0)
        return int(self._n_features)

    def derivative(self, X, coordinate: int):
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        out = np.zeros((X.shape[0], self.n_features), dtype=float)
        offset = 1 if self.include_bias else 0
        for j, col in enumerate(self.columns):
            if int(coordinate) == int(col):
                out[:, offset + j] = 1.0
        return out[0] if single else out

    def copy(self):
        return SelectedColumnsBasis(self.columns, include_bias=self.include_bias)


class ZOnlyBasis:
    """Lift a covariate-only base basis to X=[D,Z] by dropping the treatment column."""

    def __init__(self, base_basis, treatment_index: int = 0):
        self.base_basis = base_basis
        self.treatment_index = int(treatment_index)
        self._n_features: int | None = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        Z = np.delete(X, self.treatment_index, axis=1)
        self.base_basis.fit(Z, y=y)
        self._n_features = int(self.base_basis.n_features)
        return self

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        if self._n_features is None:
            self.fit(X)
        Z = np.delete(X, self.treatment_index, axis=1)
        out = np.asarray(self.base_basis(Z), dtype=float)
        return out[0] if single and out.ndim == 2 else out

    @property
    def n_features(self):
        if self._n_features is None:
            return int(self.base_basis.n_features)
        return int(self._n_features)

    def derivative(self, X, coordinate: int):
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        if int(coordinate) == self.treatment_index:
            out = np.zeros((X.shape[0], self.n_features))
            return out[0] if single else out
        z_coord = int(coordinate) - 1 if int(coordinate) > self.treatment_index else int(coordinate)
        Z = np.delete(X, self.treatment_index, axis=1)
        out = np.asarray(self.base_basis.derivative(Z, z_coord), dtype=float)
        return out[0] if single and out.ndim == 2 else out

    def copy(self):
        import copy
        return ZOnlyBasis(copy.deepcopy(self.base_basis), treatment_index=self.treatment_index)


def branch_treated(x: np.ndarray) -> int:
    return int(float(x[0]) >= 0.5)


def make_generator(loss: str, *, omega: float | None = None, C: float = 1.0):
    loss_l = str(loss).lower()
    if loss_l in {"sq", "squared", "squared_loss"}:
        return SquaredGenerator(C=0.0)
    if loss_l in {"ukl", "kl", "tailored"}:
        return UKLGenerator(C=C, branch_fn=branch_treated)
    if loss_l in {"bkl", "logistic", "mle"}:
        return BKLGenerator(C=C, branch_fn=branch_treated)
    if loss_l in {"bp", "power"}:
        return BPGenerator(C=C, omega=0.5 if omega is None else float(omega), branch_fn=branch_treated)
    raise ValueError(f"Unknown loss: {loss}")


def make_base_basis(kind: str, *, degree: int = 1, n_features: int = 80, sigma: float = 1.0, seed: int = 0):
    kind_l = str(kind).lower()
    if kind_l in {"poly", "polynomial", "poly1", "poly2"}:
        deg = degree
        if kind_l == "poly1":
            deg = 1
        if kind_l == "poly2":
            deg = 2
        return PolynomialBasis(degree=deg, include_bias=True)
    if kind_l in {"rff", "random_fourier"}:
        return RBFRandomFourierBasis(n_features=n_features, sigma=sigma, include_bias=True, random_state=seed)
    if kind_l in {"rkhs", "gaussian", "kernel"}:
        return GaussianRKHSBasis(n_centers=n_features, sigma=sigma, include_bias=True, random_state=seed)
    if kind_l in {"nystrom", "nyström"}:
        return RBFNystromBasis(n_centers=n_features, sigma=sigma, include_bias=True, random_state=seed)
    raise ValueError(f"Unknown basis kind: {kind}")


def make_treatment_basis(kind: str = "poly1", **kwargs):
    return TreatmentInteractionBasis(base_basis=make_base_basis(kind, **kwargs))


def make_zonly_basis(kind: str = "poly1", **kwargs):
    return ZOnlyBasis(make_base_basis(kind, **kwargs))


# ---------------------------------------------------------------------------
# Synthetic DGPs
# ---------------------------------------------------------------------------


def standardize_columns(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    mu = A.mean(axis=0)
    sd = A.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (A - mu) / sd


def make_ate_data(
    *,
    n: int = 500,
    d: int = 8,
    kappa: float = 1.0,
    rho: float = 0.3,
    heterogeneous: bool = True,
    seed: int = 0,
    noise_sd: float = 1.0,
) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(d)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    Z = rng.multivariate_normal(np.zeros(d), Sigma, size=n)
    h = 0.6 * Z[:, 0] - 0.4 * Z[:, 1] + 0.25 * np.sin(Z[:, 2])
    if d >= 5:
        h += 0.2 * Z[:, 3] * Z[:, 4]
    e = expit(kappa * h)
    D = rng.binomial(1, e).astype(float)
    mu0 = 0.5 * Z[:, 0] + 0.5 * np.sin(Z[:, 1]) + 0.25 * Z[:, 2] ** 2
    if heterogeneous:
        tau = 1.0 + 0.5 * Z[:, 0] + 0.25 * np.sin(Z[:, 2])
    else:
        tau = np.ones(n)
    Y = mu0 + D * tau + rng.normal(0.0, noise_sd, size=n)
    X = np.column_stack([D, Z])
    return {"X": X, "Y": Y, "D": D, "Z": Z, "e": e, "tau": tau, "theta": float(np.mean(tau))}


def make_kang_schafer_data(n: int = 300, seed: int = 0, tau: float = 1.0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 4))
    logits = -Z[:, 0] + 0.5 * Z[:, 1] - 0.25 * Z[:, 2] - 0.1 * Z[:, 3]
    e = expit(logits)
    D = rng.binomial(1, e).astype(float)
    X1 = np.exp(Z[:, 0] / 2.0)
    X2 = Z[:, 1] / (1.0 + np.exp(Z[:, 0])) + 10.0
    X3 = (Z[:, 0] * Z[:, 2] / 25.0 + 0.6) ** 3
    X4 = (Z[:, 1] + Z[:, 3] + 20.0) ** 2
    Xobs = np.column_stack([X1, X2, X3, X4])
    features = np.column_stack([Xobs, Xobs ** 2])
    features = standardize_columns(features)
    mu0 = 210.0 + 27.4 * Z[:, 0] + 13.7 * Z[:, 1] + 13.7 * Z[:, 2] + 13.7 * Z[:, 3]
    Y = mu0 + tau * D + rng.normal(scale=1.0, size=n)
    X = np.column_stack([D, features])
    return {"X": X, "Y": Y, "D": D, "Z_latent": Z, "features": features, "theta": float(tau), "e": e}


def _kernel_matrix(X: np.ndarray, Y: np.ndarray, kernel: str, param: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if kernel == "gaussian":
        d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-param * d2)
    if kernel == "laplace":
        d1 = np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)
        return np.exp(-param * d1)
    if kernel == "poly1":
        return X @ Y.T + 0.5
    if kernel == "poly3":
        return (X @ Y.T + 0.5) ** 3
    raise ValueError(kernel)


def sample_gp_function(X: np.ndarray, *, kernel: str, param: float, seed: int, jitter: float = 1e-6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    K = _kernel_matrix(X, X, kernel, param) + jitter * np.eye(len(X))
    return rng.multivariate_normal(np.zeros(len(X)), K)


def make_kernel_gp_data(
    *,
    n: int = 300,
    d: int = 5,
    f_kernel: str = "poly1",
    g_kernel: str = "gaussian",
    g_param: float = 0.1,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Xcov = rng.normal(size=(n, d))
    if f_kernel == "poly1":
        f = sample_gp_function(Xcov, kernel="poly1", param=1.0, seed=seed + 11)
    else:
        f = sample_gp_function(Xcov, kernel="gaussian", param=0.1, seed=seed + 11)
    e = np.clip(expit(f / np.std(f)), 0.05, 0.95)
    D = rng.binomial(1, e).astype(float)
    g = sample_gp_function(Xcov, kernel=g_kernel, param=g_param, seed=seed + 23)
    Y = g + rng.normal(scale=1.0, size=n)  # sharp null; ATE=0
    return {"X": np.column_stack([D, Xcov]), "Y": Y, "D": D, "Z": Xcov, "theta": 0.0, "e": e}


def make_high_dim_data(
    *,
    n: int = 400,
    p: int = 100,
    st: int = 5,
    sy: int = 20,
    rho_strength: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    idx = np.arange(p)
    Sigma = 0.5 ** np.abs(idx[:, None] - idx[None, :])
    Z = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    theta = np.zeros(p); theta[:st] = 1.0 / math.sqrt(st)
    beta = np.zeros(p); beta[:sy] = 1.0 / math.sqrt(sy)
    e = expit(rho_strength * (Z @ theta))
    D = rng.binomial(1, e).astype(float)
    Y0 = Z @ beta + rng.normal(scale=5.0, size=n)
    Y = Y0  # sharp null; ATE=0
    return {"X": np.column_stack([D, Z]), "Y": Y, "D": D, "Z": Z, "theta": 0.0, "e": e}


def make_regressor_vs_covariate_data(n: int = 800, seed: int = 0, lam_tau: float = 0.5) -> dict[str, Any]:
    """DGP modeled on Kato's covariate-vs-regressor balancing simulation."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    e = expit(0.5 * Z[:, 0] - 0.4 * Z[:, 1] + 0.2 * np.sin(Z[:, 2]))
    D = rng.binomial(1, e).astype(float)
    # Use deterministic RFF-like features for treatment heterogeneity.
    psi = np.column_stack([
        np.sin(Z[:, 0]), np.cos(Z[:, 1]), Z[:, 0] * Z[:, 2], Z[:, 1] ** 2, np.sin(Z[:, 2])
    ])
    beta0 = np.array([0.4, -0.3, 0.2, 0.1, 0.2])
    betat = np.array([0.5, 0.2, -0.2, 0.15, 0.2]) * lam_tau
    mu0 = psi @ beta0
    tau = 1.0 + psi @ betat
    Y = mu0 + D * tau + rng.normal(scale=0.1, size=n)
    return {"X": np.column_stack([D, Z]), "Y": Y, "D": D, "Z": Z, "tau": tau, "theta": float(np.mean(tau)), "e": e}


# ---------------------------------------------------------------------------
# Balance and fitting helpers
# ---------------------------------------------------------------------------


def signed_alpha_to_group_weights(alpha: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.asarray(alpha, dtype=float)
    D = np.asarray(D, dtype=float)
    w = np.abs(alpha)
    return w[D == 1], w[D == 0]


def smd_matrix(Z: np.ndarray, D: np.ndarray, weights: np.ndarray | None = None) -> pd.DataFrame:
    Z = np.asarray(Z, dtype=float)
    D = np.asarray(D, dtype=float)
    if weights is None:
        weights = np.ones(len(D))
    weights = np.asarray(weights, dtype=float)
    rows = []
    for j in range(Z.shape[1]):
        z = Z[:, j]
        z1 = z[D == 1]
        z0 = z[D == 0]
        w1 = weights[D == 1]
        w0 = weights[D == 0]
        mu1 = np.average(z1, weights=w1) if len(z1) else np.nan
        mu0 = np.average(z0, weights=w0) if len(z0) else np.nan
        pooled = math.sqrt(0.5 * (np.var(z1) + np.var(z0))) if len(z1) and len(z0) else np.nan
        if not np.isfinite(pooled) or pooled == 0:
            smd = np.nan
        else:
            smd = (mu1 - mu0) / pooled
        rows.append({"feature": f"z{j}", "smd": smd, "abs_smd": abs(smd) if np.isfinite(smd) else np.nan})
    return pd.DataFrame(rows)


def feature_balance_gap(alpha: np.ndarray, X: np.ndarray, functional, basis) -> float:
    basis = basis.copy().fit(X)
    Phi = np.asarray(basis(X), dtype=float)
    M = np.asarray(functional.m_basis_matrix(X, basis), dtype=float)
    gaps = np.mean(alpha[:, None] * Phi - M, axis=0)
    return float(np.nanmax(np.abs(gaps)))


def fit_grr_glm_alpha(X: np.ndarray, functional, basis, generator, *, lam: float, penalty: str = "l2", max_iter: int = 300):
    b = basis.copy().fit(X)
    model = GRRGLM(basis=b, generator=generator, functional=functional, penalty=penalty, lam=lam)
    model.fit(X, max_iter=max_iter, tol=1e-8)
    return model.predict_alpha(X), model


def fit_ate_once(
    data: dict[str, Any],
    *,
    loss: str,
    lam: float,
    penalty: str = "l2",
    omega: float | None = None,
    basis_kind: str = "poly1",
    basis_mode: str = "regressor",
    cross_fit: bool = True,
    folds: int = 3,
    estimators: tuple[str, ...] = ("rw", "arw"),
    outcome_lam: float = 1e-3,
    max_iter: int = 300,
) -> pd.DataFrame:
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    theta = float(data["theta"])
    if basis_mode == "regressor":
        basis = make_treatment_basis(basis_kind, degree=1, n_features=40, sigma=1.0, seed=0)
    elif basis_mode == "covariate":
        basis = make_zonly_basis(basis_kind, degree=1, n_features=40, sigma=1.0, seed=0)
    else:
        raise ValueError(basis_mode)
    gen = make_generator(loss, omega=omega)
    try:
        res = grr_ate(
            X=X,
            Y=Y,
            basis=basis,
            generator=gen,
            cross_fit=cross_fit,
            folds=folds,
            riesz_penalty=penalty,
            riesz_lam=lam,
            outcome_lam=outcome_lam,
            estimators=estimators,
            max_iter=max_iter,
        )
        df = result_to_frame(res, true_theta=theta)
        df["loss"] = loss if omega is None else f"{loss}({omega:g})"
        df["lambda"] = lam
        df["penalty"] = penalty
        df["basis"] = basis_kind
        df["basis_mode"] = basis_mode
        df["status"] = "ok"
        return df
    except Exception as exc:
        return pd.DataFrame([{
            "loss": loss if omega is None else f"{loss}({omega:g})",
            "lambda": lam,
            "penalty": penalty,
            "basis": basis_kind,
            "basis_mode": basis_mode,
            "estimator": "failed",
            "status": type(exc).__name__,
            "message": str(exc)[:180],
            "true_theta": theta,
        }])


def run_ate_grid(
    dgp_func,
    *,
    reps: int,
    seed0: int,
    losses: list[tuple[str, float | None]],
    lambdas: list[float],
    penalties: list[str],
    basis_kind: str = "poly1",
    basis_mode: str = "regressor",
    dgp_kwargs: dict[str, Any] | None = None,
    cross_fit: bool = True,
    folds: int = 3,
    estimators: tuple[str, ...] = ("rw", "arw"),
) -> pd.DataFrame:
    rows = []
    dgp_kwargs = {} if dgp_kwargs is None else dict(dgp_kwargs)
    for r in range(reps):
        data = dgp_func(seed=seed0 + r, **dgp_kwargs)
        for loss, omega in losses:
            for lam in lambdas:
                for pen in penalties:
                    out = fit_ate_once(
                        data,
                        loss=loss,
                        omega=omega,
                        lam=lam,
                        penalty=pen,
                        basis_kind=basis_kind,
                        basis_mode=basis_mode,
                        cross_fit=cross_fit,
                        folds=folds,
                        estimators=estimators,
                    )
                    out["rep"] = r
                    rows.append(out)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Real and semi-synthetic data loaders with safe fallback
# ---------------------------------------------------------------------------


def _read_remote_pandas(reader, url: str, **kwargs):
    """Read a remote dataset when explicitly enabled.

    By default, notebook smoke runs use deterministic fallback data so execution
    does not hang in offline environments. Set the environment variable
    GRR_ALLOW_REMOTE_DATA=1, or set DOWNLOAD_DATA=True in the empirical notebooks,
    to use public benchmark files.
    """
    if os.environ.get("GRR_ALLOW_REMOTE_DATA", "0") != "1":
        warnings.warn(f"Remote download disabled for {url}. Using deterministic fallback data.")
        return None, "fallback"
    try:
        return reader(url, **kwargs), "remote"
    except Exception as exc:
        warnings.warn(f"Could not download {url}. Using deterministic fallback data. Error: {exc}")
        return None, "fallback"


def load_ihdp(replication: int = 1, *, fallback_seed: int = 101) -> dict[str, Any]:
    url = f"https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{int(replication)}.csv"
    df, source = _read_remote_pandas(pd.read_csv, url, header=None)
    if df is not None and df.shape[1] >= 30:
        D = df.iloc[:, 0].to_numpy(float)
        yf = df.iloc[:, 1].to_numpy(float)
        mu0 = df.iloc[:, 3].to_numpy(float)
        mu1 = df.iloc[:, 4].to_numpy(float)
        Z = df.iloc[:, 5:].to_numpy(float)
        X = np.column_stack([D, standardize_columns(Z)])
        tau = mu1 - mu0
        return {"X": X, "Y": yf, "D": D, "Z": Z, "tau": tau, "theta": float(np.mean(tau)), "theta_ate": float(np.mean(tau)), "theta_att": float(np.mean(tau[D == 1])) if np.any(D == 1) else float(np.mean(tau)), "source": source, "name": "IHDP"}
    data = make_ate_data(n=747, d=25, kappa=1.2, heterogeneous=True, seed=fallback_seed)
    data["source"] = "deterministic fallback"
    data["name"] = "IHDP fallback"
    return data


def _standardize_lalonde_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Common variants from NBER Stata and txt files.
    rename = {
        "treat": "treat", "treatment": "treat", "age": "age", "educ": "educ", "education": "educ",
        "black": "black", "hispan": "hispan", "hispanic": "hispan", "married": "married", "nodegree": "nodegree",
        "re74": "re74", "re75": "re75", "re78": "re78",
    }
    out = {}
    for col in df.columns:
        c = str(col).lower().strip()
        if c in rename:
            out[col] = rename[c]
    df = df.rename(columns=out)
    needed = ["treat", "age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75", "re78"]
    if not all(c in df.columns for c in needed):
        # Handle unnamed text layout.
        if df.shape[1] >= 10:
            df = df.iloc[:, :10]
            df.columns = needed
    return df[needed].apply(pd.to_numeric, errors="coerce").dropna()


def load_lalonde(control_source: str = "cps3", *, fallback_seed: int = 202) -> dict[str, Any]:
    base = "https://users.nber.org/~rdehejia/data/"
    nsw_url = base + "nsw_dw.dta"
    controls = {
        "cps": "cps_controls.dta",
        "cps2": "cps_controls2.dta",
        "cps3": "cps_controls3.dta",
        "psid": "psid_controls.dta",
        "psid2": "psid_controls2.dta",
        "psid3": "psid_controls3.dta",
    }
    control_file = controls.get(str(control_source).lower(), "cps_controls3.dta")
    nsw, source1 = _read_remote_pandas(pd.read_stata, nsw_url)
    ctrl, source2 = _read_remote_pandas(pd.read_stata, base + control_file)
    if nsw is not None and ctrl is not None:
        nsw = _standardize_lalonde_columns(nsw)
        ctrl = _standardize_lalonde_columns(ctrl)
        exp_tau = float(nsw.loc[nsw.treat == 1, "re78"].mean() - nsw.loc[nsw.treat == 0, "re78"].mean())
        treated = nsw.loc[nsw.treat == 1].copy()
        control = ctrl.copy()
        control["treat"] = 0.0
        df = pd.concat([treated, control], ignore_index=True)
        covars = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]
        D = df["treat"].to_numpy(float)
        Z = standardize_columns(df[covars].to_numpy(float))
        Y = df["re78"].to_numpy(float)
        X = np.column_stack([D, Z])
        return {"X": X, "Y": Y, "D": D, "Z": Z, "theta_benchmark": exp_tau, "source": f"remote {source1}/{source2}", "name": f"Lalonde NSW + {control_source.upper()}", "covariates": covars}
    rng = np.random.default_rng(fallback_seed)
    n_t, n_c, d = 185, 429, 8
    Zt = rng.normal(0.2, 1.0, size=(n_t, d)); Zc = rng.normal(0.0, 1.1, size=(n_c, d))
    D = np.r_[np.ones(n_t), np.zeros(n_c)]
    Z = np.vstack([Zt, Zc])
    tau = 1800.0
    Y = 5000 + 1200 * Z[:, 0] - 600 * Z[:, 1] + tau * D + rng.normal(0, 3000, size=n_t + n_c)
    return {"X": np.column_stack([D, standardize_columns(Z)]), "Y": Y, "D": D, "Z": Z, "theta_benchmark": tau, "source": "deterministic fallback", "name": "Lalonde fallback", "covariates": [f"x{j}" for j in range(d)]}


def load_acic(condition: int = 1, *, fallback_seed: int = 303) -> dict[str, Any]:
    base = "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/acic_challenge_2016/"
    xdf, source1 = _read_remote_pandas(pd.read_csv, base + "x.csv")
    zdf, source2 = _read_remote_pandas(pd.read_csv, base + f"zymu_{int(condition):02d}.csv")
    if xdf is not None and zdf is not None:
        Xcov = pd.get_dummies(xdf, drop_first=True)
        Z = standardize_columns(Xcov.to_numpy(float))
        zdf_cols = {c.lower(): c for c in zdf.columns}
        D = zdf[zdf_cols.get("z", "z")].to_numpy(float)
        y0 = zdf[zdf_cols.get("y0", "y0")].to_numpy(float)
        y1 = zdf[zdf_cols.get("y1", "y1")].to_numpy(float)
        mu0 = zdf[zdf_cols.get("mu0", "mu0")].to_numpy(float)
        mu1 = zdf[zdf_cols.get("mu1", "mu1")].to_numpy(float)
        Y = D * y1 + (1.0 - D) * y0
        tau = mu1 - mu0
        return {"X": np.column_stack([D, Z]), "Y": Y, "D": D, "Z": Z, "tau": tau, "theta": float(np.mean(tau)), "theta_ate": float(np.mean(tau)), "theta_att": float(np.mean(tau[D == 1])) if np.any(D == 1) else float(np.mean(tau)), "source": f"remote {source1}/{source2}", "name": f"ACIC 2016 condition {condition}"}
    data = make_ate_data(n=800, d=20, kappa=1.5, heterogeneous=True, seed=fallback_seed)
    data["source"] = "deterministic fallback"
    data["name"] = "ACIC fallback"
    return data


def load_hdma(*, fallback_seed: int = 404) -> dict[str, Any]:
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Hmda.csv"
    df, source = _read_remote_pandas(pd.read_csv, url)
    if df is not None:
        df = df.copy()
        cols = {c.lower(): c for c in df.columns}
        # Ecdat/Hmda uses deny and afam in common versions.
        y_col = cols.get("deny")
        d_col = cols.get("afam") or cols.get("black")
        if y_col is not None and d_col is not None:
            Y = df[y_col].astype(str).str.lower().isin(["yes", "1", "true"]).astype(float).to_numpy()
            D = df[d_col].astype(str).str.lower().isin(["yes", "1", "true"]).astype(float).to_numpy()
            candidate_covars = [
                "pirat", "hirat", "lvrat", "chist", "mhist", "phist", "insurance", "selfemp", "single", "hschool", "unemp", "condomin",
            ]
            covars = [cols[c] for c in candidate_covars if c in cols]
            Zdf = pd.get_dummies(df[covars], drop_first=True)
            Z = standardize_columns(Zdf.to_numpy(float))
            return {"X": np.column_stack([D, Z]), "Y": Y, "D": D, "Z": Z, "source": source, "name": "Boston HDMA", "covariates": list(Zdf.columns)}
    rng = np.random.default_rng(fallback_seed)
    n, d = 2380, 12
    Z = rng.normal(size=(n, d))
    p_black = expit(-1.5 + 0.7 * Z[:, 0] + 0.2 * Z[:, 1])
    D = rng.binomial(1, p_black).astype(float)
    p_deny = expit(-1.5 + 0.35 * D + 0.8 * Z[:, 0] + 0.5 * Z[:, 2])
    Y = rng.binomial(1, p_deny).astype(float)
    return {"X": np.column_stack([D, standardize_columns(Z)]), "Y": Y, "D": D, "Z": Z, "source": "deterministic fallback", "name": "HDMA fallback", "covariates": [f"x{j}" for j in range(d)]}


def load_nsw_randomized(*, fallback_seed: int = 505) -> dict[str, Any]:
    base = "https://users.nber.org/~rdehejia/data/"
    nsw, source = _read_remote_pandas(pd.read_stata, base + "nsw_dw.dta")
    if nsw is not None:
        df = _standardize_lalonde_columns(nsw)
        covars = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]
        D = df["treat"].to_numpy(float)
        Z = standardize_columns(df[covars].to_numpy(float))
        Y = df["re78"].to_numpy(float)
        tau = float(Y[D == 1].mean() - Y[D == 0].mean())
        return {"X": np.column_stack([D, Z]), "Y": Y, "D": D, "Z": Z, "theta_benchmark": tau, "source": source, "name": "NSW randomized", "covariates": covars}
    return load_lalonde(control_source="cps3", fallback_seed=fallback_seed)


# ---------------------------------------------------------------------------
# Semi-synthetic conversion for real covariates
# ---------------------------------------------------------------------------


def make_semisynthetic_from_covariates(data: dict[str, Any], *, seed: int = 0, tau0: float = 1.0, kappa: float = 1.0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Z = np.asarray(data["Z"], dtype=float)
    if Z.ndim != 2:
        raise ValueError("data['Z'] must be 2D")
    Zs = standardize_columns(Z)
    score = 0.6 * Zs[:, 0]
    if Zs.shape[1] > 1:
        score -= 0.4 * Zs[:, 1]
    if Zs.shape[1] > 2:
        score += 0.25 * np.sin(Zs[:, 2])
    e = np.clip(expit(kappa * score), 0.05, 0.95)
    D = rng.binomial(1, e).astype(float)
    mu0 = 0.5 * Zs[:, 0]
    if Zs.shape[1] > 1:
        mu0 += 0.3 * Zs[:, 1] ** 2
    tau = tau0 + 0.2 * Zs[:, 0]
    Y = mu0 + D * tau + rng.normal(scale=1.0, size=len(D))
    return {"X": np.column_stack([D, Zs]), "Y": Y, "D": D, "Z": Zs, "tau": tau, "theta": float(np.mean(tau)), "theta_ate": float(np.mean(tau)), "theta_att": float(np.mean(tau[D == 1])) if np.any(D == 1) else float(np.mean(tau)), "source": f"semi-synthetic from {data.get('name', 'covariates')}", "name": f"Semi-synthetic {data.get('name', '')}"}


def fit_and_summarize_dataset(
    data: dict[str, Any],
    *,
    estimand: str = "ate",
    loss_grid: list[tuple[str, float | None]] | None = None,
    lambdas: list[float] | None = None,
    basis_kind: str = "poly1",
    cross_fit: bool = True,
    folds: int = 3,
    true_key: str = "theta",
    benchmark_key: str | None = None,
    estimators: tuple[str, ...] = ("rw", "arw"),
    max_iter: int = 300,
) -> pd.DataFrame:
    if loss_grid is None:
        loss_grid = [("SQ", None), ("UKL", None), ("BKL", None), ("BP", 0.5), ("BP", 1.0)]
    if lambdas is None:
        lambdas = [1e-3, 1e-2]
    rows = []
    for loss, omega in loss_grid:
        for lam in lambdas:
            X = np.asarray(data["X"], dtype=float); Y = np.asarray(data["Y"], dtype=float)
            basis = make_treatment_basis(basis_kind, degree=1, n_features=40, sigma=1.0, seed=0)
            gen = make_generator(loss, omega=omega)
            label = loss if omega is None else f"{loss}({omega:g})"
            try:
                if estimand.lower() == "ate":
                    res = grr_ate(X=X, Y=Y, basis=basis, generator=gen, cross_fit=cross_fit, folds=folds, riesz_lam=lam, estimators=estimators, max_iter=max_iter)
                elif estimand.lower() == "att":
                    res = grr_att(X=X, Y=Y, basis=basis, generator=gen, cross_fit=cross_fit, folds=folds, riesz_lam=lam, estimators=estimators, max_iter=max_iter)
                else:
                    raise ValueError(estimand)
                true_theta = data.get(true_key, None)
                df = result_to_frame(res, true_theta=true_theta if true_theta is not None else None, label=label)
                if benchmark_key is not None and benchmark_key in data:
                    df["benchmark"] = data[benchmark_key]
                    df["benchmark_error"] = df["estimate"] - float(data[benchmark_key])
                df["loss"] = label; df["lambda"] = lam; df["source"] = data.get("source", "")
            except Exception as exc:
                df = pd.DataFrame([{"label": label, "loss": label, "lambda": lam, "estimator": "failed", "status": type(exc).__name__, "message": str(exc)[:180], "source": data.get("source", "")}])
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def branch_squared_generator(C: float = 1.0):
    """Squared generator centered at branch-specific signs for signed ATE weights.

    This is useful for didactic covariate-balance experiments where a pure Z-only
    balancing dictionary would otherwise make the unshifted squared loss choose
    the trivial zero representer.
    """
    from genriesz import BregmanGenerator

    C = float(C)

    def _sign(X):
        X = np.asarray(X, dtype=float)
        return np.where(X[:, 0] >= 0.5, 1.0, -1.0)

    def g(X, alpha):
        X = np.asarray(X, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        s = _sign(X)
        return (alpha - s * C) ** 2

    def grad(X, alpha):
        X = np.asarray(X, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        s = _sign(X)
        return 2.0 * (alpha - s * C)

    def inv_grad(X, v):
        X = np.asarray(X, dtype=float)
        v = np.asarray(v, dtype=float)
        s = _sign(X)
        return s * C + 0.5 * v

    def grad2(X, alpha):
        return np.full_like(np.asarray(alpha, dtype=float), 2.0)

    return BregmanGenerator(g=g, grad=grad, inv_grad=inv_grad, grad2=grad2, name="branch-SQ")


def subsample_data(data: dict[str, Any], n: int, *, seed: int = 0) -> dict[str, Any]:
    """Return a shallow copy with X, Y, D, Z subsampled for fast notebook runs."""
    X = np.asarray(data["X"])
    if len(X) <= n:
        return data
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=int(n), replace=False))
    out = dict(data)
    for key in ["X", "Y", "D", "Z"]:
        if key in out:
            arr = np.asarray(out[key])
            if arr.shape[0] == len(X):
                out[key] = arr[idx]
    if "tau" in out:
        arr = np.asarray(out["tau"])
        if arr.shape[0] == len(X):
            out["tau"] = arr[idx]
            out["theta"] = float(np.mean(out["tau"]))
    return out

# ---------------------------------------------------------------------------
# ATE and ATT joint-experiment helpers
# ---------------------------------------------------------------------------


def true_theta_for_estimand(data: dict[str, Any], estimand: str) -> float | None:
    """Return the available ground-truth value for ATE or ATT.

    The simulation notebooks report ATE and ATT whenever the target is well
    defined.  For synthetic and semi-synthetic data with a stored treatment
    effect vector ``tau``, ATE is the sample average of ``tau`` and ATT is the
    sample average over units with observed treatment ``D=1``.  If only a scalar
    ``theta`` is available, it is used as a fallback, which is appropriate for
    sharp-null and homogeneous-effect designs.
    """
    e = str(estimand).lower()
    if e == "ate":
        if "theta_ate" in data:
            return float(data["theta_ate"])
        if "theta" in data:
            return float(data["theta"])
        if "tau" in data:
            return float(np.mean(np.asarray(data["tau"], dtype=float)))
        return None
    if e == "att":
        if "theta_att" in data:
            return float(data["theta_att"])
        if "tau" in data and "D" in data:
            tau = np.asarray(data["tau"], dtype=float)
            D = np.asarray(data["D"], dtype=float)
            if tau.shape[0] == D.shape[0] and np.any(D == 1):
                return float(np.mean(tau[D == 1]))
        if "theta" in data:
            return float(data["theta"])
        return None
    raise ValueError(f"Unknown estimand: {estimand}")


def _copy_with_target(data: dict[str, Any], estimand: str) -> dict[str, Any]:
    out = dict(data)
    theta = true_theta_for_estimand(data, estimand)
    if theta is not None:
        out["_theta_for_estimand"] = float(theta)
    return out


def fit_grr_estimand(
    data: dict[str, Any],
    *,
    estimand: str,
    loss: str,
    omega: float | None,
    basis,
    lam: float,
    cross_fit: bool,
    folds: int,
    estimators: tuple[str, ...],
    penalty: str = "l2",
    outcome_lam: float = 1e-3,
    max_iter: int = 300,
) -> pd.DataFrame:
    """Fit one GRR model for ATE or ATT and return an estimate-level frame."""
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    gen = make_generator(loss, omega=omega)
    estimand_l = str(estimand).lower()
    if estimand_l == "ate":
        res = grr_ate(
            X=X, Y=Y, basis=basis, generator=gen, cross_fit=cross_fit, folds=folds,
            riesz_penalty=penalty, riesz_lam=lam, outcome_lam=outcome_lam,
            estimators=estimators, max_iter=max_iter,
        )
    elif estimand_l == "att":
        res = grr_att(
            X=X, Y=Y, basis=basis, generator=gen, cross_fit=cross_fit, folds=folds,
            riesz_penalty=penalty, riesz_lam=lam, outcome_lam=outcome_lam,
            estimators=estimators, max_iter=max_iter,
        )
    else:
        raise ValueError(f"Unknown estimand: {estimand}")
    theta = true_theta_for_estimand(data, estimand_l)
    df = result_to_frame(res, true_theta=theta)
    df["estimand"] = estimand_l.upper()
    return df


def fit_ate_att_dataset(
    data: dict[str, Any],
    *,
    estimands: tuple[str, ...] = ("ate", "att"),
    loss_grid: list[tuple[str, float | None]] | None = None,
    lambdas: list[float] | None = None,
    basis_kind: str = "poly1",
    cross_fit: bool = True,
    folds: int = 3,
    estimators: tuple[str, ...] = ("rw", "arw"),
    benchmark_keys: dict[str, str | None] | None = None,
    max_iter: int = 300,
) -> pd.DataFrame:
    """Fit both ATE and ATT whenever possible for an empirical dataset."""
    if loss_grid is None:
        loss_grid = [("SQ", None), ("UKL", None), ("BKL", None), ("BP", 0.5)]
    if lambdas is None:
        lambdas = [1e-2]
    rows: list[pd.DataFrame] = []
    for estimand in estimands:
        e = str(estimand).lower()
        d = _copy_with_target(data, e)
        true_key = "_theta_for_estimand" if "_theta_for_estimand" in d else "__missing_target__"
        if benchmark_keys is None:
            benchmark_key = "theta_benchmark" if e == "att" and "theta_benchmark" in data else None
        else:
            benchmark_key = benchmark_keys.get(e)
        df = fit_and_summarize_dataset(
            d,
            estimand=e,
            loss_grid=loss_grid,
            lambdas=lambdas,
            basis_kind=basis_kind,
            cross_fit=cross_fit,
            folds=folds,
            true_key=true_key,
            benchmark_key=benchmark_key,
            estimators=estimators,
            max_iter=max_iter,
        )
        df["estimand"] = e.upper()
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
