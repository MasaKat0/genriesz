"""Replication records and descriptive outputs shared by the five paper experiments."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from . import baselines
from .seeds import CONFIRMATORY_ENTROPY


def failed(status, **details):
    return {"estimate": np.nan, "se": np.nan, "status": str(status), **details}


def score_record(psi, ddof=0, **details):
    psi = np.asarray(psi, float)
    if not np.isfinite(psi).all():
        return failed("nonfinite", **details)
    centered = psi - psi.mean()
    top = float(np.max(np.abs(centered)))
    scaled = centered / top if top > 0 else np.zeros_like(centered)
    mass = float(scaled @ scaled)
    return {
        "score_concentration": float(np.max(scaled**2) / mass) if mass > 0 else np.nan,
        "estimate": float(psi.mean()),
        "se": float(psi.std(ddof=ddof) / np.sqrt(len(psi))),
        "status": "ok",
        **details,
    }


def recorded(call):
    result, warnings, converged = baselines.run_recording_warnings(call)
    result = dict(result, warnings=warnings)
    if not converged:
        result.update(estimate=np.nan, se=np.nan, status="convergence_warning")
    elif result["status"] == "ok" and not np.isfinite([result["estimate"], result["se"]]).all():
        result.update(estimate=np.nan, se=np.nan, status="nonfinite")
    return result


class FitLog:
    def __init__(self):
        self.fits = []
        self.converged = True

    def hook(self, stage, fold, call):
        result, warnings, converged = baselines.run_recording_warnings(call)
        self.converged &= converged
        self.fits.append(
            {
                "fold": int(fold),
                "model": stage,
                "status": str(getattr(result, "status", "ok")),
                "message": str(getattr(result, "message", "")),
                "warnings": warnings,
            }
        )
        return result


def functional_record(result, log):
    detail = {"fits": log.fits, "fold_status": result.fold_status}
    if not result.success or not log.converged:
        return failed(result.status if not result.success else "convergence_warning", **detail)
    r = result["arw"]
    if not np.isfinite([r.estimate, r.se]).all():
        return failed("nonfinite", **detail)
    a = np.asarray(result.diagnostics["alpha_values"], float)
    return {
        "estimate": float(r.estimate),
        "se": float(r.se),
        "status": "ok",
        "max_abs_alpha": float(np.max(np.abs(a))),
        "ess": float(np.sum(np.abs(a)) ** 2 / (a @ a)),
        "alpha": a.tolist(),
        "score_concentration": next(
            (
                v
                for k, v in result.diagnostics.get("score_concentration", {}).items()
                if k.startswith("arw")
            ),
            np.nan,
        ),
        **detail,
    }


def tasks_for(module, entropy=CONFIRMATORY_ENTROPY):
    return [
        (c, r, entropy)
        for c in range(len(module.CELLS))
        for r in range(module.R)
    ]


def raw_frame(module, tasks, results):
    rows = []
    for (cell, rep, entropy), result in zip(tasks, results, strict=True):
        folds = result["folds"]
        for arm, record in result["arms"].items():
            row = {
                "cell": cell,
                "rep": rep,
                "entropy": entropy,
                "arm": arm,
                "n": module.CELLS[cell][-1],
                "folds": json.dumps(folds),
                **record,
            }
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    row[key] = json.dumps(value, default=lambda x: x.item())
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(raw, theta0):
    rows = []
    for (cell, n, arm), frame in raw.groupby(["cell", "n", "arm"], sort=False):
        target = theta0(int(cell)) if callable(theta0) else theta0
        ok = frame.status.eq("ok") & np.isfinite(frame.estimate) & np.isfinite(frame.se)
        estimates, ses = frame.loc[ok, "estimate"], frame.loc[ok, "se"]
        R, returned = len(frame), int(ok.sum())
        covered = int(((estimates - target).abs() <= stats.norm.ppf(0.975) * ses).sum())
        coverage, fail = covered / R, 1 - returned / R
        row = {
            "cell": cell,
            "n": n,
            "arm": arm,
            "R": R,
            "returned": returned,
            "bias_conditional": float((estimates - target).mean()),
            "bias_mcse_conditional": float(estimates.std(ddof=1) / np.sqrt(returned))
            if returned > 1
            else np.nan,
            "rmse_conditional": float(np.sqrt(np.mean((estimates - target) ** 2))),
            "coverage": coverage,
            "coverage_mcse": np.sqrt(coverage * (1 - coverage) / R),
            "length_conditional": float(2 * stats.norm.ppf(0.975) * ses.mean()),
            "failure_rate": fail,
            "failure_mcse": np.sqrt(fail * (1 - fail) / R),
            "status_counts": json.dumps(frame.status.value_counts().to_dict()),
        }
        for field in (
            "max_abs_alpha",
            "ess",
            "train_imbalance",
            "eval_imbalance",
            "score_concentration",
            "representer_l2",
        ):
            if field in frame:
                row[field + "_median"] = float(frame[field].median())
        rows.append(row)
    return pd.DataFrame(rows)


def tables(summary, exp):
    cols = [
        c
        for c in (
            "n",
            "arm",
            "bias_conditional",
            "rmse_conditional",
            "coverage",
            "failure_rate",
            "length_conditional",
        )
        if c in summary
    ]
    return {f"tab_E{exp}": summary[cols].to_latex(index=False, escape=True, float_format="%.4f")}


def figure(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for arm, frame in summary.groupby("arm", sort=False):
        ax.plot(frame["n"], frame["failure_rate"], "o-", label=arm)
    ax.set(xlabel="n", ylabel="Failure rate", ylim=(0, 1))
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
