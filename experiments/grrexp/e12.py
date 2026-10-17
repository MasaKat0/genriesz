"""E-12: balance and omitted directions in ATE estimation."""

from __future__ import annotations

import itertools

import numpy as np

from . import baselines, study
from .seeds import CONFIRMATORY_ENTROPY, Seeds, fold_ids

EXP = 12
THETA0 = 1.0
S_VALUES = (0.5, 1.25)
SETS = ("Include", "Omit")
N_VALUES = (1000, 4000)
CELLS = [list(c) for c in itertools.product(S_VALUES, SETS, N_VALUES)]
R, K = 500, 5
SQRT3 = np.sqrt(3.0)
GRR_ARMS = ("SQ", "UKL1", "EB")
ARM_LABELS = tuple(f"{a}|{e}" for a in GRR_ARMS for e in ("RW_full", "ARW_cf")) + ("LogitAIPW",)


def gr():
    import genriesz

    return genriesz


def sign(A):
    return np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)


def features(Z, regressor_set):
    """phi(z): (1, z1, z2, z3, q) for Include, (1, z1, z2, z3) for Omit; q = z2^2 - 1."""
    Z = np.atleast_2d(Z)
    cols = [np.ones(len(Z)), Z[:, 0], Z[:, 1], Z[:, 2]]
    if regressor_set == "Include":
        cols.append(Z[:, 1] ** 2 - 1.0)
    return np.column_stack(cols)


def basis(regressor_set):
    g = gr()
    b = g.TreatmentInteractionBasis(
        base_basis=g.CallableBasis(lambda Z: features(Z, regressor_set))
    )
    return b


def alpha_ref(X):
    return np.where(np.atleast_2d(X)[:, 0] == 1, 2.0, -2.0)


def eta(Z, s):
    q = Z[:, 1] ** 2 - 1.0
    return s * (Z[:, 0] - 0.5 * Z[:, 1] + 0.6 * q)


def gamma0(D, Z):
    return THETA0 * D + Z[:, 0] + 0.5 * (Z[:, 1] ** 2 - 1.0)


def draw(rng, n, s):
    Z = rng.uniform(-SQRT3, SQRT3, size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-eta(Z, s)))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = gamma0(D, Z) + rng.standard_normal(n)
    return D, Z, Y


sign.vectorized = True


def generator(arm):
    g = gr()
    return {
        "SQ": lambda: g.SquaredGenerator(C=0.0),
        "UKL1": lambda: g.UKLGenerator(C=1.0, branch_fn=sign),
        "EB": lambda: g.UKLGenerator(C=0.0, branch_fn=sign),
    }[arm]()


def replicate(task):
    cell, rep, entropy = task
    s, rs, n = CELLS[cell]
    seeds = Seeds(EXP, entropy)
    D, Z, Y = draw(seeds.data(cell, rep), n, s)
    X = np.column_stack([D, Z])
    folds = fold_ids(n, K, seeds.folds(cell, rep))
    arms = {}
    g = gr()
    for arm in GRR_ARMS:
        gen = generator(arm)

        def rw(gen=gen):
            model = g.GRRGLM(
                basis=basis(rs),
                generator=gen,
                functional=g.ATEFunctional(0),
                penalty=None,
                lam=0.0,
                offset=g.offset_from_alpha(gen, alpha_ref),
            )
            fit = model.fit(X)
            detail = {"fits": [{"status": str(fit.status), "message": str(fit.message)}]}
            if not fit.success:
                return study.failed(fit.status, **detail)
            result = g.rw_full_inference(grr=model, X=X, Y=Y, evaluate_m_alpha=False)
            a = model.predict_alpha(X)
            phi = model.basis(X)
            mphi = g.ATEFunctional(0).m_basis_matrix(X, model.basis)
            residual = np.mean(a[:, None] * phi - mphi, axis=0)
            coef = np.linalg.lstsq(phi, Y, rcond=None)[0]
            influence = mphi @ coef + a * (Y - phi @ coef) - result.estimate
            magnitude = np.max(np.abs(influence))
            scaled = influence / magnitude if magnitude > 0 else np.zeros_like(influence)
            mass = float(scaled @ scaled)
            return {
                "estimate": float(result.estimate),
                "se": float(result.se),
                "status": "ok",
                "train_imbalance": float(np.max(np.abs(residual))),
                "ess": float(np.sum(np.abs(a)) ** 2 / (a @ a)),
                "score_concentration": float(np.max(scaled**2) / mass) if mass > 0 else np.nan,
                "max_abs_alpha": float(np.max(np.abs(a))),
                **detail,
            }

        arms[f"{arm}|RW_full"] = study.recorded(rw)
        log = study.FitLog()
        res = g.grr_functional(
            X=X,
            Y=Y,
            m=g.ATEFunctional(0),
            basis=basis(rs),
            generator=gen,
            riesz_penalty=None,
            riesz_lam=0.0,
            riesz_alpha_ref=alpha_ref,
            outcome_models="shared",
            outcome_link="identity",
            outcome_penalty=None,
            outcome_lam=0.0,
            fold_ids=folds,
            estimators=("arw",),
            expose_alpha_values=True,
            evaluate_m_alpha=False,
            fit_hook=log.hook,
        )
        rec = study.functional_record(res, log)
        rec.pop("alpha", None)
        balance = res.diagnostics.get("train_imbalance", {}).get("max", [])
        rec["train_imbalance"] = float(max(balance)) if balance else np.nan
        rec["eval_imbalance"] = res.diagnostics.get("held_out_imbalance_max", np.nan)
        arms[f"{arm}|ARW_cf"] = rec
    arms["LogitAIPW"] = study.recorded(
        lambda: baselines.logit_aipw(D, features(Z, rs)[:, 1:], Y, folds)
    )
    return {"arms": arms, "folds": folds.tolist()}


def tasks_for(entropy=CONFIRMATORY_ENTROPY):
    import sys

    return study.tasks_for(sys.modules[__name__], entropy)


def summarise(raw):
    return study.summarise(raw, THETA0)


def tables(summary):
    frame = summary.copy()
    frame["s"] = [CELLS[int(c)][0] for c in frame.cell]
    frame["regressor_set"] = [CELLS[int(c)][1] for c in frame.cell]
    cols = [
        "s",
        "regressor_set",
        "n",
        "arm",
        "bias_conditional",
        "rmse_conditional",
        "coverage",
        "failure_rate",
    ]
    balance = [
        "s",
        "regressor_set",
        "n",
        "arm",
        "train_imbalance_median",
        "eval_imbalance_median",
        "max_abs_alpha_median",
    ]
    return {
        "tab_E12": frame[cols].to_latex(index=False, escape=True, float_format="%.4f"),
        "tab_E12_balance": frame.reindex(columns=balance).to_latex(
            index=False, escape=True, float_format="%.4g"
        ),
    }
