"""E-19 support code: DGPs, dictionaries and their derivatives, the SQ-Riesz fit for the
AME, the cross-fitted ARW and TMLE, the counterexamples, Stage 0, aggregation, families,
table and figure."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import e19  # noqa: E402
from grrexp.seeds import PILOT_ENTROPY, Seeds, fold_ids  # noqa: E402


@pytest.fixture(scope="module")
def pop():
    return e19.stage0()


def _rows(n=50, seed=0):
    return e19.draw_q(np.random.default_rng(seed), n)


# ---------------------------------------------------------------- DGPs and dictionaries


def test_draw_q_is_deterministic_and_finite() -> None:
    X1, Y1 = _rows(200, 4)
    X2, Y2 = _rows(200, 4)
    assert np.array_equal(X1, X2) and np.array_equal(Y1, Y2)
    assert X1.shape == (200, 3) and np.all(np.isfinite(X1)) and np.all(np.isfinite(Y1))
    assert np.all(np.abs(X1[:, 1:]) <= np.sqrt(3))


def test_draw_q_matches_the_conditional_density() -> None:
    # E[D] and E[D^2] against the quadrature of the registered density
    X, _ = e19.draw_q(np.random.default_rng(1), 200_000)
    Xq, w = e19.quadrature_q(48, e19.d_rule(48))
    for k in (1, 2):
        assert abs(np.mean(X[:, 0] ** k) - w @ Xq[:, 0] ** k) < 0.01


def test_draw_l_uses_logistic_noise() -> None:
    X, _ = e19.draw_l(np.random.default_rng(2), 100_000)
    u = X[:, 0] - e19.mu(X[:, 1:])
    assert abs(u.var() - np.pi**2 / 3) < 0.05


@pytest.mark.parametrize(
    "f, df", [(e19.feat_alpha, e19.dfeat_alpha), (e19.feat_gamma, e19.dfeat_gamma)]
)
def test_dictionary_derivatives_match_finite_differences(f, df) -> None:
    X, _ = _rows(20)
    h = 1e-6
    Xp, Xm = X.copy(), X.copy()
    Xp[:, 0] += h
    Xm[:, 0] -= h
    assert np.allclose(df(X), (f(Xp) - f(Xm)) / (2 * h), atol=1e-6)


def test_gamma0_and_alpha0_lie_in_the_spans() -> None:
    X, _ = _rows(30)
    c_g, *_ = np.linalg.lstsq(e19.feat_gamma(X), e19.gamma0(X), rcond=None)
    assert np.allclose(e19.feat_gamma(X) @ c_g, e19.gamma0(X), atol=1e-12)
    assert np.allclose(e19.feat_alpha(X) @ e19.ALPHA0_COEF, e19.alpha0_q(X), atol=1e-12)


def test_dgamma0_is_the_derivative_of_gamma0() -> None:
    X, _ = _rows(20)
    h = 1e-6
    Xp, Xm = X.copy(), X.copy()
    Xp[:, 0] += h
    Xm[:, 0] -= h
    assert np.allclose(e19.dgamma0(X), (e19.gamma0(Xp) - e19.gamma0(Xm)) / (2 * h), atol=1e-6)


# ---------------------------------------------------------------- fits and estimators


def test_sq_riesz_matches_the_closed_form_and_its_derivative() -> None:
    X, _ = _rows(400, 7)
    status, af, daf, imb = e19._fit_sq_riesz(X, e19.alpha_basis(), None, 0.0)
    assert status == "ok" and imb <= 1e-8
    Phi, M = e19.feat_alpha(X), e19.dfeat_alpha(X)
    beta = np.linalg.solve(Phi.T @ Phi / len(X), M.mean(axis=0))  # alpha = Phi beta
    assert np.allclose(af(X), Phi @ beta, atol=1e-8)
    h = 1e-6
    Xp, Xm = X.copy(), X.copy()
    Xp[:, 0] += h
    Xm[:, 0] -= h
    assert np.allclose(daf(X), (af(Xp) - af(Xm)) / (2 * h), atol=1e-5)


def test_rff_riesz_derivative_matches_finite_differences() -> None:
    X, _ = e19.draw_l(np.random.default_rng(3), 300)
    status, af, daf, _ = e19._fit_sq_riesz(X, e19.rff_basis(11), "l2", e19.RIESZ_LAM)
    assert status == "ok"
    h = 1e-6
    Xp, Xm = X.copy(), X.copy()
    Xp[:, 0] += h
    Xm[:, 0] -= h
    assert np.allclose(daf(X), (af(Xp) - af(Xm)) / (2 * h), atol=1e-5)


def test_crossfit_arw_and_tmle_follow_the_definitions() -> None:
    X, Y = _rows(500, 9)
    folds = fold_ids(len(Y), e19.K, np.random.default_rng(0))
    arw, tmle = e19._inference(X, Y, folds)
    assert arw["status"] == tmle["status"] == "ok"
    a, da, g, dg = (np.empty(len(Y)) for _ in range(4))
    for k in range(e19.K):
        tr, te = folds != k, folds == k
        _, af, daf, _ = e19._fit_sq_riesz(X[tr], e19.alpha_basis(), None, 0.0)
        coef, *_ = np.linalg.lstsq(e19.feat_gamma(X[tr]), Y[tr], rcond=None)
        a[te], da[te] = af(X[te]), daf(X[te])
        g[te], dg[te] = e19.feat_gamma(X[te]) @ coef, e19.dfeat_gamma(X[te]) @ coef
    psi = dg + a * (Y - g)
    assert math.isclose(arw["estimate"], psi.mean(), rel_tol=0, abs_tol=1e-10)
    assert math.isclose(arw["se"], np.std(psi - psi.mean()) / np.sqrt(len(Y)), abs_tol=1e-12)
    eps = np.sum(a * (Y - g)) / np.sum(a**2)
    assert math.isclose(tmle["estimate"], np.mean(dg + eps * da), abs_tol=1e-10)
    # the fluctuation solves the score equation along alpha_hat
    assert abs(np.sum(a * (Y - g - eps * a))) < 1e-8


def test_counterexamples_use_the_registered_perturbations() -> None:
    n = 1000
    X, Y = _rows(n, 5)
    out = e19._counterexamples(X, Y, n)
    j = 32
    a0 = e19.alpha0_q(X)
    d = X[:, 0]
    psi_i = e19.dgamma0(X) + np.cos(j * d) + a0 * (Y - e19.gamma0(X) - np.sin(j * d) / j)
    psi_ii = e19.dgamma0(X) + a0 * (Y - e19.gamma0(X) - n**-0.25 * (d > 0))
    assert math.isclose(out["Cex_i"]["estimate"], psi_i.mean(), abs_tol=1e-12)
    assert math.isclose(out["Cex_ii"]["estimate"], psi_ii.mean(), abs_tol=1e-12)


# ---------------------------------------------------------------- Stage 0


def test_stage0_certifies_alpha0_and_freezes_the_predictions(pop) -> None:
    inf = pop["inference"]
    assert inf["certified"] and all(inf["checks"].values())
    assert abs(inf["theta0"] - inf["theta0_48"]) <= 1e-10
    assert abs(inf["V_star"] - inf["V_star_48"]) <= 1e-10
    assert inf["riesz_identity_gap_64"] <= 1e-10
    assert all(math.isclose(c, 0.95, abs_tol=1e-3) for c in inf["c_n"].values())
    cex = pop["counterexamples"]
    assert abs(cex["f_D_at_0"] - cex["f_D_at_0_48"]) <= 1e-10
    for n in e19.N_PART["counterexamples"]:
        lo, hi = cex["i"][str(n)]["exact"]
        assert abs(lo["mean"]) < 1e-10 and abs(lo["variance"] - hi["variance"]) < 1e-8
        assert math.isclose(
            cex["ii"][str(n)]["mean_root_n_error"], -(n**0.25) * cex["f_D_at_0"], rel_tol=1e-12
        )
    assert abs(pop["illustration"]["mass"] - 1.0) < 1e-10
    json.dumps(pop, allow_nan=False)


def test_tasks_use_the_registered_replications() -> None:
    t = e19.tasks_for(PILOT_ENTROPY)
    per = pd.Series([e19.CELLS[c][0] for c, _, _ in t]).value_counts()
    for p in e19.PARTS:
        assert per[p] == e19.R_PART[p] * len(e19.N_PART[p])
    assert len(e19.tasks_for(PILOT_ENTROPY, 10)) == 10 * len(e19.CELLS)


# ---------------------------------------------------------------- aggregation, families, outputs


@pytest.fixture(scope="module")
def pilot(pop):
    tasks = e19.tasks_for(PILOT_ENTROPY, 12)
    raw = e19.raw_frame(tasks, [e19.replicate(t) for t in tasks])
    return raw


def test_replication_is_reproducible(pilot) -> None:
    again = e19.replicate((0, 3, PILOT_ENTROPY))
    row = pilot[(pilot["cell"] == 0) & (pilot["rep"] == 3) & (pilot["arm"] == "inference|ARW_cf")]
    assert again["inference|ARW_cf"]["estimate"] == float(row["estimate"].iloc[0])


def test_summary_families_table_and_figure(pop, pilot) -> None:
    from grrexp import e12

    summ = e19.summarise(pilot, pop)
    assert len(summ) == sum(len(e19.ESTIMATORS[p]) for p, _ in e19.CELLS)
    summ = summ.assign(**e12.family1_mcse(pilot, summ, Seeds(19, entropy=PILOT_ENTROPY)))
    v_inf, t_inf, _ = e19.family_inf(summ, pilot)
    v_cex, t_cex, _ = e19.family_cex(summ, pilot, pop)
    assert len(t_inf) == 3 * len(e19.N_PART["inference"])
    assert len(t_cex) == 2 * len(e19.N_PART["counterexamples"])
    assert all(0.0 <= p <= 1.0 for p in [*t_inf.values(), *t_cex.values()])
    summ["family_inf"] = json.dumps(v_inf.sentence())
    summ["family_cex"] = json.dumps(v_cex.sentence())
    tabs, fams = e19.tables(summ)
    tex = tabs["tab_E19"]
    assert tex.startswith("\\begin{tabular}") and tex.rstrip().endswith("\\end{tabular}")
    assert "-0.000" not in tex and "nan" not in tex
    assert set(fams) == {"family_inf", "family_cex"}
    import matplotlib.pyplot as plt

    plt.close(e19.figure(summ))


def test_fmt_has_no_negative_zero() -> None:
    assert e19._fmt(-0.0001) == "0.000" and e19._fmt(-0.01) == "-0.010" and e19._fmt(None) == "--"
