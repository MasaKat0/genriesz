"""E-24 support code: design checks, features, AutoDML-lasso, outcome ridge, scores,
one replication, aggregation, comparisons (family 7), tables and figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import e24, outputs  # noqa: E402
from grrexp.seeds import PILOT_ENTROPY, Seeds  # noqa: E402


def test_stage0_design_constants() -> None:
    out = e24.stage0(points=(24, 32))
    assert out["ok"], out["checks"]
    v = out["points"]["32"]
    assert abs(v["theta0"] - 1.0) < 1e-12
    assert v["V_star"] > v["E_alpha0_sq"] > 2.0


def test_draw_matches_the_dgp() -> None:
    D, Z, Y = e24.draw(np.random.default_rng(0), 200000)
    assert Z.shape == (200000, 6) and np.all(np.abs(Z) <= np.sqrt(3))
    assert abs(np.mean(e24.tau(Z)) - 1.0) < 0.01
    e = e24.propensity(Z)
    assert e.min() >= e24.E_RANGE[0] and e.max() <= e24.E_RANGE[1]
    assert abs(np.mean(Y - e24.gamma0(D, Z))) < 0.01


def test_poly_dictionary_order_and_size() -> None:
    Z = np.arange(12.0).reshape(2, 6) / 10
    P = e24.poly2(Z)
    assert P.shape == (2, 27)
    assert np.allclose(P[:, :6], Z)
    assert np.allclose(P[:, 6], Z[:, 0] ** 2) and np.allclose(P[:, 7], Z[:, 0] * Z[:, 1])
    assert np.allclose(P[:, -1], Z[:, 5] ** 2)
    X = np.column_stack([[1.0, 0.0], Z])
    b = e24.cns_dictionary(X)
    assert b.shape == (2, 56)
    assert np.allclose(b[:, :2], [[1, 1], [1, 0]])
    assert np.allclose(b[1, 29:], 0.0) and np.allclose(b[0, 29:], P[0])


def test_nystrom_reproduces_the_kernel_on_the_centres() -> None:
    rng = np.random.default_rng(1)
    Z = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(400, 6))
    nys = e24.Nystrom(Z, np.random.default_rng(2))
    F = nys(nys.centres * nys.sd + nys.mean)
    Kcc = nys._kernel(nys.centres, nys.centres)
    assert np.allclose(F @ F.T, Kcc, atol=1e-6)
    assert nys.base(Z[:3]).shape == (3, nys.dim + 1)
    # median of the pairwise distances, exactly
    from scipy.spatial.distance import pdist

    S = (Z - nys.mean) / nys.sd
    assert nys.bandwidth == pytest.approx(np.median(pdist(S)), rel=1e-12)


def test_cns_coordinate_descent_solves_the_weighted_lasso() -> None:
    rng = np.random.default_rng(3)
    A = rng.standard_normal((50, 5))
    G = A.T @ A / 50
    M = rng.standard_normal(5)
    w = np.array([0.0, 0.1, 0.3, 0.05, 1.0])
    rho, sweeps = e24._cns_coordinate_descent(G, M, w, np.zeros(5))
    assert sweeps < e24.CNS_INNER_SWEEPS
    # KKT of rho'G rho - 2 rho'M + sum w|rho|: 2(G rho - M) + w s = 0, s in subdiff
    g = 2 * (G @ rho - M)
    for j in range(5):
        if rho[j] != 0:
            assert g[j] + w[j] * np.sign(rho[j]) == pytest.approx(0, abs=1e-8)
        else:
            assert abs(g[j]) <= w[j] + 1e-8


def test_autodml_initial_point_is_the_two_group_ipw() -> None:
    rng = np.random.default_rng(4)
    D, Z, _ = e24.draw(rng, 500)
    X = np.column_stack([D, Z])
    b = e24.cns_dictionary(X)
    G = b.T @ b / 500
    mb = e24.cns_dictionary(np.column_stack([np.ones(500), Z])) - e24.cns_dictionary(
        np.column_stack([np.zeros(500), Z])
    )
    rho_low = np.linalg.solve(G[:2, :2], mb.mean(0)[:2])
    a = b[:, :2] @ rho_low
    pbar = D.mean()
    assert np.allclose(a, (D - pbar) / (pbar * (1 - pbar)))


def test_ridge_leaves_the_arm_intercepts_unpenalised() -> None:
    rng = np.random.default_rng(5)
    D, Z, Y = e24.draw(rng, 300)
    X = np.column_stack([D, Z])
    base = lambda Z_: np.column_stack([np.ones(len(np.atleast_2d(Z_))), np.atleast_2d(Z_)[:, 0]])  # noqa: E731
    r = e24.Ridge(base, 1e6, X, Y)
    # a huge penalty leaves the arm means
    for d in (0, 1):
        assert r(X[D == d][:1])[0] == pytest.approx(Y[D == d].mean(), abs=1e-3)


def test_scores_arw_and_tmle() -> None:
    rng = np.random.default_rng(6)
    n = 50
    D = (rng.uniform(size=n) < 0.5).astype(float)
    Y = rng.standard_normal(n)
    a1, a0 = np.full(n, 2.0), np.full(n, -2.0)
    a = np.where(D == 1, a1, a0)
    g1, g0 = rng.standard_normal(n), rng.standard_normal(n)
    g = np.where(D == 1, g1, g0)
    (th, se), (tt, st) = e24._scores(D, Y, a, a1, a0, g, g1, g0)
    psi = g1 - g0 + a * (Y - g)
    assert th == pytest.approx(psi.mean()) and se == pytest.approx(psi.std() / np.sqrt(n))
    eps = np.sum(a * (Y - g)) / np.sum(a**2)
    assert tt == pytest.approx(np.mean(g1 - g0 + eps * (a1 - a0)))
    # after the fluctuation, the score equation holds: mean alpha (Y - gamma*) = 0
    assert np.sum(a * (Y - g - eps * a)) == pytest.approx(0, abs=1e-10)


def _stub_tuning(n=1000):
    rng = np.random.default_rng(7)
    _, Z, _ = e24.draw(rng, 400)
    nys = e24.Nystrom(Z, rng)
    return json.dumps(
        {
            "n": n, "nystrom_dim": nys.dim, "bandwidth": nys.bandwidth,
            "lambda": {a: 1e-2 for a in e24.GRR_ARMS}, "weight_decay": 1e-3, "logit_C": 1.0,
            "outcome_lambda": 1e-3,
            "gbm": {"propensity": [0.1, 7, 100], "outcome": [0.1, 7, 100], "seed": 1},
            "nn_seed": 1, "warnings": {},
            "nystrom": {"mean": nys.mean.tolist(), "sd": nys.sd.tolist(),
                        "centres": nys.centres.tolist(), "bandwidth": nys.bandwidth},
        }
    )  # fmt: skip


@pytest.fixture(scope="module")
def two_replications():
    """Two runs of the same task in spawned single-thread workers (this process keeps
    its thread pools untouched for the other tests)."""
    from grrexp import parallel

    tj = _stub_tuning()
    task = (0, 0, PILOT_ENTROPY, tj)
    return parallel.run_tasks(e24.replicate, [task, task], 2)


def test_replication_returns_every_arm_and_is_deterministic(two_replications) -> None:
    res, again = two_replications
    assert set(res) == set(e24.ARM_LABELS)
    for lab, r in res.items():
        assert r["status"] in outputs.recorded_statuses(), (lab, r["status"])
        assert "fold_status" in r and "warnings" in r
    ok = [lab for lab, r in res.items() if r["status"] == "ok"]
    assert len(ok) >= 12
    for lab in ok:
        assert abs(res[lab]["estimate"] - 1.0) < 1.0 and res[lab]["se"] > 0
        assert again[lab]["estimate"] == res[lab]["estimate"]


def test_registered_comparison_count() -> None:
    comps = e24.comparisons()
    assert sum(c[0] == "H24-Base" for c in comps) == 80
    assert sum(c[0] == "H24-GRR" for c in comps) == 24


def _fake_raw(fail_arm=None, fail_share=0.0, reps=40):
    rng = np.random.default_rng(8)
    rows = []
    for ci, (n,) in enumerate(e24.CELLS):
        for rep in range(reps):
            for lab in e24.ARM_LABELS:
                arm = lab.split("|")[0]
                bad = arm == fail_arm and rep < fail_share * reps
                scale = 0.05 if arm != "SQ" else 0.5
                draw = 1.0 + scale * rng.standard_normal()  # drawn for failures too
                rows.append(
                    {
                        "cell": ci, "n": n, "rep": rep, "arm": lab,
                        "estimate": np.nan if bad else draw,
                        "se": np.nan if bad else scale, "max_abs_alpha": np.nan if bad else 5.0,
                        "ess": np.nan if bad else 100.0, "n_warnings": 0.0,
                        "status": "domain_prediction" if bad else "ok",
                        "warnings": json.dumps({"fit": {}, "outcome": {}}), "fold_status": "[]",
                        "info": json.dumps({"outer_iterations": 3, "outer_cap_reached": 0,
                                            "inner_sweeps_max": 5}) if arm == "AutoDML" else "{}",
                    }
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_comparisons_detect_a_large_difference_and_skip_unevaluable(monkeypatch) -> None:
    import grrexp.bootstrap as bs

    monkeypatch.setitem(bs.REGISTERED_B, 7, 2000)
    raw = _fake_raw(fail_arm="EB", fail_share=0.1)
    comps = e24.compare(raw, Seeds(24, entropy=PILOT_ENTROPY))
    by = {(c["family"], c["n"], c["A"], c["B"]): c for c in comps}
    assert by[("H24-Base", 1000, "SQ", "RieszNet")]["rejected"]  # SQ has 10x the error
    eb = by[("H24-Base", 1000, "SQ", "EB")]
    assert not eb["evaluable"] and eb["p"] == 1.0 and not eb["rejected"]
    assert not by[("H24-Base", 1000, "UKL1", "RieszNet")]["rejected"]
    fam = e24.family_sentences(comps)
    assert fam["H24-Base"]["tests"] == 80 and fam["H24-Base"]["evaluable"] == 64


def test_unevaluable_comparisons_consume_no_draws(monkeypatch) -> None:
    import grrexp.bootstrap as bs

    monkeypatch.setitem(bs.REGISTERED_B, 7, 200)
    a = e24.compare(_fake_raw(fail_arm="EB", fail_share=0.1), Seeds(24, entropy=PILOT_ENTROPY))
    b = e24.compare(_fake_raw(fail_arm="EB", fail_share=0.5), Seeds(24, entropy=PILOT_ENTROPY))
    for x, y in zip(a, b, strict=True):
        if x["B"] != "EB" and x["A"] != "EB":
            assert x["p"] == y["p"]


def test_summary_tables_macros_and_figure(monkeypatch) -> None:
    import grrexp.bootstrap as bs
    import matplotlib.pyplot as plt

    monkeypatch.setitem(bs.REGISTERED_B, 7, 200)
    raw = _fake_raw()
    S = e24.summarise(raw)
    assert len(S) == len(e24.CELLS) * len(e24.ARM_LABELS)
    assert {"rmse", "coverage", "coverage_close_to_nominal", "failure_rate"} <= set(S.columns)
    comps = e24.compare(raw, Seeds(24, entropy=PILOT_ENTROPY))
    tabs = e24.tables(S, comps)
    assert set(tabs) == {"tab_E24", "tab_E24_comparisons", "tab_E24_autodml"}
    for name in ("tab_E24", "tab_E24_comparisons"):  # the long tables; the stopping one is short
        assert "\\endfirsthead" in tabs[name] and "\\endhead" in tabs[name]
    assert tabs["tab_E24"].count(" \\\\") == 2 + len(e24.ARM_LABELS) * len(e24.N_VALUES)
    m = e24.macros(S, comps)
    assert all(outputs.MACRO_NAME.match(k) for k in m)
    fig = e24.figure(S)
    plt.close(fig)
    assert "Nystr" in e24.tuning_table([_stub_tuning()])


def test_registered_outputs_and_runs() -> None:
    assert outputs.REGISTERED_RUNS[24]["main"] == (1000, [1000, 2000, 4000, 8000])
    assert set(outputs.REGISTERED_OUTPUTS[24]) == {
        "tables/tab_E24.tex",
        "tables/tab_E24_autodml.tex",
        "figures/fig_E24_rmse.pdf",
        "macros_E-24.tex",
    }


def _replicate_with_warning_outcome(task):
    """Worker helper: the GBM outcome fit issues a ConvergenceWarning."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    class Warned(e24.GBMOutcome):
        def __init__(self, *a, **k):
            warnings.warn("injected", ConvergenceWarning, stacklevel=1)
            super().__init__(*a, **k)

    e24.GBMOutcome = Warned
    return e24.replicate(task)


def test_an_outcome_convergence_warning_fails_the_arms_that_use_it() -> None:
    from grrexp import parallel

    task = (0, 0, PILOT_ENTROPY, _stub_tuning())
    (res,) = parallel.run_tasks(_replicate_with_warning_outcome, [task], 2)
    for e in e24.ESTIMATORS:
        assert res[f"DML-GBM|{e}"]["status"] == "convergence_warning"
        assert "outcome convergence_warning" in res[f"DML-GBM|{e}"]["fold_status"]
    assert res["SQ|ARW_cf"]["status"] == "ok"


def test_cv_select_excludes_values_with_any_failed_fold_and_records_it() -> None:
    scores = {
        1e-4: (1.0, [[2, "warning", {"RuntimeWarning: x": 1}]]),
        1e-3: (2.0, []),
        1e-2: (2.0, []),
        1e-1: (None, [[0, "domain_prediction", {}]]),
        1.0: (3.0, []),
    }
    choice, rec = e24._cv_select(list(scores), lambda v: scores[v], "larger")
    assert choice == 1e-2  # 1e-4 is not eligible; 1e-3 and 1e-2 tie, the larger wins
    assert rec[repr(1e-4)]["failures"][0][1] == "warning"
    assert e24._cv_failures(rec) == 2
    choice, _ = e24._cv_select([1e-3, 1e-2], lambda v: scores[v], "smaller")
    assert choice == 1e-3
    none, _ = e24._cv_select([1e-1], lambda v: scores[v], "larger")
    assert none is None


def test_per_cell_aggregation_compares_only_the_present_cell(monkeypatch) -> None:
    import grrexp.bootstrap as bs

    monkeypatch.setitem(bs.REGISTERED_B, 7, 200)
    raw = _fake_raw(reps=12)
    one = raw[raw["n"] == 2000]
    comps = e24.compare(one, Seeds(24, entropy=PILOT_ENTROPY))
    assert {c["n"] for c in comps} == {2000} and len(comps) == 20 + 6
    S = e24.summarise(one)
    assert set(S["n"]) == {2000} and "root_n_sd" in S



def test_autodml_stopping_table_counts_folds_once():
    S = pd.DataFrame([
        {"arm": "AutoDML|ARW_cf", "n": 1000, "R": 1000, "cns_outer_cap_folds": 4997.0,
         "cns_inner_sweeps_max": 77.0},
        {"arm": "AutoDML|TMLE_cf", "n": 1000, "R": 1000, "cns_outer_cap_folds": 4997.0,
         "cns_inner_sweeps_max": 77.0},
        {"arm": "SQ|ARW_cf", "n": 1000, "R": 1000, "cns_outer_cap_folds": float("nan"),
         "cns_inner_sweeps_max": float("nan")},
    ])
    rows = [ln for ln in e24.autodml_table(S).splitlines() if ln[:1].isdigit()]
    assert rows == ["1000 & 4997 of 5000 & 77 \\\\"]
