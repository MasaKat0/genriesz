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
    assert sweeps > 0
    # KKT of rho'G rho - 2 rho'M + sum w|rho|: 2(G rho - M) + w s = 0, s in subdiff
    g = 2 * (G @ rho - M)
    for j in range(5):
        if rho[j] != 0:
            assert g[j] + w[j] * np.sign(rho[j]) == pytest.approx(0, abs=1e-4)
        else:
            assert abs(g[j]) <= w[j] + 1e-4


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
    assert set(tabs) == {"tab_E24", "tab_E24_comparisons", "tab_E24_autodml",
                         "tab_E24_matched", "tab_E24_matched_comparisons"}
    for name in ("tab_E24", "tab_E24_comparisons"):  # the long tables; the stopping one is short
        assert "\\endfirsthead" in tabs[name] and "\\endhead" in tabs[name]
    assert tabs["tab_E24"].count(" \\\\") == (
        2 + len(e24.EXISTING_ARMS) * len(e24.ESTIMATORS) * len(e24.N_VALUES)
    )
    assert tabs["tab_E24_matched"].count(" \\\\") == (
        2 + len(e24.MATCHED_ARMS) * len(e24.ESTIMATORS) * len(e24.N_VALUES)
    )
    assert "H24-PolyCV" not in tabs["tab_E24_comparisons"]
    assert "H24-Base" not in tabs["tab_E24_matched_comparisons"]
    m = e24.macros(S, comps)
    assert all(outputs.MACRO_NAME.match(k) for k in m)
    fig = e24.figure(S)
    plt.close(fig)
    assert "Nystr" in e24.tuning_table([_stub_tuning()])


def test_registered_outputs_and_runs() -> None:
    assert outputs.REGISTERED_RUNS[24]["main"] == (1000, [1000, 2000, 4000, 8000])
    assert set(outputs.REGISTERED_OUTPUTS[24]) == {
        "tables/tab_E24.tex",
        "tables/tab_E24_matched.tex",
        "tables/tab_E24_matched_comparisons.tex",
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
    assert {c["n"] for c in comps} == {2000} and len(comps) == 20 + 6 + 30
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


def test_matched_dictionary_offset_and_penalty():
    X = np.column_stack([[1., 0., 1., 0.], np.arange(24.).reshape(4, 6) / 10])
    B = e24.interaction(e24.poly_base, X)
    cns = e24.cns_dictionary(X)
    # CNS b=(1,D,P,DP) and the EB interaction dictionary span exactly the same
    # space: B=(D,DP,1-D,P-DP). Test the explicit transformation, not just rank.
    assert np.allclose(B, np.column_stack([cns[:, 1], cns[:, 29:],
                                         cns[:, 0] - cns[:, 1],
                                         cns[:, 2:29] - cns[:, 29:]]))
    for lam in (0.0, *e24.LAMBDA_GRID):
        for arm in e24.MATCHED_GENERATORS:
            mdl = e24.glm_model(arm, e24.poly_base, lam)
            mdl.basis.fit(X)
            assert np.array_equal(mdl.basis(X), B)
            from genriesz.glm import evaluate_offset

            u_ref = evaluate_offset(mdl.offset, X)
            assert np.allclose(mdl.generator.inv_grad(X, u_ref), e24.alpha_ref(X))
            # Every coordinate, including intercepts, receives lambda/2 beta^2.
            assert mdl.penalty.value(np.ones(B.shape[1])) == pytest.approx(lam * 28)
    eb = e24.fit_glm("EB", e24.poly_base, 0.0, X)
    uk = e24.fit_matched("UKL0_poly_lam0", None, X, None)
    assert eb.status == uk.status  # identical even when this tiny design fails


def test_matched_cv_uses_only_outer_training_and_common_splits(monkeypatch):
    X = np.column_stack([np.arange(25) % 2, np.arange(25), np.zeros((25, 5))])
    cv = np.arange(25) % e24.CV_FOLDS
    calls = []

    def fake_fit(gen, base, lam, Xt):
        fitted = set(Xt[:, 1])

        def fn(Xv, X1, X0):
            assert fitted.isdisjoint(Xv[:, 1])
            calls.append((gen, base, lam, fitted, set(Xv[:, 1])))
            # All candidates tie: the existing rule must select the largest.
            a = np.ones(len(Xv))
            return a, a, -a, "ok"

        return e24.Rep("ok", fn)

    monkeypatch.setattr(e24, "fit_glm", fake_fit)
    for arm in (a for a in e24.MATCHED_ARMS if e24.MATCHED_SPECS[a][2]):
        rep = e24.fit_matched(arm, e24.poly_base, X, cv)
        assert rep.status == "ok" and rep.info["lambda"] == max(e24.LAMBDA_GRID)
        assert set(rep.info["cv_scores"]) == {repr(x) for x in e24.LAMBDA_GRID}
    assert len(calls) == 10 * len(e24.LAMBDA_GRID) * e24.CV_FOLDS
    expected = [(set(X[cv != k, 1]), set(X[cv == k, 1])) for k in range(e24.CV_FOLDS)]
    for start in range(0, len(calls), e24.CV_FOLDS):
        assert [(r[3], r[4]) for r in calls[start:start + e24.CV_FOLDS]] == expected


@pytest.mark.parametrize("failure", ["maxit", "warning", "nonfinite", "domain_prediction"])
def test_matched_cv_records_failures_without_substitution(monkeypatch, failure):
    import warnings

    X = np.column_stack([np.arange(20) % 2, np.zeros((20, 6))])

    def failed_fit(*args):
        if failure == "warning":
            warnings.warn("inner CV warning", RuntimeWarning, stacklevel=1)
            return e24.Rep("ok", lambda *x: (np.ones(len(x[0])), np.ones(len(x[0])),
                                            -np.ones(len(x[0])), "ok"))
        return e24.Rep(failure)

    monkeypatch.setattr(e24, "fit_glm", failed_fit)
    rep = e24.fit_matched("SQ_poly_cv", None, X, np.arange(20) % e24.CV_FOLDS)
    assert rep.status == "cv_failed" and rep.info["lambda"] is None
    assert rep.info["cv_failures"] == len(e24.LAMBDA_GRID) * e24.CV_FOLDS
    assert all(r["score"] is None for r in rep.info["cv_scores"].values())
    assert all(f[1] == failure for r in rep.info["cv_scores"].values() for f in r["failures"])


def test_matched_comparisons_hold_dictionary_and_penalty_fixed():
    for fam, n, a, b in e24.comparisons():
        if fam in e24.MATCHED_GROUPS:
            assert n in e24.N_VALUES
            assert e24.MATCHED_SPECS[a][1:] == e24.MATCHED_SPECS[b][1:]
            assert e24.MATCHED_SPECS[a][0] != e24.MATCHED_SPECS[b][0]
    assert len(e24.MATCHED_ARMS) == 15
    assert all(sum(c[0] == fam for c in e24.comparisons()) == 40
               for fam in e24.MATCHED_GROUPS)


def test_matched_fold_tuning_is_saved_for_successes_and_failures(two_replications):
    (res, _) = two_replications
    for arm in e24.MATCHED_ARMS:
        folds = json.loads(res[f"{arm}|ARW_cf"]["info"])["folds"]
        if e24.MATCHED_SPECS[arm][2]:
            assert folds and all("lambda" in f and "cv_scores" in f for f in folds)
    nn = json.loads(res["RieszNet|ARW_cf"]["info"])["folds"]
    assert len(nn) == e24.K
    assert all(f["settings"]["lambda_targeted"] == 1 for f in nn)


@pytest.mark.parametrize("outcome_failure", [None, "convergence_warning", "outcome_nonfinite"])
def test_replication_uses_riesznet_regression_and_only_its_outcome_warnings(
    monkeypatch, outcome_failure
):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    D, Z, Y = e24.draw(Seeds(24, entropy=PILOT_ENTROPY).data(0, 0), 1000)
    X = np.column_stack([D, Z])
    outer = e24.fold_ids(1000, e24.K, Seeds(24, entropy=PILOT_ENTROPY).folds(0, 0))
    seen = []

    def fake_net(seed, Xt, Yt):
        k = len(seen)
        assert np.array_equal(Xt, X[outer != k])
        assert np.array_equal(Yt, Y[outer != k])
        seen.append(k)

        def at(Xv, X1, X0):
            return (np.where(Xv[:, 0] == 1, 2., -2.), np.full(len(Xv), 2.),
                    np.full(len(Xv), -2.), "ok")

        rep = e24.Rep("ok", at)

        def outcome(Xv):
            if outcome_failure == "convergence_warning":
                warnings.warn("network outcome", ConvergenceWarning, stacklevel=1)
            if outcome_failure == "outcome_nonfinite":
                return np.full(len(Xv), np.nan)
            return 5. + 2. * Xv[:, 0]

        rep.outcome = outcome
        return rep

    def unused_outcome(*args):
        warnings.warn("unused outcome", ConvergenceWarning, stacklevel=1)
        return lambda Xv: np.full(len(Xv), 1000.)

    monkeypatch.setattr(e24, "ARMS", ("RieszNet",))
    monkeypatch.setattr(e24, "fit_riesznet", fake_net)
    monkeypatch.setattr(e24, "Ridge", unused_outcome)
    monkeypatch.setattr(e24, "GBMOutcome", unused_outcome)
    res = e24.replicate((0, 0, PILOT_ENTROPY, _stub_tuning()))
    row = res["RieszNet|ARW_cf"]
    if outcome_failure is not None:
        assert row["status"] == outcome_failure and seen == [0]
        assert np.isnan(row["estimate"]) and np.isnan(row["se"])
        assert json.loads(row["warnings"])["outcome"] == {}
        return
    assert row["status"] == "ok" and seen == list(range(e24.K))
    psi = 2. + np.where(D == 1, 2., -2.) * (Y - 5. - 2. * D)
    assert row["estimate"] == pytest.approx(psi.mean())
    assert row["se"] == pytest.approx(psi.std(ddof=1) / np.sqrt(len(Y)))
    assert row["n_warnings"] == 0 and json.loads(row["warnings"])["outcome"] == {}
