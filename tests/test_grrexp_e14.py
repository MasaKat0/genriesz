"""E-14 support code: the planted ATT design, Stage 0 checks, the replication record,
the evaluation bound, H14-Bnd, H14-Rate, the health checks and the table."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import e14  # noqa: E402
from grrexp.seeds import PILOT_ENTROPY, Seeds  # noqa: E402


@pytest.fixture(scope="module")
def stage0():
    return e14.stage0()


@pytest.fixture(scope="module")
def designs(stage0):
    return e14.designs_from_stage0(stage0)


# ---------------------------------------------------------------- design and Stage 0


@pytest.mark.parametrize("name", e14.GENERATORS)
def test_frozen_design_is_reproduced(name) -> None:
    vals = e14.design_values(name)
    checks = e14.frozen_checks(name, vals)
    assert all(checks.values()), [k for k, v in checks.items() if not v]


@pytest.mark.parametrize("name", e14.GENERATORS)
def test_planted_dual_coordinate_is_the_generator_link(name) -> None:
    """``a0 = -(g')^{-1}(u0)`` and ``g'(-a0) = u0`` on the control branch."""
    u = np.linspace(-0.5, 0.5, 7) + e14.PLANT[name]["c"]
    a = e14.control_weight(name, u)
    assert np.all(a > e14.SHIFT[name])
    assert np.allclose(e14.dual(name, 0, -a), u, rtol=0, atol=1e-12)


def test_stage0_rows_are_certified(stage0) -> None:
    assert stage0["bkl_zero_offset_infeasible"]
    for r in stage0["generators"]:
        assert r["frozen_ok"] and r["certified"], r["generator"]
        assert r["kappa"] == pytest.approx(math.sqrt(e14.CONST[r["generator"]]["lam_min_Sigma"]))
        json.dumps(r, allow_nan=False)


def test_population_gradient_vanishes_identically() -> None:
    """At the planted beta*, (1 - e) a0 = e / pi1, so every population imbalance is 0."""
    for name in e14.GENERATORS:
        res, _ = e14.certificate(name)
        assert res["gradient_max_48"] <= e14.CERT_GRADIENT
        assert res["gradient_max_64"] <= e14.CERT_GRADIENT


def test_sigma_is_block_diagonal(designs) -> None:
    des = designs["UKL"]
    S = des.sigma(10)
    assert S.shape == (12, 12)
    assert S[0, 0] == des.pi1 and np.allclose(S[6:, 6:], (1 - des.pi1) * np.eye(6))
    assert np.all(S[0, 1:] == 0) and np.all(S[1:6, 6:] == 0)
    assert np.linalg.eigvalsh(S).min() == pytest.approx(
        e14.design_values("UKL")["lam_min_Sigma"], rel=1e-12
    )


# ---------------------------------------------------------------- sample objects


def _sample(designs, name="BP", n=3000, p_c=6, seed=1):
    des = designs[name]
    D, Z, a0 = e14.draw(np.random.default_rng(seed), n, p_c, des)
    return des, D, Z, a0


def test_target_is_the_att_functional(designs) -> None:
    des, D, Z, _ = _sample(designs)
    Phi = e14.design_matrix(D, Z)
    b = e14.target(D, Z, des.pi1)

    def m(col):  # m(W, f) = D (f(1, Z) - f(0, Z)) / pi1 for the dictionary function col
        f1 = e14.design_matrix(np.ones_like(D), Z)[:, col]
        f0 = e14.design_matrix(np.zeros_like(D), Z)[:, col]
        return np.mean(D * (f1 - f0)) / des.pi1

    assert np.allclose(b, [m(j) for j in range(Phi.shape[1])], rtol=0, atol=1e-14)


def test_gradient_at_beta_star_is_mean_xi(designs) -> None:
    import genriesz as gr

    des, D, Z, a0 = _sample(designs)
    Phi = e14.design_matrix(D, Z)
    b = e14.target(D, Z, des.pi1)
    gen = e14.generator(des.name)
    prob = gr.solvers.DualProblem(generator=gen, X=D[:, None], Phi=Phi,
                                  offset=np.where(D == 1, des.uref_t, des.uref_c), target=b)
    pt = prob.evaluate(des.beta_star(Z.shape[1]))
    alpha = np.where(D == 1, 1 / des.pi1, -a0)
    assert np.allclose(pt.alpha, alpha, rtol=1e-12, atol=1e-12)
    xi = e14.xi_matrix(alpha, Phi, D, Z, des.pi1)
    assert np.allclose(pt.grad, xi.mean(axis=0), rtol=0, atol=1e-12)


def test_penalties(designs) -> None:
    des, D, Z, a0 = _sample(designs, n=2000, p_c=4)
    n, p = len(D), e14.p_total(4)
    xi = e14.xi_matrix(np.where(D == 1, 1 / des.pi1, -a0), e14.design_matrix(D, Z), D, Z, des.pi1)
    from scipy import stats

    q = stats.norm.ppf(1 - 0.05 / (2 * p))
    assert e14.lam_sn(xi, n, p) == pytest.approx(2 * 1.1 * q * e14.s_hat(xi) / math.sqrt(n))
    B_xi = math.sqrt(3) * (des.A_alpha + des.B_m)
    assert e14.lam_theorem(des, n, p) == pytest.approx(
        2 * B_xi * math.sqrt(2 * math.log(2 * p / 0.05) / n)
    )
    assert e14.bound(des, 0.1) == pytest.approx(6 * des.kappa_g * math.sqrt(5) * 0.1 / des.kappa)


# ---------------------------------------------------------------- evaluation


def test_evaluation_columns_are_per_column_streams() -> None:
    a = e14._eval_columns(PILOT_ENTROPY, 3, 999, [0, 5], 100)
    b = e14._eval_columns(PILOT_ENTROPY, 3, 999, [5, 0, 7], 100)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[5], b[5])
    assert not np.array_equal(a[0], e14._eval_columns(PILOT_ENTROPY, 3, 998, [0], 100)[0])


@pytest.mark.parametrize("name", e14.GENERATORS)
def test_evaluation_at_beta_star_is_zero(designs, name) -> None:
    des = designs[name]
    ev = e14.evaluate(des, e14.generator(name), des.beta_star(9), PILOT_ENTROPY, 0, N=20000)
    assert ev["err2"] <= 1e-24
    assert ev["err2_lower"] <= ev["err2"]
    assert ev["eval_in_range"]


def test_bernstein_lower_bound_formula(designs) -> None:
    des = designs["SQ"]
    beta = des.beta_star(4)
    beta[2] += 0.1
    ev = e14.evaluate(des, e14.generator("SQ"), beta, PILOT_ENTROPY, 1, N=50000)
    lg = math.log(2 / e14.DELTA_EVAL)
    lower = ev["err2"] - math.sqrt(2 * ev["err2_var"] * lg / 50000) - 7 * des.B_eval**2 * lg / (
        3 * 49999
    )
    assert ev["err2_lower"] == pytest.approx(lower, rel=1e-12)
    assert ev["err2"] > 0


# ---------------------------------------------------------------- replication


@pytest.fixture(scope="module")
def small_task(stage0):
    ci = e14.CELLS.index(["BKL", 50, 4000])
    return e14.tasks_for(PILOT_ENTROPY, 1, stage0, cells=[ci])[0]


def test_replication_record_and_determinism(small_task) -> None:
    a = e14.replicate(small_task)
    b = e14.replicate(small_task)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    for pen in e14.PENALTIES:
        r = a[pen]
        assert r["status"] == "ok"
        assert r["events"] == (r["E1"] and r["E2"] and r["E3"])
        assert r["bnd_violation"] == (r["events"] and r["err2_lower"] > r["bound"] ** 2)
        assert r["prox_residual"] <= e14.FISTA_REL_TOL * r["lam"]
    assert a["BKL_zero_offset_status"] == "infeasible_start"
    assert a["n_warnings"] == 0


def test_raw_frame_and_aggregation(small_task, stage0) -> None:
    ci = small_task[0]
    tasks = e14.tasks_for(PILOT_ENTROPY, 3, stage0, cells=[ci])
    res = [e14.replicate(t) for t in tasks]
    raw = e14.raw_frame(tasks, res)
    assert len(raw) == 3 * (len(e14.PENALTIES) + 1)
    S = e14.summarise(raw)
    assert list(S["penalty"]) == list(e14.PENALTIES)
    assert (S["R"] == 3).all()
    bnd = e14.h14_bnd(S)
    assert bnd["checked"] == int(S["bnd_checked"].sum())
    hl = e14.health(raw)
    z = hl["bkl_zero_offset"]
    got = (z["verdict"], z["expected"], z["replications"], z["infeasible_start"])
    assert got == ("pass", 3, 3, 3)
    assert "CP up." in e14.tables(S, None)["tab_E14"]
    tabs = e14.tables(S, None)
    assert "\\endhead" in tabs["tab_E14"] and tabs["tab_E14"].count("\\endfirsthead") == 1


def test_bnd_violation_is_counted() -> None:
    S = pd.DataFrame({"generator": ["SQ", "UKL"], "p_c": [50, 50], "n": [4000, 4000],
                      "penalty": ["SN", "SN"], "bnd_violations": [0, 2], "bnd_checked": [5, 5]})
    out = e14.h14_bnd(S)
    assert out["verdict"] == "negative" and out["violations"] == 2
    assert out["violating_cells"] == ["UKL|p_c=50|n=4000|SN"]


def test_rate_slope_recovers_a_known_power() -> None:
    rows = []
    for ci, (_name, p_c, n) in enumerate(e14.CELLS):
        for rep in range(20):
            err = (n / 16000) ** -0.5 * (1.0 + 0.01 * ((rep % 5) - 2)) * (1 + p_c / 8000)
            rows.append({"cell": ci, "penalty": "SN", "status": "ok", "rep": rep, "err2": err**2})
    out = e14.rate_predictions(pd.DataFrame(rows), Seeds(14, entropy=PILOT_ENTROPY))
    assert len(out["slopes"]) == len(e14.GENERATORS) * len(e14.P_C_VALUES)
    for s in out["slopes"]:
        assert s["slope"] == pytest.approx(-0.5, abs=1e-12)
        assert s["lo"] <= -0.5 <= s["hi"] and s["intersects"]
    for r in out["ratios"]:
        assert r["ratio"] == pytest.approx(1.1 / 1.00625, rel=1e-12) and r["below_limit"]


# ---------------------------------------------------------------- review round 1


def test_nonfinite_evaluation_fails_the_fit(designs) -> None:
    import genriesz as gr

    des = designs["UKL"]
    beta = des.beta_star(4)
    beta[1] = -1000.0  # a = exp(-u) overflows on the evaluation sample
    ev = e14.evaluate(des, e14.generator("UKL"), beta, PILOT_ENTROPY, 0, N=1000)
    assert ev == {"eval_status": "nonfinite"}
    res = gr.solvers.SolverResult(beta=beta, status="ok", n_iter=1)
    events = {"lam_min_Sigma_hat": 1.0, "lam_min_E3": 1.0, "E2": True, "E3": True}
    rec = e14._fit_record(des, e14.generator("UKL"), res, 0.1, events, PILOT_ENTROPY, 0, 0.0)
    assert rec["status"] == "nonfinite"
    assert "bnd_violation" not in rec and "err2_lower" not in rec


def test_evaluation_outside_the_certified_range_stops(designs) -> None:
    des = designs["SQ"]
    beta = des.beta_star(4)
    beta[1] = -50.0  # far outside the l1 ball: |alpha| above A_alpha
    with pytest.raises(RuntimeError, match="certified range"):
        e14.evaluate(des, e14.generator("SQ"), beta, PILOT_ENTROPY, 0, N=1000)


def _bkl_raw(statuses):
    rows = []
    for rep, st in enumerate(statuses):
        rows.append({"cell": 45, "rep": rep, "generator": "BKL", "penalty": "SN", "status": "ok"})
        if st is not None:
            rows.append({"cell": 45, "rep": rep, "generator": "BKL", "penalty": "BKL_u0",
                         "status": st})
    return pd.DataFrame(rows)


def test_bkl_zero_offset_verdict() -> None:
    assert e14.bkl_zero_offset(_bkl_raw(["infeasible_start"] * 3))["verdict"] == "pass"
    out = e14.bkl_zero_offset(_bkl_raw(["infeasible_start", "ok", "infeasible_start"]))
    assert out["verdict"] == "fail" and out["mismatches"] == [(45, 1, "ok")]
    out = e14.bkl_zero_offset(_bkl_raw(["infeasible_start", None]))
    assert out["verdict"] == "incomplete" and out["missing"] == [[45, 1]]


def test_rate_verdict_negative_and_incomplete() -> None:
    rows = []
    for ci, (_name, p_c, n) in enumerate(e14.CELLS):
        for rep in range(20):
            err = (n / 16000) ** -1.0 * (1.0 + 0.01 * ((rep % 5) - 2)) * (1 + p_c / 50)
            rows.append({"cell": ci, "penalty": "SN", "status": "ok", "rep": rep, "err2": err**2})
    out = e14.rate_predictions(pd.DataFrame(rows), Seeds(14, entropy=PILOT_ENTROPY))
    assert out["verdict"] == "negative" and not out["incomplete"]
    assert all(f.startswith(("slope|", "ratio|")) for f in out["failed"])
    one = pd.DataFrame([r for r in rows if r["rep"] == 0])
    out = e14.rate_predictions(one, Seeds(14, entropy=PILOT_ENTROPY))
    assert out["verdict"] == "incomplete"


def test_parent_state_override(monkeypatch) -> None:
    """A genriesz worktree outside the parent names the parent with GRREXP_PARENT_REPO."""
    from grrexp import env

    calls = []

    def fake_git(repo, *args, **kw):
        calls.append((str(repo), args))
        if args[:1] == ("rev-parse",):
            return "a" * 40 + "\n"
        if args[:1] == ("ls-tree",):
            return f"160000 commit {'b' * 40}\t{args[2]}\n"
        return ""

    monkeypatch.setattr(env, "_git", fake_git)
    monkeypatch.setenv(env.PARENT_ENV, "/some/parent")
    st = env.parent_state()
    assert st["sha"] == "a" * 40 and st["gitlink"] == "b" * 40
    assert st["genriesz_worktree"] == str(env.GENRIESZ_ROOT)
    assert calls[1] == ("/some/parent", ("ls-tree", "HEAD", "genriesz"))
    monkeypatch.delenv(env.PARENT_ENV)
    st = env.parent_state()
    assert "genriesz_worktree" not in st
    assert calls[-2][1] == ("ls-tree", "HEAD", env.GENRIESZ_ROOT.name)


def test_rate_table_reports_slopes_ratios_and_health():
    import json

    import pandas as pd

    rate = {"slopes": [{"generator": "BKL", "p_c": 50, "slope": -0.27, "lo": -0.28, "hi": -0.26,
                        "intersects": False}],
            "ratios": [{"generator": "SQ", "ratio": 1.17, "lo": 1.16, "hi": 1.18,
                        "below_limit": True}]}
    health = {"failures": {"SQ|SN": [0, 3000]},
              "bkl_zero_offset": {"infeasible_start": 3000, "expected": 3000}}
    S = pd.DataFrame([{
        "generator": "SQ", "p_c": 50, "n": 4000, "penalty": "SN", "failure_rate": 0.0,
        "failure_rate_cp_upper": 0.0149, "E1_rate": 1.0, "events_rate": 0.0, "bnd_checked": 0,
        "lam_median": 0.25, "bound_median": 6.2, "err_mean": 0.44, "support_median": 3.0,
        "max_err_over_bound": float("nan"), "bnd_violations": 0,
        "h14_rate": json.dumps(rate), "health": json.dumps(health),
    }])
    out = e14.tables(S, None)
    assert "Slope & BKL & 50 & -0.270 & [-0.280, -0.260] & no" in out["tab_E14_rate"]
    assert "Ratio 800/50 & SQ & -- & 1.170" in out["tab_E14_rate"]
    assert "3000 of 3000" in out["tab_E14_rate"]
    row = out["tab_E14"].splitlines()[-3]
    assert row.rstrip(" \\").endswith("-- & --")  # nothing checked: no ratio, no count
    S2 = S.drop(columns=["h14_rate", "health"])
    assert set(e14.tables(S2, None)) == {"tab_E14"}
