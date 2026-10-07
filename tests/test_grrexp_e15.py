"""E-15 support code: Part A, the lambda rules and their KKT conditions, the Part B
identity, AutoDML-lasso, Stage 0, replication records, aggregation, families,
tables and figure."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import e12, e15  # noqa: E402
from grrexp.seeds import PILOT_ENTROPY, Seeds  # noqa: E402

import genriesz as gr  # noqa: E402


def _sample(n=400, seed=5):
    rng = np.random.default_rng(seed)
    D, Z, Y = e15.draw(rng, n)
    return np.column_stack([D, Z]), Y


def test_rho_reproduces_gamma0_on_the_interaction_basis() -> None:
    X, _ = _sample(50)
    bas = e15.basis()
    bas.fit(X)
    Phi = np.asarray(bas(X), dtype=float)
    assert Phi.shape[1] == e15.P == 52
    assert np.allclose(Phi @ e15.RHO, e15.gamma0(X[:, 0], X[:, 1:]), atol=1e-12)


def test_lambda_scale_and_s_hat() -> None:
    assert math.isclose(e15.lam_scale(1000), math.sqrt(2 * math.log(2 * 52 / 0.05) / 1000))
    a = np.array([1.0, -2.0, 3.0])
    Phi = np.array([[1.0, 0.5], [1.0, -1.0], [0.0, 2.0]])
    M = np.array([[0.5, 0.0], [0.0, 1.0], [1.0, 1.0]])
    xi = a[:, None] * Phi - M
    assert math.isclose(e15.s_hat(a, Phi, M), float(np.sqrt(np.max(np.mean(xi**2, axis=0)))))


def test_part_a_symbolic_and_float_agree() -> None:
    rows, ok = e15.part_a()
    assert ok and len(rows) == len(e15.PART_A_N) * len(e15.RULES)
    by = {(r["n"], r["rule"]): r for r in rows}
    # lambda = 0: the nominal coverage Phi(1.96) - Phi(-1.96)
    assert math.isclose(by[(100, "L_0")]["coverage"], 0.9500042097035591, rel_tol=1e-12)
    assert by[(10**6, "L_th")]["tau"] > by[(10**6, "L_us")]["tau"] > 0
    assert all(r["abs_diff"] <= 1e-12 for r in rows)


def test_fit_rules_satisfies_the_l1_kkt_conditions() -> None:
    X, _ = _sample(600)
    for gen in e15.GENERATORS:
        warn = e15._Warn()
        fits = e15.fit_rules(gen, X, warn, "t")
        assert all(fits[r][0] == "ok" for r in e15.RULES)
        bas = e15.basis()
        bas.fit(X)
        Phi = np.asarray(bas(X), dtype=float)
        M = np.asarray(gr.ATEFunctional(0).m_basis_matrix(X, bas), dtype=float)
        for rule in ("L_th", "L_us"):
            _, mdl, lam = fits[rule]
            a = np.asarray(mdl.predict_alpha(X), dtype=float)
            delta = np.mean(a[:, None] * Phi - M, axis=0)
            beta = mdl.beta_
            active = beta != 0
            # 0 in delta + lam subdiff|beta|, to the solver tolerance
            assert np.all(np.abs(delta[active] + lam * np.sign(beta[active])) <= 1e-5 * lam)
            assert np.all(np.abs(delta[~active]) <= lam * (1 + 1e-5))
        # the two-step rule
        m1 = e15.model(gen, 2.0 * e15.s_hat(e15.alpha_ref(X), Phi, M) * e15.lam_scale(len(X)))
        m1.fit(X)
        s1 = e15.s_hat(np.asarray(m1.predict_alpha(X), dtype=float), Phi, M)
        assert math.isclose(fits["L_th"][2], 2 * s1 * e15.lam_scale(len(X)), rel_tol=1e-6)
        assert math.isclose(fits["L_us"][2], s1 * len(X) ** -0.5 / math.log(len(X)), rel_tol=1e-6)
        assert fits["L_0"][2] == 0.0


def test_rw_record_satisfies_the_part_b_identity() -> None:
    X, Y = _sample(500)
    out = e15._rw_full(X, Y)
    for label in [a for a in e15.GRR_ARMS if a.endswith("RW_full")]:
        r = out[label]
        assert r["status"] == "ok" and r["anem_gap"] <= e15.ANEM_TOL
        assert r["se"] > 0 and np.isfinite(r["estimate"])


def test_coordinate_descent_solves_the_weighted_lasso() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 6))
    G = A.T @ A / 200
    Mh = rng.normal(size=6)
    thr = np.full(6, 0.3)
    rho, _, converged = e15.coordinate_descent(G, Mh, thr, np.zeros(6))
    assert converged
    grad = G @ rho - Mh  # half the gradient of the smooth part
    act = rho != 0
    assert np.allclose(grad[act], -thr[act] * np.sign(rho[act]), atol=1e-8)
    assert np.all(np.abs(grad[~act]) <= thr[~act] + 1e-8)


def test_autodml_dictionary_and_initial_representer() -> None:
    X, _ = _sample(300)
    B, Mb = e15.adml_dictionary(X)
    assert B.shape == (300, 52) and np.allclose(B[:, 0], 1) and np.allclose(B[:, 1], X[:, 0])
    assert np.allclose(Mb[:, 1], 1) and np.allclose(Mb[:, 27:], X[:, 1:]) and not np.any(Mb[:, [0]])
    # the initial representer on (1, D) is the two-group constant IPW (registration §1.4)
    G = B.T @ B / len(B)
    rho_low = np.linalg.solve(G[:2, :2], Mb.mean(axis=0)[:2])
    d = X[:, 0]
    ipw = (d - d.mean()) / (d.mean() * (1 - d.mean()))
    assert np.allclose(B[:, :2] @ rho_low, ipw, atol=1e-10)
    status, rho, info = e15.adml_fit(X)
    assert status == "ok" and rho.shape == (52,) and 1 <= info["outer"] <= e15.ADML_OUTER


def test_stage0_certificates_and_cancellation() -> None:
    s0 = e15.stage0()
    assert s0["part_a_ok"]
    for g in s0["generators"]:
        assert g["certified"], g["checks"]
        rho6 = np.array([1.0, 1.0, 0.5, 0.0, 1.0, 0.5])
        assert math.isclose(g["rho_s"], float(rho6 @ np.array(g["signs"])))
        assert g["cancellation"] == (abs(g["rho_s"]) < e15.CANCELLATION)
        assert (g["tau_pred"] is None) == g["cancellation"]
        assert max(abs(v) for v in g["irrelevant_beta_4d"]) <= e15.IRRELEVANT_TOL
    json.dumps(s0, allow_nan=False)


def test_replicate_is_deterministic_and_records_every_arm() -> None:
    a = e15.replicate((0, 3, PILOT_ENTROPY))
    b = e15.replicate((0, 3, PILOT_ENTROPY))
    assert set(a) == set(e15.ARM_LABELS)
    for k in a:
        assert a[k]["status"] == b[k]["status"]
        assert a[k]["estimate"] == b[k]["estimate"] or (
            np.isnan(a[k]["estimate"]) and np.isnan(b[k]["estimate"])
        )


@pytest.fixture(scope="module")
def small_run():
    tasks = e15.tasks_for(PILOT_ENTROPY, 3)
    res = [e15.replicate(t) for t in tasks]
    return e15.raw_frame(tasks, res), e15.stage0()


def test_aggregation_families_tables_and_figure(small_run) -> None:
    raw, s0 = small_run
    assert len(raw) == len(e15.CELLS) * 3 * len(e15.ARM_LABELS)
    anem = e15.anem_check(raw)
    assert anem["max_gap"] <= e15.ANEM_TOL
    summ = e15.summarise(raw, s0)
    summ = summ.assign(**e12.family1_mcse(raw, summ, Seeds(15, entropy=PILOT_ENTROPY)))
    verdicts, tests, _ = e15.families(summ, raw, Seeds(15, entropy=PILOT_ENTROPY))
    assert len(tests["H15-Suf"]) == len(e15.suf_arms()) * len(e15.SUF_N)
    cancel = all(g["cancellation"] for g in s0["generators"])
    assert (verdicts["H15-Bias"] is None) == cancel
    tabs = e15.tables(summ, s0["part_a"])
    assert set(tabs) == {"tab_E15_exact", "tab_E15_mc", "tab_E15_status"}
    assert "\\endfirsthead" in tabs["tab_E15_mc"] and "\\endhead" in tabs["tab_E15_mc"]
    assert "-0.000" not in tabs["tab_E15_mc"]
    import matplotlib.pyplot as plt

    plt.close(e15.figure(summ))


def test_anem_check_stops_on_a_violation(small_run) -> None:
    raw, _ = small_run
    bad = raw.copy()
    i = bad.index[bad["arm"].str.endswith("RW_full") & (bad["status"] == "ok")][0]
    bad.loc[i, "anem_gap"] = 1e-6
    with pytest.raises(AssertionError):
        e15.anem_check(bad)


def test_total_warnings_counts_each_shared_group_once() -> None:
    raw = pd.DataFrame(
        {
            "cell": [0] * 4, "rep": [0] * 4,
            "arm": ["SQ-L_th|RW_full", "SQ-L_us|RW_full", "SQ-L_0|RW_full", "AutoDML|ARW_cf"],
            "n_warnings": [2.0, 2.0, 2.0, 1.0],
        }
    )  # fmt: skip
    assert e15.total_warnings(raw) == 3


def test_fmt_has_no_negative_zero() -> None:
    assert e15._fmt(-0.0001) == "0.000" and e15._fmt(-0.01, 2) == "-0.01"
