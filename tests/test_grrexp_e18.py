"""E-18 (held-out squared-loss Riesz selection): candidates, bounds, selection and tests."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import e18, env, inference  # noqa: E402

g = env.use_submodule_genriesz()


def test_candidate_inventory():
    assert len(e18.CANDIDATES) == 64 and e18.M == 65 and e18.A0 == 64
    assert e18.CANDIDATES[0] == ("SQ", 1, 0.0) and e18.CONTROL == 0
    assert e18.CANDIDATES[-1] == ("BP0", 4, 0.1)
    assert e18.N_COLS == {0: 1, 1: 4, 2: 10, 3: 20, 4: 35}
    assert e18.CELLS[0] == ["18A", 1000] and e18.CELLS[-1] == ["18B", 8000]


def test_legendre_features_bounded_and_nested():
    rng = np.random.default_rng(0)
    Z = rng.uniform(-e18.SQRT3, e18.SQRT3, size=(5000, 3))
    P4 = e18.legendre_features(Z, 4)
    assert P4.shape == (5000, 35)
    assert np.all(np.abs(P4) <= 1.0 + 1e-12)
    np.testing.assert_allclose(P4[:, :10], e18.legendre_features(Z, 2))
    corners = np.array([[e18.SQRT3] * 3, [-e18.SQRT3] * 3])
    assert np.max(np.abs(e18.legendre_features(corners, 4))) == pytest.approx(1.0)


def test_radius_guarantee_and_constants():
    rc = e18.radius_check()
    for name, r in rc.items():
        assert r["ok"], name
        assert r["max_abs_alpha"] <= 50.0
    for d, b in e18.ETA_BOUND.items():
        exact = 2.0 * (1.0 + np.exp(b))
        assert exact <= e18.CM2[d] <= exact + 1e-6
    ukl = e18.ukl_18b_coefficients()
    assert ukl["l1"] == pytest.approx(e18.UKL_L1_18B, abs=1e-6)
    assert ukl["l1"] < e18.RADIUS["UKL1"] and ukl["max_residual"] < 1e-12


def test_closed_form_alpha_matches_genriesz():
    rng = np.random.default_rng(3)
    Z = rng.uniform(-e18.SQRT3, e18.SQRT3, size=(400, 3))
    P = e18.legendre_features(Z)
    for name in e18.GENERATORS:
        for deg in (1, 4):
            beta = rng.uniform(-1, 1, e18.N_COLS[deg])
            beta *= 0.99 * e18.RADIUS[name] / np.abs(beta).sum()
            f = e18._Fit(name, deg, {1.0: beta, 0.0: -beta})
            for d in (1.0, 0.0):
                a, out = f.alpha(d, P)
                X = np.column_stack([np.full(len(Z), d), Z])
                a2, o2, n2 = g.glm.classify_predictions(f.gen, X, f.u(d, P))
                np.testing.assert_allclose(a, a2, rtol=1e-12, atol=1e-12)
                assert not out.any() and not np.any(o2) and not np.any(n2)
                assert np.max(np.abs(a)) <= 50.0


def test_arm_separation_equals_joint_fit():
    """Each arm's objective depends on its own block only: fitting the arms separately
    reproduces the joint treatment-interaction fit (ridge, no active ball)."""
    rng = np.random.default_rng(1)
    D, Z, _ = e18.draw(rng, 600, "18A")
    X = np.column_stack([D, Z])
    a = e18.CANDIDATES.index(("UKL1", 2, 1e-2))
    status, f = e18.fit_candidate(a, X)
    assert status == "ok"
    gen = e18.generator("UKL1")
    basis = g.TreatmentInteractionBasis(
        base_basis=g.CallableBasis(lambda Zb: e18.legendre_features(Zb, 2))
    )
    joint = g.GRRGLM(
        basis=basis,
        generator=gen,
        functional=g.ATEFunctional(0),
        penalty="l2",
        lam=1e-2,
        offset=g.offset_from_alpha(gen, e18.alpha_ref),
    )
    fr = joint.fit(X, tol=1e-12)
    assert fr.status == "ok"
    b = np.asarray(joint.beta_)
    p = e18.N_COLS[2]
    np.testing.assert_allclose(f.betas[1.0], b[:p], atol=1e-7)
    np.testing.assert_allclose(f.betas[0.0], b[p:], atol=1e-7)


def test_bernstein_interval():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 4, 100_000)
    mean, lo, hi = e18.bernstein_interval(x, 4.0, 1e-6)
    assert lo <= 2.0 <= hi and lo <= mean <= hi
    with pytest.raises(ValueError):
        e18.bernstein_interval(np.array([0.0, 5.0]), 4.0, 0.01)


def test_rhs_increasing_and_huge_at_small_nv():
    r = np.linspace(0, 10, 50)
    v = e18.rhs(r, "18A", 400)
    assert np.all(np.diff(v) > 0)
    assert v[0] == pytest.approx(2 * e18.c_sel("18A") * e18.T_FOLD / 400)


def test_fold_selection_small_sample():
    rng = np.random.default_rng(4)
    D, Z, _ = e18.draw(rng, 400, "18B")
    X = np.column_stack([D, Z])
    Zev = e18.draw_eval(np.random.default_rng(5))[:3000]
    e_ev = e18.propensity(Zev, "18B")
    rec, fits = e18._fold_selection(
        "18B",
        X,
        Z,
        np.arange(300),
        np.arange(300, 400),
        e18.legendre_features(Z),
        e18.legendre_features(Zev),
        e_ev,
        (50 + e18.MAX_ALPHA0["18B"]) ** 2,
    )
    assert 0 <= rec["sel"] < e18.M
    assert rec["risk_star"] <= rec["risk_sel"] + 1e-15
    assert rec["risk_star"] <= rec["risk_control"] + 1e-15 or fits[e18.CONTROL] is None
    assert rec["n_v"] == 100
    assert sum(rec["statuses"].values()) == e18.A0
    assert not rec["violation"]


def test_selection_ties_go_to_registered_order(monkeypatch):
    crit = np.array([np.nan, 1.0, 0.5, 0.5, np.nan])
    avail = np.isfinite(crit)
    assert int(np.flatnonzero(avail)[np.argmin(crit[avail])]) == 2


def test_families_on_a_synthetic_summary():
    import pandas as pd

    rows = []
    for d in e18.DESIGNS:
        for n in e18.N_VALUES:
            for c in e18.CONSTRUCTIONS:
                rows.append(
                    {
                        "design": d,
                        "n": n,
                        "construction": c,
                        "R": e18.R[d],
                        "violations": 0,
                        "SEL_covered": int(0.95 * e18.R[d]),
                    }
                )
    S = pd.DataFrame(rows)
    thm, tt = e18.family_thm(S)
    inf, ti = e18.family_inf(S)
    assert len(tt) == 8 and len(ti) == 6
    assert not thm.negative and not inf.negative
    assert tt["a|n=1000"] == pytest.approx(inference.failure_rate_pvalue(0, 500, 0.051))
    S.loc[(S.design == "18A") & (S.n == 1000) & (S.construction == "a"), "violations"] = 100
    assert e18.family_thm(S)[0].negative


def test_emp_verdict_rules():
    iv = []
    for c in e18.CONSTRUCTIONS:
        for stat in ("b_q90_excess", "c_median_ratio", "e_median_risk"):
            iv.append({"construction": c, "statistic": stat, "n": 1000, "lower": 1.0, "upper": 2.0})
            iv.append({"construction": c, "statistic": stat, "n": 8000, "lower": 0.5, "upper": 1.5})
    v = e18.emp_verdicts(iv)
    assert v["a|b_q90_excess"] is True  # 1.5 <= 2 * 1.0
    assert v["a|c_median_ratio"] is False  # 1.5 < 1.0 fails
    assert json.dumps(v)
