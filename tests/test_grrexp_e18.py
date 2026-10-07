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


# ---------------------------------------------------------------- guards (review round 1)


def _xy(n=400, design="18B", seed=7):
    D, Z, Y = e18.draw(np.random.default_rng(seed), n, design)
    return np.column_stack([D, Z]), Z, Y


def test_outcome_arm_absent_or_too_small_fails():
    X, Z, Y = _xy()
    Q = e18.quadratic(Z)
    treated = np.flatnonzero(X[:, 0] == 1)
    st, c1, c0 = e18._arm_ols(Q, X, Y, treated)
    assert st == "degenerate_functional" and c1 is None and c0 is None
    few = np.concatenate([treated, np.flatnonzero(X[:, 0] == 0)[:5]])
    assert e18._arm_ols(Q, X, Y, few)[0] == "degenerate_functional"
    st, psi, a = e18._arw_fold(None, X, Y, e18.legendre_features(Z), Q, treated, np.arange(10))
    assert st == "degenerate_functional" and psi is None


def test_alpha_flags_rows_outside_the_ball_and_arw_reports_domain_prediction():
    X, Z, Y = _xy()
    P = e18.legendre_features(Z)
    beta = np.zeros(e18.N_COLS[1])
    beta[0] = e18.RADIUS["UKL1"] * (1 + 1e-6)  # just outside the stated ball
    f = e18._Fit("UKL1", 1, {1.0: beta, 0.0: beta})
    _, out = f.alpha(1.0, P)
    assert out.all()
    comp, te = np.arange(100, 400), np.arange(100)
    st, psi, a = e18._arw_fold(f, X, Y, P, e18.quadratic(Z), comp, te)
    assert st == "domain_prediction" and psi is None
    sq = np.zeros(e18.N_COLS[1])
    sq[0] = e18.RADIUS["SQ"]  # on the sphere: |alpha| = 50 exactly, allowed
    a, out = e18._Fit("SQ", 1, {1.0: sq, 0.0: -sq}).alpha(1.0, P)
    assert not out.any() and np.max(np.abs(a)) == pytest.approx(50.0)


def test_fitted_candidates_stay_in_the_stated_ball():
    X, Z, _ = _xy(n=300, design="18A", seed=11)
    for a in range(e18.A0):
        status, f = e18.fit_candidate(a, X)
        if status == "ok":
            for beta in f.betas.values():
                assert np.sum(np.abs(beta)) <= e18.RADIUS[f.gen_name]
    assert e18.SOLVER_RADIUS_FACTOR < 1.0


def _small_fold(monkeypatch=None):
    X, Z, _ = _xy(n=300)
    Zev = e18.draw_eval(np.random.default_rng(5))[:2000]
    return e18._fold_selection(
        "18B", X, Z, np.arange(200), np.arange(200, 300), e18.legendre_features(Z),
        e18.legendre_features(Zev), e18.propensity(Zev, "18B"),
        (50 + e18.MAX_ALPHA0["18B"]) ** 2,
    )  # fmt: skip


def test_all_candidates_failing_selects_a0(monkeypatch):
    monkeypatch.setattr(e18, "fit_candidate", lambda a, X: ("linesearch", None))
    rec, fits = _small_fold()
    assert rec["sel"] == e18.A0 and rec["raw_sel"] == e18.A0
    assert rec["n_failed"] == e18.A0 and rec["control_status"] == "linesearch"
    assert rec["risk_star"] == rec["risk_sel"]
    assert np.isnan(rec["risk_control"])


def test_convergence_warning_excludes_the_candidate(monkeypatch):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    real = e18.fit_candidate

    def warned(a, X):
        if a == e18.CONTROL:
            warnings.warn("did not converge", ConvergenceWarning, stacklevel=1)
        return real(a, X)

    monkeypatch.setattr(e18, "fit_candidate", warned)
    rec, fits = _small_fold()
    assert fits[e18.CONTROL] is None
    assert rec["control_status"] == "convergence_warning" and rec["sel"] != e18.CONTROL
    assert sum(rec["warnings"].values()) == 1


def test_emp_bootstrap_is_reproducible_and_clustered():
    import pandas as pd
    from grrexp.seeds import PILOT_ENTROPY, Seeds

    rows = []
    rng = np.random.default_rng(0)
    for d in e18.DESIGNS:
        for n in e18.N_VALUES:
            for rep in range(6):
                for c in e18.CONSTRUCTIONS:
                    star = rng.uniform(0.1, 0.2, e18.K)
                    rows.append({"design": d, "n": n, "rep": rep, "construction": c,
                                 "risk_sel": json.dumps((star * rng.uniform(1, 2, e18.K)).tolist()),
                                 "risk_star": json.dumps(star.tolist()),
                                 "n_v": json.dumps([n // 5] * e18.K)})  # fmt: skip
    raw = pd.DataFrame(rows)
    a = e18.emp_intervals(raw, Seeds(18, entropy=PILOT_ENTROPY))
    b = e18.emp_intervals(raw, Seeds(18, entropy=PILOT_ENTROPY))
    assert a == b and len(a) == 2 * 3 * 4
    assert [(r["construction"], r["statistic"], r["n"]) for r in a[:5]] == [
        ("a", "b_q90_excess", 1000), ("a", "b_q90_excess", 2000), ("a", "b_q90_excess", 4000),
        ("a", "b_q90_excess", 8000), ("a", "c_median_ratio", 1000),
    ]  # fmt: skip
    for r in a:
        assert r["lower"] <= r["upper"]
