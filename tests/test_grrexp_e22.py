"""E-22 support code: the dictionaries and the ATT arm basis, the signed control weights
and SMD, the ATT AIPW, the EB check convention, the split-median aggregation and the
majority rule, and the tables. The Lalonde file is not in the repository (registered
snapshot outside git), so these tests use synthetic data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import baselines, e22  # noqa: E402
from grrexp.outputs import STEM  # noqa: E402
from grrexp.seeds import Seeds, fold_ids  # noqa: E402


def synthetic(n=400, seed=0):
    """A Lalonde-shaped sample: D, eight covariates (two earnings with zeros), Y."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, 8))
    raw[:, 6] = np.where(rng.uniform(size=n) < 0.3, 0.0, np.exp(rng.normal(size=n)))
    raw[:, 7] = np.where(rng.uniform(size=n) < 0.3, 0.0, np.exp(rng.normal(size=n)))
    Zs = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    e = 1 / (1 + np.exp(-(-0.8 + 0.5 * Zs[:, 0] - 0.4 * Zs[:, 2])))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = 1.0 + 2.0 * D + Zs[:, 0] + 0.5 * Zs[:, 6] + rng.normal(size=n)
    return D, Y, Zs, raw, e22.design(D, Zs, raw)


# ---------------------------------------------------------------- dictionaries


def test_d2_has_the_dehejia_wahba_terms_and_zero_earnings_indicators() -> None:
    _, _, Zs, raw, _ = synthetic()
    F = e22.d2_features(Zs, raw)
    assert F.shape == (len(Zs), 14)
    np.testing.assert_array_equal(F[:, 11], (raw[:, 6] == 0).astype(float))
    np.testing.assert_array_equal(F[:, 12], (raw[:, 7] == 0).astype(float))
    np.testing.assert_allclose(F[:, 2], Zs[:, 0] ** 3)
    np.testing.assert_allclose(F[:, 13], Zs[:, 1] * Zs[:, 6])


@pytest.mark.parametrize("dictionary", ["D1", "D2", "D3"])
def test_att_arm_basis_has_a_treated_intercept_and_a_control_dictionary(dictionary) -> None:
    D, _, Zs, raw, X = synthetic()
    b = e22.ATTArmBasis(dictionary, random_state=3).copy().fit(X)
    P = b(X)
    assert P.shape == (len(D), b.n_features)
    np.testing.assert_array_equal(P[:, 0], D)
    np.testing.assert_array_equal(P[D == 1, 1:], 0.0)
    np.testing.assert_allclose(P[D == 0, 1], 1.0)  # the control arm contains the constant
    if dictionary == "D2":
        np.testing.assert_allclose(P[D == 0, 2:], e22.d2_features(Zs, raw)[D == 0])
    if dictionary == "D1":
        np.testing.assert_allclose(P[D == 0, 2:], Zs[D == 0])
        np.testing.assert_allclose(b.control_features(X), Zs)


def test_d3_centers_come_from_the_fitting_rows_only() -> None:
    _, _, _, _, X = synthetic()
    b = e22.ATTArmBasis("D3", random_state=5).fit(X[:200])
    centers = b._rkhs.centers
    assert centers.shape == (e22.RKHS_CENTERS, 8)
    assert all(any(np.allclose(c, x) for x in X[:200, 1:9]) for c in centers[:10])


# ---------------------------------------------------------------- weights and SMD


def test_control_weights_are_signed_and_normalised() -> None:
    w = e22.control_weights(np.array([-2.0, -1.0, 0.5]))
    np.testing.assert_allclose(w, [2 / 2.5, 1 / 2.5, -0.5 / 2.5])
    assert np.isclose(w.sum(), 1.0)
    assert np.all(np.isnan(e22.control_weights(np.array([1.0, 0.5]))))


def test_smd_matches_the_genriesz_convention_for_positive_weights() -> None:
    from genriesz.estimation import _covariate_balance_smd

    D, _, Zs, _, _ = synthetic()
    rng = np.random.default_rng(1)
    w = rng.uniform(0.2, 2.0, size=int((D == 0).sum()))
    ref = _covariate_balance_smd(Z=Zs, D=D, w_control=w, target="att")["smd_weighted"]
    np.testing.assert_allclose(e22._smd(Zs[D == 1], Zs[D == 0], w / w.sum()), ref)


def test_weight_summaries_counts_wrong_signs_and_ess() -> None:
    D, _, Zs, _, X = synthetic()
    alpha = np.where(D == 1, 3.0, -1.0)
    alpha[np.flatnonzero(D == 0)[:5]] = 0.5  # five control weights of the wrong sign
    out = e22.weight_summaries(D, Zs, X, alpha, e22.ATTArmBasis("D1"))
    nc = int((D == 0).sum())
    assert np.isclose(out["wrong_sign_share"], 5 / nc)
    a = -alpha[D == 0]
    assert np.isclose(out["ess_c"], a.sum() ** 2 / np.sum(a**2))
    assert len(json.loads(out["smd_raw"])) == 8


# ---------------------------------------------------------------- estimators


def test_att_aipw_matches_its_formula_and_influence_function() -> None:
    D, Y, _, _, _ = synthetic()
    rng = np.random.default_rng(2)
    e = rng.uniform(0.1, 0.6, size=len(D))
    mu0 = rng.normal(size=len(D))
    theta, se = e22._att_aipw(D, Y, e, mu0)
    r = D * (Y - mu0) - (1 - D) * e / (1 - e) * (Y - mu0)
    assert np.isclose(theta, r.sum() / D.sum())
    psi = (r - D * theta) / D.mean()
    assert np.isclose(se, np.std(psi) / np.sqrt(len(D)))
    assert abs(np.mean(psi)) < 1e-10  # the influence function is centred at theta


def test_ukl_c0_d1_balances_like_entropy_balancing() -> None:
    """The EB check's convention: the control weights -alpha of UKL (C=0) x D1 with
    lambda = 0 solve the ebal dual with target n * mean_T(1, Z)."""
    import genriesz as g

    D, _, Zs, raw, X = synthetic(n=500, seed=4)
    n, pi1 = len(D), float(D.mean())
    m = g.ATTFunctional(treatment_index=0, pi=pi1, pi_is_estimated=True)
    gen = e22.generator("UKL")
    model = g.GRRGLM(basis=e22.ATTArmBasis("D1").fit(X), generator=gen, functional=m,
                     penalty=None, lam=0.0, offset=g.offset_from_alpha(gen, e22.alpha_ref))
    assert model.fit(X, tol=e22.SAMPLE_TOL).status == "ok"
    a = np.asarray(model.predict_alpha(X), float)
    c, t = D == 0, D == 1
    F = np.column_stack([np.ones(n), Zs])
    w = baselines.eb_dual_newton(F[c], n * F[t].mean(axis=0), n)
    assert w is not None
    assert np.max(np.abs(w + a[c])) / max(1.0, w.max()) <= e22.EB_TOL
    np.testing.assert_allclose(a[t], 1 / pi1, rtol=1e-8)


def test_grr_arm_runs_on_one_split() -> None:
    D, Y, Zs, _, X = synthetic(n=400, seed=6)
    folds = fold_ids(len(Y), e22.K, np.random.default_rng(0))
    out = e22._grr_arm("SQ-D1", X, Y, D, Zs, folds, 1, 2)
    assert set(out) == set(e22.ESTIMATORS)
    assert out["ARW"]["status"] == "ok" and np.isfinite(out["ARW"]["estimate"])
    assert out["TMLE"]["n_warnings"] == 0  # counted once, on the ARW record


def test_admissibility_threshold_for_bkl() -> None:
    pi1 = e22.N_TREATED / e22.N
    c = e22.C_VALUE["BKL"]
    assert round(c * pi1 / (1 + c * pi1), 5) == 0.01484


def test_fold_splits_are_reproducible_and_distinct() -> None:
    s = Seeds(22)
    a = fold_ids(e22.N, e22.K, s.folds(0, 0))
    b = fold_ids(e22.N, e22.K, s.folds(0, 0))
    c = fold_ids(e22.N, e22.K, s.folds(0, 1))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------- aggregation and tables


def test_median_aggregate() -> None:
    theta = np.array([1.0, 2.0, 4.0])
    sigma2 = np.array([10.0, 20.0, 30.0])
    t, s2 = e22.median_aggregate(theta, sigma2, 5)
    assert t == 2.0
    assert s2 == np.median(sigma2 + 5 * (theta - 2.0) ** 2)


def fake_raw(failures):
    rows = []
    for arm in e22.ARM_LABELS:
        for s in range(e22.R):
            ok = s >= failures.get(arm, 0)
            rows.append({"cell": 0, "split": s, "arm": arm,
                         "estimate": 1000.0 + s if ok else np.nan, "se": 500.0 if ok else np.nan,
                         "status": "ok" if ok else "domain_prediction",
                         "smd_raw": json.dumps([0.01 * s] * 8) if ok else "",
                         **{k: 0.1 for k in ("smd_raw_max", "smd_dict_max", "ess_c",
                                             "wrong_sign_share", "max_abs_alpha", "lam")}})
    return pd.DataFrame(rows)


def fake_full():
    fits = {a: {"status": "ok", "smd_raw_max": -0.0001, "smd_dict_max": 0.0, "ess_c": 50.0,
                "wrong_sign_share": 0.0, "max_abs_alpha": 3.0, "lam": 0.0,
                "smd_raw": json.dumps([0.0] * 8)} for a in e22.GRR_ARMS}
    return {"fits": fits, "unweighted_smd": [-0.2] * 8, "pi1": 0.3, "n_warnings": 0,
            "eb_check": {"converged": True, "max_rel_diff": 1e-15, "agrees": True},
            "admissibility": {"bkl_threshold": 0.01484, "logit_e_min_all": 1e-5,
                              "logit_e_min_controls": 1e-5, "logit_converged": True,
                              "ukl_d1_e_min": 0.008}}


def test_majority_rule_and_split_median() -> None:
    S = e22.summarise(fake_raw({"BKL-D1|ARW": 51, "BP-D1|ARW": 50, "UKL-D1|ARW": 100}),
                      fake_full())
    by = S.set_index("arm")
    assert not by.loc["BKL-D1|ARW", "reported"] and np.isnan(by.loc["BKL-D1|ARW", "estimate"])
    assert by.loc["BP-D1|ARW", "reported"] and by.loc["BP-D1|ARW", "R_s"] == 50  # exactly half
    assert not by.loc["UKL-D1|ARW", "reported"] and by.loc["UKL-D1|ARW", "cf_smd_raw"] == ""
    assert by.loc["SQ-D1|ARW", "estimate"] == np.median(1000.0 + np.arange(100))
    cf = json.loads(by.loc["SQ-D1|ARW", "cf_smd_raw"])  # per-covariate median over splits
    np.testing.assert_allclose(cf, [np.median(0.01 * np.arange(100))] * 8)


def test_tables_have_heads_reference_no_negative_zero_and_failures() -> None:
    S = e22.summarise(fake_raw({"BKL-D1|ARW": 60}), fake_full())
    tabs = e22.tables(S)
    assert set(tabs) == {"tab_E22_main", "tab_E22_full", "tab_E22_smd", "tab_E22_diag",
                         "tab_E22_status"}
    assert "1,794" in tabs["tab_E22_main"] and "not the true effect" in tabs["tab_E22_main"]
    assert "no exact fit (40/100)" in tabs["tab_E22_main"]
    for name in ("tab_E22_full", "tab_E22_smd"):
        assert "\\endfirsthead" in tabs[name] and "\\endhead" in tabs[name]
    assert all("-0.000" not in t for t in tabs.values())
    assert "domain\\_prediction: 60" in tabs["tab_E22_status"]
    assert "(cf)" in tabs["tab_E22_smd"] and "(full)" in tabs["tab_E22_smd"]
    assert "Every full-sample fit succeeded" in tabs["tab_E22_status"]


def test_record_labels_are_recorder_stems() -> None:
    assert all(STEM.match(e22.record_label(a)) for a in e22.ARM_LABELS)
    assert len({e22.record_label(a) for a in e22.ARM_LABELS}) == len(e22.ARM_LABELS)


def test_dml_gbm_is_deterministic_and_selects_on_inner_folds() -> None:
    """With one grid point the inner CV is trivial; the estimate is a function of the
    stream-2 (inner splits) and stream-3 (GBM seed) generators only. Run in a fresh,
    single-threaded interpreter: gradient boosting starts an OpenMP pool that would
    otherwise stay in this test process (the run-recorder tests require none)."""
    import os
    import subprocess

    here = Path(__file__).resolve()
    experiments = here.parents[1] / "notebooks" / "experiments"
    code = f"""
import json, sys
import numpy as np
sys.path.insert(0, {str(here.parent)!r})
sys.path.insert(0, {str(experiments)!r})
import test_grrexp_e22 as T
from grrexp import baselines, e22
from grrexp.seeds import Seeds, fold_ids
baselines.GBM_GRID = [(0.1, 7, 50)]
D, Y, Zs, _, _ = T.synthetic(n=300, seed=8)
folds = fold_ids(len(Y), e22.K, np.random.default_rng(1))
s = Seeds(22)
a = e22.dml_gbm(D, Y, Zs, folds, s.inner_cv(0, 3), s.baseline(0, 3))
b = e22.dml_gbm(D, Y, Zs, folds, s.inner_cv(0, 3), s.baseline(0, 3))
assert a["status"] == "ok" and a["estimate"] == b["estimate"] and a["se"] == b["se"]
assert json.loads(a["gbm_params"]) == [[[0.1, 7, 50], [0.1, 7, 50]]] * e22.K
print("ok")
"""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1"}
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip().endswith("ok")


def test_failed_grr_arm_counts_its_warnings_once(monkeypatch) -> None:
    D, Y, Zs, _, X = synthetic(n=300, seed=9)
    folds = fold_ids(len(Y), e22.K, np.random.default_rng(2))

    def warned(fit):
        res = fit()
        return res, {"UserWarning: x": 3}, False  # a ConvergenceWarning: the fit fails

    monkeypatch.setattr(baselines, "run_recording_warnings", warned)
    out = e22._grr_arm("SQ-D1", X, Y, D, Zs, folds, 1, 2)
    assert out["ARW"]["status"] == "convergence_warning" and out["ARW"]["n_warnings"] == 3
    assert out["TMLE"]["n_warnings"] == 0


def test_d3_arm_selects_lambda_from_the_grid() -> None:
    D, Y, Zs, _, X = synthetic(n=300, seed=10)
    folds = fold_ids(len(Y), e22.K, np.random.default_rng(3))
    out = e22._grr_arm("SQ-D3", X, Y, D, Zs, folds, 4, 5)
    assert out["ARW"]["status"] == "ok"
    assert min(e22.LAMBDA_GRID) <= out["ARW"]["lam"] <= max(e22.LAMBDA_GRID)
    assert np.isnan(out["ARW"]["smd_dict_max"])  # fold-specific dictionaries: not defined


# ---------------------------------------------------------------- provenance


class _Recorder:
    """A stand-in for a gradient-boosting model that records the rows it sees (the row id
    is the last feature column) and predicts from a fixed rule."""

    log: list = []

    def __init__(self, kind, params):
        self.kind, self.params = kind, params

    def fit(self, X, y):
        _Recorder.log.append((self.kind, "fit", self.params, tuple(X[:, -1].astype(int))))
        return self

    def predict_proba(self, X):
        _Recorder.log.append((self.kind, "predict", self.params, tuple(X[:, -1].astype(int))))
        p = np.full(len(X), 0.3 if self.params[0] == 0.1 else 0.5)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        _Recorder.log.append((self.kind, "predict", self.params, tuple(X[:, -1].astype(int))))
        return np.zeros(len(X)) if self.params[0] == 0.1 else np.full(len(X), 50.0)


def test_dml_gbm_selects_on_inner_folds_of_the_outer_training_rows(monkeypatch) -> None:
    D, Y, Zs, _, _ = synthetic(n=200, seed=12)
    Y = Y - Y[D == 0].mean()  # the 0-predictor wins the outcome MSE
    D = np.where(np.arange(len(D)) % 10 < 3, 1.0, 0.0)  # share 0.3: the 0.3 classifier wins
    ids = np.arange(len(D), dtype=float)
    Zid = np.column_stack([Zs, ids])
    grid = [(0.05, 7, 100), (0.1, 7, 100)]
    monkeypatch.setattr(baselines, "GBM_GRID", grid)
    monkeypatch.setattr(baselines, "_gbm_classifier", lambda p, s: _Recorder("clf", p))
    monkeypatch.setattr(baselines, "_gbm_regressor", lambda p, s: _Recorder("reg", p))
    _Recorder.log = []
    folds = fold_ids(len(Y), e22.K, np.random.default_rng(4))
    out = e22.dml_gbm(D, Y, Zid, folds, np.random.default_rng(5), np.random.default_rng(6))
    assert out["status"] == "ok"
    assert json.loads(out["gbm_params"]) == [[list(grid[1]), list(grid[1])]] * e22.K
    log, per_fold = _Recorder.log, len(grid) * e22.GBM_INNER_FOLDS * 4 + 4
    assert len(log) == e22.K * per_fold
    for k in range(e22.K):
        train = set(np.flatnonzero(folds != k))
        test = set(np.flatnonzero(folds == k))
        entries = log[k * per_fold:(k + 1) * per_fold]
        inner, final = entries[:-4], entries[-4:]
        for j in range(0, len(inner), 4):  # clf fit, clf predict, reg fit, reg predict
            cf, cp, rf, rp = (set(e[3]) for e in inner[j:j + 4])
            assert cf | cp <= train and not cf & cp  # inner training and validation rows
            assert rf <= cf and rp <= cp and all(D[i] == 0 for i in rf | rp)
        cf, cp, rf, rp = (set(e[3]) for e in final)
        assert cf == train and cp == test  # the selected propensity: outer train -> eval fold
        assert rf == {i for i in train if D[i] == 0} and rp == test


def test_replicate_routes_the_registered_streams(monkeypatch) -> None:
    D, Y, Zs, raw, X = synthetic(n=300, seed=13)
    monkeypatch.setattr(e22, "data", lambda: {"D": D, "Y": Y, "Zs": Zs, "raw": raw, "X": X})
    seen = {}

    def grr(arm, X_, Y_, D_, Zs_, folds, inner_seed, center_seed):
        seen.setdefault("grr", []).append((folds.copy(), inner_seed, center_seed))
        return {e: {"status": "ok", "estimate": 0.0, "se": 1.0} for e in e22.ESTIMATORS}

    def dml(D_, Y_, Zs_, folds, inner_gen, base_gen):
        seen["dml"] = (folds.copy(), int(inner_gen.integers(0, 2**31 - 1)),
                       int(base_gen.integers(0, 2**31 - 1)))
        return {"status": "ok", "estimate": 0.0, "se": 1.0}

    monkeypatch.setattr(e22, "_grr_arm", grr)
    monkeypatch.setattr(e22, "dml_gbm", dml)
    monkeypatch.setattr(e22, "logit_aipw", lambda *a: {"status": "ok", "estimate": 0.0, "se": 1.0})
    e22.replicate((0, 7, 20261007))
    s = Seeds(22)
    folds = fold_ids(len(Y), e22.K, s.folds(0, 7))
    g2 = s.inner_cv(0, 7)
    inner_seed, center_seed = (int(g2.integers(0, 2**31 - 1)) for _ in range(2))
    next_inner = int(g2.integers(0, 2**31 - 1))  # the GBM inner splits follow the D3 seeds
    base = int(s.baseline(0, 7).integers(0, 2**31 - 1))
    assert len(seen["grr"]) == len(e22.GRR_ARMS)
    for f, a, b in seen["grr"]:
        np.testing.assert_array_equal(f, folds)
        assert (a, b) == (inner_seed, center_seed)
    np.testing.assert_array_equal(seen["dml"][0], folds)
    assert seen["dml"][1:] == (next_inner, base)


def test_d3_tunes_and_fits_on_outer_training_rows_only(monkeypatch) -> None:
    import genriesz.estimation as est

    D, Y, Zs, _, X = synthetic(n=250, seed=14)
    folds = fold_ids(len(Y), e22.K, np.random.default_rng(7))
    key = {tuple(np.round(r, 12)): i for i, r in enumerate(X)}
    tuned, fitted = [], []
    real_select = est.select_grr_hyperparams
    real_fit = e22.ATTArmBasis.fit

    def select(**kw):
        tuned.append({key[tuple(np.round(r, 12))] for r in kw["X_train"]})
        return real_select(**kw)

    def fit(self, X_, y=None):
        if self.dictionary == "D3":
            fitted.append({key[tuple(np.round(r, 12))] for r in np.atleast_2d(X_)})
        return real_fit(self, X_, y)

    monkeypatch.setattr(est, "select_grr_hyperparams", select)
    monkeypatch.setattr(e22.ATTArmBasis, "fit", fit)
    out = e22._grr_arm("SQ-D3", X, Y, D, Zs, folds, 4, 5)
    assert out["ARW"]["status"] == "ok"
    trains = [set(np.flatnonzero(folds != k)) for k in range(e22.K)]
    assert tuned == trains  # lambda is chosen on each outer training fold only
    assert fitted and all(any(f <= tr for tr in trains) for f in fitted)


def test_full_sample_counts_warnings_and_drops_a_nonconverged_logistic(monkeypatch) -> None:
    import warnings as _w

    from sklearn.exceptions import ConvergenceWarning

    D, Y, Zs, raw, X = synthetic(n=300, seed=15)
    monkeypatch.setattr(e22, "data", lambda: {"D": D, "Y": Y, "Zs": Zs, "raw": raw, "X": X})

    def logistic(F2, D_):
        _w.warn("did not converge", ConvergenceWarning, stacklevel=1)
        return np.full(len(D_), 0.3)

    monkeypatch.setattr(e22, "_logistic_full", logistic)
    full = e22.full_sample(20261007)
    ad = full["admissibility"]
    assert ad["logit_converged"] is False
    assert ad["logit_e_min_all"] is None and ad["logit_e_min_controls"] is None
    assert full["n_warnings"] >= 1
    assert full["eb_check"]["agrees"]
    assert set(full["fits"]) == set(e22.GRR_ARMS)
