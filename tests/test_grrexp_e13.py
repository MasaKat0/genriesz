"""E-13 support code: links and offsets, weighted general-link fits, Stage 0 pieces,
replication records, H13 (a)(b) checks, aggregation, table and figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import e12, e13, outputs  # noqa: E402
from grrexp.seeds import PILOT_ENTROPY, Seeds  # noqa: E402

import genriesz as gr  # noqa: E402


def _rows(n=5):
    rng = np.random.default_rng(3)
    Z = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n, 3))
    return np.column_stack([np.r_[np.ones(n // 2), np.zeros(n - n // 2)], Z])


# ---------------------------------------------------------------- links and offsets


@pytest.mark.parametrize("pair", e13.INCOMPATIBLE)
def test_registered_index_offset_gives_alpha_ref(pair) -> None:
    X = _rows()
    link = e13.LINKS[pair][0]
    assert np.allclose(link(X, np.full(len(X), e13.V_REF[pair])), e13.alpha_ref(X), atol=0)


@pytest.mark.parametrize("pair", e13.INCOMPATIBLE)
def test_link_derivatives_match_finite_differences(pair) -> None:
    X = _rows(8)
    link, dlink, d2link = e13.LINKS[pair]
    v = np.linspace(0.2, 1.4, len(X))
    h = 1e-6
    assert np.allclose(dlink(X, v), (link(X, v + h) - link(X, v - h)) / (2 * h), rtol=1e-7)
    assert np.allclose(
        d2link(X, v), (dlink(X, v + h) - dlink(X, v - h)) / (2 * h), rtol=1e-6, atol=1e-8
    )


def test_compatible_pairs_use_glm_and_incompatible_the_general_link() -> None:
    for pair in e13.COMPATIBLE:
        assert isinstance(e13.model(pair), gr.GRRGLM)
    for pair in e13.INCOMPATIBLE:
        assert isinstance(e13.model(pair), gr.GRRGeneralLink)
    assert e13.generator("I-BKLexp").C == 1.0 and e13.generator("I-UKLlin").C == 1.0


def test_gamma0_coefficients_on_the_interaction_basis() -> None:
    X = _rows(9)
    bas = e13.basis()
    bas.fit(X)
    g0 = e13.gamma0(X[:, 0], X[:, 1:])
    assert np.allclose(np.asarray(bas(X)) @ e13.RHO, g0, atol=1e-14)


# ---------------------------------------------------------------- weighted general link


def test_general_link_integer_weights_equal_replicated_rows() -> None:
    rng = np.random.default_rng(0)
    D, Z, _ = e13.draw(rng, 300)
    X = np.column_stack([D, Z])
    k = rng.integers(1, 4, size=len(X))
    Xrep = np.repeat(X, k, axis=0)
    beta = 0.05 * rng.standard_normal(e13.P)
    m = e13.model("I-BKLexp")
    m.basis.fit(X)
    assert np.isclose(m.objective(X, beta, sample_weight=k), m.objective(Xrep, beta), rtol=1e-12)
    assert np.allclose(m.gradient(X, beta, sample_weight=k), m.gradient(Xrep, beta), atol=1e-12)
    assert np.allclose(m.hessian(X, beta, sample_weight=k), m.hessian(Xrep, beta), atol=1e-11)
    r_w = e13.model("I-BKLexp").fit(X, sample_weight=k, tol=1e-11)
    r_rep = e13.model("I-BKLexp").fit(Xrep, tol=1e-11)
    assert r_w.status == r_rep.status == "ok"
    assert np.allclose(r_w.beta, r_rep.beta, atol=1e-9)
    assert np.isclose(r_w.regressor_imbalance, r_rep.regressor_imbalance, rtol=1e-6)


def test_general_link_rejects_bad_weights() -> None:
    X = _rows(6)
    m = e13.model("I-SQexp")
    m.basis.fit(X)
    for w in (np.ones(5), -np.ones(6), np.zeros(6), np.full(6, np.nan)):
        with pytest.raises(ValueError):
            m.objective(X, np.zeros(e13.P), sample_weight=w)


# ---------------------------------------------------------------- Stage 0


def test_dgp_check_reproduces_the_registered_values() -> None:
    out = e13.dgp_check(48)
    assert out["ok"], out


def test_multistart_points_are_deterministic_and_in_the_unit_ball() -> None:
    a, b = e13.multistart_points(), e13.multistart_points()
    assert a.shape == (20, 8) and np.array_equal(a, b)
    assert np.all(np.abs(a) <= 1.0) and np.array_equal(a[0], -np.ones(8))
    assert len({tuple(r) for r in a}) == 20


@pytest.mark.parametrize("pair", e13.PAIRS)
def test_support_margins_are_the_exact_range_over_the_box(pair) -> None:
    rng = np.random.default_rng(5)
    beta = 0.1 * rng.standard_normal(e13.P)
    marg = e13.support_margins(pair, beta)
    s = np.sqrt(3)
    corners = np.array(np.meshgrid([-s, s], [-s, s], [-s, s])).reshape(3, -1).T
    gen = e13.generator(pair)
    X = np.vstack([np.column_stack([np.full(8, d), corners]) for d in (1.0, 0.0)])
    bas = e13.basis()
    bas.fit(X)
    Phi = np.asarray(bas(X))
    if pair in e13.COMPATIBLE:
        u = np.asarray(gen.grad(X, e13.alpha_ref(X))) + Phi @ beta
        assert np.isclose(marg["dual"], float(np.min(gen.dual_margin(X, u))), rtol=1e-12)
    else:
        v = e13.V_REF[pair] + Phi @ beta
        a = e13.LINKS[pair][0](X, v)
        assert np.isclose(marg["alpha"], float(np.min(gen.boundary_margin(X, a))), rtol=1e-12)
        expected_index = float(np.min(v)) if pair == "I-UKLlin" else np.inf
        assert marg["index"] == pytest.approx(expected_index, rel=1e-12)


@pytest.mark.parametrize("pair", e13.COMPATIBLE)
def test_compatible_population_has_zero_bias_and_equal_variances(pair) -> None:
    out, mom = e13.population_compatible(pair, 12)
    assert out["status"] == "ok"
    assert abs(mom["b"]) < 1e-12 and np.max(np.abs(mom["delta"])) < 1e-12
    for k in ("sigma2_true_rw", "sigma2_true_arw", "sigma2_se"):
        assert np.isclose(mom[k], mom["E_alpha2"], rtol=1e-10)


def test_rw_influence_function_matches_a_gateaux_derivative() -> None:
    """The stacked Z-estimator IF of RW (incompatible pair) against a finite-difference
    derivative of theta* = E[alpha_{beta*} gamma0] under contamination of one support row."""
    pair = "I-BKLexp"
    X, w, *_ = e13.quadrature(6)

    def theta(weights):
        mdl, res = e13._general_fit(pair, X, weights)
        assert res.status == "ok"
        v = e13.V_REF[pair] + np.asarray(mdl.basis(X)) @ res.beta
        a = e13.LINKS[pair][0](X, v)
        wn = weights / weights.sum()
        return float(wn @ (a * e13.gamma0(X[:, 0], X[:, 1:]))), mdl, res, a

    th, mdl, res, a = theta(w)
    H = mdl.hessian(X, res.beta, sample_weight=w)
    v = e13.V_REF[pair] + np.asarray(mdl.basis(X)) @ res.beta
    Phi = np.asarray(mdl.basis(X))
    c, _ = mdl._coef(X, res.beta)
    M_psi = np.asarray(gr.ATEFunctional(0).m_basis_matrix(X, mdl._tangent_basis(res.beta, 8)))
    score = a[:, None] * c[:, None] * Phi - M_psi
    G = (w * e13.gamma0(X[:, 0], X[:, 1:])) @ (np.asarray(e13.LINKS[pair][1](X, v))[:, None] * Phi)
    IF = a * e13.gamma0(X[:, 0], X[:, 1:]) - th - score @ np.linalg.solve(H, G)
    eps = 1e-5
    for i in (0, 37, 250):
        e_i = np.zeros_like(w)
        e_i[i] = 1.0
        up = theta((1 - eps) * w + eps * e_i)[0]
        dn = theta((1 + eps) * w - eps * e_i)[0]
        assert (up - dn) / (2 * eps) == pytest.approx(IF[i], rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------- replications and checks


def test_fit_representer_reports_a_missing_group() -> None:
    X = _rows(10)
    X[:, 0] = 1.0
    status, mdl, rec = e13.fit_representer("C-SQ", X)
    assert status == "degenerate_functional" and rec is None


def test_replicate_is_deterministic_and_records_every_arm() -> None:
    a = e13.replicate((0, 0, PILOT_ENTROPY))
    b = e13.replicate((0, 0, PILOT_ENTROPY))
    assert set(a) == set(e13.ARM_LABELS)
    statuses = outputs.recorded_statuses()
    for label, r in a.items():
        assert r["status"] in statuses
        assert json.dumps(r, sort_keys=True) == json.dumps(b[label], sort_keys=True)
        if r["status"] == "ok":
            assert r["I_psi_max"] <= e13.BALANCE_TOL
            assert r["train_fits"] == (1 if label.endswith("RW_full") else e13.K)
            if label.startswith("C-"):
                assert r["I_phi_max"] <= e13.BALANCE_TOL
            if label.endswith("RW_full"):
                assert all(np.isfinite(r[f"delta_hat_{j}"]) for j in range(e13.P))


def _raw_stub():
    return pd.DataFrame(
        {
            "arm": ["C-SQ|RW_full", "I-SQexp|ARW_cf", "I-SQexp|RW_full"],
            "status": ["ok", "ok", "linesearch"],
            "train_fits": [1, 5, 0],
            "I_psi_max": [1e-12, 1e-9, np.nan],
            "I_phi_max": [1e-12, 0.05, np.nan],
            "scale_max": [1.0, 1.0, np.nan],
        }
    )


def test_balance_checks_pass_and_stop_on_each_violation() -> None:
    raw = _raw_stub()
    bal = e13.balance_checks(raw)
    assert bal["fits_checked"] == 6 and bal["I_phi_max_incompatible"] == 0.05
    bad_a = raw.copy()
    bad_a.loc[1, "I_psi_max"] = 2e-8
    with pytest.raises(AssertionError, match=r"\(a\)"):
        e13.balance_checks(bad_a)
    bad_b = raw.copy()
    bad_b.loc[0, ["I_psi_max", "I_phi_max"]] = [1e-12, 2e-8]
    with pytest.raises(AssertionError, match=r"\(b\)"):
        e13.balance_checks(bad_b)
    missing = raw.copy()
    missing.loc[1, "train_fits"] = 4
    with pytest.raises(AssertionError, match="balance of every fit"):
        e13.balance_checks(missing)


def _population_stub():
    rows = []
    for pair in e13.PAIRS:
        b = 0.0 if pair in e13.COMPATIBLE else 0.02
        rows.append(
            {
                "pair": pair, "certified": pair != "I-UKLlin", "b": b,
                "sigma2_true_rw": 5.0, "sigma2_true_arw": 4.8, "sigma2_se": 4.7,
                "delta": [0.01 * (j - 3) for j in range(e13.P)],
                "c_n": {e: {str(n): 0.94 for n in e13.N_VALUES} for e in e13.ESTIMATORS},
            }
        )  # fmt: skip
    return rows


def test_aggregation_table_and_figure_from_summary() -> None:
    import matplotlib.pyplot as plt

    tasks = e13.tasks_for(PILOT_ENTROPY, 3)
    raw = e13.raw_frame(tasks, [e13.replicate(t) for t in tasks])
    assert len(raw) == 6 * len(e13.ARM_LABELS)
    seeds = Seeds(13, entropy=PILOT_ENTROPY)
    e13.balance_checks(raw)
    summ = e13.summarise(raw, _population_stub())
    summ = summ.assign(**e12.family1_mcse(raw, summ, seeds))
    verdict, tests, unavailable = e13.family_h13(summ, raw, seeds)
    certified = [p for p in e13.PAIRS if p != "I-UKLlin"]
    n_c = sum(k.startswith("(c)") for k in tests)
    assert n_c == 8 * 2  # I-SQexp and I-BKLexp, eight directions each
    assert sum(k.startswith("(d)") for k in tests) == len(certified) * 2
    assert sum(k.startswith("(e)") for k in tests) == len(certified) * 2 * 2
    assert all(0.0 <= p <= 1.0 for p in tests.values())
    assert verdict.hypothesis == "H13"
    summ["family_verdict"] = json.dumps(verdict.sentence())
    tabs, fam = e13.tables(summ)
    assert tabs["tab_E13"].count(" \\\\") == 1 + len(e13.PAIRS) * 2 * 2
    fig = e13.figure(summ)
    plt.close(fig)
    assert not summ.loc[summ["pair"] == "I-UKLlin", "c_pred"].notna().any()
