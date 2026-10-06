"""grr_functional reports fold failures as statuses (registration §1.3 B/D, E-5/E-6).

- A failed fold makes every estimate NaN, with the status of the failure; no
  silent averaging over the other folds and no exception.
- Evaluation-fold rows and the counterfactual rows that ``m`` evaluates are
  checked against the domain of the fitted representer ("domain_prediction").
- Explicit, data-independent folds (``fold_ids``).
- TMLE's standard error uses the targeted (updated) regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from genriesz import (
    GRRGLM,
    ATEFunctional,
    BPGenerator,
    OutcomeGLM,
    PolynomialBasis,
    SquaredGenerator,
    TreatmentInteractionBasis,
    UKLGenerator,
    grr_ate,
    offset_from_alpha,
)
from genriesz.estimation import _DomainCheckedRepresenter


def _treated(x: np.ndarray) -> int:
    return int(x[0] == 1.0)


def _data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-1.0, 1.0, size=(n, 1))
    e = 1.0 / (1.0 + np.exp(-0.8 * Z[:, 0]))
    D = rng.binomial(1, e).astype(float)
    Y = D + Z[:, 0] + rng.normal(size=n)
    return np.column_stack([D, Z]), Y


def _basis():
    return TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    )


def _fold_ids(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % k)


def test_ok_result_has_status_and_fold_status():
    X, Y = _data()
    res = grr_ate(
        X=X, Y=Y, basis=_basis(), generator=UKLGenerator(C=1.0, branch_fn=_treated),
        riesz_penalty=None, outcome_link="identity", folds=3, random_state=0,
    )
    assert res.status == "ok" and res.success
    assert [s[2] for s in res.fold_status] == ["ok", "ok", "ok"]
    assert all(np.isfinite(e.estimate) for e in res.estimates.values())


def test_boundary_fold_makes_every_estimate_nan():
    """BP(C=1) cannot represent alpha_0 with |alpha_0| close to 1: the fit ends at the boundary."""

    rng = np.random.default_rng(3)
    n = 600
    Z = rng.normal(size=(n, 1))
    e = 1.0 / (1.0 + np.exp(-2.5 * Z[:, 0]))
    D = rng.binomial(1, e).astype(float)
    X = np.column_stack([D, Z])
    Y = D + Z[:, 0] + rng.normal(size=n)
    res = grr_ate(
        X=X, Y=Y, basis=_basis(), generator=BPGenerator(C=1.0, omega=0.5, branch_fn=_treated),
        riesz_penalty=None, outcome_link="identity", folds=2, random_state=0,
        estimators=("rw", "arw", "tmle", "ra"),
    )
    assert res.status == "boundary"
    assert set(res.estimates) == {"rw", "arw", "tmle", "ra"}
    assert all(np.isnan(v.estimate) for v in res.estimates.values())
    assert res.diagnostics["failure"]["status"] == "boundary"


def test_evaluation_row_outside_the_bp_domain_is_domain_prediction():
    X, Y = _data(n=400, seed=1)
    fold_ids = _fold_ids(len(X), 2, seed=0)
    # Two extreme rows in fold 0's evaluation set; the training rows are bounded.
    X = np.vstack([X, [[1.0, 1e4], [1.0, -1e4]]])
    Y = np.concatenate([Y, [0.0, 0.0]])
    fold_ids = np.concatenate([fold_ids, [0, 0]])
    res = grr_ate(
        X=X, Y=Y, basis=_basis(), generator=BPGenerator(C=0.0, omega=1.0, branch_fn=_treated),
        riesz_penalty=None, outcome_link="identity", fold_ids=fold_ids,
    )
    assert res.status == "domain_prediction"
    assert np.isnan(res.arw.estimate)
    stage = [s for s in res.fold_status if s[2] == "domain_prediction"][0]
    assert stage[0] == 0 and stage[1] == "prediction"


def test_counterfactual_rows_are_domain_checked():
    """A row inside the domain on its own arm but outside on the toggled arm is caught."""

    X, _ = _data(n=400, seed=2)
    gen = BPGenerator(C=0.0, omega=1.0, branch_fn=_treated)
    basis = _basis().fit(X)
    model = GRRGLM(basis=basis, generator=gen, functional=ATEFunctional(0), penalty=None)
    assert model.fit(X).status == "ok"
    # Control-arm dual coordinate u0(z) = s * (b0 + b1 z) must stay above -2 on the
    # negative branch (s = -1): find z with the control arm outside, treated inside.
    zs = np.linspace(-1e3, 1e3, 200001)
    rows1 = np.column_stack([np.ones_like(zs), zs])
    rows0 = np.column_stack([np.zeros_like(zs), zs])
    ok1 = model.domain_mask(rows1)
    ok0 = model.domain_mask(rows0)
    pick = np.flatnonzero(ok1 & ~ok0)
    assert pick.size > 0
    row = rows1[pick[:1]]
    assert model.domain_mask(row)[0]  # the observed row itself is fine
    rep = _DomainCheckedRepresenter(model)
    vals = ATEFunctional(0).m_from_function(row, predict=rep.predict)
    assert rep.n_outside == 1
    assert np.isnan(vals[0])


def test_outcome_failure_is_a_status():
    X, _ = _data(n=200, seed=4)
    rng = np.random.default_rng(0)
    Yb = (rng.random(len(X)) < 0.5).astype(float)
    res = grr_ate(
        X=X, Y=Yb, basis=_basis(), generator=SquaredGenerator(), outcome_link="logit",
        estimators=("arw",), folds=2, random_state=0, max_iter=1,
    )
    assert res.status == "outcome_optimizer_failure"
    assert np.isnan(res.arw.estimate)


def test_fold_ids_define_the_partition():
    X, Y = _data(n=300, seed=5)
    ids = _fold_ids(len(X), 5, seed=1)
    kw = dict(X=X, Y=Y, basis=_basis(), generator=SquaredGenerator(), outcome_link="identity",
              estimators=("arw",), expose_alpha_values=True)
    res = grr_ate(fold_ids=ids, **kw)
    assert res.status == "ok"
    # The fold-0 representer is fitted on the complement of fold 0 only.
    basis = _basis().fit(X[ids != 0])
    model = GRRGLM(basis=basis, generator=SquaredGenerator(), functional=ATEFunctional(0))
    assert model.fit(X[ids != 0]).status == "ok"
    alpha0 = model.predict_alpha(X[ids == 0])
    assert np.allclose(res.diagnostics["alpha_values"][ids == 0], alpha0, atol=1e-12)
    with pytest.raises(ValueError, match="fold_ids"):
        grr_ate(fold_ids=np.zeros(len(X), dtype=int), **kw)
    with pytest.raises(ValueError, match="fold_ids"):
        grr_ate(fold_ids=ids.astype(float), **kw)


def test_riesz_offset_is_used_in_every_fold():
    X, Y = _data(n=300, seed=6)
    gen = UKLGenerator(C=1.0, branch_fn=_treated)
    # alpha_ref = +-3 gives u_ref = +-log 2 (alpha_ref = +-2 would give u_ref = 0).
    off = offset_from_alpha(gen, lambda X_: np.where(X_[:, 0] == 1.0, 3.0, -3.0))
    kw = dict(X=X, Y=Y, basis=_basis(), generator=gen, outcome_link="identity",
              estimators=("rw",), folds=2, random_state=0, riesz_lam=0.5)
    a = grr_ate(riesz_offset=off, **kw)
    b = grr_ate(**kw)
    assert a.status == b.status == "ok"
    # With a penalty the offset changes the shrinkage target and hence the estimate.
    assert abs(a.rw.estimate - b.rw.estimate) > 1e-6


def test_tmle_standard_error_uses_the_targeted_regression():
    X, Y = _data(n=300, seed=7)
    res = grr_ate(
        X=X, Y=Y, basis=_basis(), generator=SquaredGenerator(), riesz_penalty=None,
        outcome_link="identity", outcome_penalty="l2", outcome_lam=1.0, cross_fit=False,
        estimators=("tmle",),
    )
    basis = _basis().fit(X)
    rr = GRRGLM(basis=basis, generator=SquaredGenerator(), functional=ATEFunctional(0),
                penalty=None)
    assert rr.fit(X).status == "ok"
    alpha = rr.predict_alpha(X)
    # A heavily penalized regression leaves a nonzero targeting step epsilon.
    out = OutcomeGLM(basis=basis, link="identity", penalty="l2", lam=1.0)
    out.fit(X, Y)
    mu = out.predict(X)
    eps = float(np.sum(alpha * (Y - mu)) / np.sum(alpha * alpha))
    m_mu_star = ATEFunctional(0).m_from_predictor(X, out.predict) + eps * ATEFunctional(
        0
    ).m_from_predictor(X, rr.predict_alpha)
    theta = float(np.mean(m_mu_star))
    psi_updated = m_mu_star + alpha * (Y - (mu + eps * alpha)) - theta
    psi_initial = m_mu_star + alpha * (Y - mu) - theta
    se_updated = float(np.sqrt(np.var(psi_updated, ddof=1) / len(Y)))
    se_initial = float(np.sqrt(np.var(psi_initial, ddof=1) / len(Y)))
    assert res.tmle.estimate == pytest.approx(theta, rel=1e-10)
    assert res.tmle.se == pytest.approx(se_updated, rel=1e-10)
    assert abs(eps) > 1e-3
    assert abs(se_updated - se_initial) > 1e-6
