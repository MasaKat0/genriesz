"""GRRGeneralLink: arbitrary loss-link pairs (registration E-13, §1.3 A-4, E-8).

- The gradient equals ``Delta(alpha, psi_j)`` of ``prop:arbitrary_pair_foc`` and
  the finite-difference derivative of the objective; the analytic Hessian
  equals the finite-difference derivative of the gradient.
- A compatible pair reproduces GRRGLM (tangent directions = regressors).
- An incompatible pair converges to a stationary point with zero tangent
  imbalance but nonzero imbalance in the intended regressors.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from genriesz import (
    GRRGLM,
    ATEFunctional,
    BKLGenerator,
    CallableBasis,
    GRRGeneralLink,
    SquaredGenerator,
    UKLGenerator,
)


def _treated(x):
    return int(x[0] == 1.0)


def _dgp13(n: int, seed: int):
    rng = np.random.default_rng(seed)
    a = np.sqrt(3.0)
    Z = rng.uniform(-a, a, size=(n, 3))
    q = Z[:, 1] ** 2 - 1.0
    e = expit(0.75 * (Z[:, 0] - 0.5 * Z[:, 1] + 0.6 * q))
    D = rng.binomial(1, e).astype(float)
    return np.column_stack([D, Z])


def _cols(X):
    X = np.atleast_2d(X)
    d = X[:, [0]]
    F = np.column_stack([np.ones(len(X)), X[:, 1:]])
    return np.concatenate([d * F, (1.0 - d) * F], axis=1)


def _s(X):
    return 2.0 * np.atleast_2d(X)[:, 0] - 1.0


# "logit" link alpha = s (1 + exp(s eta)) and the linear link alpha = s (1 + eta).
def _exp_link(X, eta):
    s = _s(X)
    return s * (1.0 + np.exp(s * eta))


def _exp_dlink(X, eta):
    return np.exp(_s(X) * eta)


def _exp_d2link(X, eta):
    s = _s(X)
    return s * np.exp(s * eta)


def _lin_link(X, eta):
    return _s(X) * (1.0 + eta)


def _lin_dlink(X, eta):
    return _s(X) * np.ones_like(eta)


def _lin_d2link(X, eta):
    return np.zeros_like(eta)


PAIRS = {
    "I-SQexp": (lambda: SquaredGenerator(), _exp_link, _exp_dlink, _exp_d2link),
    "I-BKLexp": (lambda: BKLGenerator(C=1.0, branch_fn=_treated), _exp_link, _exp_dlink,
                 _exp_d2link),
    "I-UKLlin": (lambda: UKLGenerator(C=1.0, branch_fn=_treated), _lin_link, _lin_dlink,
                 _lin_d2link),
    "C-UKL": (lambda: UKLGenerator(C=1.0, branch_fn=_treated), _exp_link, _exp_dlink,
              _exp_d2link),
}


# Registered index offsets (§1.3 A-4): v_ref = link^{-1}(alpha_ref) with |alpha_ref| = 2,
# i.e. 0 for the exponential link and 1 for the linear link s(1 + v).
V_REF = {"I-SQexp": 0.0, "I-BKLexp": 0.0, "I-UKLlin": 1.0, "C-UKL": 0.0}


def _model(name, *, index_offset="registered"):
    make_gen, link, dlink, d2link = PAIRS[name]
    off = V_REF[name] if index_offset == "registered" else index_offset
    return GRRGeneralLink(
        make_gen(), link, dlink, d2link, basis=CallableBasis(_cols), functional=ATEFunctional(0),
        index_offset=off,
    )


def _beta_inside(name: str) -> np.ndarray:
    return np.zeros(8)


@pytest.mark.parametrize("name", list(PAIRS))
def test_gradient_is_the_tangent_imbalance_and_matches_finite_differences(name):
    X = _dgp13(400, seed=0)
    model = _model(name)
    rng = np.random.default_rng(1)
    beta = _beta_inside(name) + 0.05 * rng.normal(size=8)
    grad = model.gradient(X, beta)

    # prop:arbitrary_pair_foc: Delta(alpha, psi_j), psi_j = g''(alpha) link'(eta) phi_j.
    gen = model.generator
    Phi = _cols(X)
    eta = V_REF[name] + Phi @ beta
    alpha = model.link(X, eta)

    def psi_basis(Z):
        Z = np.atleast_2d(Z)
        e = V_REF[name] + _cols(Z) @ beta
        a = model.link(Z, e)
        return (gen.grad2(Z, a) * model.dlink(Z, e))[:, None] * _cols(Z)

    M_psi = ATEFunctional(0).m_basis_matrix(X, CallableBasis(psi_basis))
    explicit = (alpha[:, None] * psi_basis(X) - M_psi).mean(axis=0)
    assert np.allclose(grad, explicit, rtol=1e-12, atol=1e-13)

    h = 1e-6
    fd = np.array([
        (model.objective(X, beta + h * np.eye(8)[j]) - model.objective(X, beta - h * np.eye(8)[j]))
        / (2 * h)
        for j in range(8)
    ])
    assert np.allclose(fd, grad, rtol=1e-6, atol=1e-8)

    H = model.hessian(X, beta)
    fdH = np.column_stack([
        (model.gradient(X, beta + h * np.eye(8)[k]) - model.gradient(X, beta - h * np.eye(8)[k]))
        / (2 * h)
        for k in range(8)
    ])
    assert np.allclose(H, fdH, rtol=1e-5, atol=1e-7)


def test_compatible_pair_reproduces_grrglm():
    X = _dgp13(1000, seed=2)
    model = _model("C-UKL")
    r = model.fit(X)
    assert r.status == "ok"
    glm = GRRGLM(
        basis=CallableBasis(_cols), generator=UKLGenerator(C=1.0, branch_fn=_treated),
        functional=ATEFunctional(0), penalty=None,
    )
    assert glm.fit(X).status == "ok"
    assert np.max(np.abs(r.beta - glm.beta_)) < 1e-8
    # Compatible: tangent directions are the regressors, so both imbalances vanish.
    assert r.tangent_imbalance <= 1e-10 and r.regressor_imbalance <= 1e-9


@pytest.mark.parametrize("name", ["I-SQexp", "I-BKLexp", "I-UKLlin"])
def test_incompatible_pair_balances_tangents_but_not_regressors(name):
    X = _dgp13(1000, seed=7)  # a sample on which all three pairs have interior solutions
    model = _model(name)
    r = model.fit(X, beta0=_beta_inside(name))
    assert r.status == "ok", r.message
    assert r.tangent_imbalance <= 1e-10
    assert r.hessian_min_eig > 0.0
    assert r.regressor_imbalance > 1e-3
    assert np.allclose(
        model.predict_alpha(X[:5]), model.link(X[:5], V_REF[name] + _cols(X[:5]) @ r.beta)
    )


def test_failures_are_statuses_not_exceptions_or_warnings():
    """I-UKLlin at the domain boundary, and I-SQexp on an unbounded sample objective."""

    import warnings

    X = _dgp13(1000, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no floating-point warning may escape
        r_lin = _model("I-UKLlin").fit(X, beta0=_beta_inside("I-UKLlin"))
        r_sq = _model("I-SQexp").fit(X)
    assert r_lin.status == "uncertified_numerical_boundary" and r_lin.n_boundary > 0
    assert r_sq.status in {"linesearch", "nonfinite", "maxit"}
    assert not r_sq.success


def test_i_ukllin_registered_index_offset_gives_the_reference_at_beta_zero():
    """v_ref = 1: beta = 0 is alpha_ref = s * 2, inside |alpha| > 1."""

    X = _dgp13(300, seed=4)
    model = _model("I-UKLlin")
    assert model.objective(X, np.zeros(8)) == model.objective(X, np.zeros(8))  # finite
    model.beta_ = np.zeros(8)
    assert np.allclose(model.predict_alpha(X), 2.0 * (2.0 * X[:, 0] - 1.0))
    # The offset is part of the index everywhere: also at the counterfactual rows.
    X1 = X.copy()
    X1[:, 0] = 1.0 - X1[:, 0]
    assert np.allclose(model.predict_alpha(X1), 2.0 * (2.0 * X1[:, 0] - 1.0))
    # Without the offset, beta = 0 gives |alpha| = 1 = C: not in the open domain.
    bare = _model("I-UKLlin", index_offset=None)
    r = bare.fit(X)
    assert r.status == "infeasible_start"
    with pytest.raises(RuntimeError, match="not fit"):
        bare.predict_alpha(X)


def test_prediction_validates_branch_and_domain():
    """A value on the wrong branch (or inside |alpha| <= C) is a domain failure."""

    from genriesz import DomainError

    X = _dgp13(300, seed=5)
    model = _model("I-UKLlin")
    model.beta_ = np.zeros(8)
    model.beta_[0] = -10.0  # treated arm: v = 1 - 10 < 0 -> alpha = 1 + v < 0 on the + branch
    treated = X[X[:, 0] == 1.0][:3]
    alpha, outside, nonfinite = model.classify(treated)
    assert np.all(outside) and not np.any(nonfinite) and np.all(np.isnan(alpha))
    with pytest.raises(DomainError):
        model.predict_alpha(treated)
    assert np.all(np.isnan(model.predict_alpha(treated, out_of_domain="nan")))


def test_rejects_unsupported_penalty():
    with pytest.raises(ValueError):
        GRRGeneralLink(SquaredGenerator(), _exp_link, _exp_dlink, _exp_d2link,
                       basis=CallableBasis(_cols), functional=ATEFunctional(0), penalty="l1")
