"""Step 5 (fix/kl-cap-and-domain): KL-type link domain / cap correctness.

Design references:
- library_improvement_design.md item E and §10.5.1 (only BKL's domain-violation
  clip broke the conjugate identity, exploding alpha to ~2e8).
- coverage_failure_improvement_design_revised.md "KL系lossとcapの修正設計":
  方針A (default BKLGenerator: exact uncapped link, raise on domain violation)
  and 方針B (BoundedBKLGenerator: bounded smooth link, consistent objective and
  gradient, a target-sensitivity candidate per §9-4).

The central invariant these tests protect: the ``alpha`` returned by
``conjugate`` must equal ``d g*(v)/d v`` wherever the link is used, so that the
GRR objective ``mean(g*(v)) - mean(M beta)`` and its gradient
``mean(alpha * Phi - M)`` are mutually consistent. A post-hoc clip that pins the
pre-image (the old BKL behavior) violates this; raising or a consistent bounded
link does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from genriesz import (
    GRRGLM,
    ATEFunctional,
    BKLGenerator,
    BoundedBKLGenerator,
    BregmanGenerator,
    DomainError,
    PolynomialBasis,
    SquaredGenerator,
    TreatmentInteractionBasis,
)
from genriesz.glm import DomainError as GLMDomainError


def _pos_branch(x: np.ndarray) -> int:
    return 1 if x[0] >= 0.0 else 0


def _make_ate(n: int = 200, d_z: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d_z))
    e = 1.0 / (1.0 + np.exp(-(0.6 * Z[:, 0] - 0.3 * Z[:, 1])))
    D = rng.binomial(1, e, size=n).astype(float)
    return np.column_stack([D, Z])


# ---------------------------------------------------------------------------
# 方針A: default BKL is the exact uncapped link and raises on domain violation.
# ---------------------------------------------------------------------------
def test_domainerror_is_shared_between_generators_and_glm():
    # glm re-exports the generators.DomainError so existing imports keep working.
    assert DomainError is GLMDomainError


def test_bkl_valid_region_is_finite_and_not_exploding():
    gen = BKLGenerator(C=1.0, branch_fn=_pos_branch)
    X = np.ones((5, 2))  # positive branch -> u = v
    v = np.array([-3.0, -1.0, -0.5, -0.2, -0.1])
    _, alpha = gen.conjugate(X, v)
    assert np.all(np.isfinite(alpha))
    # The old clip produced ~2e8 near the boundary; the exact link stays modest.
    assert np.max(np.abs(alpha)) < 1e3


def test_bkl_raises_on_domain_violation_instead_of_clipping():
    gen = BKLGenerator(C=1.0, branch_fn=_pos_branch)
    X = np.ones((4, 2))  # positive branch -> u = v
    # u = v >= 0 is outside the BKL domain; the exact inverse does not exist.
    with pytest.raises(DomainError, match="domain violation"):
        gen.inv_grad(X, np.array([-1.0, -0.5, 0.0, 0.3]))
    with pytest.raises(DomainError):
        gen.conjugate(X, np.full(4, 0.5))


def test_bkl_grrglm_fit_fails_honestly_from_cold_start():
    # beta0 = 0 -> v = 0 sits exactly on the BKL boundary (g*(0) = +inf). The fit
    # must report an explicit domain_error failure, not a silent broken success.
    X = _make_ate(n=150, seed=1)
    m = ATEFunctional(treatment_index=0)
    basis = TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    ).fit(X)
    gen = BKLGenerator(C=1e-2, branch_fn=lambda x: int(x[0] == 1.0))
    res = GRRGLM(functional=m, basis=basis, generator=gen, penalty="l2", lam=1e-3).fit(X)
    assert not res.success
    assert res.status == "domain_error"


# ---------------------------------------------------------------------------
# 方針B: BoundedBKLGenerator is bounded, consistent, and optimizable.
# ---------------------------------------------------------------------------
def test_bounded_bkl_requires_alpha_max_above_C():
    with pytest.raises(ValueError, match="alpha_max"):
        BoundedBKLGenerator(C=1.0, alpha_max=0.5, branch_fn=_pos_branch)


def test_bounded_bkl_alpha_is_bounded_everywhere():
    amax = 25.0
    gen = BoundedBKLGenerator(C=1.0, alpha_max=amax, branch_fn=_pos_branch)
    X = np.ones((7, 2))  # positive branch -> u = v
    # Include deep-valid, near-boundary, and domain-violating v; none may explode.
    v = np.array([-5.0, -1.0, -0.1, -1e-3, 0.0, 0.5, 5.0])
    _, alpha = gen.conjugate(X, v)
    assert np.all(np.isfinite(alpha))
    assert np.max(np.abs(alpha)) <= amax + 1e-9


def test_bounded_bkl_conjugate_identity_including_binding_region():
    # The cap-consistency test: analytic alpha == finite-difference d g*/dv, even
    # where the bound binds (constant alpha there keeps the envelope identity).
    gen = BoundedBKLGenerator(C=1.0, alpha_max=20.0, branch_fn=_pos_branch)
    X = np.ones((11, 2))
    v = np.linspace(-4.0, 0.8, 11)  # spans deep-valid through binding/violation
    _, alpha = gen.conjugate(X, v)
    h = 1e-6
    gp, _ = gen.conjugate(X, v + h)
    gm, _ = gen.conjugate(X, v - h)
    fd = (gp - gm) / (2.0 * h)
    rel = np.max(np.abs(fd - alpha) / np.maximum(1.0, np.abs(alpha)))
    assert rel < 1e-5
    # And the bound genuinely binds somewhere in this range (otherwise the test
    # would not exercise the cap).
    assert np.any(gen.domain_binding(X, v))


def test_bounded_bkl_matches_exact_bkl_in_interior():
    # Where the bound does not bind, the bounded link must equal the exact link.
    exact = BKLGenerator(C=1.0, branch_fn=_pos_branch)
    bounded = BoundedBKLGenerator(C=1.0, alpha_max=1e4, branch_fn=_pos_branch)
    X = np.ones((6, 2))
    v = np.array([-4.0, -2.0, -1.0, -0.5, -0.3, -0.2])  # strictly interior
    assert not np.any(bounded.domain_binding(X, v))
    _, a_exact = exact.conjugate(X, v)
    _, a_bounded = bounded.conjugate(X, v)
    assert np.allclose(a_exact, a_bounded, rtol=1e-9, atol=1e-9)


def test_bounded_bkl_grrglm_gradient_matches_finite_difference():
    # The GRR objective/gradient wiring must be consistent for the bounded link:
    # analytic jac == finite difference of fun at a random (feasible) beta.
    X = _make_ate(n=120, seed=2)
    m = ATEFunctional(treatment_index=0)
    basis = TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    ).fit(X)
    gen = BoundedBKLGenerator(C=1e-2, alpha_max=30.0, branch_fn=lambda x: int(x[0] == 1.0))
    model = GRRGLM(functional=m, basis=basis, generator=gen, penalty="l2", lam=1e-3)

    Phi = np.asarray(basis(X), dtype=float)
    M = np.asarray(m.m_basis_matrix(X, basis), dtype=float)
    p = Phi.shape[1]

    def fun(beta):
        v = Phi @ beta
        g_star, _ = gen.conjugate(X, v)
        return float(np.mean(g_star - (M @ beta))) + model.penalty.value(beta)

    def jac(beta):
        v = Phi @ beta
        _, alpha = gen.conjugate(X, v)
        return (alpha[:, None] * Phi - M).mean(axis=0) + model.penalty.grad(beta)

    rng = np.random.default_rng(3)
    beta = 0.1 * rng.normal(size=p)
    analytic = jac(beta)
    h = 1e-6
    fd = np.empty(p)
    for j in range(p):
        e = np.zeros(p)
        e[j] = h
        fd[j] = (fun(beta + e) - fun(beta - e)) / (2.0 * h)
    rel = np.max(np.abs(fd - analytic) / np.maximum(1.0, np.abs(analytic)))
    assert rel < 1e-5


def test_bounded_bkl_is_optimizable_and_bounded_from_cold_start():
    X = _make_ate(n=200, seed=4)
    m = ATEFunctional(treatment_index=0)
    basis = TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    ).fit(X)
    amax = 40.0
    gen = BoundedBKLGenerator(C=1e-2, alpha_max=amax, branch_fn=lambda x: int(x[0] == 1.0))
    model = GRRGLM(functional=m, basis=basis, generator=gen, penalty="l2", lam=1e-3)
    res = model.fit(X)
    assert res.success
    assert res.status == "converged"
    alpha = model.predict_alpha(X)
    assert np.max(np.abs(alpha)) <= amax + 1e-9


# ---------------------------------------------------------------------------
# §9-4 seam: bounded/capped variants are flagged as modifying the estimand so
# model selection can keep them out of the admissible set.
# ---------------------------------------------------------------------------
def test_modifies_estimand_flag():
    assert BoundedBKLGenerator(C=1.0, alpha_max=10.0, branch_fn=_pos_branch).modifies_estimand
    assert not BKLGenerator(C=1.0, branch_fn=_pos_branch).modifies_estimand
    assert not SquaredGenerator(C=0.0).modifies_estimand
    assert not BregmanGenerator(g=lambda a: a * a).modifies_estimand
