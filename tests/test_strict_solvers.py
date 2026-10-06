"""Strict solvers (damped Newton, FISTA), offsets, generator identities, statuses.

Registration §1.3 E (items 1-7) for the coefficient problem of GRRGLM:

1. Newton and FISTA reproduce known solutions to 1e-10 (the SQ closed form and
   the n = p = 1 examples of PR-4a, including an active l1 ball).
2. Offset invariance only for offsets absorbable by the span, with lambda = 0.
3. Conjugate and derivative identities of every built-in generator.
4. Basis derivatives (AME) by finite differences.
5./6. Failure statuses instead of exceptions or clipping; held-out domain.
7. No silent clipping (BP in particular), and BKL with C > 0 fits.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.special import expit

from genriesz import (
    GRRGLM,
    ATEFunctional,
    BKLGenerator,
    BPGenerator,
    BregmanGenerator,
    CallableBasis,
    DomainError,
    PolynomialBasis,
    PUGenerator,
    SquaredGenerator,
    TreatmentInteractionBasis,
    UKLGenerator,
    offset_from_alpha,
)
from genriesz.functionals import LinearFunctional
from genriesz.solvers import (
    DualProblem,
    fista_solve,
    l1_kkt_residual,
    newton_solve,
    prox_l1_ball,
)


def _treated(x: np.ndarray) -> int:
    return int(x[0] == 1.0)


def _pos(_x: np.ndarray) -> int:
    return 1


def _probe_ate(n: int, s: float, seed: int):
    """The registration probe's ATE design (bounded covariates, overlap s)."""

    rng = np.random.default_rng(seed)
    a = np.sqrt(3.0)
    Z = rng.uniform(-a, a, size=(n, 3))
    q = Z[:, 1] ** 2 - 1.0
    e = expit(s * (Z[:, 0] - 0.5 * Z[:, 1] + 0.6 * q))
    D = rng.binomial(1, e).astype(float)
    Y = Z[:, 0] + 0.5 * q + D + rng.normal(size=n)
    return np.column_stack([D, Z]), Y


def _interaction_basis(X: np.ndarray):
    return TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    ).fit(X)


class _ConstantFunctional(LinearFunctional):
    """m(W, f) = c * f(X): with phi = 1 this gives mean m(W, phi) = c."""

    def __init__(self, c: float):
        super().__init__(name="const")
        object.__setattr__(self, "c", float(c))

    def m_basis_matrix(self, X, basis):
        return self.c * np.asarray(basis(X), dtype=float)

    def m_from_predictor(self, X, predict):
        return self.c * np.asarray(predict(X), dtype=float)


_ONES = CallableBasis(lambda X: np.ones((np.atleast_2d(X).shape[0], 1)))


# ---------------------------------------------------------------------------
# 1. Known solutions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lam", [0.0, 1e-2])
def test_newton_reproduces_the_sq_closed_form(lam):
    X, _ = _probe_ate(400, 0.5, seed=0)
    basis = _interaction_basis(X)
    m = ATEFunctional(0)
    pen = None if lam == 0.0 else "l2"
    legacy = GRRGLM(
        basis=basis, generator=SquaredGenerator(), functional=m, penalty=pen, lam=lam,
        solver="lbfgs",
    )
    r_cf = legacy.fit(X)
    assert r_cf.status == "closed_form"
    strict = GRRGLM(basis=basis, generator=SquaredGenerator(), functional=m, penalty=pen, lam=lam)
    r = strict.fit(X)
    assert r.status == "ok" and r.solver == "newton"
    assert np.max(np.abs(r.beta - r_cf.beta)) < 1e-10
    assert r.kkt_residual <= 1e-10 * max(1.0, float(np.max(np.abs(r.beta))))


def test_fista_reproduces_the_scalar_lasso_solution():
    """SQ, phi = 1, mean m = 1: minimize beta^2/4 - beta + lam |beta| -> beta = 2(1 - lam)."""

    lam = 0.3
    X = np.zeros((5, 1))
    model = GRRGLM(
        basis=_ONES, generator=SquaredGenerator(), functional=_ConstantFunctional(1.0),
        penalty="l1", lam=lam,
    )
    r = model.fit(X)
    assert r.status == "ok" and r.solver == "fista"
    assert abs(r.beta[0] - 2.0 * (1.0 - lam)) < 1e-10
    assert r.kkt_residual < 1e-10
    # alpha_hat = 1 - lam and the imbalance equals -lam (PR-4a M7 example).
    assert np.allclose(model.predict_alpha(X), 1.0 - lam, atol=1e-10)
    assert abs(r.max_abs_imbalance - lam) < 1e-10


def test_fista_with_an_active_l1_ball_reproduces_pr4a_example():
    """PR-4a [M5]: g(a) = a^2/2, u_ref = 0, mean m = 3, lam = 1/2, B0 = [-1, 1].

    The solution is beta = 1 with imbalance -2, so the ball is active with
    multiplier mu = 3/2 and the constrained KKT residual is zero.
    """

    gen = BregmanGenerator(
        g=lambda a: 0.5 * a * a,
        grad=lambda a: a,
        inv_grad=lambda v: v,
        grad2=lambda a: np.ones_like(a),
        name="half-square",
    )
    X = np.zeros((1, 1))
    model = GRRGLM(
        basis=_ONES, generator=gen, functional=_ConstantFunctional(3.0),
        penalty="l1", lam=0.5, l1_radius=1.0,
    )
    r = model.fit(X)
    assert r.status == "ok"
    assert abs(r.beta[0] - 1.0) < 1e-10
    assert r.l1_ball_active
    assert abs(r.ball_multiplier - 1.5) < 1e-10
    assert r.kkt_residual < 1e-10
    assert abs(r.max_abs_imbalance - 2.0) < 1e-10
    # Without the ball the unrestricted lasso solution is 5/2.
    free = GRRGLM(
        basis=_ONES, generator=gen, functional=_ConstantFunctional(3.0), penalty="l1", lam=0.5
    ).fit(X)
    assert abs(free.beta[0] - 2.5) < 1e-10 and not free.l1_ball_active


def test_prox_scales_the_l1_threshold_by_the_step():
    z = np.array([3.0, -1.0, 0.2])
    assert np.allclose(prox_l1_ball(z, 0.5, 1.0, None), [2.5, -0.5, 0.0])
    # Ball active: threshold is the ball level, larger than step * lam.
    w = prox_l1_ball(z, 0.5, 1.0, 1.0)
    assert abs(np.sum(np.abs(w)) - 1.0) < 1e-12
    assert np.allclose(w, [1.0, 0.0, 0.0])


def test_l1_kkt_residual_uses_the_normal_cone_only_when_the_ball_is_active():
    beta = np.array([1.0, 0.0])
    grad = np.array([-2.0, 0.1])
    r_free, act, _ = l1_kkt_residual(grad, beta, 0.5, None)
    assert not act and abs(r_free - 1.5) < 1e-15
    r_ball, act, mu = l1_kkt_residual(grad, beta, 0.5, 1.0)
    assert act and abs(r_ball) < 1e-15 and abs(mu - 1.5) < 1e-15


@pytest.mark.parametrize("radius", [None, 2.0])
def test_fista_matches_a_reference_lasso_on_an_ate_design(radius):
    """UKL ATE lasso: KKT certified and domain kept; ridge-free reference by tiny steps."""

    X, _ = _probe_ate(600, 0.75, seed=1)
    basis = _interaction_basis(X)
    gen = UKLGenerator(C=1.0, branch_fn=_treated)
    off = offset_from_alpha(gen, lambda X_: np.where(X_[:, 0] == 1.0, 2.0, -2.0))
    lam = 0.02
    model = GRRGLM(
        basis=basis, generator=gen, functional=ATEFunctional(0), penalty="l1", lam=lam,
        offset=off, l1_radius=radius,
    )
    r = model.fit(X)
    assert r.status == "ok"
    assert r.kkt_residual <= 1e-6 * lam
    # Subgradient conditions directly: |Delta_j| <= lam (+ ball multiplier).
    Phi = basis(X)
    M = ATEFunctional(0).m_basis_matrix(X, basis)
    alpha = model.predict_alpha(X)
    delta = (alpha[:, None] * Phi - M).mean(axis=0)
    lam_eff = lam + r.ball_multiplier
    assert np.all(np.abs(delta) <= lam_eff + 1e-9)
    on = r.beta != 0.0
    assert np.allclose(delta[on], -lam_eff * np.sign(r.beta[on]), atol=1e-9)
    # The Newton polish on the support brings the residual to ~1e-10.
    assert r.polished and r.kkt_residual < 1e-10
    if radius is not None:
        assert np.sum(np.abs(r.beta)) <= radius + 1e-12


# ---------------------------------------------------------------------------
# 2. Offset invariance (absorbable offset, lambda = 0) -- and its failure otherwise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "make_gen",
    [
        lambda: SquaredGenerator(),
        lambda: UKLGenerator(C=1.0, branch_fn=_treated),
        lambda: BPGenerator(C=0.0, omega=0.5, branch_fn=_treated),
    ],
    ids=["sq", "ukl", "bp"],
)
def test_absorbable_offset_leaves_alpha_unchanged_without_penalty(make_gen):
    X, _ = _probe_ate(500, 0.5, seed=2)
    basis = _interaction_basis(X)
    gen = make_gen()
    # Arm-specific constants lie in the span (the basis has per-arm intercepts).
    off = lambda X_: np.where(X_[:, 0] == 1.0, 0.3, -0.2)  # noqa: E731
    r0 = GRRGLM(basis=basis, generator=gen, functional=ATEFunctional(0), penalty=None)
    r1 = GRRGLM(basis=basis, generator=gen, functional=ATEFunctional(0), penalty=None, offset=off)
    assert r0.fit(X).status == "ok" and r1.fit(X).status == "ok"
    Xt, _ = _probe_ate(200, 0.5, seed=3)
    assert np.max(np.abs(r0.predict_alpha(Xt) - r1.predict_alpha(Xt))) < 1e-9

    # With a penalty the offset moves the shrinkage target, so alpha changes.
    p0 = GRRGLM(basis=basis, generator=gen, functional=ATEFunctional(0), penalty="l2", lam=0.5)
    p1 = GRRGLM(
        basis=basis, generator=gen, functional=ATEFunctional(0), penalty="l2", lam=0.5,
        offset=off,
    )
    assert p0.fit(X).status == "ok" and p1.fit(X).status == "ok"
    assert np.max(np.abs(p0.predict_alpha(Xt) - p1.predict_alpha(Xt))) > 1e-3


def test_offset_enters_predictions_at_held_out_and_counterfactual_rows():
    X, _ = _probe_ate(300, 0.5, seed=4)
    basis = _interaction_basis(X)
    gen = UKLGenerator(C=1.0, branch_fn=_treated)
    off = lambda X_: 0.1 * X_[:, 1]  # noqa: E731  (not in the span? it is: z1 * arm)
    model = GRRGLM(basis=basis, generator=gen, functional=ATEFunctional(0), lam=1e-2, offset=off)
    assert model.fit(X).status == "ok"
    Xt = X[:10].copy()
    Xt[:, 0] = 1.0 - Xt[:, 0]
    v = model.predict_v(Xt)
    assert np.allclose(v, off(Xt) + basis(Xt) @ model.beta_)


# ---------------------------------------------------------------------------
# 3. Conjugate and derivative identities for every generator
# ---------------------------------------------------------------------------
GENERATORS = [
    ("sq", lambda: SquaredGenerator(C=0.3), np.array([-1.5, -0.2, 0.4, 2.0])),
    ("ukl0", lambda: UKLGenerator(C=0.0, branch_fn=_pos), np.array([-2.0, -0.5, 0.3, 1.5])),
    ("ukl1", lambda: UKLGenerator(C=1.0, branch_fn=_pos), np.array([-2.0, -0.5, 0.3, 1.5])),
    ("bkl", lambda: BKLGenerator(C=0.7, branch_fn=_pos), np.array([-3.0, -1.0, -0.4, -0.05])),
    ("bp05", lambda: BPGenerator(C=0.0, omega=0.5, branch_fn=_pos),
     np.array([-2.5, -1.0, 0.0, 2.0])),
    ("bp2", lambda: BPGenerator(C=1.0, omega=2.0, branch_fn=_pos),
     np.array([-1.2, -0.3, 0.5, 2.0])),
    ("pu", lambda: PUGenerator(C=0.8, branch_fn=_pos), np.array([-3.0, -0.5, 0.2, 2.5])),
]


@pytest.mark.parametrize("name, make_gen, v", GENERATORS, ids=[g[0] for g in GENERATORS])
def test_conjugate_and_derivative_identities(name, make_gen, v):
    gen = make_gen()
    X = np.ones((v.size, 1))
    assert np.all(gen.link_domain(X, v))
    g_star, alpha, dalpha = gen.dual_eval(X, v)
    # g*(v) = v alpha - g(alpha) and alpha = (g')^{-1}(v).
    assert np.allclose(g_star, v * alpha - gen.g(X, alpha), rtol=1e-10, atol=1e-12)
    assert np.allclose(gen.grad(X, alpha), v, rtol=1e-10, atol=1e-12)
    assert np.allclose(gen.inv_grad(X, v), alpha, rtol=1e-12, atol=1e-14)
    h = 1e-6
    gp, ap, _ = gen.dual_eval(X, v + h)
    gm, am, _ = gen.dual_eval(X, v - h)
    # (g*)'(v) = alpha and d alpha/dv = 1/g''(alpha).
    assert np.allclose((gp - gm) / (2 * h), alpha, rtol=1e-6, atol=1e-8)
    assert np.allclose((ap - am) / (2 * h), dalpha, rtol=1e-6, atol=1e-8)
    assert np.allclose(dalpha, 1.0 / gen.grad2(X, alpha), rtol=1e-10)
    # g' = dg/dalpha, g'' = dg'/dalpha, g''' = dg''/dalpha.
    ha = 1e-6 * np.maximum(1.0, np.abs(alpha))
    fd1 = (gen.g(X, alpha + ha) - gen.g(X, alpha - ha)) / (2 * ha)
    fd2 = (gen.grad(X, alpha + ha) - gen.grad(X, alpha - ha)) / (2 * ha)
    fd3 = (gen.grad2(X, alpha + ha) - gen.grad2(X, alpha - ha)) / (2 * ha)
    assert np.allclose(fd1, gen.grad(X, alpha), rtol=1e-6, atol=1e-7)
    assert np.allclose(fd2, gen.grad2(X, alpha), rtol=1e-6, atol=1e-7)
    assert np.allclose(fd3, gen.grad3(X, alpha), rtol=1e-5, atol=1e-6)


def test_ukl_and_bp_require_nonnegative_shift():
    with pytest.raises(ValueError):
        UKLGenerator(C=-0.1, branch_fn=_pos)
    with pytest.raises(ValueError):
        BPGenerator(C=-0.1, branch_fn=_pos)
    with pytest.raises(ValueError):
        BKLGenerator(C=0.0, branch_fn=_pos)


def test_sq_is_excluded_from_the_boundary_rule():
    gen = SquaredGenerator(C=0.0)
    X = np.ones((3, 1))
    alpha = np.array([0.0, 1e-12, -3.0])  # alpha = 0 is interior for SQ
    assert not np.any(gen.boundary_mask(X, 2.0 * alpha, alpha))
    # ... and an SQ fit whose representer crosses zero is "ok".
    X2, _ = _probe_ate(300, 0.5, seed=5)
    basis = _interaction_basis(X2)
    r = GRRGLM(basis=basis, generator=gen, functional=ATEFunctional(0), penalty=None).fit(X2)
    assert r.status == "ok" and r.n_boundary == 0


# ---------------------------------------------------------------------------
# 4. Basis derivatives (AME) and the representer derivative
# ---------------------------------------------------------------------------
def test_basis_and_representer_derivatives_match_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    basis = PolynomialBasis(degree=3, include_bias=True).fit(X)
    h = 1e-6
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        fd = (basis(X + e) - basis(X - e)) / (2 * h)
        assert np.allclose(basis.derivative(X, k), fd, rtol=1e-6, atol=1e-6)

    from genriesz import AMEFunctional

    Y = X[:, 0] + rng.normal(size=50)
    del Y
    gen = SquaredGenerator()

    class _Off:
        def __call__(self, X_):
            return 0.2 * np.sin(X_[:, 1])

        def derivative(self, X_, c):
            return 0.2 * np.cos(X_[:, 1]) if c == 1 else np.zeros(len(X_))

    model = GRRGLM(basis=basis, generator=gen, functional=AMEFunctional(0), lam=1e-2, offset=_Off())
    assert model.fit(X).status == "ok"
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        fd = (model.predict_alpha(X + e) - model.predict_alpha(X - e)) / (2 * h)
        assert np.allclose(model.derivative_alpha(X, k), fd, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 5./6. Statuses, held-out domain
# ---------------------------------------------------------------------------
def test_bkl_without_offset_is_infeasible_start_and_with_offset_fits():
    X, _ = _probe_ate(400, 0.5, seed=6)
    basis = _interaction_basis(X)
    gen = BKLGenerator(C=1.0, branch_fn=_treated)
    r = GRRGLM(basis=basis, generator=gen, functional=ATEFunctional(0), penalty=None).fit(X)
    assert r.status == "infeasible_start"
    off = offset_from_alpha(gen, lambda X_: np.where(X_[:, 0] == 1.0, 2.0, -2.0))
    r = GRRGLM(
        basis=basis, generator=gen, functional=ATEFunctional(0), penalty=None, offset=off
    ).fit(X)
    assert r.status == "ok"


@pytest.mark.parametrize("n", [1000, 4000])
@pytest.mark.parametrize("s", [0.5, 1.25])
def test_bkl_with_positive_c_fits_the_probe_design_without_domain_errors(n, s):
    """The registration probe found BKL(C=1) failing with domain_error in every case."""

    X, _ = _probe_ate(n, s, seed=1)

    def cols(X_):
        X_ = np.atleast_2d(X_)
        d = X_[:, [0]]
        z = X_[:, 1:]
        F = np.concatenate([np.ones((len(X_), 1)), z, z[:, [1]] ** 2 - 1.0], axis=1)
        return np.concatenate([d * F, (1 - d) * F], axis=1)

    gen = BKLGenerator(C=1.0, branch_fn=_treated)
    off = offset_from_alpha(gen, lambda X_: np.where(X_[:, 0] == 1.0, 2.0, -2.0))
    model = GRRGLM(
        basis=CallableBasis(cols), generator=gen, functional=ATEFunctional(0),
        penalty=None, offset=off,
    )
    r = model.fit(X)
    assert r.status == "ok", r.message
    M = ATEFunctional(0).m_basis_matrix(X, CallableBasis(cols))
    scale = max(1.0, float(np.max(np.abs(M.mean(axis=0)))))
    assert r.kkt_residual <= 1e-10 * scale
    assert r.max_abs_imbalance <= 1e-10 * scale


def test_newton_reports_maxit_and_linesearch():
    X, _ = _probe_ate(300, 1.25, seed=7)
    basis = _interaction_basis(X)
    gen = UKLGenerator(C=1.0, branch_fn=_treated)
    model = GRRGLM(basis=basis, generator=gen, functional=ATEFunctional(0), penalty=None)
    r = model.fit(X, max_iter=1)
    assert r.status == "maxit" and model.beta_ is None
    problem = DualProblem(
        generator=gen, X=X, Phi=basis(X), offset=np.zeros(len(X)),
        target=ATEFunctional(0).m_basis_matrix(X, basis).mean(axis=0),
    )
    r = newton_solve(problem, beta0=np.zeros(basis.n_features), max_halvings=0, armijo=0.9999)
    assert r.status == "linesearch"


def test_nonfinite_offset_is_reported():
    X, _ = _probe_ate(100, 0.5, seed=8)
    basis = _interaction_basis(X)
    model = GRRGLM(
        basis=basis, generator=SquaredGenerator(), functional=ATEFunctional(0),
        offset=lambda X_: np.full(len(X_), np.nan),
    )
    assert model.fit(X).status == "nonfinite"


def test_prediction_outside_the_bp_domain_raises_or_returns_nan():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(30), rng.uniform(0.0, 1.0, 30)])
    gen = BPGenerator(C=0.0, omega=1.0, branch_fn=_pos)  # u > -2 on the positive branch
    basis = CallableBasis(lambda X_: np.atleast_2d(X_)[:, 1:2])
    model = GRRGLM(basis=basis, generator=gen, functional=_ConstantFunctional(0.6), penalty=None)
    r = model.fit(X)
    assert r.status == "ok"
    beta = float(model.beta_[0])
    assert beta < 0.0  # small target mean -> negative slope
    x_far = np.array([[1.0, -3.0 / beta]])  # u = beta * x = -3 < -2: outside the domain
    assert not model.domain_mask(x_far)[0]
    with pytest.raises(DomainError):
        model.predict_alpha(x_far)
    assert np.isnan(model.predict_alpha(x_far, out_of_domain="nan")[0])


# ---------------------------------------------------------------------------
# 7. No silent clipping
# ---------------------------------------------------------------------------
def test_built_in_links_are_exact_by_default_and_clip_only_on_request():
    X = np.ones((1, 1))
    ukl = UKLGenerator(C=1.0, branch_fn=_pos)
    assert ukl.inv_grad(X, np.array([-40.0]))[0] - 1.0 == pytest.approx(np.exp(-40.0), rel=1e-12)
    assert UKLGenerator(C=1.0, branch_fn=_pos, legacy_clip=True).inv_grad(
        X, np.array([-40.0])
    )[0] - 1.0 == pytest.approx(1e-12)
    bp = BPGenerator(C=1.0, omega=0.5, branch_fn=_pos)  # k = 3
    with pytest.raises(DomainError):
        bp.inv_grad(X, np.array([-3.5]))
    legacy = BPGenerator(C=1.0, omega=0.5, branch_fn=_pos, legacy_clip=True)
    assert legacy.inv_grad(X, np.array([-3.5]))[0] == pytest.approx(1.0 + 1e-12)
    pu = PUGenerator(C=1.0, branch_fn=_pos)
    assert pu.inv_grad(X, np.array([-30.0]))[0] == pytest.approx(expit(-30.0), rel=1e-12)
    assert np.isnan(ukl.g(X, np.array([0.5]))[0])  # |alpha| < C is outside the domain


def test_bp_no_longer_clips_silently_on_the_probe_design():
    """Probe: legacy BP(C=1) 'converged' with 25-31% of rows clipped (weak overlap).

    The strict solver either reaches an interior KKT point or says "boundary";
    it never reports success with a clipped link.
    """

    def cols(X_):
        X_ = np.atleast_2d(X_)
        d = X_[:, [0]]
        z = X_[:, 1:]
        F = np.concatenate([np.ones((len(X_), 1)), z, z[:, [1]] ** 2 - 1.0], axis=1)
        return np.concatenate([d * F, (1 - d) * F], axis=1)

    X, _ = _probe_ate(1000, 1.25, seed=1)
    basis = CallableBasis(cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = GRRGLM(
            basis=basis,
            generator=BPGenerator(C=1.0, omega=0.5, branch_fn=_treated, legacy_clip=True),
            functional=ATEFunctional(0),
            penalty=None,
            solver="lbfgs",
        ).fit(X, max_iter=2000, tol=1e-12)
    assert legacy.status == "converged" and legacy.clip_binding_rate > 0.2

    strict = GRRGLM(
        basis=basis,
        generator=BPGenerator(C=1.0, omega=0.5, branch_fn=_treated),
        functional=ATEFunctional(0),
        penalty=None,
    ).fit(X)
    assert strict.status == "boundary"
    assert not strict.success and strict.n_boundary > 0


def test_pr4a_bp_example_reports_boundary_not_success():
    """PR-4a §8 [S2] remark: BP(omega=1, C=0), n=2, phi=(1,-1), b=3/2, lambda=0.

    On B_dom = (-2, 2) the objective beta^2/4 + 1 - 3 beta/2 decreases toward the
    boundary, so no GRR minimizer exists. The solver must say "boundary".
    """

    gen = BPGenerator(C=0.0, omega=1.0, branch_fn=_pos)
    X = np.array([[1.0], [-1.0]])
    problem = DualProblem(
        generator=gen, X=X, Phi=X.copy(), offset=np.zeros(2), target=np.array([1.5])
    )
    r = newton_solve(problem, beta0=np.zeros(1))
    assert r.status == "boundary"
    assert 2.0 - r.beta[0] < 1e-7
    assert np.min(np.abs(r.alpha)) <= 1e-8
    r = fista_solve(problem, beta0=np.zeros(1), lam=0.0)
    assert r.status == "boundary"
