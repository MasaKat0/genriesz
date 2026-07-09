import numpy as np

from genriesz import (
    GRRGLM,
    ATEFunctional,
    BregmanGenerator,
    PolynomialBasis,
    SquaredGenerator,
    TreatmentInteractionBasis,
    UKLGenerator,
)


def _make_synthetic_ate(n: int = 200, d_z: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d_z))
    logits = 0.5 * Z[:, 0] - 0.25 * Z[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n).astype(float)
    X = np.column_stack([D, Z])
    return X


def test_grrglm_runs_with_squared_and_ukl_generators():
    X = _make_synthetic_ate(n=150, d_z=2, seed=0)

    m = ATEFunctional(treatment_index=0)
    psi = PolynomialBasis(degree=1, include_bias=True)
    basis = TreatmentInteractionBasis(base_basis=psi, treatment_index=0)

    # SQ-Riesz
    sq = SquaredGenerator(C=0.0).as_generator()
    model_sq = GRRGLM(functional=m, basis=basis, generator=sq, penalty="l2", lam=1e-3)
    model_sq.fit(X, max_iter=200, tol=1e-9)
    alpha_sq = model_sq.predict_alpha(X)
    assert alpha_sq.shape == (len(X),)
    assert np.all(np.isfinite(alpha_sq))

    # UKL-Riesz
    ukl = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1.0)).as_generator()
    model_ukl = GRRGLM(functional=m, basis=basis, generator=ukl, penalty="l2", lam=1e-3)
    model_ukl.fit(X, max_iter=200, tol=1e-9)
    alpha_ukl = model_ukl.predict_alpha(X)
    assert alpha_ukl.shape == (len(X),)
    assert np.all(np.isfinite(alpha_ukl))


def test_custom_bregman_generator_quadratic_link_matches_identity():
    # g(alpha) = 0.5 * alpha^2 => grad(alpha)=alpha, inv_grad(v)=v
    def g(_x, a: float) -> float:
        return 0.5 * a * a

    def grad(_x, a: float) -> float:
        return a

    def inv_grad(_x, v: float) -> float:
        return v

    X = _make_synthetic_ate(n=80, d_z=2, seed=1)
    m = ATEFunctional(treatment_index=0)
    psi = PolynomialBasis(degree=1, include_bias=True)
    basis = TreatmentInteractionBasis(base_basis=psi, treatment_index=0)

    gen = BregmanGenerator(g=g, grad=grad, inv_grad=inv_grad, name="quad")

    model = GRRGLM(functional=m, basis=basis, generator=gen, penalty="l2", lam=1e-3)
    model.fit(X, max_iter=150, tol=1e-9)
    alpha = model.predict_alpha(X)
    assert np.all(np.isfinite(alpha))


def test_sq_closed_form_matches_numeric_lbfgs_solution():
    # The SQ + l2 case is solved in closed form. A custom generator with the
    # same g/link takes the numeric L-BFGS path, so both must agree.
    X = _make_synthetic_ate(n=200, d_z=2, seed=3)
    m = ATEFunctional(treatment_index=0)

    def make_basis():
        return TreatmentInteractionBasis(
            base_basis=PolynomialBasis(degree=2, include_bias=True), treatment_index=0
        )

    sq = SquaredGenerator(C=0.0)
    model_cf = GRRGLM(functional=m, basis=make_basis(), generator=sq, penalty="l2", lam=1e-2)
    res_cf = model_cf.fit(X)
    assert res_cf.status == "closed_form"
    assert res_cf.success

    gen_numeric = BregmanGenerator(
        g=lambda _x, a: a * a,
        grad=lambda _x, a: 2.0 * a,
        inv_grad=lambda _x, v: 0.5 * v,
        grad2=lambda _x, a: 2.0,
        name="sq-numeric",
    )
    model_num = GRRGLM(
        functional=m, basis=make_basis(), generator=gen_numeric, penalty="l2", lam=1e-2
    )
    res_num = model_num.fit(X, max_iter=500, tol=1e-12)
    assert res_num.success

    assert np.max(np.abs(res_cf.beta - res_num.beta)) < 1e-4
    assert abs(res_cf.objective_value - res_num.objective_value) < 1e-8


def test_fit_result_reports_solution_diagnostics():
    X = _make_synthetic_ate(n=150, d_z=2, seed=4)
    m = ATEFunctional(treatment_index=0)
    basis = TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    )
    model = GRRGLM(
        functional=m, basis=basis, generator=SquaredGenerator(C=0.0), penalty="l2", lam=1e-3
    )
    res = model.fit(X)

    assert np.isfinite(res.objective_value)
    assert np.isfinite(res.gradient_norm) and res.gradient_norm < 1e-8
    assert np.isfinite(res.kkt_residual) and res.kkt_residual < 1e-8
    assert res.clip_binding_rate == 0.0
    assert np.isfinite(res.fit_time)


def test_domain_violation_yields_explicit_failure_not_silent_success():
    # A generator with a bounded domain and no analytic inv_grad: starting
    # far outside the domain used to return success=True with beta unchanged
    # (zero gradient was reported at the infeasible start).
    def g_dom(a: float) -> float:
        if abs(a) >= 1.0:
            return float("nan")
        return -float(np.log(1.0 - a * a))

    X = _make_synthetic_ate(n=60, d_z=2, seed=5)
    m = ATEFunctional(treatment_index=0)
    basis = TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    ).fit(X)
    gen = BregmanGenerator(g=g_dom, name="bounded-domain")
    model = GRRGLM(functional=m, basis=basis, generator=gen, penalty="l2", lam=1e-3)

    bad_start = np.full(basis.n_features, 50.0)
    res = model.fit(X, beta0=bad_start)

    assert not res.success
    assert res.status == "domain_error"
