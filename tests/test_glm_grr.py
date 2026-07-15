import numpy as np
import pytest

from genriesz import (
    GRRGLM,
    ATEFunctional,
    BregmanGenerator,
    PolynomialBasis,
    SquaredGenerator,
    TreatmentInteractionBasis,
    UKLGenerator,
)
from genriesz.functionals import LinearFunctional


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


class _DuplicateColumnBasis:
    """Rank-1 feature map: the two columns are identical."""

    def fit(self, X, y=None):
        return self

    def copy(self):
        return _DuplicateColumnBasis()

    def __call__(self, X):
        return np.ones((len(np.asarray(X, dtype=float)), 2), dtype=float)

    def derivative(self, X, coordinate):
        return np.zeros((len(np.asarray(X, dtype=float)), 2), dtype=float)

    @property
    def n_features(self):
        return 2


class _UnrepresentableFunctional(LinearFunctional):
    """M rows point outside the span of the (rank-1) basis rows."""

    def __init__(self):
        super().__init__(name="unrepresentable")

    def m_basis_matrix(self, X, basis):
        M = np.zeros((len(np.asarray(X, dtype=float)), 2), dtype=float)
        M[:, 0] = 1.0
        return M


# A wide feature map (p = 3 > n = 2): the Gram matrix is rank-deficient in exact
# arithmetic, but its smallest eigenvalue lands at ~1e-16 rather than 0, so
# np.linalg.solve accepts it. This is the case a "did solve raise?" guard misses.
_PHI_WIDE = np.array(
    [
        [-0.98912135, -0.36778665, 1.28792526],
        [0.19397442, 0.92023090, 0.57710379],
    ]
)
_NULL_DIR = np.linalg.svd(_PHI_WIDE, full_matrices=True)[2][-1]


class _WideBasis:
    """Returns the fixed rank-deficient feature matrix above."""

    def fit(self, X, y=None):
        return self

    def copy(self):
        return _WideBasis()

    def __call__(self, X):
        return _PHI_WIDE.copy()

    def derivative(self, X, coordinate):
        return np.zeros_like(_PHI_WIDE)

    @property
    def n_features(self):
        return _PHI_WIDE.shape[1]


class _NullComponentFunctional(LinearFunctional):
    """mean(M) has a component along the null direction of the feature map."""

    def __init__(self):
        super().__init__(name="null-component")

    def m_basis_matrix(self, X, basis):
        return np.tile(_PHI_WIDE.mean(axis=0) + _NULL_DIR, (len(_PHI_WIDE), 1))


def test_singular_closed_form_without_a_stationary_point_fails_instead_of_succeeding():
    """No representer in the span + no penalty -> honest failure, not a fake fit.

    The closed form solves ``(0.5 Phi'Phi/n + lam I) beta = mean(M) - C mean(Phi)``.
    Unlike a least-squares normal equation, the right-hand side need not lie in
    the range of the left, and here it does not: the basis is rank 1 with row
    space span{[1, 1]} while mean(M) = [1, 0]. The objective then has no
    stationary point and runs to -inf along [1, -1]. ``lstsq`` would still hand
    back a finite vector, which used to be reported as ``success=True``.
    """
    X = _make_synthetic_ate(n=40, d_z=2, seed=9)
    model = GRRGLM(
        functional=_UnrepresentableFunctional(),
        basis=_DuplicateColumnBasis(),
        generator=SquaredGenerator(C=0.0),
        penalty=None,  # lam = 0 -> A stays singular
    )
    res = model.fit(X)

    assert not res.success
    assert res.status == "singular"
    assert "unbounded below" in res.message

    # No solution was ever computed, so the model must not be predictable: the
    # old behaviour kept beta_ = beta0 (zeros) and predict_alpha() silently
    # evaluated the generator at that meaningless point.
    assert model.beta_ is None
    with pytest.raises(RuntimeError, match="not fit"):
        model.predict_alpha(X)

    # An l2 penalty makes A positive definite again, and the fit succeeds.
    ok = GRRGLM(
        functional=_UnrepresentableFunctional(),
        basis=_DuplicateColumnBasis(),
        generator=SquaredGenerator(C=0.0),
        penalty="l2",
        lam=1e-3,
    ).fit(X)
    assert ok.success
    assert np.all(np.isfinite(ok.beta))


def test_singular_closed_form_fails_even_when_numpy_solve_does_not_raise():
    """The rank deficiency LAPACK does not report.

    ``_PHI_WIDE`` has p = 3 > n = 2, so its Gram matrix is singular; but rounding
    leaves the smallest eigenvalue at ~1e-16 instead of 0 and ``np.linalg.solve``
    returns a ~1e16-norm vector without raising. Keying the guard on "did solve
    raise?" let this through as ``success=True, status='closed_form'`` with a KKT
    residual of ~1. The check must be on where the range of A actually ends.
    """
    X = np.zeros((len(_PHI_WIDE), 2), dtype=float)  # the basis ignores X

    A = 0.5 * _PHI_WIDE.T @ _PHI_WIDE / len(_PHI_WIDE)
    b = _PHI_WIDE.mean(axis=0) + _NULL_DIR
    assert np.linalg.norm(np.linalg.solve(A, b)) > 1e15  # LAPACK does not object

    res = GRRGLM(
        functional=_NullComponentFunctional(),
        basis=_WideBasis(),
        generator=SquaredGenerator(C=0.0),
        penalty=None,
    ).fit(X)

    assert not res.success
    assert res.status == "singular"


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

    # Same contract as the singular closed form: a fit that never produced a
    # solution must not leave a predictable state behind.
    assert model.beta_ is None
    with pytest.raises(RuntimeError, match="not fit"):
        model.predict_alpha(X)


def test_all_zero_functional_matrix_is_a_loud_failure_not_a_zero_fit():
    """M identically zero => degenerate Riesz problem, not a "successful" beta=0.

    An ATT-style M matrix vanishes on training data with no treated unit; the
    closed form then solves A beta = 0 and used to report the penalty artifact
    beta = 0 as a converged fit, giving alpha == const on the eval fold and a
    deceptively tight downstream CI (audit EST-07 / K-01).
    """
    from genriesz import ATTFunctional

    rng = np.random.default_rng(3)
    n = 40
    Z = rng.normal(size=(n, 2))
    X = np.column_stack([np.zeros(n), Z])  # no treated unit at all

    m = ATTFunctional(treatment_index=0, pi=0.5)
    basis = TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    ).fit(X)
    model = GRRGLM(
        functional=m,
        basis=basis,
        generator=SquaredGenerator(C=0.0).as_generator(),
        penalty="l2",
        lam=1e-3,
    )
    res = model.fit(X)

    assert not res.success
    assert res.status == "degenerate_functional"
    assert "identically zero" in res.message
    assert model.beta_ is None
    with pytest.raises(RuntimeError, match="not fit"):
        model.predict_alpha(X)


def test_optimizer_failure_leaves_the_model_unpredictable():
    """A fit stopped at max_iter must not stay silently predictable (audit P0-07).

    PR #23 fixed the "singular" and "domain_error" paths (no solution ever
    computed); the optimizer-failure path still pinned the last iterate to
    ``beta_`` and let ``predict_alpha`` evaluate it.
    """
    X = _make_synthetic_ate(n=80, d_z=2, seed=4)
    m = ATEFunctional(treatment_index=0)
    basis = TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    ).fit(X)
    # l1 forces the numeric (L-BFGS) path; maxiter=1 cannot converge here.
    model = GRRGLM(
        functional=m,
        basis=basis,
        generator=SquaredGenerator(C=0.0).as_generator(),
        penalty="l1",
        lam=1e-2,
    )
    res = model.fit(X, max_iter=1)

    assert not res.success
    assert res.status == "optimizer_failure"
    # The iterate stays available for diagnostics on the FitResult itself.
    assert np.all(np.isfinite(res.beta))
    assert model.beta_ is None
    with pytest.raises(RuntimeError, match="not fit"):
        model.predict_alpha(X)


def test_outcome_glm_optimizer_failure_leaves_the_model_unpredictable():
    from genriesz import OutcomeGLM

    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 2))
    y = (rng.random(60) < 0.5).astype(float)
    basis = PolynomialBasis(degree=2, include_bias=True).fit(X)
    model = OutcomeGLM(basis=basis, link="logit", penalty="l2", lam=1e-3)
    res = model.fit(X, y, max_iter=1)

    assert not res.success
    assert res.status == "optimizer_failure"
    assert model.theta_ is None
    with pytest.raises(RuntimeError, match="not fit"):
        model.predict(X)
