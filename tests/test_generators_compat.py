"""Loss/link compatibility (design doc item K) and route separation (items O, 9-4).

Three invariants are protected here.

1. **Dual linearity.** The GRR objective is
   ``L(beta) = mean(g*(Phi beta)) - mean(M beta)``. Only the first term is
   nonlinear, and the envelope identity ``d g*(v)/d v = alpha = inv_grad(v)``
   makes its beta-gradient ``mean(alpha * Phi)``. Hence

       grad L(beta) = mean(alpha * Phi - M),

   which is exactly the *balancing imbalance* on the working span. A loss and a
   link are "compatible" precisely when this holds: the link must be the
   derivative of the conjugate of the loss generator. (The branch sign ``s``
   must not depend on ``v`` for this to be exact, which is why the generators
   warn when ``branch_fn`` is absent.)

2. **KKT residual and imbalance.** At an interior solution, the optimizer
   diagnostic ``FitResult.kkt_residual`` equals the balancing diagnostic
   ``max_j |mean(alpha * phi_j - M_j)|`` without a penalty, and the penalty
   gradient gives the corresponding shift under regularization. BP and BKL
   have open dual domains. If the fitted linear predictor reaches the numerical
   margin that represents such a boundary, the constrained KKT system replaces
   the unconstrained balance equation. Model selection uses the reported KKT
   residual and the held-out imbalance without treating a boundary solution as
   exact balance.

3. **The BKL logistic-MLE route is not one of them.** ``fit_density_ratio``
   fits BKL as a probabilistic classifier and predicts ``prior * exp(v)``; its
   ``v`` is a logistic logit, not a Bregman dual linear predictor, and
   ``generator.inv_grad`` is never called. It must stay distinguishable
   (``route``) and must never be mixed into the balancing candidates. Bounded
   links are separated by the same principle for a different reason: they change
   the estimand (design section 9-4), so they are never admissible.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from genriesz import (
    GRRGLM,
    ATEFunctional,
    BKLGenerator,
    BoundedBKLGenerator,
    BPGenerator,
    BregmanGenerator,
    GaussianRKHSBasis,
    GRRCVConfig,
    GRRCVResult,
    PolynomialBasis,
    SquaredGenerator,
    TreatmentInteractionBasis,
    UKLGenerator,
    fit_density_ratio,
    grr_ate,
    select_grr_hyperparams,
)


def _treated_branch(x: np.ndarray) -> int:
    """Positive branch for treated rows: alpha > 0 iff D = 1."""

    return int(x[0] == 1.0)


def _make_ate(n: int = 300, d_z: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d_z))
    e = 1.0 / (1.0 + np.exp(-(0.6 * Z[:, 0] - 0.3 * Z[:, 1])))
    D = rng.binomial(1, e, size=n).astype(float)
    X = np.column_stack([D, Z])
    Y = D + Z[:, 0] + 0.5 * Z[:, 1] + rng.normal(scale=0.5, size=n)
    return X, Y


def _design(X: np.ndarray):
    """Fitted basis, functional, and the (Phi, M) pair the balance is stated on."""

    m = ATEFunctional(treatment_index=0)
    basis = TreatmentInteractionBasis(
        base_basis=PolynomialBasis(degree=1, include_bias=True), treatment_index=0
    ).fit(X)
    Phi = np.asarray(basis(X), dtype=float)
    M = np.asarray(m.m_basis_matrix(X, basis), dtype=float)
    return basis, m, Phi, M


def _imbalance(gen, X: np.ndarray, Phi: np.ndarray, M: np.ndarray, beta: np.ndarray):
    """The balancing residual mean(alpha * phi_j - M_j), recomputed from scratch."""

    alpha = gen.inv_grad(X, Phi @ beta)
    return (alpha[:, None] * Phi - M).mean(axis=0)


# The three generators the design names as ATE balancing candidates. BKL is
# excluded on purpose: uncapped it raises at v = 0, and its density-ratio route
# is the logistic classifier tested at the bottom of this file.
BALANCING_GENERATORS = [
    ("sq", lambda: SquaredGenerator(C=0.0)),
    ("ukl", lambda: UKLGenerator(C=1.0, branch_fn=_treated_branch)),
    ("bp", lambda: BPGenerator(C=1.0, omega=0.5, branch_fn=_treated_branch)),
]
_GEN_IDS = [g[0] for g in BALANCING_GENERATORS]


# ---------------------------------------------------------------------------
# 1. Dual linearity: grad_beta of the objective is the balancing imbalance.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, make_gen", BALANCING_GENERATORS, ids=_GEN_IDS)
def test_dual_linearity_gradient_is_the_balancing_imbalance(name, make_gen):
    X, _ = _make_ate(n=200, seed=1)
    _, _, Phi, M = _design(X)
    gen = make_gen()
    p = Phi.shape[1]

    rng = np.random.default_rng(11)
    beta = 0.1 * rng.normal(size=p)

    # Stay in the smooth region so the identity is exercised, not the clip.
    assert not np.any(gen.domain_binding(X, Phi @ beta))

    def objective(b: np.ndarray) -> float:
        g_star, _ = gen.conjugate(X, Phi @ b)
        return float(np.mean(g_star - (M @ b)))

    h = 1e-6
    fd = np.empty(p)
    for j in range(p):
        step = np.zeros(p)
        step[j] = h
        fd[j] = (objective(beta + step) - objective(beta - step)) / (2.0 * h)

    analytic = _imbalance(gen, X, Phi, M, beta)
    rel = np.max(np.abs(fd - analytic) / np.maximum(1.0, np.abs(analytic)))
    assert rel < 1e-5

    # The linear part really is linear: the M-term contributes a constant
    # -mean(M) to the gradient at every beta.
    beta2 = beta + 0.05 * rng.normal(size=p)
    d_alpha = _imbalance(gen, X, Phi, M, beta2) - _imbalance(gen, X, Phi, M, beta)
    alpha1 = gen.inv_grad(X, Phi @ beta)
    alpha2 = gen.inv_grad(X, Phi @ beta2)
    d_expected = ((alpha2 - alpha1)[:, None] * Phi).mean(axis=0)
    assert np.allclose(d_alpha, d_expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. KKT residual == imbalance (unpenalized), and == imbalance + penalty grad.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, make_gen", BALANCING_GENERATORS, ids=_GEN_IDS)
def test_kkt_residual_equals_working_span_imbalance_unpenalized(name, make_gen):
    X, _ = _make_ate(n=300, seed=2)
    basis, m, Phi, M = _design(X)
    gen = make_gen()

    model = GRRGLM(basis=basis, generator=gen, functional=m, penalty=None, lam=0.0)
    fr = model.fit(X, max_iter=2000, tol=1e-12)
    assert fr.success

    delta = _imbalance(gen, X, Phi, M, model.beta_)
    assert fr.kkt_residual == pytest.approx(float(np.max(np.abs(delta))), rel=1e-9, abs=1e-12)
    # And the solution actually balances the working span.
    assert np.max(np.abs(delta)) < 1e-5


@pytest.mark.parametrize("name, make_gen", BALANCING_GENERATORS, ids=_GEN_IDS)
def test_l2_penalty_shifts_the_balance_by_exactly_the_penalty_gradient(name, make_gen):
    X, _ = _make_ate(n=300, seed=3)
    basis, m, Phi, M = _design(X)
    gen = make_gen()
    lam = 1e-2

    model = GRRGLM(basis=basis, generator=gen, functional=m, penalty="l2", lam=lam)
    fr = model.fit(X, max_iter=2000, tol=1e-12)
    assert fr.success

    delta = _imbalance(gen, X, Phi, M, model.beta_)
    stationarity = delta + lam * model.beta_
    if fr.clip_binding_rate == 0.0:
        # Interior stationarity of the penalized objective gives
        # delta = -lam * beta.
        assert np.allclose(delta, -lam * model.beta_, atol=1e-5)
        assert fr.kkt_residual == pytest.approx(
            float(np.max(np.abs(stationarity))), rel=1e-6, abs=1e-9
        )
    else:
        # BP can attain the numerical margin of its open dual domain. The
        # unconstrained gradient then need not vanish, but the constrained KKT
        # residual must still be small.
        assert name == "bp"
        assert fr.kkt_residual < 1e-5
        assert np.max(np.abs(stationarity)) > 1e-4
    # The penalty genuinely moves the balance away from zero, so the previous
    # test's unpenalized identity is not vacuous.
    assert np.max(np.abs(delta)) > 1e-4


def test_compatible_generators_share_the_balance_equations_in_the_interior():
    """Compatible generators use the same equations when no domain bound is active."""

    X, _ = _make_ate(n=300, seed=4)
    basis, m, Phi, M = _design(X)
    boundary_cases = 0

    for name, make_gen in BALANCING_GENERATORS:
        gen = make_gen()
        model = GRRGLM(basis=basis, generator=gen, functional=m, penalty=None, lam=0.0)
        fr = model.fit(X, max_iter=2000, tol=1e-12)
        assert fr.success
        delta = _imbalance(gen, X, Phi, M, model.beta_)
        if fr.clip_binding_rate == 0.0:
            assert np.max(np.abs(delta)) < 1e-5
        else:
            # This seed makes the BP solution attain the numerical margin of
            # its exact dual domain. The constrained KKT condition, rather than
            # unconstrained exact balance, is the relevant first-order result.
            assert name == "bp"
            boundary_cases += 1
            assert fr.kkt_residual < 1e-5
            assert np.max(np.abs(delta)) > 1e-4

    assert boundary_cases == 1


# ---------------------------------------------------------------------------
# 3a. Route separation: the BKL density-ratio path is a logistic classifier.
# ---------------------------------------------------------------------------
def _shifted_samples(n: int = 250, seed: int = 5):
    rng = np.random.default_rng(seed)
    X_den = rng.normal(size=(n, 2))
    X_num = rng.normal(loc=0.5, size=(n, 2))
    return X_num, X_den


def test_bkl_density_ratio_takes_the_logistic_classification_route():
    X_num, X_den = _shifted_samples()
    res = fit_density_ratio(X_num, X_den, generator="bkl", sigma=1.0, n_centers=40, lam=1e-2)

    assert res.route == "logistic_classification"
    assert res.class_prior_ratio is not None
    assert res.class_prior_ratio == pytest.approx(len(X_den) / len(X_num))

    # predict_ratio is prior * exp(v), i.e. the classifier's odds -- not inv_grad.
    v = res.predict_v(X_den)
    expected = res.class_prior_ratio * np.exp(np.clip(v, -700.0, 700.0))
    assert np.allclose(res.predict_ratio(X_den, clip_nonnegative=False), expected)


def test_logistic_route_never_calls_the_generator_link():
    """The docstring promises `generator.inv_grad` is unused on this route."""

    X_num, X_den = _shifted_samples(seed=6)
    res = fit_density_ratio(X_num, X_den, generator="bkl", sigma=1.0, n_centers=40, lam=1e-2)
    assert isinstance(res.generator, BKLGenerator)

    def _boom(*_args, **_kwargs):
        raise AssertionError("inv_grad must not be called on the logistic route")

    res.generator.inv_grad = _boom  # instance attribute shadows the method
    assert np.all(np.isfinite(res.predict_ratio(X_den)))


def test_bregman_route_does_call_the_generator_link():
    """The mirror image: on the Bregman route the link is what produces r_hat."""

    X_num, X_den = _shifted_samples(seed=7)
    res = fit_density_ratio(X_num, X_den, generator="sq", sigma=1.0, n_centers=40, lam=1e-2)

    assert res.route == "bregman"
    assert res.class_prior_ratio is None

    sentinel = np.full(len(X_den), 3.0)
    res.generator.inv_grad = lambda *_a, **_k: sentinel
    assert np.allclose(res.predict_ratio(X_den), sentinel)


def test_bounded_bkl_is_not_misrouted_to_the_classifier():
    """BoundedBKL is not a BKLGenerator, so it keeps the Bregman route."""

    X_num, X_den = _shifted_samples(seed=8)
    gen = BoundedBKLGenerator(C=1.0, alpha_max=20.0, branch_fn=lambda _x: 1)
    res = fit_density_ratio(X_num, X_den, generator=gen, sigma=1.0, n_centers=40, lam=1e-2)

    assert not isinstance(gen, BKLGenerator)
    assert res.route == "bregman"
    assert res.class_prior_ratio is None


def test_route_and_class_prior_ratio_agree():
    """A caller can key off either field; they never disagree."""

    X_num, X_den = _shifted_samples(seed=9)
    for spec in ("sq", "ukl", "bp", "bkl"):
        res = fit_density_ratio(X_num, X_den, generator=spec, sigma=1.0, n_centers=40, lam=1e-2)
        is_classifier = res.route == "logistic_classification"
        assert is_classifier == (res.class_prior_ratio is not None)
        assert is_classifier == (spec == "bkl")


# ---------------------------------------------------------------------------
# 3b. Section 9-4: estimand-modifying generators are never admissible.
# ---------------------------------------------------------------------------
def _select(gen, *, n: int = 250, seed: int = 10, admissibility_thresholds=None):
    X, Y = _make_ate(n=n, seed=seed)
    return select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(treatment_index=0),
        basis=GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0),
        generator=gen,
        config=GRRCVConfig(
            lam_grid=[1e-2, 1e-1],
            cv_folds=2,
            return_path=True,
            random_state=0,
            admissibility_thresholds=admissibility_thresholds,
        ),
        outcome_link="identity",
    )


def test_bounded_generator_is_excluded_from_the_admissible_set():
    gen = BoundedBKLGenerator(C=1e-2, alpha_max=30.0, branch_fn=_treated_branch)
    with pytest.raises(RuntimeError, match="modifying the estimand"):
        _select(gen)


def test_exclusion_holds_even_when_the_bound_never_binds():
    """The screen is the flag, not the binding rate: a loose bound is still out.

    A binding-rate-only screen (``max_cap_binding_rate``) would admit this
    candidate, because ``alpha_max`` is far above any alpha the fit produces.
    Section 9-4 says bounded links are *always* target-sensitivity.

    The effectively-unclipped fit concentrates its weight on a couple of
    observations (max alpha ~1e6, ESS ratio ~0.01), which genuinely violates
    the ESS floor -- a *quality* failure the estimand flag must not mask (audit
    CV-11). This test pins the flag alone, so it disables that floor; the
    quality gating itself is covered by
    ``test_modifies_estimand_does_not_bypass_the_quality_checks``.
    """

    gen = BoundedBKLGenerator(C=1e-2, alpha_max=1e6, branch_fn=_treated_branch)
    with pytest.raises(RuntimeError, match="modifying the estimand"):
        _select(gen, admissibility_thresholds={"min_ess_ratio": None})


def test_unbounded_generator_stays_admissible():
    res = _select(SquaredGenerator(C=0.0))
    assert res.modifies_estimand is False
    assert res.n_admissible >= 1
    assert all(not r["modifies_estimand"] for r in res.path)


def test_all_candidates_failing_raises_without_the_sensitivity_warning():
    """The warning describes "the selection below"; there must be one to describe."""

    X, Y = _make_ate(n=120, seed=13)

    gen = BregmanGenerator(
        g=lambda alpha: alpha * alpha,
        grad=lambda alpha: 2.0 * alpha,
        inv_grad=lambda _value: float("nan"),
        grad2=lambda _alpha: 2.0,
    )
    gen.modifies_estimand = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(
            RuntimeError, match="No Riesz candidate passed the admissibility screen"
        ):
            _select(gen, n=120, seed=13)

    assert not [w for w in caught if "modifies the estimand" in str(w.message)]
    assert not [w for w in caught if "admissibility screen" in str(w.message)]


@pytest.mark.parametrize(
    "make_gen, expected",
    [
        (lambda: SquaredGenerator(C=0.0), False),
        (lambda: BoundedBKLGenerator(C=1e-2, alpha_max=30.0, branch_fn=_treated_branch), True),
    ],
    ids=["sq", "bounded_bkl"],
)
def test_grr_functional_reports_the_estimand_flag(make_gen, expected):
    X, Y = _make_ate(n=250, seed=12)
    with warnings.catch_warnings():
        # A bounded link may legitimately bind; that warning is not under test.
        warnings.simplefilter("ignore", UserWarning)
        res = grr_ate(
            X=X,
            Y=Y,
            basis=GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0),
            generator=make_gen(),
            riesz_lam=1e-2,
            folds=3,
            random_state=0,
        )
    assert res.diagnostics["riesz_modifies_estimand"] is expected


def test_grrcvresult_keeps_its_positional_signature():
    """`modifies_estimand` is keyword-only, so old positional callers still work."""

    assert "modifies_estimand" not in GRRCVResult.__match_args__
    res = GRRCVResult(None, 1e-2, None, "bias_variance", 0.5, 1, 1, [])
    assert res.modifies_estimand is False
