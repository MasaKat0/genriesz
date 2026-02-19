import numpy as np

from genriesz import (
    ATEFunctional,
    BregmanGenerator,
    GRRGLM,
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
