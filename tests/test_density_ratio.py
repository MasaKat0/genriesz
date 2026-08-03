import numpy as np
import pytest

from genriesz import fit_density_ratio


def test_fit_density_ratio_basic_shapes():
    rng = np.random.default_rng(0)
    X_num = rng.normal(loc=0.0, scale=1.0, size=(200, 2))
    X_den = rng.normal(loc=0.5, scale=1.0, size=(200, 2))

    res = fit_density_ratio(
        X_num,
        X_den,
        n_centers=50,
        sigma=1.0,
        lam=1e-2,
        cv=False,
        random_state=0,
        max_iter=100,
        tol=1e-8,
    )

    r_den = res.predict_ratio(X_den)
    r_num = res.predict_ratio(X_num)

    assert r_den.shape == (200,)
    assert r_num.shape == (200,)
    assert np.all(np.isfinite(r_den))
    assert np.all(np.isfinite(r_num))


def test_fit_density_ratio_cv_runs():
    rng = np.random.default_rng(1)
    X_num = rng.normal(loc=0.0, scale=1.0, size=(120, 1))
    X_den = rng.normal(loc=0.7, scale=1.0, size=(120, 1))

    res = fit_density_ratio(
        X_num,
        X_den,
        n_centers=30,
        sigma_grid=[0.3, 0.6, 1.0],
        lam_grid=[1e-3, 1e-2],
        cv=True,
        folds=3,
        random_state=1,
        max_iter=80,
        tol=1e-6,
    )

    assert res.sigma in {0.3, 0.6, 1.0}
    assert res.lam in {1e-3, 1e-2}
    assert res.centers.ndim == 2


def test_fit_density_ratio_ukl_can_return_values_below_one():
    rng = np.random.default_rng(2)
    X_num = rng.normal(loc=0.0, scale=1.0, size=(160, 1))
    X_den = rng.normal(loc=1.0, scale=1.0, size=(160, 1))

    res = fit_density_ratio(
        X_num,
        X_den,
        generator="ukl",
        n_centers=40,
        sigma=1.0,
        lam=1e-2,
        cv=False,
        random_state=2,
        max_iter=120,
        tol=1e-6,
    )

    vals = res.predict_ratio(np.array([[1.5], [2.0]]), clip_nonnegative=True)
    assert np.any(vals < 1.0)


def test_fit_density_ratio_bkl_can_return_values_below_one():
    rng = np.random.default_rng(3)
    X_num = rng.normal(loc=0.0, scale=1.0, size=(180, 1))
    X_den = rng.normal(loc=1.0, scale=1.0, size=(180, 1))

    res = fit_density_ratio(
        X_num,
        X_den,
        generator="bkl",
        n_centers=40,
        sigma=1.0,
        lam=1e-2,
        cv=False,
        random_state=3,
        max_iter=120,
        tol=1e-6,
    )

    vals = res.predict_ratio(np.array([[1.5], [2.0]]), clip_nonnegative=True)
    assert np.any(vals < 1.0)


def test_fit_density_ratio_reports_route():
    rng = np.random.default_rng(9)
    X_num = rng.normal(0.2, 1.0, size=(80, 1))
    X_den = rng.normal(0.0, 1.0, size=(80, 1))

    res_sq = fit_density_ratio(X_num, X_den, n_centers=20, sigma=1.0, lam=1e-2)
    assert res_sq.route == "bregman"

    # BKL is fit as a probabilistic classifier; the result must say so because
    # predictions then bypass generator.inv_grad entirely.
    res_bkl = fit_density_ratio(
        X_num, X_den, generator="bkl", n_centers=20, sigma=1.0, lam=1e-2
    )
    assert res_bkl.route == "logistic_classification"


def test_fit_density_ratio_cv_handles_more_centers_than_fold_size():
    rng = np.random.default_rng(10)
    X_num = rng.normal(0.0, 1.0, size=(60, 1))
    X_den = rng.normal(0.5, 1.0, size=(60, 1))

    # n_centers larger than any training fold: fold-local center selection
    # must cap at the fold size instead of reaching into validation data.
    res = fit_density_ratio(
        X_num,
        X_den,
        n_centers=500,
        sigma_grid=[0.5, 1.0],
        lam_grid=[1e-2],
        cv=True,
        folds=3,
        random_state=0,
    )
    assert res.sigma in {0.5, 1.0}


def test_singular_but_solvable_closed_form_returns_the_minimum_norm_minimizer():
    """``0.5 H + lam I`` singular but ``b`` in its range: minimizers exist."""
    from genriesz.density_ratio import _Penalty, _solve_squared_closed_form

    # Every row identical -> rank 1. b = mean(Phi_num) = [1,1,1] lies in the
    # range of A = 0.5 * ones(3,3) / 1, so the minimum-norm solution is a real
    # minimizer and must be returned rather than rejected.
    Phi_den = np.ones((8, 3), dtype=float)
    Phi_num = np.ones((6, 3), dtype=float)

    A = 0.5 * (Phi_den.T @ Phi_den) / len(Phi_den)
    b = Phi_num.mean(axis=0)

    beta = _solve_squared_closed_form(
        Phi_num=Phi_num,
        Phi_den=Phi_den,
        C=0.0,
        penalty=_Penalty(None, lam=0.0, p_norm=2.0),
    )
    assert np.all(np.isfinite(beta))
    # It really solves the stationarity condition, which is what makes it a minimizer.
    np.testing.assert_allclose(A @ beta, b, atol=1e-10)
    # And it is the minimum-norm one: the minimizers are {beta : sum(beta) = 2},
    # whose smallest member is the constant vector.
    np.testing.assert_allclose(beta, [2 / 3, 2 / 3, 2 / 3], atol=1e-10)


def test_singular_and_unsolvable_closed_form_raises_instead_of_returning_a_non_solution():
    """``b`` outside the range of ``A``: no stationary point, unbounded below.

    ``lstsq`` still returns a finite vector here -- the least-squares fit of an
    unsolvable system -- but it does not solve ``A beta = b``, and the objective
    runs to -inf along the null space of ``A``. Returning it would dress a
    divergent candidate up as a successful fit.
    """
    from genriesz.density_ratio import _Penalty, _solve_squared_closed_form

    # A = [[0.5, 0], [0, 0]], b = [1, 1]: the second equation reads 0 = 1.
    Phi_den = np.array([[1.0, 0.0], [1.0, 0.0]])
    Phi_num = np.array([[1.0, 1.0]])

    with pytest.raises(np.linalg.LinAlgError, match="unbounded below"):
        _solve_squared_closed_form(
            Phi_num=Phi_num,
            Phi_den=Phi_den,
            C=0.0,
            penalty=_Penalty(None, lam=0.0, p_norm=2.0),
        )


def test_cv_excludes_a_failed_candidate_without_substitution(monkeypatch):
    import genriesz.density_ratio as dr

    real = dr._solve_squared_closed_form_status

    def failed_status(*, Phi_num, Phi_den, C, penalty):
        if penalty.lam == 1e-3:
            return dr._DensityRatioFit(
                beta=None,
                success=False,
                status="singular",
                message="Singular matrix",
            )
        return real(Phi_num=Phi_num, Phi_den=Phi_den, C=C, penalty=penalty)

    monkeypatch.setattr(dr, "_solve_squared_closed_form_status", failed_status)

    rng = np.random.default_rng(11)
    X_num = rng.normal(0.0, 1.0, size=(60, 1))
    X_den = rng.normal(0.5, 1.0, size=(60, 1))

    res = fit_density_ratio(
        X_num,
        X_den,
        generator="sq",
        n_centers=20,
        sigma_grid=[1.0],
        lam_grid=[1e-3, 1e-1],
        cv=True,
        folds=3,
        random_state=0,
    )

    assert res.lam == 1e-1
    assert np.all(np.isfinite(res.beta))
    assert res.n_failed_candidates == 1
    failed = [row for row in res.cv_path if not bool(row["success"])]
    assert len(failed) == 1
    assert failed[0]["lam"] == 1e-3
    assert failed[0]["fold_status"] == ("singular", "singular", "singular")


def test_cv_excludes_a_candidate_with_invalid_validation_scores(monkeypatch):
    import genriesz.density_ratio as dr
    from genriesz import SquaredGenerator
    from genriesz.generators import ConjugateEvaluation

    real_status = SquaredGenerator.conjugate_status
    calls = {"n": 0}

    def invalid_then_valid(self, X, v):
        calls["n"] += 1
        if calls["n"] <= 3:
            values = np.full(len(np.asarray(v).reshape(-1)), np.nan, dtype=float)
            alpha = np.full_like(values, np.nan)
            valid = np.zeros(values.shape, dtype=bool)
            return ConjugateEvaluation(conjugate=values, alpha=alpha, valid=valid)
        return real_status(self, X, v)

    monkeypatch.setattr(dr.SquaredGenerator, "conjugate_status", invalid_then_valid)

    rng = np.random.default_rng(12)
    X_num = rng.normal(0.0, 1.0, size=(60, 1))
    X_den = rng.normal(0.5, 1.0, size=(60, 1))

    res = fit_density_ratio(
        X_num,
        X_den,
        generator="sq",
        n_centers=20,
        sigma_grid=[1.0],
        lam_grid=[1e-3, 1e-1],
        cv=True,
        folds=3,
        random_state=0,
    )

    assert res.lam == 1e-1
    assert res.n_failed_candidates == 1
    failed = [row for row in res.cv_path if not bool(row["success"])]
    assert len(failed) == 1
    assert failed[0]["fold_status"] == (
        "validation_domain_error",
        "validation_domain_error",
        "validation_domain_error",
    )
