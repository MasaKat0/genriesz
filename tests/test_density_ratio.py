import numpy as np

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


def test_squared_closed_form_falls_back_to_lstsq_when_singular():
    """An unpenalized closed form on rank-deficient features must not raise.

    ``0.5 H + lam I`` is singular when ``lam = 0`` and the design is
    rank-deficient (duplicated rows collapse the kernel columns). GRRGLM already
    falls back to the minimum-norm ``lstsq`` solution for this exact system; the
    density-ratio solver now matches it instead of propagating ``LinAlgError``.
    """
    from genriesz.density_ratio import _Penalty, _solve_squared_closed_form

    # Every row identical -> Phi has rank 1, and with the bias column H is singular.
    Phi_den = np.ones((8, 3), dtype=float)
    Phi_num = np.ones((6, 3), dtype=float)

    beta = _solve_squared_closed_form(
        Phi_num=Phi_num,
        Phi_den=Phi_den,
        C=0.0,
        penalty=_Penalty(None, lam=0.0, p_norm=2.0),
    )
    assert beta.shape == (3,)
    assert np.all(np.isfinite(beta))


def test_cv_excludes_a_linalgerror_candidate_instead_of_aborting(monkeypatch, recwarn):
    """LinAlgError derives from ValueError, so ``except RuntimeError`` missed it.

    A singular solve on one candidate used to escape the per-candidate handler
    and abort the entire sweep, violating the "failing candidates are excluded
    and counted" contract.
    """
    import genriesz.density_ratio as dr

    real = dr._solve_squared_closed_form

    def flaky(*, Phi_num, Phi_den, C, penalty):
        # Fail every fit at the first sigma candidate, succeed at the second.
        if penalty.lam == 1e-3:
            raise np.linalg.LinAlgError("Singular matrix")
        return real(Phi_num=Phi_num, Phi_den=Phi_den, C=C, penalty=penalty)

    monkeypatch.setattr(dr, "_solve_squared_closed_form", flaky)

    rng = np.random.default_rng(11)
    X_num = rng.normal(0.0, 1.0, size=(60, 1))
    X_den = rng.normal(0.5, 1.0, size=(60, 1))

    res = fit_density_ratio(
        X_num,
        X_den,
        generator="sq",
        n_centers=20,
        sigma_grid=[1.0],
        lam_grid=[1e-3, 1e-1],  # the 1e-3 candidate raises LinAlgError
        cv=True,
        folds=3,
        random_state=0,
    )

    # The surviving candidate is selected and the failures are reported, not raised.
    assert res.lam == 1e-1
    assert np.all(np.isfinite(res.beta))
    messages = [str(w.message) for w in recwarn.list]
    assert any("candidate fit(s) failed" in msg for msg in messages)
