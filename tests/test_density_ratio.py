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
