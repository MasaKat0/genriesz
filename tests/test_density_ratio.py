import numpy as np

from genriesz import grr_density_ratio


def test_grr_density_ratio_basic_shapes():
    rng = np.random.default_rng(0)
    X_num = rng.normal(loc=0.0, scale=1.0, size=(200, 2))
    X_den = rng.normal(loc=0.5, scale=1.0, size=(200, 2))

    res = grr_density_ratio(
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


def test_grr_density_ratio_cv_runs():
    rng = np.random.default_rng(1)
    X_num = rng.normal(loc=0.0, scale=1.0, size=(120, 1))
    X_den = rng.normal(loc=0.7, scale=1.0, size=(120, 1))

    res = grr_density_ratio(
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
