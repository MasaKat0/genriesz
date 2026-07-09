"""Tests for RKHS bandwidth selection (Step 3a, item A).

Covers ``sigma="auto"`` (median heuristic, resolved on the training sample only)
across the RBF bases, backward compatibility with a fixed float bandwidth, and
``GaussianRKHSBasis.copy_with_params`` for building CV candidates.
"""

from __future__ import annotations

import numpy as np
import pytest

from genriesz import GaussianRKHSBasis, RBFNystromBasis, RBFRandomFourierBasis
from genriesz.basis import _median_pairwise_distance


def _data(n: int = 300, d: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d))


def test_sigma_auto_matches_median_heuristic():
    X = _data()
    b = GaussianRKHSBasis(n_centers=50, sigma="auto", standardize=True, random_state=0).fit(X)
    # After standardization the data are (approximately) already unit-scaled here,
    # but the resolved sigma must equal the median pairwise distance of Xs.
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0, ddof=0) > 0, X.std(axis=0, ddof=0), 1.0)
    expected = _median_pairwise_distance(Xs, random_state=0)
    assert b.sigma_ == pytest.approx(expected, rel=1e-9)


def test_sigma_auto_works_on_all_rbf_bases():
    X = _data()
    for basis in (
        GaussianRKHSBasis(n_centers=40, sigma="auto", random_state=0),
        RBFRandomFourierBasis(n_features=80, sigma="auto", random_state=0),
        RBFNystromBasis(n_centers=40, sigma="auto", random_state=0),
    ):
        fitted = basis.fit(X)
        assert np.isfinite(fitted.sigma_) and fitted.sigma_ > 0
        # The feature map is usable and finite.
        Phi = np.asarray(fitted(X[:5]), dtype=float)
        assert np.all(np.isfinite(Phi))


def test_float_sigma_is_unchanged_backward_compatible():
    X = _data()
    b = GaussianRKHSBasis(n_centers=40, sigma=2.5, random_state=0).fit(X)
    assert b.sigma_ == 2.5
    # A default (float) basis is byte-for-byte identical to the pre-"auto" API.
    default = GaussianRKHSBasis(n_centers=40, random_state=0).fit(X)
    assert default.sigma == 1.0 and default.sigma_ == 1.0


def test_invalid_sigma_string_rejected():
    with pytest.raises(ValueError, match="positive float or 'auto'"):
        GaussianRKHSBasis(sigma="median")
    with pytest.raises(ValueError, match="positive float or 'auto'"):
        RBFRandomFourierBasis(sigma="scott")


def test_sigma_property_requires_fit():
    b = GaussianRKHSBasis(n_centers=10, sigma="auto")
    with pytest.raises(RuntimeError, match="fit"):
        _ = b.sigma_


def test_sigma_auto_is_data_dependent():
    b0 = GaussianRKHSBasis(n_centers=40, sigma="auto", random_state=0)
    # Fitting on data of different spread gives different resolved bandwidths.
    narrow = b0.copy_with_params().fit(_data(seed=1) * 0.5)
    wide = b0.copy_with_params().fit(_data(seed=1) * 4.0)
    # standardize=True rescales, so compare with standardize off to see the effect.
    n2 = GaussianRKHSBasis(n_centers=40, sigma="auto", standardize=False, random_state=0)
    s_narrow = n2.copy_with_params(standardize=False).fit(_data(seed=1) * 0.5).sigma_
    s_wide = n2.copy_with_params(standardize=False).fit(_data(seed=1) * 4.0).sigma_
    assert s_wide > s_narrow
    assert np.isfinite(narrow.sigma_) and np.isfinite(wide.sigma_)


def test_copy_with_params_pins_centers_and_overrides():
    X = _data()
    base = GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=None)

    c1 = base.copy_with_params(sigma=0.5)
    c2 = base.copy_with_params(sigma=2.0)
    # New objects, original left unfitted.
    assert c1 is not base and base._centers is None
    assert c1.sigma == 0.5 and c2.sigma == 2.0

    c1.fit(X)
    c2.fit(X)
    # Candidates differ only in sigma: the center subsample is shared because the
    # seed is pinned when random_state is None.
    assert np.allclose(c1.centers, c2.centers)
    assert c1.sigma_ == 0.5 and c2.sigma_ == 2.0

    # Overriding n_centers changes the candidate as requested.
    c3 = base.copy_with_params(n_centers=10).fit(X)
    assert c3.centers.shape[0] == 10


def test_copy_with_params_preserves_explicit_random_state():
    X = _data()
    base = GaussianRKHSBasis(n_centers=30, sigma=1.0, random_state=7)
    a = base.copy_with_params(sigma=0.5).fit(X)
    b = base.copy_with_params(sigma=0.5).fit(X)
    assert np.allclose(a.centers, b.centers)
