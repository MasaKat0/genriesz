"""Tests for inner Riesz-hyperparameter cross-validation (Step 3b, item C).

Covers the no-leakage guarantees, the two-stage selection (admissibility screen
+ criterion), grid normalization, and backward compatibility of ``grr_functional``
when no grid is supplied.
"""

from __future__ import annotations

import numpy as np
import pytest

from genriesz import (
    ATEFunctional,
    BPGenerator,
    GaussianRKHSBasis,
    GRRCVConfig,
    PolynomialBasis,
    SquaredGenerator,
    grr_ate,
    select_grr_hyperparams,
)
from genriesz.basis import _median_pairwise_distance
from genriesz.functionals import AMEFunctional, ATTFunctional
from genriesz.glm import GRRGLM
from genriesz.model_selection import (
    DEFAULT_LAM_GRID,
    DEFAULT_SIGMA_MULTIPLIERS,
    GRRCVResult,
    _require_positive_median,
    _standardized,
    make_candidate_basis,
    normalize_grid,
    score_grr_candidate,
    select_kernel_centers,
)
from genriesz.utils import Fold


def _make_ate(n: int = 400, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    ps = 1.0 / (1.0 + np.exp(-(0.6 * Z[:, 0] - 0.4 * Z[:, 1])))
    D = (rng.uniform(size=n) < ps).astype(float)
    X = np.column_stack([D, Z])
    Y = D + Z[:, 0] + 0.5 * Z[:, 1] + rng.normal(scale=0.5, size=n)
    return X, Y, 1.0


# ---------------------------------------------------------------------------
# Grid normalization
# ---------------------------------------------------------------------------

def test_normalize_grid_variants():
    assert normalize_grid([0.1, 0.2], kind="sigma") == [0.1, 0.2]
    assert normalize_grid(0.5, kind="sigma") == [0.5]
    auto = normalize_grid("auto", kind="sigma", median=2.0)
    assert auto == [2.0 * m for m in (0.25, 0.5, 1.0, 2.0, 4.0)]

    assert normalize_grid("auto", kind="lam") == list(
        (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
    )
    assert normalize_grid(1e-2, kind="lam") == [1e-2]
    assert normalize_grid([1e-3, 1e-1], kind="lam") == [1e-3, 1e-1]

    # n_centers is capped at n and de-duplicated.
    assert normalize_grid("auto", kind="n_centers", n=100) == [80, 100]
    assert normalize_grid([50, 500], kind="n_centers", n=200) == [50, 200]


def test_normalize_grid_auto_sigma_requires_median():
    with pytest.raises(ValueError, match="positive median"):
        normalize_grid("auto", kind="sigma", median=0.0)


def test_normalize_grid_lam_rejects_none_and_unknown_string():
    # None means "do not vary lambda" and must be resolved by the caller against
    # riesz_lam, never silently expanded into the default sweep.
    with pytest.raises(ValueError, match="do not vary lambda"):
        normalize_grid(None, kind="lam")
    with pytest.raises(ValueError, match="must be 'auto'"):
        normalize_grid("default", kind="lam")


# Audit K-19 / CV-03: bad grid values must fail at the entrance with a named
# error, not deep in the solver ("negative dimensions", "A and b must be
# finite") and never by silent reinterpretation (True -> one center, 40.7 -> 40,
# an empty grid -> "no candidate could be fitted").
@pytest.mark.parametrize(
    "kind, bad, match",
    [
        ("n_centers", [0, 40], "integers >= 1"),
        ("n_centers", [-5], "integers >= 1"),
        ("n_centers", [True], "boolean"),
        ("n_centers", [np.bool_(True)], "boolean"),
        ("n_centers", [40.7], "integers >= 1"),
        ("n_centers", [float("inf")], "integers >= 1"),
        ("n_centers", [], "empty list"),
        ("n_centers", float("nan"), "integers >= 1"),
        ("n_centers", 0, "integers >= 1"),
        ("lam", [float("nan")], "finite"),
        ("lam", [float("-inf")], "finite"),
        ("lam", [-1e-2], ">= 0"),
        ("lam", [True], "boolean"),
        ("lam", [np.bool_(False)], "boolean"),
        ("lam", [], "empty list"),
        ("lam", float("inf"), "finite"),
        ("sigma", [float("nan")], "finite and positive"),
        ("sigma", [-1.0], "finite and positive"),
        ("sigma", [0.0], "finite and positive"),
        ("sigma", [float("inf")], "finite and positive"),
        ("sigma", [True], "boolean"),
        ("sigma", [np.bool_(True)], "boolean"),
        ("sigma", [], "empty list"),
        ("sigma", float("nan"), "finite and positive"),
    ],
)
def test_normalize_grid_rejects_garbage_at_the_entrance(kind, bad, match):
    with pytest.raises(ValueError, match=match):
        normalize_grid(bad, kind=kind, n=200)


def test_normalize_grid_accepts_reasonable_numeric_types():
    # Integral floats and NumPy scalars are honest spellings of the same
    # candidate; only value-changing coercions are rejected.
    assert normalize_grid([np.int64(50), 40.0], kind="n_centers", n=200) == [40, 50]
    assert normalize_grid([np.float64(0.5)], kind="sigma") == [0.5]
    assert normalize_grid([0.0, np.float64(1e-2)], kind="lam") == [0.0, 1e-2]


def test_select_grr_hyperparams_rejects_garbage_grids_before_fitting():
    # The same validation must guard the real entry point: the sigma path goes
    # through _sigma_candidate_specs, not normalize_grid.
    X, y, _ = _make_ate(n=120)
    common = dict(
        X_train=X,
        y_train=y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=20, random_state=0),
        generator=SquaredGenerator(),
    )
    # The matches pin the *entrance* messages: before the fix a NaN reached the
    # solver and died with "A and b must be finite", which a loose match like
    # "finite" would also accept.
    for grids, match in [
        (dict(sigma_grid=[float("nan")], lam_grid=[1e-2]), "^sigma_grid values must be finite"),
        (dict(sigma_grid=[], lam_grid=[1e-2]), "^sigma_grid is an empty list"),
        (dict(n_centers_grid=[0, 40], lam_grid=[1e-2]), "^n_centers_grid values must be integers"),
        (dict(lam_grid=[float("nan")]), "^lam_grid values must be finite"),
    ]:
        with pytest.raises(ValueError, match=match):
            select_grr_hyperparams(
                config=GRRCVConfig(cv_folds=2, random_state=0, **grids), **common
            )


# ---------------------------------------------------------------------------
# No leakage: centers within training, selection is a pure function of X_train
# ---------------------------------------------------------------------------

def test_select_kernel_centers_are_within_training():
    X, _, _ = _make_ate(n=200)
    centers = select_kernel_centers(X, n_centers=30, random_state=0)
    assert centers.shape == (30, X.shape[1])
    # Every returned center is an actual training row.
    for c in centers:
        assert np.any(np.all(np.isclose(X, c), axis=1))


def test_selection_is_deterministic_function_of_training():
    X, Y, _ = _make_ate(n=300, seed=1)
    cfg = GRRCVConfig(sigma_grid="auto", lam_grid=[1e-2, 1e-1], random_state=0)
    common = dict(
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        config=cfg,
        outcome_link="identity",
    )
    r1 = select_grr_hyperparams(X_train=X, y_train=Y, **common)
    r2 = select_grr_hyperparams(X_train=X, y_train=Y, **common)
    assert (r1.sigma, r1.lam, r1.n_centers) == (r2.sigma, r2.lam, r2.n_centers)


# ---------------------------------------------------------------------------
# Strict nested CV (audit P0-05): no inner-validation row enters the feature map
# that scores it -- centers and the median heuristic come from inner-training only
# ---------------------------------------------------------------------------

def _nested_common(**over):
    base = dict(
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=60, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        outcome_link="identity",
    )
    base.update(over)
    return base


def test_strict_nested_center_and_preprocess_disjoint_from_validation():
    X, Y, _ = _make_ate(n=240, seed=3)
    cfg = GRRCVConfig(
        sigma_grid="auto", lam_grid=[1e-2], n_centers_grid=[40, 60], random_state=0
    )
    res = select_grr_hyperparams(X_train=X, y_train=Y, config=cfg, **_nested_common())

    assert res.strict_nested is True
    assert len(res.fold_provenance) == cfg.cv_folds
    n = X.shape[0]
    for prov in res.fold_provenance:
        val = set(prov["validation_index"].tolist())
        train = set(prov["preprocess_fit_index"].tolist())
        centers = set(prov["center_index"].tolist())
        # Centers are drawn from inner-training rows only.
        assert centers  # a positive n_centers pool was selected
        assert centers <= train
        assert centers.isdisjoint(val)
        # Preprocessing (standardization / median heuristic) is fit on the
        # inner-training rows, which partition the training sample with the
        # validation rows -- no overlap, and together they cover everything.
        assert train.isdisjoint(val)
        assert train | val == set(range(n))


def test_strict_nested_training_feature_map_ignores_validation_rows():
    # Perturbing an inner-validation row of a fold must not change that fold's
    # inner-training feature map: its centers, their coordinates, and its "auto"
    # bandwidth anchor are all functions of the inner-training rows alone.
    X, Y, _ = _make_ate(n=240, seed=5)
    cfg = GRRCVConfig(sigma_grid="auto", lam_grid=[1e-2], n_centers_grid=[50], random_state=0)
    res = select_grr_hyperparams(X_train=X, y_train=Y, config=cfg, **_nested_common())
    prov0 = res.fold_provenance[0]
    train0 = prov0["preprocess_fit_index"]
    val0 = prov0["validation_index"]
    centers0 = prov0["center_index"]

    med0 = _median_pairwise_distance(_standardized(X[train0]), random_state=0)

    X2 = X.copy()
    # Blow up every fold-0 validation row's *covariates*. The treatment column
    # stays intact: the inner split is stratified on it, so corrupting D would
    # change the fold membership itself rather than test feature-map leakage.
    X2[np.ix_(val0, np.arange(1, X.shape[1]))] += 1000.0

    res2 = select_grr_hyperparams(X_train=X2, y_train=Y, config=cfg, **_nested_common())
    prov0b = res2.fold_provenance[0]

    # Same fold-0 center indices, and their coordinates are untouched (the
    # perturbed rows are validation rows, never centers of their own fold).
    assert np.array_equal(prov0b["center_index"], centers0)
    assert np.array_equal(X2[centers0], X[centers0])
    # The fold-0 bandwidth anchor (median over standardized inner-training) is
    # unchanged, because it never reads the validation rows.
    med0b = _median_pairwise_distance(_standardized(X2[train0]), random_state=0)
    assert med0b == pytest.approx(med0)


class _RecordingBasis:
    """Wraps a basis and records the (X, y) every ``fit`` saw, for leakage tests.

    Records live at class level so the copies ``make_candidate_basis`` produces
    still report to the same list. Delegates everything else to the inner basis,
    so a genuinely *supervised* fit (one that reads ``y``) is exercised too.
    """

    fits: list = []

    def __init__(self, inner):
        self._inner = inner

    def fit(self, X, y=None):
        Xa = np.asarray(X, dtype=float).copy()
        ya = None if y is None else np.asarray(y, dtype=float).copy()
        _RecordingBasis.fits.append((Xa, ya))
        self._inner.fit(X, y)
        return self

    def __call__(self, X):
        return self._inner(X)

    def copy(self):
        return _RecordingBasis(self._inner.copy())

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


def test_strict_nested_supervised_basis_fits_on_inner_training_only():
    # A supervised basis (whose fit reads y) must never see an inner-validation
    # row -- not in X and not in y. score_grr_candidate always fits on the fold's
    # inner-training slice, so the recorded fit rows equal X[train]/y[train].
    X, Y, _ = _make_ate(n=200, seed=7)
    n = X.shape[0]
    idx = np.arange(n)
    fold = Fold(train=idx[: 3 * n // 4], test=idx[3 * n // 4 :])

    _RecordingBasis.fits = []
    template = _RecordingBasis(PolynomialBasis(degree=2))
    row = score_grr_candidate(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(0),
        template_basis=template,  # lambda-only CV -> make_candidate_basis uses copy()
        generator=SquaredGenerator(),
        sigma=None,
        lam=1e-2,
        centers=None,
        inner_folds=[fold],
        riesz_penalty="l2",
        riesz_p_norm=None,
        outcome_link="identity",
        outcome_penalty="l2",
        outcome_lam=1e-3,
        max_iter=500,
        tol=1e-8,
        want_kernel=False,
    )
    assert row["success"]
    assert _RecordingBasis.fits  # the basis was fit at least once
    supervised_fits = 0
    for Xf, yf in _RecordingBasis.fits:
        assert np.array_equal(Xf, X[fold.train])  # never the validation rows
        if yf is not None:
            supervised_fits += 1
            assert np.array_equal(yf, Y[fold.train])
    assert supervised_fits > 0


def test_non_nested_feature_map_is_rejected():
    with pytest.raises(ValueError, match="inner-validation"):
        GRRCVConfig(strict_nested=False)


def test_strict_nested_rejects_degenerate_inner_fold(monkeypatch):
    # A fold whose inner-training rows are identical has no positive median
    # pairwise distance. Automatic bandwidth selection is undefined in that
    # fold, so the selection must stop rather than substitute an unrelated
    # bandwidth.
    import genriesz.model_selection as ms

    k = 30
    ident_z = np.tile(np.array([0.5, -0.3, 1.2]), (k, 1))
    rng = np.random.default_rng(0)
    dist_z = rng.normal(size=(k, 3)) + 5.0
    Z = np.vstack([ident_z, dist_z])
    D = np.full(2 * k, 0.25)
    X = np.column_stack([D, Z])
    Y = rng.normal(size=2 * k)
    A = np.arange(k)
    B = np.arange(k, 2 * k)

    def fake_splits(n, *, folds, random_state=None, shuffle=True):
        yield Fold(train=A, test=B)
        yield Fold(train=B, test=A)

    monkeypatch.setattr(ms, "kfold_splits", fake_splits)
    cfg = GRRCVConfig(
        sigma_grid="auto",
        lam_grid=[1e-2],
        cv_folds=2,
        random_state=0,
    )
    with pytest.raises(ValueError, match="no positive pairwise-distance median"):
        select_grr_hyperparams(
            X_train=X,
            y_train=Y,
            m=ATEFunctional(0),
            basis=GaussianRKHSBasis(n_centers=20, sigma=1.0, random_state=0),
            generator=SquaredGenerator(),
            config=cfg,
            outcome_link="identity",
        )


@pytest.mark.parametrize("value", [0.0, float("nan"), -2.0, float("inf")])
def test_require_positive_median_rejects_invalid_anchor(value):
    with pytest.raises(ValueError, match="Automatic bandwidth selection is undefined"):
        _require_positive_median(value, context="Inner training fold 0")


def test_require_positive_median_accepts_positive_anchor():
    assert _require_positive_median(2.5, context="Inner training fold 0") == pytest.approx(2.5)


def test_grrcvconfig_positional_signature_is_stable():
    # strict_nested must be keyword-only so it does not shift the positional
    # arguments of a previously-valid call (the 5th positional stays
    # selection_score, not strict_nested).
    cfg = GRRCVConfig(None, [0.1], None, 3, "bregman_validation")
    assert cfg.selection_score == "bregman_validation"
    assert cfg.strict_nested is True
    assert "strict_nested" not in GRRCVConfig.__match_args__
    assert GRRCVConfig.__match_args__[:5] == (
        "sigma_grid",
        "lam_grid",
        "n_centers_grid",
        "cv_folds",
        "selection_score",
    )
    # GRRCVResult's added fields are keyword-only too: the positional constructor
    # and __match_args__ of the previous release still hold.
    res = GRRCVResult(None, 1e-2, None, "bias_variance", 0.5, 1, 1, [])
    assert res.strict_nested is True and res.fold_provenance == []
    assert "strict_nested" not in GRRCVResult.__match_args__
    assert "fold_provenance" not in GRRCVResult.__match_args__
    assert not hasattr(res, "used_fallback")
    assert "modifies_estimand" not in GRRCVResult.__match_args__


# --------------------------------------------------------------------------- #
# Audit N-03: the "auto" sigma anchor follows the basis's coordinate system
# --------------------------------------------------------------------------- #


def _large_scale_sample(n: int = 240, scale: float = 1000.0):
    rng = np.random.default_rng(0)
    D = rng.binomial(1, 0.5, size=n).astype(float)
    Z = rng.normal(0.0, scale, size=(n, 3))
    X = np.column_stack([D, Z])
    y = Z[:, 0] / scale + D + rng.normal(size=n)
    return X, y


def test_auto_sigma_anchor_follows_the_basis_coordinate_system():
    # A basis with standardize=False measures kernel distances on the raw
    # features, so the "auto" grid must anchor at the *raw* median. Anchoring at
    # the standardized median (audit N-03) puts every candidate bandwidth a
    # factor of the feature scale too small, the kernel saturates, and the whole
    # grid is rejected by the admissibility screen.
    from genriesz.model_selection import _standardized as std

    X, y = _large_scale_sample()
    cfg = GRRCVConfig(sigma_grid="auto", cv_folds=3, return_path=True, random_state=0)
    res = select_grr_hyperparams(
        X_train=X,
        y_train=y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=40, standardize=False, random_state=0),
        generator=SquaredGenerator(),
        config=cfg,
    )
    raw_median = _median_pairwise_distance(X, random_state=0)
    std_median = _median_pairwise_distance(std(X), random_state=0)
    assert raw_median > 100 * std_median  # the scale gap the anchor must bridge
    expected = sorted(float(m) * raw_median for m in DEFAULT_SIGMA_MULTIPLIERS)
    got = sorted(row["sigma"] for row in res.path)
    assert np.allclose(got, expected, rtol=1e-12)
    # Candidates in the right coordinate system remain admissible.
    assert res.n_admissible > 0


def test_auto_sigma_anchor_stays_standardized_for_a_standardizing_basis():
    # The default (standardize=True) basis keeps the standardized anchor: the
    # N-03 fix must not move it.
    from genriesz.model_selection import _standardized as std

    X, y = _large_scale_sample()
    cfg = GRRCVConfig(sigma_grid="auto", cv_folds=3, return_path=True, random_state=0)
    res = select_grr_hyperparams(
        X_train=X,
        y_train=y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=40, random_state=0),
        generator=SquaredGenerator(),
        config=cfg,
    )
    std_median = _median_pairwise_distance(std(X), random_state=0)
    expected = sorted(float(m) * std_median for m in DEFAULT_SIGMA_MULTIPLIERS)
    got = sorted(row["sigma"] for row in res.path)
    assert np.allclose(got, expected, rtol=1e-12)


# --------------------------------------------------------------------------- #
# Admissibility is mandatory; no failed or inadmissible candidate is returned
# --------------------------------------------------------------------------- #


def _healthy_sample(n: int = 150):
    rng = np.random.default_rng(1)
    D = rng.binomial(1, 0.5, size=n).astype(float)
    X = np.column_stack([D, rng.normal(size=(n, 2))])
    y = X[:, 1] + D + rng.normal(size=n)
    return X, y


_IMPOSSIBLE = {"min_ess_ratio": 2.0}


def test_no_admissible_candidate_halts_selection():
    X, y = _healthy_sample()
    cfg = GRRCVConfig(
        sigma_grid="auto",
        cv_folds=2,
        random_state=0,
        admissibility_thresholds=_IMPOSSIBLE,
    )
    with pytest.raises(RuntimeError, match="No Riesz candidate passed the admissibility screen"):
        select_grr_hyperparams(
            X_train=X,
            y_train=y,
            m=ATEFunctional(0),
            basis=GaussianRKHSBasis(n_centers=30, random_state=0),
            generator=SquaredGenerator(),
            config=cfg,
        )


def test_removed_fallback_policy_keyword_is_rejected():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        GRRCVConfig(fallback_policy="best_criterion")


def test_a_stray_estimand_flag_attribute_no_longer_screens_candidates():
    # The estimand-flag exclusion machinery is removed: an attribute with the
    # old name has no effect, and selection judges the candidate on quality.
    X, y = _healthy_sample()
    gen = SquaredGenerator()
    gen.modifies_estimand = True
    res = select_grr_hyperparams(
        X_train=X,
        y_train=y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=30, random_state=0),
        generator=gen,
        config=GRRCVConfig(sigma_grid="auto", cv_folds=2, random_state=0),
    )
    assert res.n_admissible >= 1


def test_estimation_entry_point_does_not_accept_fallback_policy():
    X, y = _healthy_sample(n=180)
    common = dict(
        X=X,
        Y=y,
        treatment_index=0,
        basis=GaussianRKHSBasis(n_centers=25, random_state=0),
        generator=SquaredGenerator(),
        riesz_sigma_grid="auto",
        riesz_cv_folds=2,
        riesz_admissibility_thresholds=_IMPOSSIBLE,
        estimators=("arw",),
        cross_fit=True,
        folds=2,
        random_state=0,
    )
    with pytest.raises(RuntimeError, match="admissibility screen"):
        grr_ate(**common)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        grr_ate(**common, riesz_fallback_policy="best_criterion")


def test_outer_refit_reselects_centers_when_n_centers_cv_active(monkeypatch):
    # When the inner CV selects an ``n_centers``, the outer refit must drop any
    # fixed centers the basis carried, so it reselects that many centers from the
    # outer-training fold. Otherwise copy_with_params keeps the original centers
    # and silently ignores the selected count.
    X, Y, _ = _make_ate(n=300, seed=4)
    explicit = X[:120].copy()  # a basis built with fixed centers
    base = GaussianRKHSBasis(n_centers=300, sigma=1.0, random_state=0, centers=explicit)

    captured: list[dict] = []
    orig = GaussianRKHSBasis.copy_with_params

    def spy(self, **overrides):
        captured.append(dict(overrides))
        return orig(self, **overrides)

    monkeypatch.setattr(GaussianRKHSBasis, "copy_with_params", spy)
    grr_ate(
        X=X,
        Y=Y,
        basis=base,
        generator=SquaredGenerator(),
        folds=2,
        cross_fit=True,
        random_state=0,
        riesz_n_centers_grid=[20, 40],
        riesz_lam=1e-2,
    )
    # The outer-refit copies are exactly those overriding ``n_centers``; each must
    # also clear the fixed centers so the count actually takes effect.
    refit_calls = [c for c in captured if "n_centers" in c]
    assert refit_calls
    assert all(c.get("centers", "MISSING") is None for c in refit_calls)


# ---------------------------------------------------------------------------
# lam_grid=None means "keep riesz_lam", exactly as for sigma_grid/n_centers_grid
# ---------------------------------------------------------------------------

def test_lam_grid_none_keeps_riesz_lam_and_does_not_sweep():
    X, Y, _ = _make_ate(n=300, seed=6)
    res = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        config=GRRCVConfig(sigma_grid="auto", lam_grid=None, return_path=True, random_state=0),
        riesz_lam=0.037,  # a value in no default grid, so a sweep would show up
        outcome_link="identity",
    )
    # One row per sigma candidate -- lambda is not a swept dimension. Under the
    # old behavior this was len(DEFAULT_LAM_GRID) times larger.
    assert len(res.path) == len(DEFAULT_SIGMA_MULTIPLIERS)
    assert {r["lam"] for r in res.path} == {0.037}
    assert res.lam == pytest.approx(0.037)


def test_lam_grid_auto_sweeps_the_default_grid():
    X, Y, _ = _make_ate(n=300, seed=6)
    res = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        config=GRRCVConfig(lam_grid="auto", return_path=True, random_state=0),
        riesz_lam=0.037,
        outcome_link="identity",
    )
    assert sorted(r["lam"] for r in res.path) == sorted(DEFAULT_LAM_GRID)
    assert res.lam in DEFAULT_LAM_GRID


# ---------------------------------------------------------------------------
# Two-stage selection: kernel-health screen excludes degenerate bandwidths
# ---------------------------------------------------------------------------

def test_kernel_health_screen_excludes_degenerate_bandwidths():
    X, Y, _ = _make_ate(n=400, seed=2)
    cfg = GRRCVConfig(
        sigma_grid="auto",  # median * (0.25, 0.5, 1, 2, 4): includes both traps
        lam_grid=[1e-2, 1e-1],
        n_centers_grid=[60],
        return_path=True,
        random_state=0,
    )
    res = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=60, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        config=cfg,
        outcome_link="identity",
    )
    by_sigma = {round(r["sigma"], 3): r for r in res.path}
    sigmas = sorted(by_sigma)
    # Smallest bandwidth collapses the kernel (median ~ 0): inadmissible.
    assert by_sigma[sigmas[0]]["kernel_median"] < 1e-3
    assert all(not by_sigma[sigmas[0]]["admissible"] for _ in [0])
    # Largest bandwidth saturates the kernel (median -> 1): inadmissible.
    assert by_sigma[sigmas[-1]]["kernel_median"] > 0.8
    assert not by_sigma[sigmas[-1]]["admissible"]
    # At least one healthy candidate survives, and the selected sigma is not a
    # degenerate extreme.
    assert res.n_admissible >= 1
    assert sigmas[0] < res.sigma < sigmas[-1]


def test_make_candidate_basis_requires_rkhs_for_sigma():
    poly = PolynomialBasis(degree=2)
    with pytest.raises(ValueError, match="copy_with_params"):
        make_candidate_basis(poly, sigma=1.0, centers=None)
    # Lambda-only CV (no sigma/centers) works for any basis via copy().
    clone = make_candidate_basis(poly, sigma=None, centers=None)
    assert isinstance(clone, PolynomialBasis)


# ---------------------------------------------------------------------------
# grr_functional integration
# ---------------------------------------------------------------------------

def test_grr_functional_without_grids_is_unchanged():
    X, Y, _ = _make_ate(n=300, seed=3)
    kw = dict(
        X=X,
        Y=Y,
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        riesz_lam=1e-2,
        folds=4,
        random_state=0,
    )
    r1 = grr_ate(**kw)
    r2 = grr_ate(**kw)
    assert r1.arw.estimate == pytest.approx(r2.arw.estimate, abs=0.0)
    # No CV requested -> no CV diagnostics.
    assert "riesz_cv" not in r1.diagnostics


def test_grr_functional_cv_reports_selection_and_beats_fixed_trap():
    X, Y, tau = _make_ate(n=400, seed=4)
    fixed = grr_ate(
        X=X,
        Y=Y,
        basis=GaussianRKHSBasis(n_centers=60, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        riesz_lam=1e-2,
        folds=4,
        random_state=0,
    )
    tuned = grr_ate(
        X=X,
        Y=Y,
        basis=GaussianRKHSBasis(n_centers=60, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        folds=4,
        random_state=0,
        riesz_sigma_grid="auto",
        riesz_lam_grid=[1e-2, 1e-1],
        riesz_n_centers_grid=[60, 120],
    )
    cv = tuned.diagnostics["riesz_cv"]
    assert len(cv["selected"]) == 4
    for s in cv["selected"]:
        assert s["sigma"] is not None and s["sigma"] > 0
        assert s["n_admissible"] >= 1
    assert tuned.diagnostics["riesz_cv_selection_score"] == "bias_variance"
    # The tuned estimate is at least as close to the truth as the fixed baseline,
    # and its SE is not the fake-tiny value of a saturated kernel.
    assert abs(tuned.arw.estimate - tau) <= abs(fixed.arw.estimate - tau) + 0.05
    assert tuned.arw.se > 1e-2


def test_grr_functional_cv_lambda_only_with_any_basis():
    X, Y, _ = _make_ate(n=300, seed=5)

    def phi(A):
        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            d, z = A[0], A[1:]
            return np.concatenate([[1.0], [d], z, d * z])
        d, z = A[:, [0]], A[:, 1:]
        return np.concatenate([np.ones((len(A), 1)), d, z, d * z], axis=1)

    res = grr_ate(
        X=X,
        Y=Y,
        basis=phi,
        generator=SquaredGenerator(C=0.0).as_generator(),
        folds=3,
        random_state=0,
        riesz_lam_grid=[1e-3, 1e-2, 1e-1],
    )
    sel = res.diagnostics["riesz_cv"]["selected"]
    assert len(sel) == 3
    for s in sel:
        assert s["sigma"] is None  # no bandwidth CV for a callable basis
        assert s["lam"] in (1e-3, 1e-2, 1e-1)


def test_grr_functional_sigma_only_cv_holds_riesz_lam_fixed():
    X, Y, _ = _make_ate(n=300, seed=7)
    res = grr_ate(
        X=X,
        Y=Y,
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        folds=3,
        random_state=0,
        riesz_sigma_grid="auto",
        riesz_lam=0.037,  # must be honored: riesz_lam_grid is None
        return_riesz_cv_path=True,
    )
    cv = res.diagnostics["riesz_cv"]
    for s in cv["selected"]:
        assert s["lam"] == pytest.approx(0.037)
    for fold_path in cv["path"]:
        assert len(fold_path) == len(DEFAULT_SIGMA_MULTIPLIERS)  # no lambda sweep
        assert {r["lam"] for r in fold_path} == {0.037}


# ---------------------------------------------------------------------------
# squared_loss_validation: a generator-agnostic (uLSIF-style) selection score
# ---------------------------------------------------------------------------

def test_squared_loss_validation_selects_and_is_consistent():
    X, Y, _ = _make_ate(n=350, seed=6)
    cfg = GRRCVConfig(
        sigma_grid="auto",
        lam_grid=[1e-2, 1e-1],
        selection_score="squared_loss_validation",
        return_path=True,
        random_state=0,
    )
    common = dict(
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        config=cfg,
        outcome_link="identity",
    )
    res = select_grr_hyperparams(X_train=X, y_train=Y, **common)

    assert res.selection_score == "squared_loss_validation"
    assert np.isfinite(res.best_score)
    assert res.sigma is not None and res.sigma > 0.0

    # Deterministic function of the training sample.
    res2 = select_grr_hyperparams(X_train=X, y_train=Y, **common)
    assert (res.sigma, res.lam, res.n_centers) == (res2.sigma, res2.lam, res2.n_centers)

    # Every fitted candidate carries a finite LSIF score, and the criterion used
    # for selection is exactly that score.
    fitted = [r for r in res.path if r["success"]]
    assert fitted
    for r in fitted:
        assert np.isfinite(r["squared_loss_validation"])
        assert r["criterion"] == pytest.approx(r["squared_loss_validation"])

    # The winner minimizes the score over the admissible candidates.
    adm = [r["criterion"] for r in res.path if r["admissible"] and np.isfinite(r["criterion"])]
    assert adm
    assert res.best_score == pytest.approx(min(adm))


def test_squared_loss_validation_scores_non_squared_generator():
    # The LSIF risk is built from the fitted representer, not the generator's own
    # Bregman conjugate, so a candidate fit with a *non-squared* generator (here
    # BP) is still scored on the common yardstick.
    X, Y, _ = _make_ate(n=400, seed=7)

    def branch(x):  # treated branch where D == 1 (first column of X)
        return 1 if float(x[0]) > 0.5 else 0

    cfg = GRRCVConfig(
        lam_grid=[1e-2, 1e-1],
        selection_score="squared_loss_validation",
        return_path=True,
        random_state=0,
    )
    res = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=BPGenerator(C=1.0, omega=0.5, branch_fn=branch),
        config=cfg,
        outcome_link="identity",
    )

    assert res.selection_score == "squared_loss_validation"
    assert np.isfinite(res.best_score)
    fitted = [r for r in res.path if r["success"]]
    assert fitted  # at least one BP candidate fit and received a finite LSIF score
    assert all(np.isfinite(r["squared_loss_validation"]) for r in fitted)

    # Deterministic function of the training sample for a non-squared generator.
    res2 = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(0),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=BPGenerator(C=1.0, omega=0.5, branch_fn=branch),
        config=cfg,
        outcome_link="identity",
    )
    assert (res.sigma, res.lam) == (res2.sigma, res2.lam)
    assert res.best_score == pytest.approx(res2.best_score)


def test_squared_loss_validation_matches_hand_computed_lsif_risk():
    # Directly validate the LSIF formula (and alpha_iva == inv_grad) against an
    # independent reconstruction of the fit on a single fold.
    X, Y, _ = _make_ate(n=240, seed=11)
    n = X.shape[0]
    idx = np.arange(n)
    fold = Fold(train=idx[: n // 2], test=idx[n // 2 :])

    m = ATEFunctional(0)
    gen = SquaredGenerator()
    template = GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0)
    kw = dict(riesz_penalty="l2", lam=1e-2)

    row = score_grr_candidate(
        X_train=X,
        y_train=Y,
        m=m,
        template_basis=template,
        generator=gen,
        sigma=None,
        centers=None,
        inner_folds=[fold],
        riesz_p_norm=None,
        outcome_link="identity",
        outcome_penalty="l2",
        outcome_lam=1e-3,
        max_iter=500,
        tol=1e-8,
        want_kernel=False,
        want_squared_loss=True,
        **kw,
    )
    assert row["success"]

    # Independent reconstruction of the same single-fold fit.
    cb = make_candidate_basis(template, sigma=None, centers=None)
    cb.fit(X[fold.train], Y[fold.train])
    grr = GRRGLM(basis=cb, generator=gen, functional=m, penalty="l2", lam=1e-2)
    fr = grr.fit(X[fold.train], max_iter=500, tol=1e-8)
    assert fr.success and grr.beta_ is not None
    beta = grr.beta_

    X_te = X[fold.test]
    v = np.asarray(cb(X_te), dtype=float) @ beta
    _, alpha = gen.conjugate(X_te, v)
    # The representer identity: conjugate's maximizer equals inv_grad(v).
    assert np.allclose(alpha, gen.inv_grad(X_te, v))

    def rep(A):
        return gen.inv_grad(A, np.asarray(cb(A), dtype=float) @ beta)

    m_rep = np.asarray(m.m_from_function(X_te, predict=rep, derivative=None), dtype=float)
    expected = 0.5 * float(np.mean(alpha**2)) - float(np.mean(m_rep))
    assert row["squared_loss_validation"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_squared_loss_validation_surfaces_functional_incompatibility():
    # A functional whose m(alpha) needs a derivative but is *not* AME slips past
    # the isinstance guard, so the scorer must raise a clear ValueError rather
    # than swallow it into a misleading "all candidates failed to fit".
    class _DerivOnlyFunctional(ATEFunctional):
        requires_function_derivative = True

        def m_from_function(self, X, *, predict, derivative=None):
            if derivative is None:
                raise NotImplementedError("needs a derivative")
            return super().m_from_function(X, predict=predict, derivative=derivative)

    X, Y, _ = _make_ate(n=200, seed=12)
    cfg = GRRCVConfig(
        lam_grid=[1e-2],
        selection_score="squared_loss_validation",
        random_state=0,
    )
    with pytest.raises(ValueError, match="representer alone"):
        select_grr_hyperparams(
            X_train=X,
            y_train=Y,
            m=_DerivOnlyFunctional(0),
            basis=GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0),
            generator=SquaredGenerator(),
            config=cfg,
            outcome_link="identity",
        )


def test_squared_loss_validation_does_not_swallow_valueerror():
    # A genuine ValueError from the functional (not a NotImplementedError about a
    # missing derivative) must propagate unchanged, never be turned into NaN.
    class _RaisesValueError(ATEFunctional):
        def m_from_function(self, X, *, predict, derivative=None):
            raise ValueError("boom-from-functional")

    X, Y, _ = _make_ate(n=200, seed=14)
    cfg = GRRCVConfig(
        lam_grid=[1e-2],
        selection_score="squared_loss_validation",
        random_state=0,
    )
    with pytest.raises(ValueError, match="boom-from-functional"):
        select_grr_hyperparams(
            X_train=X,
            y_train=Y,
            m=_RaisesValueError(0),
            basis=GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0),
            generator=SquaredGenerator(),
            config=cfg,
            outcome_link="identity",
        )


def test_squared_loss_score_is_nan_when_a_fold_fails_to_fit():
    # When the primary Riesz fit fails on a fold, that fold is never scored, so
    # the LSIF aggregate must be NaN (strict: len(sq_risks) == len(folds)), not a
    # partial-fold average. Here max_iter=0 on the L-BFGS (l1) path fails both
    # folds, and the candidate is both success=False and NaN-scored.
    X, Y, _ = _make_ate(n=240, seed=13)
    n = X.shape[0]
    idx = np.arange(n)
    folds = [
        Fold(train=idx[: n // 2], test=idx[n // 2 :]),
        Fold(train=idx[n // 2 :], test=idx[: n // 2]),
    ]
    row = score_grr_candidate(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(0),
        template_basis=GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0),
        generator=SquaredGenerator(C=0.0).as_generator(),
        sigma=None,
        lam=1e-3,
        centers=None,
        inner_folds=folds,
        riesz_penalty="l1",  # numeric path; SQ + l2 is closed form and cannot fail
        riesz_p_norm=None,
        outcome_link="identity",
        outcome_penalty="l2",
        outcome_lam=1e-3,
        max_iter=0,  # zero iterations -> the fit fails on every fold
        tol=1e-8,
        want_kernel=False,
        want_squared_loss=True,
    )
    assert row["success"] is False
    assert np.isnan(row["squared_loss_validation"])


def test_incompatible_functional_selects_fine_under_other_scores():
    # want_squared_loss=False must skip the LSIF path entirely, so an AME
    # functional (whose m(alpha) needs a derivative) still selects under a
    # different score without raising the squared-loss incompatibility error.
    # A derivative-capable basis (Polynomial) is required for AME at all.
    X, Y, _ = _make_ate(n=300, seed=15)
    cfg = GRRCVConfig(
        lam_grid=[1e-2, 1e-1],
        selection_score="bregman_validation",
        random_state=0,
    )
    res = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=AMEFunctional(coordinate=1),
        basis=PolynomialBasis(degree=2),
        generator=SquaredGenerator(),
        config=cfg,
        outcome_link="identity",
    )
    assert res.selection_score == "bregman_validation"
    assert np.isfinite(res.best_score)


def test_squared_loss_validation_works_for_att():
    X, Y, _ = _make_ate(n=350, seed=9)
    pi = float(np.mean(X[:, 0]))
    cfg = GRRCVConfig(
        lam_grid=[1e-2, 1e-1],
        selection_score="squared_loss_validation",
        random_state=0,
    )
    res = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATTFunctional(treatment_index=0, pi=pi, pi_is_estimated=True),
        basis=GaussianRKHSBasis(n_centers=50, sigma=1.0, random_state=0),
        generator=SquaredGenerator(),
        config=cfg,
        outcome_link="identity",
    )
    assert np.isfinite(res.best_score)
    assert res.lam in (1e-2, 1e-1)


def test_squared_loss_validation_rejects_ame():
    X, Y, _ = _make_ate(n=200, seed=8)
    cfg = GRRCVConfig(
        lam_grid=[1e-2],
        selection_score="squared_loss_validation",
        random_state=0,
    )
    with pytest.raises(ValueError, match="not defined for"):
        select_grr_hyperparams(
            X_train=X,
            y_train=Y,
            m=AMEFunctional(coordinate=1),
            basis=GaussianRKHSBasis(n_centers=40, sigma=1.0, random_state=0),
            generator=SquaredGenerator(),
            config=cfg,
            outcome_link="identity",
        )


def test_criterion_is_nan_when_required_metrics_are_missing():
    """A candidate whose bias/variance metrics are missing must not win by default.

    NaN-metric rows used to have b/v replaced by 0 -- the best possible value --
    so an un-evaluable candidate would beat every honestly-evaluated one.
    """

    from genriesz.model_selection import _criterion

    row_missing = {"b_hat": float("nan"), "v_hat": float("nan"), "r_hat": 0.01, "k_hat": 0.0}
    row_valid = {"b_hat": 0.5, "v_hat": 2.0, "r_hat": 0.01, "k_hat": 0.0}

    c_missing = _criterion(row_missing, score="bias_variance", n=100, tau_R=1e-2, tau_K=1e-3)
    c_valid = _criterion(row_valid, score="bias_variance", n=100, tau_R=1e-2, tau_K=1e-3)

    assert np.isnan(c_missing)
    assert np.isfinite(c_valid)

    # Partial missingness (only the variance piece) is just as unscoreable.
    row_partial = {"b_hat": 0.5, "v_hat": float("nan"), "r_hat": 0.01, "k_hat": 0.0}
    assert np.isnan(_criterion(row_partial, score="bias_variance", n=100, tau_R=1e-2, tau_K=1e-3))

    # imbalance_validation requires the standardized imbalance; the raw
    # imbalance is on a different scale and must not be silently substituted.
    row_no_std = {"std_imbalance": float("nan"), "held_out_imbalance": 0.1}
    assert np.isnan(
        _criterion(row_no_std, score="imbalance_validation", n=100, tau_R=1e-2, tau_K=1e-3)
    )


def test_selection_prefers_scoreable_candidates_over_missing_metric_ones(monkeypatch):
    """Integration: a candidate whose outcome fit fails on every inner fold has
    no b_hat/v_hat and must lose the bias_variance selection, not win it."""

    from genriesz.glm import FitResult, OutcomeGLM

    X, Y, _ = _make_ate(n=240, seed=21)
    basis = GaussianRKHSBasis(n_centers=30, sigma=1.0, random_state=0)
    cfg = GRRCVConfig(
        sigma_grid=[0.5, 1.0, 2.0], lam_grid=[1e-2], return_path=True, random_state=0
    )

    def run():
        return select_grr_hyperparams(
            X_train=X,
            y_train=Y,
            m=ATEFunctional(treatment_index=0),
            basis=basis,
            generator=SquaredGenerator(),
            config=cfg,
            riesz_lam=1e-2,
            outcome_link="identity",
        )

    winner = run().sigma  # the candidate that wins when everything is scoreable

    real_fit = OutcomeGLM.fit

    def poisoned_fit(self, X_, y_, **kw):
        sigma = getattr(self.basis, "sigma", None)
        if sigma is not None and np.isclose(float(sigma), float(winner)):
            self.theta_ = None
            return FitResult(
                beta=np.zeros(1), success=False, message="poisoned", n_iter=0,
                status="optimizer_failure",
            )
        return real_fit(self, X_, y_, **kw)

    monkeypatch.setattr(OutcomeGLM, "fit", poisoned_fit)
    res = run()

    # The previous winner is now un-evaluable: NaN criterion (it used to get
    # b = v = 0, the best possible score) and a different candidate is chosen.
    assert res.sigma != winner
    poisoned_rows = [r for r in res.path if r["sigma"] == pytest.approx(winner)]
    assert poisoned_rows
    assert all(np.isnan(r["criterion"]) for r in poisoned_rows)


def test_all_candidates_unscoreable_raises_for_bias_variance_only(monkeypatch):
    """Integration: with every outcome fit failing, bias_variance has nothing it
    can score (RuntimeError), while bregman_validation does not need the
    outcome side and still selects."""

    from genriesz.glm import FitResult, OutcomeGLM

    X, Y, _ = _make_ate(n=200, seed=22)
    basis = GaussianRKHSBasis(n_centers=30, sigma=1.0, random_state=0)

    def failing_fit(self, X_, y_, **kw):
        self.theta_ = None
        return FitResult(
            beta=np.zeros(1), success=False, message="always fails", n_iter=0,
            status="optimizer_failure",
        )

    monkeypatch.setattr(OutcomeGLM, "fit", failing_fit)

    def run(score):
        cfg = GRRCVConfig(lam_grid=[1e-2, 1e-1], selection_score=score, random_state=0)
        return select_grr_hyperparams(
            X_train=X,
            y_train=Y,
            m=ATEFunctional(treatment_index=0),
            basis=basis,
            generator=SquaredGenerator(),
            config=cfg,
            riesz_lam=1e-2,
            outcome_link="identity",
        )

    with pytest.raises(RuntimeError, match="admissibility screen"):
        run("bias_variance")

    res = run("bregman_validation")
    assert np.isfinite(res.best_score)


# ---------------------------------------------------------------------------
# Stratified inner splits (audit CV-13)
# ---------------------------------------------------------------------------


def _make_rare_ate(n: int = 60, n_treated: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    D = np.zeros(n)
    D[:n_treated] = 1.0
    rng.shuffle(D)
    X = np.column_stack([D, Z])
    Y = D + Z[:, 0] + rng.normal(scale=0.5, size=n)
    return X, Y


def test_inner_cv_is_stratified_on_a_binary_treatment():
    # 6 treated / 60 rows / 3 inner folds: plain K-fold can hand an inner fold
    # zero treated units and its metrics silently stop being comparable across
    # candidates. The inner split must keep both groups in every inner-training
    # fold, with the treated units spread across the folds as evenly as the
    # counts allow.
    X, Y = _make_rare_ate()
    cfg = GRRCVConfig(sigma_grid="auto", lam_grid=[1e-2], cv_folds=3, random_state=0)
    res = select_grr_hyperparams(X_train=X, y_train=Y, config=cfg, **_nested_common())

    treated_in_validation = []
    for prov in res.fold_provenance:
        d_tr = X[prov["preprocess_fit_index"], 0]
        assert np.any(d_tr == 1.0) and np.any(d_tr == 0.0)
        treated_in_validation.append(int(np.sum(X[prov["validation_index"], 0])))
    assert max(treated_in_validation) - min(treated_in_validation) <= 1


def test_inner_cv_with_a_single_treated_unit_raises():
    # Whichever inner fold's validation receives the only treated unit leaves
    # its inner-training all-control; that cannot be scored comparably and must
    # fail loudly rather than run through.
    X, Y = _make_rare_ate(n_treated=1)
    cfg = GRRCVConfig(sigma_grid="auto", lam_grid=[1e-2], cv_folds=3, random_state=0)
    with pytest.raises(ValueError, match="single unit"):
        select_grr_hyperparams(X_train=X, y_train=Y, config=cfg, **_nested_common())


def test_inner_cv_rejects_a_single_group_binary_sample():
    # A binary treatment column with only one group must not silently fall
    # back to a plain, unchecked K-fold: every inner-training fold would be
    # single-group, which is exactly what the stratified guard rejects.
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(40, 3))
    Y = Z[:, 0] + rng.normal(size=40)
    cfg = GRRCVConfig(sigma_grid="auto", lam_grid=[1e-2], cv_folds=3, random_state=0)
    for d_value in (1.0, 0.0):
        X = np.column_stack([np.full(40, d_value), Z])
        with pytest.raises(ValueError, match="single-group"):
            select_grr_hyperparams(X_train=X, y_train=Y, config=cfg, **_nested_common())
