import numpy as np
import pytest

from genriesz import (
    local_polynomial_nn_lsif_density_ratio,
    local_polynomial_nn_lsif_inverse_propensity_weights,
    nn_matching_inverse_propensity_weights,
)


def test_nn_matching_inverse_propensity_weights_simple():
    # Two treated, two controls, 1D covariate. No distance ties.
    Z = np.array([[0.0], [10.0], [2.0], [8.0]], dtype=float)
    D = np.array([1, 1, 0, 0], dtype=int)

    out = nn_matching_inverse_propensity_weights(X=Z, D=D, M=1, standardize=False)
    w = out.w

    assert w.shape == (4,)
    assert np.all(np.isfinite(w))

    # Matching weights are >= 1 for both groups in this construction.
    assert np.min(w) >= 1.0


def test_local_polynomial_nn_lsif_inverse_propensity_weights_runs():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    D = (rng.random(n) < 0.5).astype(int)

    out = local_polynomial_nn_lsif_inverse_propensity_weights(
        X=X,
        D=D,
        M=10,
        degree=1,
        kernel="ball",
        standardize=True,
    )

    assert out.w.shape == (n,)
    assert np.all(np.isfinite(out.w))


def test_local_polynomial_nn_lsif_inverse_propensity_weight_scale():
    rng = np.random.default_rng(1)
    n = 600
    X = rng.normal(size=(n, 2))
    D = rng.binomial(1, 0.5, size=n).astype(int)

    out = local_polynomial_nn_lsif_inverse_propensity_weights(
        X=X,
        D=D,
        M=40,
        degree=1,
        kernel="ball",
        standardize=True,
        clip_min=None,
        ridge=1e-8,
    )

    pi1 = float(np.mean(D))
    pi0 = 1.0 - pi1
    assert np.isclose(float(np.mean(out.w1)), 1.0 / pi1, rtol=0.35)
    assert np.isclose(float(np.mean(out.w0)), 1.0 / pi0, rtol=0.35)


def test_local_polynomial_nn_lsif_exclude_self_requires_room_for_neighbor():
    X = np.array([[0.0], [1.0], [2.0]], dtype=float)

    with pytest.raises(ValueError, match="exclude_self=True"):
        local_polynomial_nn_lsif_density_ratio(
            numerator=X,
            denominator=X,
            eval_points=X,
            M=len(X),
            degree=0,
            exclude_self=True,
        )


def test_exclude_self_uses_the_Mth_neighbor_radius_at_out_of_sample_points():
    """exclude_self must not widen the ball where there is no self to exclude.

    With ``degree=0`` and ``ridge=0`` the estimate is exactly the ratio of the
    in-ball counts, ``(n_Z/N1) / (n_X/N0)``, so it pins the radius. The M+1
    neighbors are queried up front; the code used to take the last of them at
    every eval point without a coincident neighbor, i.e. the (M+1)-th, so an
    out-of-sample point silently got an (M+1)-neighbor ball.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])  # denominator, N0 = 5
    Z = np.array([[0.4], [1.8]])  # numerator, N1 = 2
    kw = dict(numerator=Z, denominator=X, M=2, degree=0, ridge=0.0)

    # x = 0.5 is not a denominator point. Its 2nd-neighbor radius is 0.5, which
    # holds X = {0, 1} and Z = {0.4}: r = (1/2) / (2/5) = 1.25. The 3rd-neighbor
    # radius 1.5 would hold X = {0, 1, 2} and Z = {0.4, 1.8}: r = 1 / (3/5) = 1.667.
    oos = np.array([[0.5]])
    r_excl = local_polynomial_nn_lsif_density_ratio(eval_points=oos, exclude_self=True, **kw)
    r_keep = local_polynomial_nn_lsif_density_ratio(eval_points=oos, exclude_self=False, **kw)
    assert r_excl[0] == pytest.approx(1.25)
    # With no self match, excluding it changes nothing.
    assert r_excl[0] == pytest.approx(r_keep[0])

    # x = 0.0 *is* a denominator point: the coincident neighbor is dropped from
    # the radius definition *and* from the ball count, so the radius steps out
    # to the 3rd queried distance (2.0) and the ball holds X = {1, 2} (self
    # excluded) and Z = {0.4, 1.8}: r = (2/2) / (2/5) = 2.5. Keeping self in
    # the count would leave M+1 = 3 denominator points in every in-sample ball
    # and attenuate the estimate by ~ M/(M+1) (audit N-01). Without exclusion
    # the radius is the 2nd distance (1.0), holding X = {0, 1} and Z = {0.4}:
    # r = 1.25.
    ins = np.array([[0.0]])
    r_excl_in = local_polynomial_nn_lsif_density_ratio(eval_points=ins, exclude_self=True, **kw)
    r_keep_in = local_polynomial_nn_lsif_density_ratio(eval_points=ins, exclude_self=False, **kw)
    assert r_excl_in[0] == pytest.approx(2.5)
    assert r_keep_in[0] == pytest.approx(1.25)


def test_non_euclidean_metric_raises_instead_of_being_ignored():
    # An unsupported metric fails loudly with NotImplementedError, rather than
    # being accepted and quietly ignored.
    Z = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    D = np.array([1, 1, 0, 0], dtype=int)

    with pytest.raises(NotImplementedError, match="euclidean"):
        nn_matching_inverse_propensity_weights(X=Z, D=D, M=1, metric="manhattan")


def _algorithm_deprecation_cases():
    Z = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    D = np.array([1, 1, 0, 0], dtype=int)
    return [
        lambda alg: nn_matching_inverse_propensity_weights(X=Z, D=D, M=1, algorithm=alg),
        lambda alg: local_polynomial_nn_lsif_density_ratio(
            numerator=Z, denominator=Z, eval_points=Z, M=2, algorithm=alg
        ),
        lambda alg: local_polynomial_nn_lsif_inverse_propensity_weights(
            X=Z, D=D, M=1, algorithm=alg
        ),
    ]


@pytest.mark.parametrize("call", _algorithm_deprecation_cases())
def test_algorithm_keyword_is_deprecated_but_still_accepted(call):
    # ``algorithm`` is a no-op kept for backward compatibility. A non-default
    # value must warn (rather than be silently ignored) but still succeed.
    with pytest.warns(DeprecationWarning, match="algorithm") as record:
        call("kd_tree")
    # Exactly one warning: the IPW wrapper must not double-warn by also
    # forwarding `algorithm` to the inner density-ratio helper.
    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1


@pytest.mark.parametrize("call", _algorithm_deprecation_cases())
def test_algorithm_default_does_not_warn(call, recwarn):
    # The default "auto" must stay quiet, so ordinary callers see no warning.
    call("auto")
    assert not [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]


def test_exclude_self_drops_one_self_per_side_and_keeps_duplicates():
    # The eval point x=2 occurs twice in the denominator and once in the
    # numerator. Exactly one coincident row is dropped from each side; the
    # remaining duplicate is a distinct observation and stays. Radius: the
    # 6 denominator distances are [0, 0, 1, 1, 2, 2] and with a self match the
    # M-th usable neighbor is entry M -> rho = 1. Denominator ball {1, 2, 2, 3}
    # loses one coincident row -> H00 = 3/6; numerator ball {2.0, 1.5} loses
    # the coincident 2.0 -> h = 1/3. r = (1/3) / (3/6) = 2/3.
    X = np.array([[0.0], [1.0], [2.0], [2.0], [3.0], [4.0]])
    Z = np.array([[2.0], [1.5], [10.0]])
    r = local_polynomial_nn_lsif_density_ratio(
        numerator=Z,
        denominator=X,
        eval_points=np.array([[2.0]]),
        M=2,
        degree=0,
        ridge=0.0,
        exclude_self=True,
    )
    assert r[0] == pytest.approx(2.0 / 3.0)
