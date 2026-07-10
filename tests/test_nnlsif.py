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
