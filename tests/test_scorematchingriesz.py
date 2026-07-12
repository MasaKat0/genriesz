from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch optional dependency missing",
)


def test_scorematchingriesz_core_module_imports() -> None:
    import genriesz

    smr = genriesz.load_scorematchingriesz()
    assert hasattr(smr, "TimeScoreNet")
    assert hasattr(smr, "fit_time_smr_dre_infinity")
    assert not hasattr(smr, "run_trials_ame")


def test_crossfit_splits_cover_each_index_once() -> None:
    from genriesz.scorematchingriesz import crossfit_splits

    splits = crossfit_splits(11, n_folds=3, seed=123)
    test_indices = np.concatenate([test for _, test in splits])
    assert sorted(test_indices.tolist()) == list(range(11))
    for train, test in splits:
        assert set(train).isdisjoint(set(test))


def test_small_time_smr_ratio_training_runs() -> None:
    import torch

    from genriesz.scorematchingriesz import fit_time_smr_dre_infinity, log_ratio_from_time_score

    rng = np.random.default_rng(0)
    x_p = rng.normal(size=(24, 2)).astype("float32")
    x_q = rng.normal(loc=0.2, size=(24, 2)).astype("float32")

    model = fit_time_smr_dre_infinity(
        x_q,
        x_p,
        hidden_dims=(8,),
        t_emb_dim=8,
        n_steps=1,
        batch_size=8,
        lambda_kind="bump",
        seed=0,
        device=torch.device("cpu"),
    )
    log_r = log_ratio_from_time_score(
        model,
        x_p[:5],
        steps=4,
        normalize=True,
        x_p_for_norm=x_p,
        device=torch.device("cpu"),
    )
    assert tuple(log_r.shape) == (5, 1)
    assert torch.isfinite(log_r).all()


def test_small_data_smr_shift_ratio_runs() -> None:
    import torch

    from genriesz.scorematchingriesz import fit_data_smr_score_dsm, log_ratio_from_data_score_shift

    rng = np.random.default_rng(1)
    x = rng.normal(size=(24, 3)).astype("float32")
    model = fit_data_smr_score_dsm(
        x,
        hidden_dims=(8,),
        n_steps=1,
        batch_size=8,
        sigma_min=0.1,
        sigma_max=0.2,
        seed=1,
        device=torch.device("cpu"),
    )
    out = log_ratio_from_data_score_shift(
        model,
        x[:6],
        delta=0.1,
        steps=4,
        normalize=False,
        device=torch.device("cpu"),
    )
    assert out.shape == (6, 1)
    assert np.isfinite(out).all()


SMR_EPS = 1e-4  # interior truncation used by time_smr_loss / joint_smr_loss


class _ScoreRecorder:
    """Stand-in score model that records every ``(x, t)`` it is evaluated at."""

    def __init__(self, joint: bool = False) -> None:
        self.calls: list = []
        self._joint = joint

    def __call__(self, x, t):
        self.calls.append((x.detach().clone(), t.detach().clone()))
        # Returning t keeps the output differentiable w.r.t. t, which the losses
        # need for ds/dt. For the joint loss the first output must track x.
        return (x, t) if self._joint else t

    def boundary_calls(self) -> list:
        """The two calls made at a fixed time (the interior samples t at random)."""

        fixed = [(x, t) for x, t in self.calls if float(t.min()) == float(t.max())]
        return sorted(fixed, key=lambda c: float(c[1].flatten()[0]))


def _assert_boundary_at_bridge_endpoints(rec: _ScoreRecorder, xq, xp) -> None:
    import torch

    lo, hi = SMR_EPS, 1.0 - SMR_EPS
    boundary = rec.boundary_calls()
    assert len(boundary) == 2
    (x_lo, t_lo), (x_hi, t_hi) = boundary

    # The integration-by-parts boundary must be taken at the interior truncation
    # endpoints [eps, 1 - eps] -- never at t = 1 -- and evaluated on the *bridge*
    # samples at those times, not on the raw q/p endpoints. Getting either wrong
    # leaves a linear-in-s residual that makes the objective unbounded below.
    assert torch.allclose(t_lo, torch.full_like(t_lo, lo))
    assert torch.allclose(t_hi, torch.full_like(t_hi, hi))
    assert torch.allclose(x_lo, (1.0 - lo) * xq + lo * xp)
    assert torch.allclose(x_hi, (1.0 - hi) * xq + hi * xp)


def test_fit_defaults_use_bounded_lambda_kind() -> None:
    import inspect

    from genriesz.scorematchingriesz import (
        fit_joint_smr_dre_infinity,
        fit_time_smr_dre_infinity,
    )

    # "inv" blows up at the endpoints, so its mini-batch estimate has catastrophic
    # variance and training diverges; "bump" is the safe default.
    for fn in (fit_time_smr_dre_infinity, fit_joint_smr_dre_infinity):
        assert inspect.signature(fn).parameters["lambda_kind"].default == "bump"


def test_time_smr_loss_boundary_uses_bridge_endpoints() -> None:
    import torch

    from genriesz.scorematchingriesz import time_smr_loss

    torch.manual_seed(0)
    xq = torch.randn(16, 2)
    xp = torch.randn(16, 2)
    rec = _ScoreRecorder()
    _ = time_smr_loss(rec, xq, xp, lambda_kind="inv")
    _assert_boundary_at_bridge_endpoints(rec, xq, xp)


def test_joint_smr_loss_boundary_uses_bridge_endpoints() -> None:
    import torch

    from genriesz.scorematchingriesz import joint_smr_loss

    torch.manual_seed(0)
    xq = torch.randn(16, 2)
    xp = torch.randn(16, 2)
    rec = _ScoreRecorder(joint=True)
    _ = joint_smr_loss(rec, xq, xp, lambda_kind="inv")
    _assert_boundary_at_bridge_endpoints(rec, xq, xp)


@pytest.mark.parametrize("loss_name", ["time_smr_loss", "joint_smr_loss"])
def test_smr_loss_weights_the_boundary_at_the_truncated_endpoints(
    loss_name: str, monkeypatch
) -> None:
    """``lambda`` itself must be evaluated at ``[eps, 1 - eps]``.

    The recorder tests pin the times the *model* is handed; this pins the times the
    *weight* is handed, which is exactly where the original bug sat -- ``lambda(1)``
    rather than ``lambda(1 - eps)``. A "const" weight cannot see this (it is flat), so
    it has to be asserted on the calls themselves rather than on a loss value.
    """

    import torch

    import genriesz.scorematchingriesz as smr

    seen: list = []
    real_lambda_fn = smr.lambda_fn

    def spy(t, kind="const"):
        seen.append(t.detach().clone())
        return real_lambda_fn(t, kind=kind)

    monkeypatch.setattr(smr, "lambda_fn", spy)

    torch.manual_seed(0)
    xq = torch.randn(16, 2)
    xp = torch.randn(16, 2)
    rec = _ScoreRecorder(joint=loss_name == "joint_smr_loss")
    _ = getattr(smr, loss_name)(rec, xq, xp, lambda_kind="inv")

    # The interior weight is drawn at random times; the two boundary weights are the
    # ones evaluated at a single time held fixed across the batch.
    fixed = sorted(float(t.flatten()[0]) for t in seen if float(t.min()) == float(t.max()))
    assert fixed == pytest.approx([SMR_EPS, 1.0 - SMR_EPS])


_A, _B = SMR_EPS, 1.0 - SMR_EPS  # interior truncation endpoints


@pytest.mark.parametrize("loss_name", ["time_smr_loss", "joint_smr_loss"])
@pytest.mark.parametrize(
    ("score_kind", "expected", "tol"),
    [
        # s == 1: interior == 2*lam*0 + 2*lam'*1 + lam*1 == 1 pointwise, and the
        # boundary 2*lam(a)*1 - 2*lam(b)*1 cancels, so loss == int_a^b 1 dt == b - a.
        # Exact (no MC noise). Pins the interior's (b - a) scaling: drop it and the
        # loss becomes 1.0. It cannot see the boundary's sign -- both ends are equal.
        ("one", _B - _A, 1e-9),
        # s == t: interior == 2 + t^2 and the boundary is 2a - 2b != 0, so
        # loss == int_a^b (2 + t^2) dt + 2(a - b) == (b^3 - a^3) / 3. This one *does*
        # pin the boundary's sign and times: flipping the sign would give ~4.33.
        ("t", (_B**3 - _A**3) / 3.0, 5e-3),
    ],
)
def test_smr_loss_matches_integration_by_parts_identity(
    loss_name: str, score_kind: str, expected: float, tol: float
) -> None:
    """Both losses must reproduce the closed form the IBP identity implies.

    With ``lambda == "const"`` an analytic score makes the whole loss computable, so
    this checks the value itself -- the interior's ``(b - a)`` integral scaling and the
    boundary's times and sign -- which the recorder tests above cannot see.
    """

    import torch

    import genriesz.scorematchingriesz as smr

    joint = loss_name == "joint_smr_loss"

    def model(x, t):
        # s == 1 must stay differentiable w.r.t. t (ds/dt == 0), hence ``t * 0.0 + 1``.
        s_t = t * 0.0 + 1.0 if score_kind == "one" else t
        # For the joint loss the data score must track x; zero it out and pair with
        # data_weight=0 so only the time-score part contributes.
        return (x * 0.0, s_t) if joint else s_t

    torch.manual_seed(0)
    n = 65536
    xq = torch.randn(n, 2, dtype=torch.float64)
    xp = torch.randn(n, 2, dtype=torch.float64)

    kwargs = {"lambda_kind": "const"}
    if joint:
        kwargs["data_weight"] = 0.0
    loss = getattr(smr, loss_name)(model, xq, xp, **kwargs).item()

    assert loss == pytest.approx(expected, abs=tol)


@pytest.mark.parametrize("fit_name", ["fit_time_smr_dre_infinity", "fit_joint_smr_dre_infinity"])
def test_fit_warns_on_unstable_inv_lambda_kind(fit_name: str) -> None:
    import torch

    import genriesz.scorematchingriesz as smr

    fit = getattr(smr, fit_name)
    rng = np.random.default_rng(0)
    x_p = rng.normal(size=(24, 2)).astype("float32")
    x_q = rng.normal(loc=0.2, size=(24, 2)).astype("float32")

    with pytest.warns(RuntimeWarning, match="inv") as caught:
        fit(
            x_q,
            x_p,
            hidden_dims=(8,),
            t_emb_dim=8,
            n_steps=1,
            batch_size=8,
            lambda_kind="inv",
            seed=0,
            device=torch.device("cpu"),
        )

    # stacklevel must blame the caller, not a frame inside the library.
    assert caught[0].filename == __file__


def test_fit_functions_do_not_mutate_global_numpy_rng() -> None:
    from genriesz.scorematchingriesz import fit_sq_riesz_ame

    rng = np.random.default_rng(0)
    x = rng.normal(size=(48, 3))

    np.random.seed(12345)
    before = np.random.get_state()[1].copy()
    fit_sq_riesz_ame(x, n_steps=2, batch_size=16, device="cpu")
    after = np.random.get_state()[1].copy()

    assert np.array_equal(before, after)


def test_fit_functions_do_not_mutate_global_torch_rng() -> None:
    import torch

    from genriesz.scorematchingriesz import fit_sq_riesz_ame, fit_sq_riesz_ratio

    rng = np.random.default_rng(0)
    x = rng.normal(size=(48, 3))
    x_q = rng.normal(size=(48, 3))
    x_p = rng.normal(loc=0.4, size=(48, 3))

    # Put the global torch RNG in a non-trivial state, unrelated to any fit seed.
    torch.manual_seed(777)
    _ = torch.rand(5)
    before = torch.get_rng_state().clone()

    # torch.manual_seed would also seed MPS/XPU; the fits must not, since only
    # CPU+CUDA are saved and restored. Capture the MPS state too where present so
    # a regression back to torch.manual_seed in _seed_torch is caught here.
    mps_available = torch.backends.mps.is_available()
    if mps_available:
        import torch.mps

        torch.mps.manual_seed(222)
        mps_before = torch.mps.get_rng_state().clone()

    # A directly decorated fit and one that delegates to _fit_ratio_template must
    # both leave the global torch RNG exactly as they found it.
    fit_sq_riesz_ame(x, n_steps=2, batch_size=16, device="cpu")
    fit_sq_riesz_ratio(x_q, x_p, n_steps=2, batch_size=16, device="cpu")
    after = torch.get_rng_state()

    assert torch.equal(before, after)
    if mps_available:
        assert torch.equal(mps_before, torch.mps.get_rng_state())


def test_fit_is_reproducible_regardless_of_global_torch_rng() -> None:
    import torch

    from genriesz.scorematchingriesz import fit_sq_riesz_ratio

    rng = np.random.default_rng(2)
    x_q = rng.normal(size=(48, 2)).astype("float32")
    x_p = rng.normal(loc=0.3, size=(48, 2)).astype("float32")

    def params(seed: int) -> torch.Tensor:
        m = fit_sq_riesz_ratio(
            x_q, x_p, n_steps=5, batch_size=16, seed=seed, device="cpu"
        )
        return torch.cat([p.detach().reshape(-1) for p in m.parameters()])

    torch.manual_seed(1)
    a = params(0)
    torch.manual_seed(999999)
    _ = torch.rand(23)  # a very different global state before the second fit
    b = params(0)

    # Same seed -> identical fit, independent of the global RNG state going in.
    assert torch.allclose(a, b)
