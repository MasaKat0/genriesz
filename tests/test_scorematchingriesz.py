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
