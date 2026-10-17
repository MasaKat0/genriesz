"""RieszNet architecture, Eq. (5), Appendix A.1 training and DR integration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from grrexp import e24, riesznet  # noqa: E402


def sample():
    Z = np.random.default_rng(19).normal(size=(40, 6))
    X = np.column_stack([np.arange(40) % 2, Z])
    return X, X[:, 0] + 0.2 * Z[:, 0]


def test_architecture_shapes_seed_and_shared_representation():
    state = torch.random.get_rng_state().clone()
    net = riesznet.RieszNet(7, 101)
    assert torch.equal(state, torch.random.get_rng_state())
    linear = [m for m in net.shared if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear] == [(7, 200), (200, 200), (200, 200)]
    assert net.riesz.in_features == 200 and net.riesz.out_features == 1
    for head in net.regression:
        layers = [m for m in head if isinstance(m, torch.nn.Linear)]
        assert [(m.in_features, m.out_features) for m in layers] == [
            (200, 100),
            (100, 100),
            (100, 1),
        ]
    X, _ = sample()
    a, g = net(torch.tensor(X))
    assert a.shape == g.shape == (40,) and a.dtype == g.dtype == torch.float64
    assert sum(isinstance(m, torch.nn.ELU) for m in net.modules()) == 7
    again = riesznet.RieszNet(7, 101)
    assert all(torch.equal(p, q) for p, q in zip(net.parameters(), again.parameters(), strict=True))


def test_loss_terms_and_gradients_match_equation_five():
    net = riesznet.RieszNet(7, 101)
    with torch.no_grad():
        net.epsilon.fill_(0.3)
    X, Y = (torch.tensor(v) for v in sample())
    X1, X0 = X.clone(), X.clone()
    X1[:, 0], X0[:, 0] = 1.0, 0.0
    a, g = net(X)
    a1, _ = net(X1)
    a0, _ = net(X0)
    terms = riesznet.loss_terms(net, X, Y)
    assert torch.allclose(terms["regression"], ((Y - g) ** 2).mean())
    assert torch.allclose(terms["riesz"], (a**2 - 2 * (a1 - a0)).mean())
    assert torch.allclose(terms["targeted"], ((Y - g - 0.3 * a) ** 2).mean())
    pen = 1e-3 * sum(p.square().sum() for name, p in net.named_parameters() if name != "epsilon")
    assert torch.allclose(terms["l2"], pen)
    assert torch.allclose(
        terms["total"], terms["regression"] + 0.1 * terms["riesz"] + terms["targeted"] + pen
    )
    for name in ("regression", "riesz", "targeted"):
        (grad,) = torch.autograd.grad(terms[name], net.shared[0].weight, retain_graph=True)
        assert torch.count_nonzero(grad) > 0
    assert torch.autograd.grad(pen, net.epsilon, allow_unused=True)[0] is None
    terms["total"].backward()
    assert net.epsilon.grad != 0
    assert all(p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in net.parameters())


def test_tiny_training_decreases_loss_and_runs_the_published_schedule():
    e24._torch()
    X, Y = sample()
    seed = 101
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    tr, va = perm[:8], perm[8:]
    initial = riesznet.RieszNet(7, seed)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    initial_loss = float(riesznet.loss_terms(initial, Xt[va], Yt[va])["total"].detach())
    net, info = riesznet.fit(seed, X, Y)
    final_loss = float(riesznet.loss_terms(net, Xt[va], Yt[va])["total"].detach())
    assert final_loss < initial_loss
    assert info["train_rows"] == len(tr) == 8 and info["validation_rows"] == len(va) == 32
    assert [r["lr"] for r in info["stages"]] == [1e-4, 1e-5]
    assert all(
        1 <= r["epochs"] <= cap
        for r, (_, cap, _) in zip(info["stages"], riesznet.SCHEDULE, strict=True)
    )
    assert all(r["best_validation_loss"] <= r["initial_validation_loss"] for r in info["stages"])
    assert info["settings"]["schedule"] == ((1e-4, 100, 2), (1e-5, 600, 40))
    assert info["settings"]["l2"] == 1e-3
    assert net.epsilon.item() != 0
    again, other = riesznet.fit(seed, X, Y)
    assert info == other
    assert all(torch.equal(p, q) for p, q in zip(net.parameters(), again.parameters(), strict=True))


def test_training_nonfinite_and_degenerate_samples_surface():
    X, Y = sample()
    with pytest.raises(FloatingPointError, match="nonfinite"):
        riesznet.fit(101, X, np.full(len(Y), np.nan))
    X[:, 0] = 0.0
    with pytest.raises(ValueError, match="degenerate_functional"):
        riesznet.fit(101, X, Y)


def test_riesznet_rep_returns_both_nuisances(monkeypatch):
    X, Y = sample()
    net = riesznet.RieszNet(7, 101)
    monkeypatch.setattr(riesznet, "fit", lambda seed, Xt, Yt: (net, {"epsilon": 0.0}))
    rep = e24.fit_riesznet(101, X, Y)
    a, a1, a0, st = rep.at(X)
    assert st == "ok" and a.shape == a1.shape == a0.shape == (40,)
    assert np.array_equal(rep.outcome(X), net(torch.tensor(X))[1].detach().numpy())


def test_dr_estimator_uses_sample_sd_and_the_network_regression():
    X, Y = sample()
    net = riesznet.RieszNet(7, 101)
    Xt = torch.tensor(X)
    X1, X0 = Xt.clone(), Xt.clone()
    X1[:, 0], X0[:, 0] = 1.0, 0.0
    with torch.no_grad():
        a, g = (p.numpy() for p in net(Xt))
        a1, g1 = (p.numpy() for p in net(X1))
        a0, g0 = (p.numpy() for p in net(X0))
    from grrexp import study

    moment = g1 - g0 + a * (Y - g)
    result = study.score_record(moment, ddof=1)
    assert result["estimate"] == pytest.approx(moment.mean())
    assert result["se"] == pytest.approx(moment.std(ddof=1) / np.sqrt(len(Y)))
