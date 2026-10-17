"""RieszNet for the ATE: Chernozhukov et al. (ICML 2022), §3, §5 and Appendix A.1."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

LAMBDA_RIESZ = 0.1
LAMBDA_TARGETED = 1.0
L2 = 1e-3
SCHEDULE = ((1e-4, 100, 2), (1e-5, 600, 40))  # learning rate, epochs, patience
MIN_DELTA = 1e-4
TRAIN_FRACTION = 0.2
BATCH_SIZE = 256


class RieszNet(nn.Module):
    """Three shared 200-unit ELU layers, linear RR head, two regression heads.

    Each treatment-specific regression head has two 100-unit ELU layers and a
    linear output (Appendix A.1). Inputs are raw (D, Z), with no feature or outcome
    scaling and no representer offset. Linear layers include bias intercepts.
    The scalar epsilon is learned jointly and is excluded from L2 (Eq. (5)).
    """

    def __init__(self, input_dim, seed):
        super().__init__()
        # fork_rng keeps construction from consuming other arms' random streams.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 200), nn.ELU(),
                nn.Linear(200, 200), nn.ELU(), nn.Linear(200, 200), nn.ELU(),
            )
            self.riesz = nn.Linear(200, 1)
            self.regression = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(200, 100), nn.ELU(), nn.Linear(100, 100), nn.ELU(),
                    nn.Linear(100, 1),
                ) for _ in range(2)
            ])
            self.epsilon = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.double()

    def forward(self, X):
        h = self.shared(X)
        alpha = self.riesz(h).squeeze(1)
        g0, g1 = (head(h).squeeze(1) for head in self.regression)
        g = torch.where(X[:, 0] == 1, g1, g0)
        return alpha, g


def loss_terms(net, X, Y):
    """Eq. (5), with sample means for each loss and L2 = 1e-3 * sum weights^2.

    RRloss = mean(alpha(D,Z)^2 - 2*(alpha(1,Z)-alpha(0,Z)));
    REGloss = mean((Y-g(D,Z))^2);
    TMLEloss = mean((Y-g(D,Z)-epsilon*alpha(D,Z))^2).
    All network parameters, including biases, enter L2, but epsilon does not.
    This explicit L2 convention avoids optimizer-dependent weight-decay scaling.
    No gradient is detached: all three losses train the shared representation.
    """
    X1, X0 = X.clone(), X.clone()
    X1[:, 0], X0[:, 0] = 1.0, 0.0
    a, g = net(X)
    a1, _ = net(X1)
    a0, _ = net(X0)
    reg = ((Y - g) ** 2).mean()
    rr = (a**2 - 2.0 * (a1 - a0)).mean()
    targeted = ((Y - g - net.epsilon * a) ** 2).mean()
    penalty = L2 * sum((p**2).sum() for name, p in net.named_parameters()
                       if name != "epsilon")
    return {"regression": reg, "riesz": rr, "targeted": targeted, "l2": penalty,
            "total": reg + LAMBDA_RIESZ * rr + LAMBDA_TARGETED * targeted + penalty}


def fit(seed, X, Y):
    """Adam and the two-stage schedule in Appendix A.1, inside an outer train fold.

    We follow the appendix's literal 20% training and 80% validation split; neither
    contains outer held-out observations. The appendix's 'test error' rule is
    interpreted as patience of 2 then 40 epochs without a decrease of at least
    1e-4 in validation Eq. (5) loss. Each stage restores its best checkpoint.
    Batch size (256), float64, raw input scaling and default PyTorch linear-layer
    initialization are E-24 choices where the paper gives no value. Adam defaults
    apply, with explicit L2 in the loss and no optimizer weight decay.
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.shape != (len(X),) or len(X) < 5:
        raise ValueError("RieszNet needs at least five rows and one outcome per row")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise FloatingPointError("nonfinite RieszNet input")
    if not np.all(np.isin(X[:, 0], [0.0, 1.0])):
        raise ValueError("RieszNet ATE requires binary treatment in column zero")
    rng = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(X), generator=rng)
    n_train = int(math.floor(TRAIN_FRACTION * len(X)))
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Yt = torch.as_tensor(Y, dtype=torch.float64)
    tr, va = perm[:n_train], perm[n_train:]
    if len(torch.unique(Xt[tr, 0])) != 2:
        raise ValueError("degenerate_functional")
    net = RieszNet(X.shape[1], seed)
    stages = []
    for lr, epochs, patience in SCHEDULE:
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        with torch.no_grad():
            best = float(loss_terms(net, Xt[va], Yt[va])["total"])
        if not np.isfinite(best):
            raise FloatingPointError("nonfinite RieszNet validation loss")
        checkpoint = copy.deepcopy(net.state_dict())
        initial_loss = anchor = best
        wait = 0
        for _epoch in range(epochs):
            net.train()
            order = tr[torch.randperm(len(tr), generator=rng)]
            for start in range(0, len(tr), BATCH_SIZE):
                idx = order[start : start + BATCH_SIZE]
                loss = loss_terms(net, Xt[idx], Yt[idx])["total"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite RieszNet training loss")
                opt.zero_grad()
                loss.backward()
                if not all(p.grad is None or torch.all(torch.isfinite(p.grad))
                           for p in net.parameters()):
                    raise FloatingPointError("nonfinite RieszNet gradient")
                opt.step()
            net.eval()
            with torch.no_grad():
                val = float(loss_terms(net, Xt[va], Yt[va])["total"])
            if not np.isfinite(val):
                raise FloatingPointError("nonfinite RieszNet validation loss")
            if val < best:
                best, checkpoint = val, copy.deepcopy(net.state_dict())
            if val < anchor - MIN_DELTA:
                anchor, wait = val, 0
            else:
                wait += 1
            if wait >= patience:
                break
        net.load_state_dict(checkpoint)
        stages.append({"lr": lr, "epochs": _epoch + 1,
                       "initial_validation_loss": initial_loss, "best_validation_loss": best,
                       "last_validation_loss": val, "early_stopped": wait >= patience})
    net.eval()
    return net, {"train_rows": len(tr), "validation_rows": len(va), "stages": stages,
                 "epsilon": float(net.epsilon.detach()),
                 "settings": {"shared_widths": [200, 200, 200], "head_widths": [100, 100],
                              "activation": "ELU", "lambda_riesz": LAMBDA_RIESZ,
                              "lambda_targeted": LAMBDA_TARGETED, "l2": L2,
                              "schedule": SCHEDULE, "min_delta": MIN_DELTA,
                              "train_fraction": TRAIN_FRACTION, "batch_size": BATCH_SIZE,
                              "dtype": "float64", "input_scaling": "none", "seed": int(seed)}}
