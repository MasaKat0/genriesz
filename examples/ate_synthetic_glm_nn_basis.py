"""Synthetic ATE example using a neural-network embedding as the basis.

Run:

    python examples/ate_synthetic_glm_nn_basis.py

This example illustrates how to keep the *GLM-style* GRR solver (and its
automatic link construction) while still using a neural network:

1) Train an embedding network psi(Z) on an auxiliary task (here: predict D).
2) Freeze psi.
3) Use psi(Z) as the basis inside :class:`genriesz.glm.GRR`.

This keeps the GRR optimization convex in beta and preserves the ACB structure
for the fixed features.
"""

from __future__ import annotations

import numpy as np

from genriesz import ATEFunctional, GRR, TreatmentInteractionBasis, UKLGenerator
from genriesz.torch_basis import MLPEmbeddingNet, TorchEmbeddingBasis


def make_synthetic_data(n: int = 3000, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))

    logits = 0.6 * Z[:, 0] - 0.4 * Z[:, 1] + 0.2 * Z[:, 2]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n)

    tau = 1.0
    mu0 = np.sin(Z[:, 0]) + 0.25 * Z[:, 1] ** 2
    Y = mu0 + tau * D + rng.normal(scale=1.0, size=n)

    X = np.concatenate([D.reshape(-1, 1), Z], axis=1)
    return X, Z, D, Y, tau


def train_embedding_on_treatment(Z: np.ndarray, D: np.ndarray, *, seed: int = 0) -> MLPEmbeddingNet:
    """Train a small embedding network psi(Z) to predict D.

    This is just one reasonable way to learn a representation.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)

    n, d = Z.shape
    embed = MLPEmbeddingNet(input_dim=d, hidden_dim=64, output_dim=32)
    head = nn.Linear(32, 1)
    model = nn.Sequential(embed, head)

    xt = torch.tensor(Z, dtype=torch.float32)
    yt = torch.tensor(D.reshape(-1, 1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=256, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _epoch in range(5):
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    # Freeze the embedding and return it.
    embed.eval()
    for p in embed.parameters():
        p.requires_grad_(False)
    return embed


def main() -> None:
    X, Z, D, Y, tau = make_synthetic_data(n=3000, d=5, seed=0)

    # 1) Train psi(Z)
    embed = train_embedding_on_treatment(Z, D, seed=0)

    # 2) Wrap as a NumPy-returning basis
    psi = TorchEmbeddingBasis(embed, device="cpu")

    # 3) Add treatment interactions so phi(W) is a basis on W=[D,Z]
    phi = TreatmentInteractionBasis(base_basis=psi)

    # ATE functional (vectorized)
    m = ATEFunctional(treatment_index=0)

    # UKL generator (automatic link)
    gen = UKLGenerator(C=1.0, branch_fn=lambda w: int(w[0] == 1)).as_generator()

    est = GRR(basis=phi, m=m, generator=gen, penalty="l2", lam=1e-3)
    est.fit(X, max_iter=400, tol=1e-9)

    ate_hat = est.estimate_linear_functional(Y, X)

    print("True ATE (by construction):", tau)
    print("Estimated ATE:", float(ate_hat))


if __name__ == "__main__":
    main()
