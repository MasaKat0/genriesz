"""Optional PyTorch-based basis functions.

This module is only needed if you want to use a neural network as a *feature map*
for :class:`genriesz.glm.GRRGLM`.

The recommended pattern for keeping the GLM solver convex in `beta` (and thus
preserving the exact ACB structure for the chosen basis) is:

1) train an embedding network psi(Z) however you like,
2) freeze it,
3) feed its output as a fixed basis into GRRGLM.

If you instead train the embedding network jointly inside the GRR objective, you
leave the GLM setting and should not expect finite-sample *exact* covariate
balancing for the network features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for genriesz.torch_basis. Install with: pip install genriesz[torch]"
    ) from e


Array = np.ndarray


@dataclass
class TorchEmbeddingBasis:
    """Wrap a PyTorch module as a NumPy-returning basis function.

    Parameters
    ----------
    model:
        A `torch.nn.Module` mapping a tensor of shape (n, d) to (n, p).
    device:
        Device on which to run the model. If None, uses CPU.
    include_bias:
        If True, append a constant feature 1.
    """

    model: nn.Module
    device: Optional[str] = None
    include_bias: bool = False

    def __post_init__(self) -> None:
        self.device = self.device or "cpu"
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, X: Array) -> Array:
        X_in = np.asarray(X, dtype=float)
        if X_in.ndim == 1:
            X2 = X_in.reshape(1, -1)
        elif X_in.ndim == 2:
            X2 = X_in
        else:
            raise ValueError(f"X must be 1D or 2D. Got shape {X_in.shape}.")

        with torch.no_grad():
            xt = torch.tensor(X2, dtype=torch.float32, device=self.device)
            out = self.model(xt)
            out_np = out.detach().cpu().numpy()

        if out_np.ndim != 2:
            raise ValueError(
                "TorchEmbeddingBasis expects the model to return a 2D tensor (n,p). "
                f"Got shape {out_np.shape}."
            )

        if self.include_bias:
            out_np = np.concatenate([out_np, np.ones((len(out_np), 1))], axis=1)

        if X_in.ndim == 1:
            return out_np.reshape(-1)
        return out_np


class MLPEmbeddingNet(nn.Module):
    """A small default MLP for building embeddings.

    This is provided only as a convenience for examples.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
