"""Optional PyTorch-based bases.

Neural networks can be used as flexible feature maps. The GRR solvers remain
linear in the coefficients, but the basis itself can be learned.

This module provides:

- :class:`MLPEmbeddingNet`  - a simple MLP producing an embedding
- :class:`TorchEmbeddingBasis` - trains the embedding network (optionally) and
  exposes the embedding as a NumPy feature map.

The implementation is intentionally minimal and is meant for examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import BaseBasis

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


if torch is not None:

    class MLPEmbeddingNet(nn.Module):
        """A simple feed-forward network that outputs an embedding."""

        def __init__(
            self,
            *,
            input_dim: int,
            hidden_dims: tuple[int, ...] = (32, 16),
            output_dim: int = 8,
            activation: str = "relu",
        ) -> None:
            super().__init__()

            activation_lower = activation.lower()
            if activation_lower not in {"relu", "tanh"}:
                raise ValueError(f"Unknown activation: {activation}")

            def _make_act() -> nn.Module:
                return nn.ReLU() if activation_lower == "relu" else nn.Tanh()

            layers: list[nn.Module] = []
            d_in = int(input_dim)
            for h in hidden_dims:
                layers.append(nn.Linear(d_in, int(h)))
                layers.append(_make_act())
                d_in = int(h)
            layers.append(nn.Linear(d_in, int(output_dim)))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            return self.net(x)

else:  # pragma: no cover - exercised only without the optional dependency

    class MLPEmbeddingNet:  # type: ignore[no-redef]
        """Placeholder that raises a clean ImportError when PyTorch is absent."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "MLPEmbeddingNet requires PyTorch. Install it with `pip install torch` "
                "or `pip install genriesz[torch]`."
            ) from _IMPORT_ERROR


@dataclass
class TorchEmbeddingBasis(BaseBasis):
    """A basis given by a learned PyTorch embedding network."""

    net: MLPEmbeddingNet
    include_bias: bool = True
    device: str | None = None

    def __post_init__(self) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("TorchEmbeddingBasis requires PyTorch") from _IMPORT_ERROR

        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self._device)
        self._is_fit = False

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 128,
        verbose: bool = False,
    ):
        """Optionally train the embedding to predict ``y``.

        If ``y`` is None, this method only marks the basis as fit.
        """

        if torch is None:  # pragma: no cover
            raise ImportError("TorchEmbeddingBasis requires PyTorch") from _IMPORT_ERROR

        X_ = np.asarray(X, dtype=float)
        if y is None:
            self._is_fit = True
            return self

        y_ = np.asarray(y, dtype=float).reshape(-1, 1)

        # Simple supervised training with a linear prediction head.
        last_layer = self.net.net[-1]
        assert isinstance(last_layer, nn.Linear)
        head = nn.Linear(last_layer.out_features, 1).to(self._device)
        params = list(self.net.parameters()) + list(head.parameters())
        opt = optim.Adam(params, lr=float(lr))
        loss_fn = nn.MSELoss()

        ds = torch.utils.data.TensorDataset(
            torch.tensor(X_, dtype=torch.float32),
            torch.tensor(y_, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=int(batch_size), shuffle=True)

        self.net.train()
        head.train()
        for ep in range(int(epochs)):
            total = 0.0
            for xb, yb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)

                opt.zero_grad(set_to_none=True)
                emb = self.net(xb)
                pred = head(emb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu().item()) * len(xb)

            if verbose:
                print(f"[TorchEmbeddingBasis] epoch {ep+1}/{epochs} loss={total/len(ds):.6f}")

        self._is_fit = True
        return self

    @property
    def n_features(self) -> int:
        last_layer = self.net.net[-1]
        assert isinstance(last_layer, nn.Linear)
        out_dim = int(last_layer.out_features)
        return out_dim + (1 if self.include_bias else 0)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        if torch is None:  # pragma: no cover
            raise ImportError("TorchEmbeddingBasis requires PyTorch") from _IMPORT_ERROR
        if not self._is_fit:
            raise RuntimeError("TorchEmbeddingBasis must be fit before use")

        X_ = np.asarray(X, dtype=float)
        single = X_.ndim == 1
        if single:
            X_ = X_.reshape(1, -1)

        self.net.eval()
        with torch.no_grad():
            xb = torch.tensor(X_, dtype=torch.float32, device=self._device)
            emb = self.net(xb).detach().cpu().numpy().astype(float)

        if self.include_bias:
            emb = np.concatenate([np.ones((emb.shape[0], 1), dtype=float), emb], axis=1)
        return emb[0] if single else emb
