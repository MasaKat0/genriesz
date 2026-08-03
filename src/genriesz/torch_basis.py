"""PyTorch-based feature maps for generalized Riesz regression.

Neural networks can provide fitted feature maps while the final generalized
Riesz regression remains linear in its coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import ArrayLike, NDArray

from .basis import BaseBasis


class MLPEmbeddingNet(nn.Module):
    """Feed-forward network that returns an embedding."""

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

        def make_activation() -> nn.Module:
            return nn.ReLU() if activation_lower == "relu" else nn.Tanh()

        layers: list[nn.Module] = []
        d_in = int(input_dim)
        for hidden in hidden_dims:
            layers.append(nn.Linear(d_in, int(hidden)))
            layers.append(make_activation())
            d_in = int(hidden)
        layers.append(nn.Linear(d_in, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TorchEmbeddingBasis(BaseBasis):
    """Feature map given by a fitted PyTorch embedding network."""

    net: MLPEmbeddingNet
    include_bias: bool = True
    device: str | None = None

    def __post_init__(self) -> None:
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
        """Fit the embedding to predict ``y`` when an outcome is supplied."""

        X_ = np.asarray(X, dtype=float)
        if y is None:
            self._is_fit = True
            return self

        y_ = np.asarray(y, dtype=float).reshape(-1, 1)
        last_layer = self.net.net[-1]
        if not isinstance(last_layer, nn.Linear):
            raise TypeError("The embedding network must end with a linear layer.")
        head = nn.Linear(last_layer.out_features, 1).to(self._device)
        optimizer = optim.Adam(
            list(self.net.parameters()) + list(head.parameters()), lr=float(lr)
        )
        loss_function = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_, dtype=torch.float32),
            torch.tensor(y_, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=int(batch_size), shuffle=True
        )

        self.net.train()
        head.train()
        for epoch in range(int(epochs)):
            total = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                prediction = head(self.net(x_batch))
                loss = loss_function(prediction, y_batch)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu().item()) * len(x_batch)
            if verbose:
                print(
                    f"[TorchEmbeddingBasis] epoch {epoch + 1}/{epochs} "
                    f"loss={total / len(dataset):.6f}"
                )

        self._is_fit = True
        return self

    @property
    def n_features(self) -> int:
        last_layer = self.net.net[-1]
        if not isinstance(last_layer, nn.Linear):
            raise TypeError("The embedding network must end with a linear layer.")
        return int(last_layer.out_features) + (1 if self.include_bias else 0)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        if not self._is_fit:
            raise RuntimeError("TorchEmbeddingBasis must be fit before use")
        X_ = np.asarray(X, dtype=float)
        single = X_.ndim == 1
        if single:
            X_ = X_.reshape(1, -1)
        if X_.ndim != 2:
            raise ValueError(f"X must be one- or two-dimensional, got {X_.shape}.")

        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_, dtype=torch.float32, device=self._device)
            embedding = self.net(x_tensor).detach().cpu().numpy().astype(float)
        if self.include_bias:
            embedding = np.concatenate(
                [np.ones((embedding.shape[0], 1), dtype=float), embedding], axis=1
            )
        return embedding[0] if single else embedding
