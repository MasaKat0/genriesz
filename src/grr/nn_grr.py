from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "NN_GRR requires PyTorch. Install torch first, or install this project with the extra 'torch'."
    ) from e


class NN_GRR:
    """Neural-network learner for a Riesz representer and a regression model.

    The class implements two learned components:

    - A Riesz representer model (used for IPW / AIPW)
    - A regression model for E[Y | X, T] (used for DM / AIPW)

    The public methods are:

    - :meth:`riesz_fit` / :meth:`riesz_predict`
    - :meth:`reg_fit` / :meth:`reg_predict` / :meth:`reg_predict_diff`

    Notes
    -----
    This module intentionally keeps the original training logic.
    """

    def __init__(self) -> None:
        # Attributes are created during fitting
        pass

    # ------------------------------------------------------------------
    # Riesz model construction and losses
    # ------------------------------------------------------------------
    def construct_riesz_model(self) -> None:
        """Construct the neural network used for the Riesz representer."""
        self.riesz_fc1 = nn.Linear(self.riesz_input_dim, self.riesz_hidden_dim)
        self.riesz_fc2 = nn.Linear(self.riesz_hidden_dim, 1)

        self.riesz_model = nn.Sequential(
            self.riesz_fc1,
            nn.ReLU(),
            self.riesz_fc2,
        )

    def _sq_loss_function(self, outputs, outputs_one, outputs_zero, targets, upperbound: float = 20):
        """Squared-distance-style loss used for Riesz representer fitting.

        The original code uses a piecewise form (with a heuristic stabilization).
        """

        loss1 = -torch.mean(
            2 * (outputs_one - outputs_zero) - (outputs_one**2 + outputs_zero**2) / upperbound
        )
        loss2 = torch.mean(
            -(outputs_one**2 + outputs_zero**2) / upperbound + outputs**2
        )
        if loss2 > 0:
            loss = loss1 + loss2
        else:
            loss = -0.01 * loss2

        return loss + self.lbd * self._l2_regularization()

    def _ukl_loss_function(self, outputs, outputs_one, outputs_zero, targets, upperbound: float = 20):
        """Unnormalized-KL-style loss used for Riesz representer fitting."""

        loss1 = torch.mean(
            (
                torch.log(torch.abs(outputs_one) - 1)
                + torch.log(torch.abs(outputs_zero) - 1)
                + torch.abs(outputs_one)
                + torch.abs(outputs_zero)
            )
            / upperbound
            - torch.log(torch.abs(outputs_one) - 1)
            - torch.log(torch.abs(outputs_zero) - 1)
        )
        loss2 = torch.mean(
            -(
                torch.log(torch.abs(outputs_one) - 1)
                + torch.log(torch.abs(outputs_zero) - 1)
                + torch.abs(outputs_one)
                + torch.abs(outputs_zero)
            )
            / upperbound
            + torch.log(torch.abs(outputs) - 1)
            + torch.abs(outputs)
        )

        if loss2 > 0:
            loss = loss1 + loss2
        else:
            loss = -0.01 * loss2

        return torch.mean(loss) + self.lbd * self._l2_regularization()

    def _bkl_loss_function(self, outputs, outputs_one, outputs_zero, targets):
        """Binary-KL / MLE-type loss using the ATE Riesz representer parameterization."""

        loss = -targets * torch.log(1 / outputs_one) - (1 - targets) * torch.log(-1 / outputs_zero)
        return torch.mean(loss) + self.lbd * self._l2_regularization()

    def _logistic_loss_function(self, outputs, outputs_one, outputs_zero, targets):
        """Placeholder for a custom logistic-style loss."""
        raise NotImplementedError(
            "Custom 'Logit' loss is not implemented. Use 'SQ', 'UKL', or 'BKL', "
            "or implement _logistic_loss_function."
        )

    def _l2_regularization(self):
        """Compute L2 regularization term over the parameters of the Riesz model."""
        return sum(torch.sum(param**2) for param in self.riesz_model.parameters())

    # ------------------------------------------------------------------
    # Riesz fit / predict
    # ------------------------------------------------------------------
    def riesz_fit(
        self,
        X,
        T,
        riesz_loss: str,
        riesz_with_D: bool,
        riesz_link_name: str,
        riesz_hidden_dim: int = 100,
        riesz_max_iter: int = 3000,
        tol: float = 1e-10,
        lbd: float = 0.01,
        lr: float = 0.01,
        batch_size: int = 1000,
    ) -> None:
        """Fit the Riesz model using mini-batch optimization.

        Parameters
        ----------
        X:
            Covariates, shape (n, d).
        T:
            Binary treatment indicator, shape (n,) or (n, 1).
        riesz_loss:
            Loss function name. Supported values: ``"SQ"``, ``"UKL"``, ``"BKL"``.
        riesz_with_D:
            If True, concatenate treatment to covariates as an input feature.
        riesz_link_name:
            If ``"Logit"``, interpret the network outputs as propensity logits and map
            them to the ATE Riesz representer.
        riesz_hidden_dim, riesz_max_iter, tol, lbd, lr, batch_size:
            Standard training hyperparameters.
        """

        self.riesz_hidden_dim = riesz_hidden_dim
        self.riesz_max_iter = riesz_max_iter
        self.tol = tol
        self.lbd = lbd
        self.lr = lr
        self.riesz_link_name = riesz_link_name
        self.riesz_with_D = riesz_with_D

        riesz_input_dim = np.asarray(X).shape[1]

        # Construct input dimension with or without treatment indicator
        self.riesz_input_dim = riesz_input_dim + 1 if self.riesz_with_D else riesz_input_dim

        # Build the Riesz neural network
        self.construct_riesz_model()

        # Select loss function
        if riesz_loss == "Logit":
            self.criterion = self._logistic_loss_function
        elif riesz_loss == "SQ":
            self.criterion = self._sq_loss_function
        elif riesz_loss == "UKL":
            self.criterion = self._ukl_loss_function
        elif riesz_loss == "BKL":
            self.criterion = self._bkl_loss_function
        else:
            raise ValueError(f"Invalid loss function specified: {riesz_loss}")

        self.optimizer = optim.Adam(self.riesz_model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        T_tensor = torch.tensor(T, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_tensor, T_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        prev_loss = float("inf")
        for _epoch in range(riesz_max_iter):
            for X_batch, T_batch in dataloader:
                self.optimizer.zero_grad()

                if self.riesz_with_D:
                    # Create three inputs: using actual T, forcing T=1, forcing T=0
                    T_batch_zero = torch.zeros_like(T_batch)
                    T_batch_one = torch.ones_like(T_batch)

                    X_batch_one = torch.cat([X_batch, T_batch_one], dim=1)
                    X_batch_zero = torch.cat([X_batch, T_batch_zero], dim=1)
                    X_batch_full = torch.cat([X_batch, T_batch], dim=1)
                else:
                    X_batch_one = X_batch
                    X_batch_zero = X_batch
                    X_batch_full = X_batch

                # Forward pass for all three inputs
                raw_outputs = self.riesz_model(X_batch_full)
                raw_outputs_one = self.riesz_model(X_batch_one)
                raw_outputs_zero = self.riesz_model(X_batch_zero)

                if self.riesz_link_name == "Logit":
                    # Interpret raw outputs as propensity logits and map to Riesz representer.
                    prop_outputs_one = torch.sigmoid(raw_outputs_one)
                    prop_outputs_zero = torch.sigmoid(raw_outputs_zero)

                    prop_outputs_one = torch.clamp(prop_outputs_one, min=0.05, max=0.95)
                    prop_outputs_zero = torch.clamp(prop_outputs_zero, min=0.05, max=0.95)

                    # ATE Riesz representer:
                    #   g(X, T) = T / p(X) - (1 - T) / (1 - p(X))
                    outputs = T_batch / prop_outputs_one - (1 - T_batch) / (1 - prop_outputs_zero)
                    outputs_one = 1.0 / prop_outputs_one
                    outputs_zero = -1.0 / (1.0 - prop_outputs_zero)
                else:
                    outputs = raw_outputs
                    outputs_one = raw_outputs_one
                    outputs_zero = raw_outputs_zero

                loss = self.criterion(outputs, outputs_one, outputs_zero, T_batch)
                loss.backward()
                self.optimizer.step()

            # Convergence check (per epoch, based on last mini-batch loss)
            current_loss = float(loss.item())
            if abs(prev_loss - current_loss) < self.tol:
                break
            prev_loss = current_loss

    def riesz_predict(self, X, T):
        """Predict the Riesz representer g(X, T)."""

        X_tensor = torch.tensor(X, dtype=torch.float32)
        T_tensor = torch.tensor(T, dtype=torch.float32).view(-1, 1)

        if self.riesz_with_D:
            X_tensor = torch.cat([X_tensor, T_tensor], dim=1)

        raw_outputs = self.riesz_model(X_tensor)

        if self.riesz_link_name == "Logit":
            prop_outputs = torch.sigmoid(raw_outputs)
            prop_outputs = torch.clamp(prop_outputs, min=0.05, max=0.95)
            outputs = T_tensor / prop_outputs - (1 - T_tensor) / (1 - prop_outputs)
        else:
            outputs = raw_outputs

        return outputs.detach().numpy().copy().T[0]

    # ------------------------------------------------------------------
    # Regression model construction / fit / predict
    # ------------------------------------------------------------------
    def construct_reg_model(self) -> None:
        """Construct the neural network model for regression E[Y | X, T]."""

        self.reg_fc1 = nn.Linear(self.reg_input_dim, self.reg_hidden_dim)
        self.reg_fc2 = nn.Linear(self.reg_hidden_dim, 1)

        self.reg_model = nn.Sequential(
            self.reg_fc1,
            nn.ReLU(),
            self.reg_fc2,
        )

    def reg_fit(
        self,
        X,
        T,
        Y,
        reg_hidden_dim: int = 100,
        reg_max_iter: int = 3000,
        tol: float = 1e-10,
        lbd: float = 0.01,
        lr: float = 0.01,
        batch_size: int = 1000,
    ) -> None:
        """Fit the regression model m(X, T) = E[Y | X, T] using mini-batch MSE.

        Notes
        -----
        The argument ``lbd`` is currently unused (kept to preserve the research-code
        signature).
        """

        self.reg_hidden_dim = reg_hidden_dim
        self.reg_input_dim = np.asarray(X).shape[1] + 1  # X plus treatment

        X_tensor = torch.tensor(X, dtype=torch.float32)
        T_tensor = torch.tensor(T, dtype=torch.float32).view(-1, 1)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

        # Concatenate treatment to covariates
        X_tensor = torch.cat([X_tensor, T_tensor], dim=1)

        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Build regression network
        self.construct_reg_model()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.reg_model.parameters(), lr=lr)

        self.reg_model.train()
        prev_loss = float("inf")
        for _epoch in range(reg_max_iter):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = self.reg_model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

            current_loss = float(loss.item())
            if abs(prev_loss - current_loss) < tol:
                break
            prev_loss = current_loss

    def reg_predict(self, X, T):
        """Predict m(X, T) = E[Y | X, T]."""

        X_tensor = torch.tensor(X, dtype=torch.float32)
        T_tensor = torch.tensor(T, dtype=torch.float32).view(-1, 1)

        X_tensor = torch.cat([X_tensor, T_tensor], dim=1)

        outputs = self.reg_model(X_tensor)
        return outputs.detach().numpy().copy().T[0]

    def reg_predict_diff(self, X, T):
        """Return m(X, T), m(X, 1) and m(X, 0)."""

        T_zero = np.asarray(T) * 0
        T_one = T_zero + 1

        est_reg = self.reg_predict(X, T)
        est_reg_one = self.reg_predict(X, T_one)
        est_reg_zero = self.reg_predict(X, T_zero)

        return est_reg, est_reg_one, est_reg_zero
