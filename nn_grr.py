import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NN_GRR:
    """
    Neural network learner for Riesz representer and regression models.
    """

    def __init__(self):
        # Attributes are set later in fit methods
        pass

    # ------------------------------------------------------------------
    # Riesz model construction and losses
    # ------------------------------------------------------------------
    def construct_riesz_model(self):
        """
        Construct the neural network used for the Riesz representer.
        """
        self.fc1 = nn.Linear(self.riesz_input_dim, self.riesz_hidden_dim)
        self.fc2 = nn.Linear(self.riesz_hidden_dim, 1)

        self.riesz_model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )

    def _sq_loss_function(self, outputs, outputs_one, outputs_zero, targets, upperbound=20):
        """
        Least-squares type loss:
        E[-2 * (g(X, 1) - g(X, 0)) + g(X, T)^2] + L2 regularization.
        """
        loss1 = - torch.mean(2*(outputs_one - outputs_zero) - (outputs_one**2 + outputs_zero**2)/upperbound)
        loss2 = torch.mean(-(outputs_one**2 + outputs_zero**2)/upperbound + outputs**2)
        if loss2 > 0:
            loss = loss1 + loss2
        else:
            loss = -0.01*loss2

        return loss + self.lbd * self._l2_regularization()

    def _ukl_loss_function(self, outputs, outputs_one, outputs_zero, targets, upperbound=20):
        """
        UKL-type loss for density-ratio-style objectives.
        Assumes outputs_one > 0 and outputs_zero < 0.
        """
        loss1 = torch.mean(
            (torch.log(torch.abs(outputs_one) - 1) + torch.log(torch.abs(outputs_zero) - 1)
            + torch.abs(outputs_one) + torch.abs(outputs_zero)) / upperbound
            - torch.log(torch.abs(outputs_one) - 1)
            - torch.log(torch.abs(outputs_zero) - 1)
        )
        loss2 = torch.mean(
            - (torch.log(torch.abs(outputs_one) - 1) + torch.log(torch.abs(outputs_zero) - 1)
            + torch.abs(outputs_one) + torch.abs(outputs_zero)) / upperbound
            + torch.log(torch.abs(outputs) - 1)
            + torch.abs(outputs)
        )
        if loss2 > 0:
            loss = loss1 + loss2
        else:
            loss = -0.01*loss2
            
      
        loss = torch.mean(loss)
        return loss + self.lbd * self._l2_regularization()

    def _bkl_loss_function(self, outputs, outputs_one, outputs_zero, targets):
        """
        BKL (Maximum-likelihood loss) using logits.

        Here `outputs` are treated as logits of P(T=1 | X),
        and `targets` are binary treatment indicators.
        """
        loss = -targets * torch.log(1/outputs_one) - (1 - targets) * torch.log(-1/outputs_zero)
        loss = torch.mean(loss)
        return loss + self.lbd * self._l2_regularization()

    def _logistic_loss_function(self, outputs, outputs_one, outputs_zero, targets):
        """
        Placeholder for a custom 'Logit' loss.

        If you need a specific logistic-style risk for the Riesz function,
        implement it here and use `riesz_loss='Logit'` in `riesz_fit`.
        """
        raise NotImplementedError(
            "Custom 'Logit' loss is not implemented. "
            "Please implement _logistic_loss_function or use 'LS', 'KL', 'TL', or 'MLE'."
        )

    def _l2_regularization(self):
        """
        Compute L2 regularization term over the parameters of the Riesz model.
        """
        reg_loss = sum(torch.sum(param**2) for param in self.riesz_model.parameters())
        return reg_loss

    # ------------------------------------------------------------------
    # Riesz fit / predict
    # ------------------------------------------------------------------
    def riesz_fit(
        self,
        X,
        T,
        riesz_loss,
        riesz_with_D,
        riesz_link_name,
        riesz_hidden_dim=100,
        riesz_max_iter=3000,
        tol=1e-10,
        lbd=0.01,
        lr=0.01,
        batch_size=1000,
    ):
        """
        Train the Riesz model using mini-batch optimization.

        Parameters
        ----------
        X : array-like, shape (N, d)
            Covariates.
        T : array-like, shape (N,) or (N, 1)
            Binary treatment indicator.
        riesz_loss : {"LS", "KL", "TL", "MLE", "Logit"}
            Type of loss function used for training.
        riesz_with_D : bool
            If True, the treatment indicator is concatenated to X as an input.
        riesz_link_name : {"Logit", ...}
            Link function name; when "Logit", outputs are interpreted as logits
            for the propensity score and transformed to the Riesz representer.
        riesz_hidden_dim : int
            Number of hidden units.
        riesz_max_iter : int
            Maximum number of epochs.
        tol : float
            Convergence tolerance based on the change in loss.
        lbd : float
            L2 regularization weight.
        lr : float
            Learning rate for Adam optimizer.
        batch_size : int
            Mini-batch size.
        """
        self.riesz_hidden_dim = riesz_hidden_dim
        self.riesz_max_iter = riesz_max_iter
        self.tol = tol
        self.lbd = lbd
        self.lr = lr
        self.riesz_link_name = riesz_link_name
        self.riesz_with_D = riesz_with_D

        riesz_input_dim = X.shape[1]

        # Construct input dimension with or without treatment indicator
        if self.riesz_with_D:
            self.riesz_input_dim = riesz_input_dim + 1
        else:
            self.riesz_input_dim = riesz_input_dim

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
            raise ValueError("Invalid loss function specified: {}".format(riesz_loss))

        self.optimizer = optim.Adam(self.riesz_model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        T_tensor = torch.tensor(T, dtype=torch.float32).view(-1, 1)

        dataset = TensorDataset(X_tensor, T_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        prev_loss = float("inf")
        for epoch in range(riesz_max_iter):
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

                # Forward pass for all three inputs (avoid double computation)
                raw_outputs = self.riesz_model(X_batch_full)
                raw_outputs_one = self.riesz_model(X_batch_one)
                raw_outputs_zero = self.riesz_model(X_batch_zero)

                if self.riesz_link_name == "Logit":
                    # Interpret raw_outputs as logits of propensity score p(X)
                    prop_outputs_one = torch.sigmoid(raw_outputs_one)
                    prop_outputs_zero = torch.sigmoid(raw_outputs_zero)
                    #prop_outputs_one = torch.sigmoid(raw_outputs_one)
                    #prop_outputs_zero = torch.sigmoid(raw_outputs_zero)
                    #print("one", prop_outputs_one[0:3])
                    #print("zero", prop_outputs_zero[0:3])
                    
                    prop_outputs_one = torch.clamp(prop_outputs_one, min=0.05, max=0.95)
                    prop_outputs_zero = torch.clamp(prop_outputs_zero, min=0.05, max=0.95)

                    # Riesz representer for the ATE:
                    # g(X, T) = T / p(X) - (1 - T) / (1 - p(X))
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
            current_loss = loss.item()
            if abs(prev_loss - current_loss) < self.tol:
                break
            prev_loss = current_loss

    def riesz_predict(self, X, T):
        """
        Predict the Riesz representer g(X, T).

        Parameters
        ----------
        X : array-like, shape (N, d)
            Covariates.
        T : array-like or scalar, broadcastable to shape (N, 1)
            Treatment indicator(s).

        Returns
        -------
        outputs : ndarray, shape (N,)
            Predicted values of the Riesz representer.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        T_tensor = torch.tensor(T, dtype=torch.float32).view(-1, 1)

        if self.riesz_with_D:
            X_tensor = torch.cat([X_tensor, T_tensor], dim=1)

        raw_outputs = self.riesz_model(X_tensor)

        if self.riesz_link_name == "Logit":
            # Again interpret as logits of propensity and map to Riesz representer
            prop_outputs = torch.sigmoid(raw_outputs)
            prop_outputs = torch.clamp(prop_outputs, min=0.05, max=0.95)
            outputs = T_tensor / prop_outputs - (1 - T_tensor) / (1 - prop_outputs)
        else:
            outputs = raw_outputs

        outputs = outputs.detach().numpy().copy().T[0]
        return outputs

    # ------------------------------------------------------------------
    # Regression model construction / fit / predict
    # ------------------------------------------------------------------
    def construct_reg_model(self):
        """
        Construct the neural network model for regression E[Y | X, T].
        """
        self.fc1 = nn.Linear(self.reg_input_dim, self.reg_hidden_dim)
        self.fc2 = nn.Linear(self.reg_hidden_dim, 1)

        self.reg_model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )

    def reg_fit(
        self,
        X,
        T,
        Y,
        reg_hidden_dim=100,
        reg_max_iter=3000,
        tol=1e-10,
        lbd=0.01,
        lr=0.01,
        batch_size=1000,
    ):
        """
        Train the regression model m(X, T) = E[Y | X, T] using mini-batch MSE.

        Parameters
        ----------
        X : array-like, shape (N, d)
            Covariates.
        T : array-like, shape (N,) or (N, 1)
            Treatment indicator.
        Y : array-like, shape (N,) or (N, 1)
            Outcome.
        reg_hidden_dim : int
            Number of hidden units.
        reg_max_iter : int
            Maximum number of epochs.
        tol : float
            Convergence tolerance based on the change in loss.
        lbd : float
            (Currently unused) L2 regularization weight for regression model.
        lr : float
            Learning rate for Adam optimizer.
        batch_size : int
            Mini-batch size.
        """
        self.reg_hidden_dim = reg_hidden_dim
        self.reg_input_dim = X.shape[1] + 1  # X plus treatment

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
        for epoch in range(reg_max_iter):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = self.reg_model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

            # Convergence check (per epoch, based on last mini-batch loss)
            current_loss = loss.item()
            if abs(prev_loss - current_loss) < tol:
                break
            prev_loss = current_loss

    def reg_predict(self, X, T):
        """
        Predict the regression function m(X, T) = E[Y | X, T].

        Parameters
        ----------
        X : array-like, shape (N, d)
            Covariates.
        T : array-like or scalar, broadcastable to shape (N, 1)
            Treatment indicator(s).

        Returns
        -------
        outputs : ndarray, shape (N,)
            Predicted conditional mean of Y.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        T_tensor = torch.tensor(T, dtype=torch.float32).view(-1, 1)

        X_tensor = torch.cat([X_tensor, T_tensor], dim=1)

        outputs = self.reg_model(X_tensor)
        outputs = outputs.detach().numpy().copy().T[0]
        return outputs
    
    def reg_predict_diff(self, X, T):
        T_zero = T * 0
        T_one = T_zero + 1
        
        est_reg = self.reg_predict(X, T)
        est_reg_one = self.reg_predict(X, T_one)
        est_reg_zero = self.reg_predict(X, T_zero)
        
        return est_reg, est_reg_one, est_reg_zero
        
