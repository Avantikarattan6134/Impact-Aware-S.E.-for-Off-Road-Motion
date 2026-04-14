"""
model/lstm_noise_model.py
=========================
LSTM predicting TWO outputs per window:
  alpha (α) : Q scale factor        in [alpha_min, alpha_max]
  beta  (β) : NHC relaxation factor in [0, beta_max]

Architecture
------------
  Input  : (batch, window, 7)  [ax, ay, az, gx, gy, gz, jerk_mag]
  LSTM   : hidden_size x num_layers
  Head   : Linear → Sigmoid × 2 → [alpha, beta]
"""

import torch
import torch.nn as nn
import numpy as np


class LSTMNoiseAdapter(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg["lstm"]
        jc = cfg["jerk"]
        self.window    = int(lc["window"])
        self.n_feat    = int(lc["features"])
        self.hidden    = int(lc["hidden_size"])
        self.n_layers  = int(lc["num_layers"])
        self.alpha_min = float(lc["alpha_min"])
        self.alpha_max = float(lc["alpha_max"])
        self.beta_max  = float(jc.get("beta_max", 0.85))

        self.input_norm = nn.LayerNorm(self.n_feat)

        self.lstm = nn.LSTM(
            input_size=self.n_feat,
            hidden_size=self.hidden,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=lc["dropout"] if self.n_layers > 1 else 0.0,
        )

        self.head_alpha = nn.Sequential(
            nn.Linear(self.hidden, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.head_beta = nn.Sequential(
            nn.Linear(self.hidden, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        alpha : (batch, 1) in [alpha_min, alpha_max]
        beta  : (batch, 1) in [0, beta_max]
        """
        x    = self.input_norm(x)
        out, _ = self.lstm(x)
        last   = out[:, -1, :]
        alpha  = self.alpha_min + (self.alpha_max - self.alpha_min) * self.head_alpha(last)
        beta   = self.beta_max * self.head_beta(last)
        return alpha, beta

    @torch.no_grad()
    def predict(self, window_np: np.ndarray, device: str = "cpu"):
        """
        Returns (alpha_float, beta_float) for online filter use.
        """
        self.eval()
        x = torch.tensor(window_np[np.newaxis], dtype=torch.float32, device=device)
        alpha, beta = self.forward(x)
        return alpha.item(), beta.item()
