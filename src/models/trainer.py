"""
Dual-Path TFT Training and Evaluation Engine.
Handles 48-hour sliding window construction, PyTorch training loops, 
and computes MAE, RMSE, and R2 evaluation metrics (Table 3).
"""

from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import TFTConfig
from src.models.tft_model import DualPathTFTModel


class TFTTrainerEngine:
    """Manages training loops and forecasting evaluations for the dual-path TFT architecture."""

    def __init__(self, config: TFTConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualPathTFTModel(
            hidden_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate
        ).to(self.device)

    def _build_sliding_windows(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates sliding historical lookback windows and future step targets."""
        X, Y = [], []
        w = self.cfg.lookback_window
        for i in range(len(series) - w):
            X.append(series[i : i + w])
            Y.append(series[i + w])
        return np.array(X)[..., np.newaxis], np.array(Y)

    def train_and_forecast(
        self, p_long: np.ndarray, p_short: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Executes dual-path training on 80% data split and tests on 20% split.

        :return: Tuple of (metrics_dict, pred_long_test, pred_short_test).
        """
        X_l, Y_l = self._build_sliding_windows(p_long)
        X_s, Y_s = self._build_sliding_windows(p_short)

        split_idx = int(len(X_l) * self.cfg.train_split)

        # Datasets
        train_ds = TensorDataset(
            torch.FloatTensor(X_l[:split_idx]),
            torch.FloatTensor(X_s[:split_idx]),
            torch.FloatTensor(Y_l[:split_idx]),
            torch.FloatTensor(Y_s[:split_idx])
        )
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        criterion = nn.MSELoss()

        # Training Loop
        self.model.train()
        for epoch in range(self.cfg.training_epochs):
            for xl, xs, yl, ys in train_loader:
                xl, xs, yl, ys = xl.to(self.device), xs.to(self.device), yl.to(self.device), ys.to(self.device)
                optimizer.zero_grad()
                pred_l, pred_s = self.model(xl, xs)
                loss = criterion(pred_l, yl) + criterion(pred_s, ys)
                loss.backward()
                optimizer.step()

        # Testing & Evaluation
        self.model.eval()
        with torch.no_grad():
            test_xl = torch.FloatTensor(X_l[split_idx:]).to(self.device)
            test_xs = torch.FloatTensor(X_s[split_idx:]).to(self.device)
            pred_l, pred_s = self.model(test_xl, test_xs)

            pred_l_np = pred_l.cpu().numpy()
            pred_s_np = pred_s.cpu().numpy()

        y_actual = Y_l[split_idx:] + Y_s[split_idx:]
        y_pred = pred_l_np + pred_s_np

        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)

        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
        return metrics, pred_l_np, pred_s_np