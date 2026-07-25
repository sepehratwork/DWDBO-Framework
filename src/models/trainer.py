"""
Model Training and Forecasting Execution Manager.
Manages dataset windowing, PyTorch training loops, and evaluation metrics.
"""

from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import ModelConfig
from src.models.tft_model import DualPathTFTPredictor


class TFTForecastingEngine:
    """Handles dataset construction, neural network optimization, and forecast generation."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualPathTFTPredictor(
            hidden_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            num_layers=config.num_layers,
            dropout=config.dropout
        ).to(self.device)

    def _create_sliding_windows(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates sliding temporal window sequences (Lookback history -> Forecast target)."""
        x, y = [], []
        w = self.config.lookback_window
        h = self.config.forecast_horizon
        for i in range(len(data) - w - h + 1):
            x.append(data[i : i + w])
            y.append(data[i + w + h - 1])
        return np.array(x)[..., np.newaxis], np.array(y)

    def train_and_evaluate(
        self, p_long: np.ndarray, p_short: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Trains dual-path TFT and returns test accuracy metrics and predicted series."""
        # Build sliding windows
        x_long, y_long = self._create_sliding_windows(p_long)
        x_short, y_short = self._createsliding_windows(p_short)

        split_idx = int(len(x_long) * self.config.train_split)

        # Datasets
        train_ds = TensorDataset(
            torch.FloatTensor(x_long[:split_idx]),
            torch.FloatTensor(x_short[:split_idx]),
            torch.FloatTensor(y_long[:split_idx]),
            torch.FloatTensor(y_short[:split_idx]),
        )
        test_ds = TensorDataset(
            torch.FloatTensor(x_long[split_idx:]),
            torch.FloatTensor(x_short[split_idx:]),
            torch.FloatTensor(y_long[split_idx:]),
            torch.FloatTensor(y_short[split_idx:]),
        )

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(self.config.epochs):
            for xl, xs, yl, ys in train_loader:
                xl, xs, yl, ys = xl.to(self.device), xs.to(self.device), yl.to(self.device), ys.to(self.device)
                optimizer.zero_grad()
                pred_l, pred_s = self.model(xl, xs)
                loss = criterion(pred_l, yl) + criterion(pred_s, ys)
                loss.backward()
                optimizer.step()

        # Inference / Testing
        self.model.eval()
        with torch.no_grad():
            test_xl = torch.FloatTensor(x_long[split_idx:]).to(self.device)
            test_xs = torch.FloatTensor(x_short[split_idx:]).to(self.device)
            pred_l, pred_s = self.model(test_xl, test_xs)

            pred_l = pred_l.cpu().numpy()
            pred_s = pred_s.cpu().numpy()

        y_actual = y_long[split_idx:] + y_short[split_idx:]
        y_pred = pred_l + pred_s

        # Evaluation metrics
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)

        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
        return metrics, pred_l, pred_s