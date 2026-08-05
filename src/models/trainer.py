"""
Dual-Path TFT Training and Evaluation Engine.
Handles sliding lookback windows with cyclic temporal feature engineering, PyTorch training loops with tqdm progress bars,
and computes MAE, RMSE, and R2 evaluation metrics (Table 3).
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import TFTConfig
from src.models.tft_model import DualPathTFTModel


class TFTTrainerEngine:
    """Manages training loops and forecasting evaluations for the dual-path TFT architecture."""

    def __init__(self, config: TFTConfig):
        self.cfg = config
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

    def _build_sliding_windows_with_time(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Vectorized construction of sliding historical lookback windows and future targets with temporal features."""
        scaler = StandardScaler()
        series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

        N = len(series)
        w = getattr(self.cfg, "lookback_window", 48)

        # 15-minute time steps = 96 steps per day
        time_steps = np.arange(N) % 96
        sin_tod = np.sin(2.0 * np.pi * time_steps / 96.0)
        cos_tod = np.cos(2.0 * np.pi * time_steps / 96.0)

        X, Y = [], []
        for i in range(N - w):
            window_feat = np.column_stack([
                series_scaled[i : i + w],
                sin_tod[i : i + w],
                cos_tod[i : i + w]
            ])
            X.append(window_feat)
            Y.append(series_scaled[i + w])

        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), scaler

    def train_and_forecast_single_source(
        self, p_long: np.ndarray, p_short: np.ndarray, source_label: str = "RES"
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, Dict[str, list], Dict[str, Any]]:
        """
        Executes dual-path TFT training with tqdm progress bar logging.
        """
        X_l, Y_l, scaler_long = self._build_sliding_windows_with_time(p_long)
        X_s, Y_s, scaler_short = self._build_sliding_windows_with_time(p_short)

        train_split = getattr(self.cfg, "train_split", 0.80)
        split_idx = int(len(X_l) * train_split)

        X_l_train, Y_l_train = torch.tensor(X_l[:split_idx]), torch.tensor(Y_l[:split_idx])
        X_s_train, Y_s_train = torch.tensor(X_s[:split_idx]), torch.tensor(Y_s[:split_idx])

        train_ds = TensorDataset(X_l_train, X_s_train, Y_l_train, Y_s_train)
        batch_size = getattr(self.cfg, "batch_size", 64)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        hidden_dim = getattr(self.cfg, "hidden_dim", 32)
        num_heads = getattr(self.cfg, "attention_heads", 2)
        num_layers = getattr(self.cfg, "num_layers", 2)
        dropout_rate = getattr(self.cfg, "dropout_rate", 0.12)
        learning_rate = getattr(self.cfg, "learning_rate", 0.001)
        epochs = getattr(self.cfg, "training_epochs", 150)

        model = DualPathTFTModel(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.MSELoss()

        history = {"epoch": [], "loss_total": [], "loss_long": [], "loss_short": []}

        model.train()
        pbar = tqdm(
            range(1, epochs + 1),
            desc=f"[Step 3] Dual-Path TFT Training ({source_label})",
            unit="epoch",
            bar_format="{l_bar}{bar:30}{r_bar}"
        )

        for epoch in pbar:
            running_loss = 0.0
            running_l_loss = 0.0
            running_s_loss = 0.0

            for xl, xs, yl, ys in train_loader:
                xl, xs = xl.to(self.device), xs.to(self.device)
                yl, ys = yl.to(self.device), ys.to(self.device)

                optimizer.zero_grad()
                pred_l, pred_s = model(xl, xs)

                loss_l = criterion(pred_l, yl)
                loss_s = criterion(pred_s, ys)
                loss = loss_l + loss_s

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                running_l_loss += loss_l.item()
                running_s_loss += loss_s.item()

            scheduler.step()
            n_batches = len(train_loader)
            avg_tot = running_loss / n_batches
            avg_l = running_l_loss / n_batches
            avg_s = running_s_loss / n_batches

            history["epoch"].append(epoch)
            history["loss_total"].append(avg_tot)
            history["loss_long"].append(avg_l)
            history["loss_short"].append(avg_s)

            pbar.set_postfix({
                "Loss": f"{avg_tot:.4f}",
                "Long": f"{avg_l:.4f}",
                "Short": f"{avg_s:.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.5f}"
            })

        # Evaluation on Test Split
        model.eval()
        with torch.no_grad():
            X_l_test = torch.tensor(X_l[split_idx:]).to(self.device)
            X_s_test = torch.tensor(X_s[split_idx:]).to(self.device)
            pred_l_test_scaled, pred_s_test_scaled = model(X_l_test, X_s_test)

            pred_l_test_scaled = pred_l_test_scaled.cpu().numpy().reshape(-1, 1)
            pred_s_test_scaled = pred_s_test_scaled.cpu().numpy().reshape(-1, 1)

        pred_long_unscaled = scaler_long.inverse_transform(pred_l_test_scaled).flatten()
        pred_short_unscaled = scaler_short.inverse_transform(pred_s_test_scaled).flatten()

        y_l_actual = scaler_long.inverse_transform(Y_l[split_idx:].reshape(-1, 1)).flatten()
        y_s_actual = scaler_short.inverse_transform(Y_s[split_idx:].reshape(-1, 1)).flatten()

        y_actual_total = y_l_actual + y_s_actual
        y_pred_total = pred_long_unscaled + pred_short_unscaled

        # Enforce physical non-negativity constraint for generation power
        y_pred_total = np.maximum(0.0, y_pred_total)

        mae = float(mean_absolute_error(y_actual_total, y_pred_total))
        rmse = float(np.sqrt(mean_squared_error(y_actual_total, y_pred_total)))
        r2 = float(r2_score(y_actual_total, y_pred_total))

        slope, intercept = np.polyfit(y_actual_total, y_pred_total, 1)
        r_val = float(np.corrcoef(y_actual_total, y_pred_total)[0, 1])

        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
        eval_details = {
            "y_actual": y_actual_total,
            "y_pred": y_pred_total,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": r_val
        }

        return metrics, pred_long_unscaled, pred_short_unscaled, history, eval_details

    def train_and_forecast(
        self, p_long: np.ndarray, p_short: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        metrics, pred_l, pred_s, _, _ = self.train_and_forecast_single_source(p_long, p_short)
        return metrics, pred_l, pred_s