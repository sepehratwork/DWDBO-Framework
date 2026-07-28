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
        
        # Hyperparameters with default fallbacks matching Table 2 of the paper
        hidden_dim = getattr(config, "hidden_dim", getattr(config, "hidden_layer_dimensions", 32))
        num_heads = getattr(config, "attention_heads", 2)
        num_layers = getattr(config, "num_layers", getattr(config, "number_of_layers", 2))
        dropout_rate = getattr(config, "dropout_rate", 0.12)
        
        # GPU detection & hardware acceleration configuration
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        else:
            self.device = torch.device("cpu")

        print(f"[TFTTrainerEngine] Hardware Device Configured: {self.device}")

        self.model = DualPathTFTModel(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Optional PyTorch 2.0+ Model Compiler for CUDA
        if hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

    def _build_sliding_windows(self, series: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized construction of sliding historical lookback windows and future targets."""
        w = getattr(self.cfg, "lookback_window", getattr(self.cfg, "sliding_window", 48))
        series_t = torch.as_tensor(series, dtype=torch.float32)
        
        # Fast tensor unfold view (dimension, size, step) replacing slow Python loops
        X = series_t.unfold(0, w, 1)[:-1].unsqueeze(-1)
        Y = series_t[w:]
        return X, Y

    def train_and_forecast(
        self, p_long: np.ndarray, p_short: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Executes dual-path training on 80% data split and tests on 20% split.

        :return: Tuple of (metrics_dict, pred_long_test, pred_short_test).
        """
        X_l, Y_l = self._build_sliding_windows(p_long)
        X_s, Y_s = self._build_sliding_windows(p_short)

        # 80% Train / 20% Test split as specified in paper Section 3.2
        train_split = getattr(self.cfg, "train_split", getattr(self.cfg, "train_ratio", 0.80))
        split_idx = int(len(X_l) * train_split)

        # Store training dataset directly on GPU memory to eliminate per-batch CPU-to-GPU transfer overhead
        train_ds = TensorDataset(
            X_l[:split_idx].to(self.device, non_blocking=True),
            X_s[:split_idx].to(self.device, non_blocking=True),
            Y_l[:split_idx].to(self.device, non_blocking=True),
            Y_s[:split_idx].to(self.device, non_blocking=True)
        )
        batch_size = getattr(self.cfg, "batch_size", 128)
        train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=False
        )

        learning_rate = getattr(self.cfg, "learning_rate", 0.001)
        epochs = getattr(self.cfg, "training_epochs", getattr(self.cfg, "epochs", 150))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Mixed Precision AMP setup for accelerated GPU execution
        use_amp = (self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # Training Loop
        self.model.train()
        print(f"[TFTTrainerEngine] Training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            running_loss = 0.0
            for xl, xs, yl, ys in train_loader:
                optimizer.zero_grad(set_to_none=True)
                
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        pred_l, pred_s = self.model(xl, xs)
                        pred_l_flat = pred_l.squeeze(-1) if pred_l.dim() > 1 else pred_l
                        pred_s_flat = pred_s.squeeze(-1) if pred_s.dim() > 1 else pred_s
                        loss = criterion(pred_l_flat, yl) + criterion(pred_s_flat, ys)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred_l, pred_s = self.model(xl, xs)
                    pred_l_flat = pred_l.squeeze(-1) if pred_l.dim() > 1 else pred_l
                    pred_s_flat = pred_s.squeeze(-1) if pred_s.dim() > 1 else pred_s
                    loss = criterion(pred_l_flat, yl) + criterion(pred_s_flat, ys)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            # Periodic logging eliminates stdio buffering overheads
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch: {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")

        # Testing & Evaluation
        self.model.eval()
        with torch.no_grad():
            test_xl = X_l[split_idx:].to(self.device, non_blocking=True)
            test_xs = X_s[split_idx:].to(self.device, non_blocking=True)
            
            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred_l, pred_s = self.model(test_xl, test_xs)
            else:
                pred_l, pred_s = self.model(test_xl, test_xs)

            pred_l_np = pred_l.detach().cpu().numpy().reshape(-1)
            pred_s_np = pred_s.detach().cpu().numpy().reshape(-1)

        y_actual = (Y_l[split_idx:] + Y_s[split_idx:]).cpu().numpy()
        y_pred = pred_l_np + pred_s_np

        mae = float(mean_absolute_error(y_actual, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
        r2 = float(r2_score(y_actual, y_pred))

        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
        return metrics, pred_l_np, pred_s_np