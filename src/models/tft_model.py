"""
Temporal Fusion Transformer (TFT) Architecture in PyTorch.
Implements Gated Residual Networks (GRN), Multi-Head Attention, and Dual-Path predictors.
"""

import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) for adaptive feature gating."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc(x).chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) module for nonlinear context integration."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.12):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.glu(out)
        return self.layer_norm(residual + out)


class SinglePathTFT(nn.Module):
    """Single-Path Temporal Fusion Transformer for sequence-to-sequence prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_encoder = nn.Linear(input_dim, hidden_dim)
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        out = self.input_encoder(x)
        out = self.grn(out)
        out = self.transformer_encoder(out)
        pred = self.output_head(out[:, -1, :])  # Predict step h from final temporal state
        return pred.squeeze(-1)


class DualPathTFTPredictor(nn.Module):
    """
    Dual-Path TFT network encapsulating separate paths for 
    Long-Term (smooth trend) and Short-Term (high-frequency volatility) forecasting.
    """

    def __init__(self, hidden_dim: int = 32, num_heads: int = 2, num_layers: int = 2, dropout: float = 0.12):
        super().__init__()
        self.tft_long = SinglePathTFT(1, hidden_dim, num_heads, num_layers, dropout)
        self.tft_short = SinglePathTFT(1, hidden_dim, num_heads, num_layers, dropout)

    def forward(self, x_long: torch.Tensor, x_short: torch.Tensor):
        pred_long = self.tft_long(x_long)
        pred_short = self.tft_short(x_short)
        return pred_long, pred_short