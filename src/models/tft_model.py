"""
Temporal Fusion Transformer (TFT) PyTorch Architecture.
Includes Gated Residual Networks (GRN), Gated Linear Units (GLU), Multi-Head Attention,
and Static Context Injection as formulated in Equations (7)-(9).
"""

import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) for adaptive feature gating."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) integrating temporal features with static context cs (Eq. 7)."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.12):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.glu = GatedLinearUnit(input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor, c_s: torch.Tensor = None) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        if c_s is not None:
            out = out + c_s
        out = self.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.glu(out)
        return self.layer_norm(residual + out)


class TemporalFusionTransformerPath(nn.Module):
    """Single-Path TFT module with attention mechanisms and sequence modeling."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout_rate: float):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        h = self.input_layer(x)
        h = self.grn(h)
        h = self.transformer_encoder(h)
        out = self.fc_out(h[:, -1, :])  # Extract last hidden temporal state
        return out.squeeze(-1)


class DualPathTFTModel(nn.Module):
    """
    Dual-Path Temporal Fusion Transformer framework separately predicting
    Long-term approximation (Eq. 8) and Short-term fluctuations (Eq. 9).
    """

    def __init__(self, hidden_dim: int = 32, num_heads: int = 2, num_layers: int = 2, dropout_rate: float = 0.12):
        super().__init__()
        self.tft_long = TemporalFusionTransformerPath(1, hidden_dim, num_heads, num_layers, dropout_rate)
        self.tft_short = TemporalFusionTransformerPath(1, hidden_dim, num_heads, num_layers, dropout_rate)

    def forward(self, x_long: torch.Tensor, x_short: torch.Tensor):
        pred_long = self.tft_long(x_long)
        pred_short = self.tft_short(x_short)
        return pred_long, pred_short