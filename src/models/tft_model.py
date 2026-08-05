"""
Temporal Fusion Transformer (TFT) PyTorch Architecture.
Includes Gated Residual Networks (GRN), Gated Linear Units (GLU), Multi-Head Attention,
Positional Encoding, and Dual-Path Context Injection as formulated in Equations (7)-(9).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Injects positional encoding to preserve temporal order across sequences."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) for adaptive feature gating."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) integrating temporal features with static/cross-path context (Eq. 7).
    GRN(x, c) = LayerNorm(x + GLU(W2 * ELU(W1 * x + Wc * c + b1) + b2))
    """

    def __init__(self, input_dim: int, hidden_dim: int, context_dim: Optional[int] = None, dropout_rate: float = 0.12):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_fc = None
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.glu = GatedLinearUnit(input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        if context is not None and self.context_fc is not None:
            if context.dim() == 2 and x.dim() == 3:
                c = self.context_fc(context).unsqueeze(1)
            else:
                c = self.context_fc(context)
            out = out + c
        out = self.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.glu(out)
        # Element-wise feature selection / gating (Eq. 7)
        return self.layer_norm(residual + out)


class TemporalFusionTransformerPath(nn.Module):
    """
    Single-Path TFT module with positional encoding, multi-head self-attention,
    GRN feature gating, and temporal context extraction (Eq. 8 & Eq. 9).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float = 0.12,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.grn_input = GatedResidualNetwork(hidden_dim, hidden_dim, context_dim=context_dim, dropout_rate=dropout_rate)

        # Multi-Head Self-Attention Layer
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.glu_attn = GatedLinearUnit(hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)

        # Post-Attention GRN
        self.grn_post = GatedResidualNetwork(hidden_dim, hidden_dim, context_dim=context_dim, dropout_rate=dropout_rate)

        # Output Forecast Layer
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, input_dim)
        h = self.input_layer(x)
        h = self.pos_encoder(h)
        h = self.grn_input(h, context=context)

        # Self-Attention Gating Residual Connection
        attn_out, _ = self.mha(h, h, h)
        h = self.norm_attn(h + self.glu_attn(attn_out))

        # Post-Attention GRN
        h = self.grn_post(h, context=context)

        # Extract last hidden temporal state as context vector
        h_context = h[:, -1, :]
        out = self.fc_out(h_context).squeeze(-1)

        return out, h_context


class DualPathTFTModel(nn.Module):
    """
    Dual-Path Temporal Fusion Transformer framework separately predicting
    Long-term approximation (Eq. 8) and Short-term fluctuations (Eq. 9) with cross-path context injection.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout_rate: float = 0.12
    ):
        super().__init__()
        # Long-term path (Eq. 8)
        self.tft_long = TemporalFusionTransformerPath(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            context_dim=None
        )

        # Short-term path conditioned on long-term context (Eq. 9)
        self.tft_short = TemporalFusionTransformerPath(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            context_dim=hidden_dim
        )

    def forward(self, x_long: torch.Tensor, x_short: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Long-term prediction and temporal context extraction (Eq. 8)
        pred_long, context_long = self.tft_long(x_long)
        # Short-term prediction injected with long-term context (Eq. 9)
        pred_short, _ = self.tft_short(x_short, context=context_long)
        return pred_long, pred_short