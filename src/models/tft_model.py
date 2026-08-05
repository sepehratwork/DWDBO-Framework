"""
Temporal Fusion Transformer (TFT) PyTorch Architecture.
Includes Gated Residual Networks (GRN), Gated Linear Units (GLU), Variable Selection Networks (VSN),
Multi-Head Temporal Self-Attention, Positional Encoding, and Dual-Path Structure as formulated in Equations (7)-(9).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Injects temporal sequence position embeddings into hidden representations."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


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
    Gated Residual Network (GRN) integrating temporal features with static context cs (Eq. 7).
    GRN(x, c_s) = LayerNorm(Residual(x) + GLU(W_2 * ELU(W_1 * x + W_c * c_s + b_1) + b_2))
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None, dropout_rate: float = 0.12):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.glu = GatedLinearUnit(output_dim)

        if input_dim != output_dim:
            self.res_skip = nn.Linear(input_dim, output_dim)
        else:
            self.res_skip = nn.Identity()

        self.layer_norm = nn.LayerNorm(output_dim)
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor, c_s: torch.Tensor = None) -> torch.Tensor:
        residual = self.res_skip(x)
        out = self.fc1(x)
        if c_s is not None:
            out = out + c_s
        out = self.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.glu(out)
        return self.layer_norm(residual + out)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) providing instance-wise feature selection and dynamic weighting.
    """

    def __init__(self, num_features: int, hidden_dim: int, dropout_rate: float = 0.12):
        super().__init__()
        self.num_features = num_features
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, dropout_rate=dropout_rate) for _ in range(num_features)
        ])
        self.flattened_grn = GatedResidualNetwork(num_features * hidden_dim, hidden_dim, dropout_rate=dropout_rate)
        self.weight_fc = nn.Linear(hidden_dim, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, num_features, hidden_dim)
        processed_features = []
        for i in range(self.num_features):
            processed_features.append(self.feature_grns[i](x[:, :, i, :]))

        flat = torch.cat(processed_features, dim=-1)
        weights = self.weight_fc(self.flattened_grn(flat))
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)

        stacked = torch.stack(processed_features, dim=2)
        selected = torch.sum(weights * stacked, dim=2)
        return selected


class TemporalFusionTransformerPath(nn.Module):
    """
    Single-Path TFT module with Variable Selection, LSTM locality enhancement,
    Multi-Head Temporal Self-Attention, and Gated Residual Networks.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout_rate: float):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.feature_embed = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(input_dim)
        ])
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim, dropout_rate)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.lstm_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout_rate=dropout_rate)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.post_attention_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout_rate=dropout_rate)
        self.final_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout_rate=dropout_rate)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, feature_dim = x.shape

        embedded = []
        for i in range(feature_dim):
            feat_i = x[:, :, i : i + 1]
            embedded.append(self.feature_embed[i](feat_i))
        stacked = torch.stack(embedded, dim=2)
        selected = self.vsn(stacked)

        lstm_out, _ = self.lstm(selected)
        lstm_out = self.lstm_grn(lstm_out)

        attn_input = self.pos_encoder(lstm_out)
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        attn_out = self.post_attention_grn(attn_out + lstm_out)

        final_out = self.final_grn(attn_out[:, -1, :])
        out = self.fc_out(final_out)
        return out.squeeze(-1)


class DualPathTFTModel(nn.Module):
    """
    Dual-Path Temporal Fusion Transformer framework separately predicting
    Long-term approximation (Eq. 8) and Short-term fluctuations (Eq. 9).
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, num_heads: int = 2, num_layers: int = 2, dropout_rate: float = 0.12):
        super().__init__()
        self.tft_long = TemporalFusionTransformerPath(input_dim, hidden_dim, num_heads, num_layers, dropout_rate)
        self.tft_short = TemporalFusionTransformerPath(input_dim, hidden_dim, num_heads, num_layers, dropout_rate)

    def forward(self, x_long: torch.Tensor, x_short: torch.Tensor):
        pred_long = self.tft_long(x_long)
        pred_short = self.tft_short(x_short)
        return pred_long, pred_short