"""
Global Configuration File for DWDBO Framework.
Contains hyperparameters, system constants, and optimization bounds.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    """Hyperparameters for Temporal Fusion Transformer (TFT)."""
    num_layers: int = 2
    attention_heads: int = 2
    hidden_dim: int = 32
    dropout: float = 0.12
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 150
    lookback_window: int = 48  # 48 hours lookback window
    forecast_horizon: int = 1  # Next hour prediction
    train_split: float = 0.80  # 80% train, 20% test


@dataclass
class BESSConfig:
    """Battery Energy Storage System physical & operational bounds."""
    capacity_bounds_mwh: Tuple[float, float] = (0.0, 50.0)
    power_rating_bounds_mw: Tuple[float, float] = (0.0, 15.0)
    eta_charge: float = 0.90
    eta_discharge: float = 0.95
    soc_min: float = 0.10
    soc_max: float = 0.90
    wear_cost_per_mwh: float = 1.141  # BESS operational/degradation cost coefficient ($/MWh)


@dataclass
class AOAConfig:
    """Adaptive Arithmetic Optimization Algorithm parameters."""
    population_size: int = 30
    max_iterations: int = 40
    moa_min: float = 0.2
    moa_max: float = 0.9
    alpha: float = 5.0
    weights: Tuple[float, float, float, float] = (0.5, 0.2, 0.15, 0.15)  # w1(Cop), w2(Cinv), w3(Vdev), w4(Lloss)


@dataclass
class RiskConfig:
    """Conditional Value-at-Risk (CVaR) risk control settings."""
    confidence_level_alpha: float = 0.95  # Confidence level alpha
    num_scenarios: int = 100
    error_std_dev: float = 0.05  # Standard deviation of renewable forecast error


@dataclass
class ConvergenceConfig:
    """Convergence threshold settings."""
    tolerance: float = 1e-6
    max_outer_iterations: int = 20