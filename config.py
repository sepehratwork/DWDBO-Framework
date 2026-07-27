"""
Global Configuration File for DWDBO Framework.
Contains exact hyperparameters, physical parameters, and optimization bounds 
specified throughout the paper.
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class DatasetConfig:
    """Open Power System Data (OPSD) time-series specifications."""
    num_samples: int = 5000  # Sample dataset length (Section 3.2)
    time_step_minutes: int = 15  # 15-minute resolution
    train_split: float = 0.80  # 80% train, 20% test (Section 3.2)


@dataclass
class TFTConfig:
    """Temporal Fusion Transformer (TFT) hyperparameters (Table 2)."""
    num_layers: int = 2
    attention_heads: int = 2
    hidden_dim: int = 32
    dropout_rate: float = 0.12
    training_epochs: int = 150
    batch_size: int = 128
    learning_rate: float = 0.001
    lookback_window: int = 192  # 48 hours lookback (48 * 4 steps at 15-min res)
    forecast_horizon: int = 4   # 1 hour forecast (4 steps at 15-min res)


@dataclass
class BESSConfig:
    """Battery Energy Storage System physical & operational bounds (Section 2.4, Section 3.3)."""
    num_units: int = 2
    capacity_min_mwh: float = 0.0
    capacity_max_mwh: float = 50.0  # [0.0 - 50.0 MWh]
    power_min_mw: float = 0.0
    power_max_mw: float = 15.0      # [0.0 - 15.0 MW]
    eta_charge: float = 0.90        # Charging efficiency eta_ch = 0.90
    eta_discharge: float = 0.95     # Discharging efficiency eta_dis = 0.95
    soc_min: float = 0.10           # Minimum SOC (10%)
    soc_max: float = 0.90           # Maximum SOC (90%)
    soc_initial: float = 0.50       # Initial state of charge (50%)
    degradation_cost: float = 1.141 # BESS wear cost C_BESS ($/MWh)


@dataclass
class AOAConfig:
    """Adaptive Arithmetic Optimization Algorithm parameters (Algorithm 1, Section 2.6)."""
    population_size: int = 30
    max_iterations: int = 40
    moa_min: float = 0.2
    moa_max: float = 0.9
    alpha: float = 5.0  # Regulates decay profile of MOP(t)
    # Weights for multi-objective fitness function F_BESS = w1*Cop + w2*Cinv + w3*Vdev + w4*Lloss
    weights: Tuple[float, float, float, float] = (0.50, 0.20, 0.15, 0.15)


@dataclass
class CVaRConfig:
    """Lower-level Conditional Value-at-Risk parameters (Section 2.5, Section 3.5)."""
    confidence_level_alpha: float = 0.95  # Confidence level alpha (tested across 0.90, 0.95, 0.99)
    num_scenarios: int = 100
    forecast_error_std: float = 0.05      # Zero-mean Gaussian error variance proxy


@dataclass
class ConvergenceConfig:
    """Bi-level stopping condition settings (Eq. 23)."""
    tau_tolerance: float = 1e-6
    max_outer_iterations: int = 10
    epsilon_stabilizer: float = 1e-8