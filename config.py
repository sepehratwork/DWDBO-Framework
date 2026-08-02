"""
Global Configuration File for DWDBO Framework.
Contains exact hyperparameters, physical parameters, parallelization settings,
cache options, and output configurations specified throughout the paper.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class ParallelConfig:
    """Parallelization configuration for multi-core acceleration."""
    num_workers: int = -1
    use_multiprocessing: bool = True

    def get_effective_workers(self) -> int:
        """Resolves negative worker counts to physical CPU core limits."""
        if self.num_workers <= 0:
            cpu_cnt = os.cpu_count() or 4
            return max(1, cpu_cnt - 2 if cpu_cnt > 2 else 1)
        return self.num_workers


@dataclass
class CacheConfig:
    """Disk caching configuration to support checkpoint resumption."""
    enable_cache: bool = True
    cache_dir: str = "cache"


@dataclass
class OutputConfig:
    """Output directory and visualization export settings."""
    results_dir: str = "results"
    export_figures: bool = True
    export_tables: bool = True
    figure_dpi: int = 300


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
    lookback_window: int = 48  # 48 lookback time steps (Section 3)
    forecast_horizon: int = 1  # 1-step forecast horizon


@dataclass
class BESSConfig:
    """Battery Energy Storage System physical & operational bounds (Section 2.4, Section 3.3)."""
    num_units: int = 2
    capacity_min_mwh: float = 0.0
    capacity_max_mwh: float = 50.0  # [0.0 - 50.0 MWh]
    power_min_mw: float = 0.0
    power_max_mw: float = 15.0      # [0.0 - 15.0 MW]
    eta_charge: float = 0.90        # Charging efficiency eta_ch = 0.90 (Section 2.4)
    eta_discharge: float = 0.95     # Discharging efficiency eta_dis = 0.95 (Section 2.4)
    soc_min: float = 0.10           # Minimum SOC (10%) (Eq. 15)
    soc_max: float = 0.90           # Maximum SOC (90%) (Eq. 15)
    soc_initial: float = 0.50       # Initial state of charge (50%)
    degradation_cost: float = 1.141 # BESS wear cost C_BESS ($/MWh) (Table 5)
    capital_cost_per_mwh: float = 15.0 # Investment cost coefficient ($/MWh)


@dataclass
class AOAConfig:
    """Adaptive Arithmetic Optimization Algorithm parameters (Algorithm 1, Section 2.6)."""
    population_size: int = 30
    max_iterations: int = 40
    moa_min: float = 0.2
    moa_max: float = 0.9
    alpha: float = 5.0  # Regulates decay profile of MOP(t) (Eq. 20)
    weights: Tuple[float, float, float, float] = (0.50, 0.20, 0.15, 0.15)


@dataclass
class CVaRConfig:
    """Lower-level Conditional Value-at-Risk parameters (Section 2.5, Section 3.5)."""
    confidence_level_alpha: float = 0.95
    alpha_sensitivity_levels: List[float] = field(
        default_factory=lambda: [0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
    )
    num_scenarios: int = 100
    forecast_error_std: float = 0.05


@dataclass
class ConvergenceConfig:
    """Bi-level stopping condition settings (Eq. 23)."""
    tau_tolerance: float = 1e-6
    max_outer_iterations: int = 10
    epsilon_stabilizer: float = 1e-8