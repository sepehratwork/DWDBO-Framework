
***

# DWDBO-Framework: A Wavelet-Driven Bi-Level Optimization Framework for BESS and Renewable Energy Planning under CVaR Risk Management

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE 30-Bus System](https://img.shields.io/badge/Benchmark-IEEE%2030--Bus-green.svg)](#)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Official PyTorch and Python implementation of the paper:  
**"A Wavelet-Driven Bi-Level Optimization Framework for Battery Energy Storage System and Renewable Energy Planning under CVaR-Based Risk Management"**

---

## 📋 Table of Contents
1. [Executive Overview](#-executive-overview)
2. [Key Features](#-key-features)
3. [Mathematical & Algorithmic Architecture](#-mathematical--algorithmic-architecture)
    - [1. Data Imputation & Signal Decomposition](#1-data-imputation--signal-decomposition)
    - [2. Dual-Path Temporal Fusion Transformer (TFT)](#2-dual-path-temporal-fusion-transformer-tft)
    - [3. Upper-Level Optimization: BESS Planning via Adaptive AOA](#3-upper-level-optimization-bess-planning-via-adaptive-aoa)
    - [4. Lower-Level Optimization: CVaR-Aided Real-Time Dispatch](#4-lower-level-optimization-cvar-aided-real-time-dispatch)
4. [Repository Structure](#-repository-structure)
5. [Installation & Setup](#-installation--setup)
6. [Dataset Acquisition & Preprocessing](#-dataset-acquisition--preprocessing)
7. [Execution & Usage Guide](#-execution--usage-guide)
8. [Experimental Results & Benchmarks](#-experimental-results--benchmarks)
9. [Citation & Contact](#-citation--contact)

---

## 📑 Executive Overview

Modern power systems face significant operational volatility due to the high penetration of intermittent renewable energy sources (RESs), such as Photovoltaic (PV) systems and Wind Farms (WFs). Traditional single-level deterministic optimization frameworks fail to separate long-term strategic investments from short-term real-time corrective actions, leading to excessive renewable curtailment and unhedged operational exposure to tail-risk events.

The **Deep Wavelet-Driven Bi-Level Optimization (DWDBO)** framework resolves these challenges through an end-to-end multi-timescale strategy:
- **Multi-Scale Feature Signal Processing**: Decomposes raw renewable energy time-series into persistent long-term trends and high-frequency short-term fluctuations using Discrete Wavelet Transform (DWT).
- **Dual-Path Deep Learning**: Utilizes separate Temporal Fusion Transformer (TFT) networks for multi-horizon trend and fluctuation predictions ($R^2 = 0.9765$).
- **Strategic Upper-Level Planning**: Employs an **Adaptive Arithmetic Optimization Algorithm (AOA)** to optimize the placement, power rating, and energy capacity of Battery Energy Storage Systems (BESS) alongside unit commitment on the IEEE 30-bus test system.
- **Risk-Averse Lower-Level Real-Time Dispatch**: Integrates **Conditional Value-at-Risk (CVaR)** to hedge against tail-end forecast errors and extreme renewable intermittency scenarios.

---

## ✨ Key Features

- **End-to-End Modular Pipeline**: Seamlessly combines missing data imputation (KNN Imputer), multi-rate signal analysis (DWT), deep learning forecasting (TFT), metaheuristic optimization (Adaptive AOA), and risk modeling (CVaR).
- **Adaptive Exploration-Exploitation Mechanism**: Time-varying Math Optimizer Accelerated (MOA) and Math Optimizer Probability (MOP) schedules prevent premature convergence in combinatorial BESS siting and sizing problems.
- **Tail-Risk Control**: Lower-level CVaR-constrained real-time dispatch minimizes operating costs in the worst $(1-\alpha)\%$ scenarios.
- **Power Flow Integration**: Built-in non-linear AC/DC Optimal Power Flow (OPF) evaluation enforcing thermal, voltage deviation, ramp rate, generator, and storage state-of-charge (SOC) physical constraints.

---

## 📐 Mathematical & Algorithmic Architecture

The DWDBO model decouples strategic planning decisions from operational dispatch decisions via a two-stage hierarchical bi-level optimization scheme:

```
                                 DWDBO Framework Pipeline
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ 1. Data Acquisition & Preprocessing                                                    │
 │    - Open Power System Data (OPSD) -> KNN Imputation -> Cleaned Time-Series            │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ 2. Multi-Scale Signal Decomposition (DWT)                                              │
 │    - P_RES[n] = P_RES_long[n] (Low Frequency) + P_RES_short[n] (High Frequency)        │
 └─────────────────────────────┬───────────────────────────────┬──────────────────────────┘
                               │                               │
                               ▼                               ▼
 ┌──────────────────────────────────────────┐    ┌────────────────────────────────────────┐
 │ 3a. TFT Long-Term Path                   │    │ 3b. TFT Short-Term Path                │
 │     - Predicts P̂_long (Trend/Scheduling) │    │     - Predicts P̂_short (Fluctuations)  │
 └─────────────────────────────┬────────────┘    └────────────────┬───────────────────────┘
                               │                                  │
                               ▼                                  │
 ┌─────────────────────────────────────────────────────────────┐  │
 │ 4. Upper-Level Optimization (Adaptive AOA)                  │  │
 │    - BESS Placement (Bus) & Sizing (Capacity, Power Rating) │  │
 │    - Long-Term Optimal Power Flow Scheduling                │  │
 └─────────────────────────────┬───────────────────────────────┘  │
                               │                                  │
                               │ Upper-Level Parameters           │ Feedback
                               ▼                                  │ Iteration
 ┌────────────────────────────────────────────────────────────┐   │ Loop
 │ 5. Lower-Level Optimization (CVaR Risk Control)            │◀─┘
 │    - Scenario-based stochastic adjustment (ΔP_short)       │
 │    - Minimizes CVaR_α of real-time operating cost          │
 └────────────────────────────────────────────────────────────┘
```

### 1. Data Imputation & Signal Decomposition
Raw time-series from the ENTSO-E platform processed via Open Power System Data (OPSD) contain missing entries filled via distance-weighted K-Nearest Neighbors (KNN). Cleaned signals $x[n]$ undergo Discrete Wavelet Transform (DWT):
$W_\psi(j,k) = \frac{1}{\sqrt{2^j}} \sum_n x[n] \psi^*\left(\frac{n - 2^j k}{2^j}\right)$

$P_{\text{RES}}[n] = P_{\text{RES}}^{\text{long}}[n] + P_{\text{RES}}^{\text{short}}[n]$

where $P_{\text{RES}}^{\text{long}}[n] = \sum_k cA_J[k] \phi_{J,k}[n]$ represents smooth trends, and $P_{\text{RES}}^{\text{short}}[n] = \sum_{j=1}^J \sum_k cD_j[k] \psi_{j,k}[n]$ represents high-frequency volatility.

### 2. Dual-Path Temporal Fusion Transformer (TFT)
Separate TFT models process static context information $c_s$ and dynamic temporal sequences $\xi_t$ via Gated Residual Networks (GRN) and Gated Linear Units (GLU):
$\tilde{\xi}_t = \text{GLU}(\text{GRN}_\xi(\xi_t, c_s)) \odot \xi_t$

Independent transformer paths yield:
$\hat{P}_{t+h}^{\text{long}} = \text{TFT}_{\text{long}}\left(P_{t-\tau:t}^{\text{long}}\right), \quad \hat{P}_{t+h}^{\text{short}} = \text{TFT}_{\text{short}}\left(P_{t-\tau:t}^{\text{short}}\right)$

### 3. Upper-Level Optimization: BESS Planning via Adaptive AOA
The upper-level objective minimizes total operational costs, investment costs, voltage deviations, and grid line losses:
$\min_{P_G, P_{\text{BESS}}} C_{op} = \sum_{t,g} \left( a_g P_{G,g}(t)^2 + b_g P_{G,g}(t) + c_g \right) + \sum_{t,b} C_{\text{BESS},b} |P_{\text{BESS},b}(t)|$

$\text{Fitness } F_{\text{BESS}} = w_1 C_{op} + w_2 C_{inv} + w_3 V_{dev} + w_4 L_{loss}$

Siting vector $X = [L_1, \dots, L_{N_{\text{BESS}}}, S_1, \dots, S_{N_{\text{BESS}}}]$ is navigated using adaptive AOA coefficient updates:
$\text{MOA}(t) = \text{MOA}_{\min} + \frac{(\text{MOA}_{\max} - \text{MOA}_{\min}) t}{T_{\max}}$

$\text{MOP}(t) = 1 - \left( \frac{t}{T_{\max}} \right)^{1/\alpha}$

### 4. Lower-Level Optimization: CVaR-Aided Real-Time Dispatch
The lower-level corrects for real-time short-term deviations $\Delta P^{\text{short}}$ using Conditional Value-at-Risk (CVaR) at confidence level $\alpha$:
$\min_{\zeta, \eta_s} \text{CVaR}_\alpha = \zeta + \frac{1}{1 - \alpha} \sum_{s=1}^{N_S} \pi_s \eta_s$

$\text{Subject to: } \eta_s \ge C_s - \zeta, \quad \eta_s \ge 0, \quad \epsilon_{r,s}(t) \sim \mathcal{N}\left(0, \sigma_r^2(t)\right)$

---

## 📁 Repository Structure

The code implementation strictly reflects the file layout below:

```text
DWDBO-Framework/
├── data/                      # Raw and processed time-series datasets
├── src/                       # Source modules
│   ├── data_processing/       # Data cleaning, imputation, and signal processing
│   │   ├── __init__.py
│   │   ├── imputer.py         # K-Nearest Neighbors (KNN) data imputer
│   │   └── wavelet.py         # Discrete Wavelet Transform (DWT) signal decomposition
│   ├── models/                # Deep learning forecasting models
│   │   ├── __init__.py        # Temporal Fusion Transformer (TFT) dual-path module
│   ├── optimization/          # Metaheuristic and mathematical solvers
│   │   ├── __init__.py
│   │   ├── adaptive_aoa.py    # Adaptive Arithmetic Optimization Algorithm (Upper Level)
│   │   └── cvar_lower_level.py# Conditional Value-at-Risk (CVaR) real-time solver
│   ├── pipeline/              # Bi-level integration framework
│   │   ├── __init__.py
│   │   └── dwdbo_solver.py    # Main DWDBO bi-level execution solver
│   ├── power_system/          # Power system simulation environment
│   │   ├── __init__.py
│   │   ├── ieee30_data.py     # IEEE 30-bus test system definitions and parameters
│   │   └── power_flow.py      # Non-linear AC/DC Optimal Power Flow (OPF) module
│   └── __init__.py
├── .gitignore                 # Version control git ignore rules
├── config.py                  # Global configurations, hyperparameters, and path definitions
├── main.py                    # Top-level executable script for end-to-end execution
├── README.md                  # System documentation
└── requirements.txt           # Environment dependencies
```

### Module Descriptions

- `src/data_processing/imputer.py`: Identifies missing entries in load/solar/wind data and performs KNN-based spatial-temporal distance imputation.
- `src/data_processing/wavelet.py`: Implements forward and inverse DWT decomposition to separate renewable generation into low-frequency ($P^{\text{long}}$) and high-frequency ($P^{\text{short}}$) signals.
- `src/models/`: Implements the dual-path TFT architecture with multi-head self-attention, variable selection, and static context gating.
- `src/optimization/adaptive_aoa.py`: Contains the Adaptive Arithmetic Optimization Algorithm used for BESS siting and sizing.
- `src/optimization/cvar_lower_level.py`: Linear programming/convex optimization engine constructing scenario-based CVaR risk bounds.
- `src/power_system/ieee30_data.py`: Provides generator quadratic cost profiles, bus topology, branch limits, and candidate locations for the benchmark IEEE 30-bus grid.
- `src/power_system/power_flow.py`: Computes network load flow, bus voltage deviations, and line transmission losses.
- `src/pipeline/dwdbo_solver.py`: Bridges upper-level strategic sizing with lower-level operational CVaR dispatch in an iterative feedback loop.
- `config.py`: Central repository setting hyperparameters (learning rates, batch size, DWT levels, CVaR confidence parameter $\alpha$, and AOA population size).

---

## ⚙️ Installation & Setup

### Prerequisites
- Operating System: Linux, macOS, or Windows 11 (64-bit)
- Python Version: 3.12
- CUDA Toolkit (Optional, for GPU acceleration of TFT models)

### 1. Clone the Repository
```bash
git clone https://github.com/sepehratwork/DWDBO-Framework.git
cd DWDBO-Framework
```

### 2. Create Virtual Environment
Using `conda`:
```bash
conda create -n dwdbo_env python=3.12 -y
conda activate dwdbo_env
```
Or using `venv`:
```bash
python -m venv dwdbo_env
source dwdbo_env/bin/activate  # On Windows: dwdbo_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💾 Dataset Acquisition & Preprocessing

The framework uses European electricity market datasets provided by the Open Power System Data (OPSD) platform (derived from the ENTSO-E Transparency Portal). 

Execute the following bash commands to download the raw hourly single-index time-series directly into the target `data/` directory:

```bash
mkdir -p data
cd data
wget https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv -O opsd_raw_data.csv
cd ..
```

---

## 🚀 Execution & Usage Guide

### Full Pipeline Execution
To execute the complete end-to-end framework—including missing data imputation, DWT signal decomposition, dual-path TFT training, Adaptive AOA BESS optimization, and lower-level CVaR real-time dispatch—run `main.py`:

```bash
python main.py
```

---

## 📊 Experimental Results & Benchmarks

The framework was evaluated on the IEEE 30-bus transmission test system. Below are key experimental findings reported in the paper:

### 1. Renewable Forecasting Performance (DWT + TFT)
Tested on 5,000 samples at a 15-minute temporal resolution (80% training / 20% testing split):

| Target Feature | MAE | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **PV / Wind Generation Prediction** | **84.3946** | **111.1693** | **0.9765** |

### 2. Adaptive AOA Siting & Sizing Solution
Optimal placement and parameters determined by the Adaptive AOA over 40 iterations (improving objective fitness by **41.76%**):

| Selected Bus Number | Installed Capacity (MWh) | Power Rating (MW) |
| :---: | :---: | :---: |
| **Bus 1** | 10.01 | 2.272 |
| **Bus 4** | 10.09 | 1.113 |

### 3. Grid Performance Comparison (24-Hour Scheduling Horizon)

| Operational Metric | Baseline (Without BESS) | Proposed (With BESS) | Net Improvement (%) |
| :--- | :---: | :---: | :---: |
| **Operating Cost ($C_{op}$)** | \$5,667.658 | **\$5,640.171** | **+0.484%** |
| **Renewable Curtailment (MW)** | 2.654 MW | **2.576 MW** | **+2.966% (~3%)** |
| **Total System Losses ($L_{loss}$)** | 48.290 MW | **48.283 MW** | **+0.015%** |
| **Voltage Deviation ($V_{dev}$)** | 9.056 | 9.063 | -0.079% |

### 4. CVaR Tail-Risk Sensitivity Analysis
Impact of confidence level ($\alpha$) on operational vs. risk costs:

| CVaR Confidence Level ($\alpha$) | Expected Operational Cost (\$) | CVaR Tail Cost (\$) | Risk Aversion Behavior |
| :---: | :---: | :---: | :--- |
| **0.90** | \$657.28 | \$988.88 | Moderate Risk Neutrality |
| **0.95** | \$664.14 | \$997.33 | Balanced Risk-Averse |
| **0.99** | \$669.73 | \$1007.27 | Conservative (High Tail Protection) |

---

## 📜 Citation & Contact

If you find this repository or paper useful in your research, please cite:

```bibtex
@article{DWDBO2026,
  title     = {A Wavelet-Driven Bi-Level Optimization Framework for Battery Energy Storage System and Renewable Energy Planning under CVaR-Based Risk Management},
  author    = {Saeed Abazari, Ali Abdollahi, Sepehr Kerachi},
  journal   = {IEEE Transactions / Energy Systems},
  year      = {2026},
  publisher = {IEEE}
}
```

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.