"""
Multi-Period Optimal Power Flow (OPF) Engine.
Solves network economic dispatch with BESS SOC dynamics (Eq. 14-15),
generator ramping constraints (Eq. 12-13), and power loss calculations over 24h / 48h horizons.
Accelerated using multiprocessing and progress tracking.
"""

from typing import Tuple, Optional, List
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import tqdm

from src.power_system.ieee30_data import IEEE30BusData
from config import BESSConfig, ParallelConfig


def _eval_timestep_power_flow(
    t: int,
    P_g_opt_t: np.ndarray,
    P_RES_t: float,
    P_curt_t: float,
    P_D_t: float,
    pch_t: np.ndarray,
    pdis_t: np.ndarray,
    num_buses: int,
    num_generators: int,
    gen_bus_indices: np.ndarray,
    bess_placements: np.ndarray,
    sys_data: IEEE30BusData
) -> Tuple[float, float]:
    """Top-level helper function for multiprocessing evaluation of timestep power flow."""
    P_inj = np.zeros(num_buses)
    for g in range(num_generators):
        bus_idx = gen_bus_indices[g]
        P_inj[bus_idx] += P_g_opt_t[g]
    P_inj[12] += (P_RES_t - P_curt_t)
    
    num_bess = len(bess_placements)
    for b in range(num_bess):
        b_bus = int(bess_placements[b]) - 1
        if 0 <= b_bus < num_buses:
            P_inj[b_bus] += (pdis_t[b] - pch_t[b])
            
    P_inj -= (P_D_t / num_buses)
    return sys_data.compute_network_flow_and_losses(P_inj)


class MultiPeriodOPFSolver:
    """Solves multi-period network constrained cost minimization problem (Upper-Level OPF)."""

    def __init__(self, sys_data: IEEE30BusData, bess_cfg: BESSConfig,
                 parallel_config: Optional[ParallelConfig] = None):
        self.sys = sys_data
        self.bess_cfg = bess_cfg
        self.parallel_cfg = parallel_config or ParallelConfig()
        self.n_workers = self.parallel_cfg.get_effective_workers()

    def solve_multi_period_dispatch(
        self,
        horizon_hours: int,
        demand_profile: np.ndarray,
        p_res_long_profile: np.ndarray,
        bess_placements: np.ndarray,
        bess_capacities: np.ndarray
    ) -> Tuple[float, float, float, float, float, np.ndarray]:
        """
        Solves multi-period OPF minimization problem (Eq. 10 - Eq. 15) using multiprocessing.

        :return: Tuple of (C_op, C_inv, V_dev, L_loss, curtailment_pct, commitment_matrix).
        """
        T = horizon_hours
        num_gen = self.sys.num_generators
        num_bess = len(bess_placements)

        P_D = demand_profile[:T] if len(demand_profile) >= T else np.pad(demand_profile, (0, T - len(demand_profile)), mode='edge')
        P_RES = p_res_long_profile[:T] if len(p_res_long_profile) >= T else np.pad(p_res_long_profile, (0, T - len(p_res_long_profile)), mode='edge')

        idx_pg = 0
        idx_pch = idx_pg + num_gen * T
        idx_pdis = idx_pch + num_bess * T
        idx_soc = idx_pdis + num_bess * T
        idx_curtail = idx_soc + num_bess * T
        num_vars = idx_curtail + T

        def objective(x: np.ndarray) -> float:
            P_g = x[idx_pg : idx_pch].reshape((T, num_gen))
            P_ch = x[idx_pch : idx_pdis].reshape((T, num_bess))
            P_dis = x[idx_pdis : idx_soc].reshape((T, num_bess))

            cost_gen = np.sum(self.sys.cost_a * (P_g ** 2) + self.sys.cost_b * P_g + self.sys.cost_c)
            cost_bess = np.sum(self.bess_cfg.degradation_cost * (P_ch + P_dis))
            return float(cost_gen + cost_bess)

        constraints = []

        # Power balance constraints (Eq. 11)
        for t in range(T):
            def make_balance_rule(time_step):
                def balance_rule(x: np.ndarray) -> float:
                    pg_t = x[idx_pg + time_step * num_gen : idx_pg + (time_step + 1) * num_gen]
                    pch_t = x[idx_pch + time_step * num_bess : idx_pch + (time_step + 1) * num_bess]
                    pdis_t = x[idx_pdis + time_step * num_bess : idx_pdis + (time_step + 1) * num_bess]
                    p_curt_t = x[idx_curtail + time_step]
                    
                    gen_total = np.sum(pg_t)
                    bess_net = np.sum(pdis_t - pch_t)
                    res_net = P_RES[time_step] - p_curt_t
                    loss_est = 0.02 * (P_D[time_step] / self.sys.base_demand) * 10.0
                    return float(gen_total + res_net + bess_net - (P_D[time_step] + loss_est))
                return balance_rule

            constraints.append({'type': 'eq', 'fun': make_balance_rule(t)})

        # BESS SOC Dynamics (Eq. 14)
        dt = 1.0
        for b in range(num_bess):
            cap_b = max(1.0, bess_capacities[b])
            for t in range(T):
                def make_soc_rule(unit_b, time_step, E_cap):
                    def soc_rule(x: np.ndarray) -> float:
                        pch = x[idx_pch + time_step * num_bess + unit_b]
                        pdis = x[idx_pdis + time_step * num_bess + unit_b]
                        soc_curr = x[idx_soc + time_step * num_bess + unit_b]
                        soc_prev = self.bess_cfg.soc_initial if time_step == 0 else x[idx_soc + (time_step - 1) * num_bess + unit_b]
                        delta_soc = ((self.bess_cfg.eta_charge * pch) - (pdis / self.bess_cfg.eta_discharge)) * dt / E_cap
                        return float(soc_curr - (soc_prev + delta_soc))
                    return soc_rule

                constraints.append({'type': 'eq', 'fun': make_soc_rule(b, t, cap_b)})

        # Ramping limits (Eq. 13)
        for t in range(1, T):
            for g in range(num_gen):
                def make_ramp_up_rule(time_step, gen_unit):
                    def ramp_up(x: np.ndarray) -> float:
                        return float(self.sys.ramp_limits[gen_unit] - (x[idx_pg + time_step * num_gen + gen_unit] - x[idx_pg + (time_step - 1) * num_gen + gen_unit]))
                    return ramp_up

                def make_ramp_down_rule(time_step, gen_unit):
                    def ramp_down(x: np.ndarray) -> float:
                        return float(self.sys.ramp_limits[gen_unit] - (x[idx_pg + (time_step - 1) * num_gen + gen_unit] - x[idx_pg + time_step * num_gen + gen_unit]))
                    return ramp_down

                constraints.append({'type': 'ineq', 'fun': make_ramp_up_rule(t, g)})
                constraints.append({'type': 'ineq', 'fun': make_ramp_down_rule(t, g)})

        # Variable Bounds
        bounds = []
        for t in range(T):
            for g in range(num_gen):
                bounds.append((self.sys.p_min[g], self.sys.p_max[g]))
        for t in range(T):
            for b in range(num_bess):
                bounds.append((self.bess_cfg.power_min_mw, self.bess_cfg.power_max_mw))
        for t in range(T):
            for b in range(num_bess):
                bounds.append((self.bess_cfg.power_min_mw, self.bess_cfg.power_max_mw))
        for t in range(T):
            for b in range(num_bess):
                bounds.append((self.bess_cfg.soc_min, self.bess_cfg.soc_max))
        for t in range(T):
            bounds.append((0.0, max(0.0, P_RES[t])))

        x0 = np.zeros(num_vars)
        for t in range(T):
            x0[idx_pg + t * num_gen : idx_pg + (t + 1) * num_gen] = (self.sys.p_min + self.sys.p_max) / 2.0
            x0[idx_soc + t * num_bess : idx_soc + (t + 1) * num_bess] = self.bess_cfg.soc_initial

        # SLSQP Optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        x_opt = res.x if res.success else x0
        c_op = float(objective(x_opt))
        c_inv = float(np.sum(bess_capacities) * self.bess_cfg.capital_cost_per_mwh) if num_bess > 0 else 0.0

        P_g_opt = x_opt[idx_pg : idx_pch].reshape((T, num_gen))
        P_curt_opt = x_opt[idx_curtail :]

        # Multiprocessing calculation of network flow across timesteps
        if self.n_workers > 1:
            flow_results = Parallel(n_jobs=self.n_workers)(
                delayed(_eval_timestep_power_flow)(
                    t,
                    P_g_opt[t],
                    P_RES[t],
                    P_curt_opt[t],
                    P_D[t],
                    x_opt[idx_pch + t * num_bess : idx_pch + (t + 1) * num_bess],
                    x_opt[idx_pdis + t * num_bess : idx_pdis + (t + 1) * num_bess],
                    self.sys.num_buses,
                    self.sys.num_generators,
                    self.sys.gen_bus_indices,
                    bess_placements,
                    self.sys
                )
                for t in range(T)
            )
        else:
            flow_results = [
                _eval_timestep_power_flow(
                    t, P_g_opt[t], P_RES[t], P_curt_opt[t], P_D[t],
                    x_opt[idx_pch + t * num_bess : idx_pch + (t + 1) * num_bess],
                    x_opt[idx_pdis + t * num_bess : idx_pdis + (t + 1) * num_bess],
                    self.sys.num_buses, self.sys.num_generators,
                    self.sys.gen_bus_indices, bess_placements, self.sys
                )
                for t in range(T)
            ]

        v_dev_total = sum(r[0] for r in flow_results)
        l_loss_total = sum(r[1] for r in flow_results)

        v_dev_avg = float(v_dev_total / T)
        l_loss_total = float(l_loss_total)
        curtailment_pct = float((np.sum(P_curt_opt) / max(1e-5, np.sum(P_RES))) * 100.0)
        commitment_matrix = (P_g_opt > (self.sys.p_min * 0.1)).astype(int).T

        return c_op, c_inv, v_dev_avg, l_loss_total, curtailment_pct, commitment_matrix