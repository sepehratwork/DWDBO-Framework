"""
Multi-Period Optimal Power Flow (OPF) Engine.
Solves network economic dispatch with BESS SOC dynamics (Eq. 14-15),
generator ramping constraints (Eq. 12-13), and power loss calculations over 24h / 48h horizons.
Parallelized using multiprocessing for fast execution.
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from src.power_system.ieee30_data import IEEE30BusData
from config import BESSConfig, ParallelConfig


def _eval_timestep_flow_worker(args: Tuple[int, np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, Any]) -> Tuple[float, float]:
    """Worker function executing parallel DC power flow, loss, and voltage deviation calculations per timestep."""
    t, x_opt, P_D, P_RES_t, P_curt_t, bess_placements, sys_data = args
    num_gen = sys_data.num_generators
    num_bess = len(bess_placements)

    T = len(P_D)
    idx_pg = 0
    idx_pch = idx_pg + num_gen * T
    idx_pdis = idx_pch + num_bess * T

    P_inj = np.zeros(sys_data.num_buses)
    
    # Generator injections
    for g in range(num_gen):
        bus_idx = sys_data.gen_bus_indices[g]
        P_inj[bus_idx] += x_opt[idx_pg + t * num_gen + g]
        
    # Renewable injection at bus 12
    P_inj[12] += (P_RES_t - P_curt_t)
    
    # BESS injections
    for b in range(num_bess):
        b_bus = int(bess_placements[b]) - 1
        if 0 <= b_bus < sys_data.num_buses:
            pch = x_opt[idx_pch + t * num_bess + b]
            pdis = x_opt[idx_pdis + t * num_bess + b]
            P_inj[b_bus] += (pdis - pch)
            
    # Load demand
    P_inj -= (P_D[t] / sys_data.num_buses)

    v_d, l_l = sys_data.compute_network_flow_and_losses(P_inj)
    return v_d, l_l


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

        :param horizon_hours: Scheduling time horizon T (24 or 48 hours).
        :param demand_profile: Hourly load demand vector P_D(t).
        :param p_res_long_profile: Hourly long-term renewable forecast P_long(t).
        :param bess_placements: Bus indices for installed BESS units.
        :param bess_capacities: Storage capacities E_cap (MWh) for installed BESS units.
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
            """Vectorized multi-period objective function computation (Eq. 10)."""
            P_g = x[idx_pg : idx_pch].reshape((T, num_gen))
            P_ch = x[idx_pch : idx_pdis].reshape((T, num_bess))
            P_dis = x[idx_pdis : idx_soc].reshape((T, num_bess))

            cost_gen = np.sum(
                self.sys.cost_a * (P_g ** 2) + self.sys.cost_b * P_g + self.sys.cost_c
            )
            cost_bess = np.sum(self.bess_cfg.degradation_cost * (P_ch + P_dis))
            return float(cost_gen + cost_bess)

        constraints = []

        # 1. Power Balance Constraints (Eq. 11)
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

        # 2. BESS State of Charge Dynamics (Eq. 14)
        dt = 1.0
        for b in range(num_bess):
            cap_b = max(1.0, bess_capacities[b])
            for t in range(T):
                def make_soc_rule(unit_b, time_step, E_cap):
                    def soc_rule(x: np.ndarray) -> float:
                        pch = x[idx_pch + time_step * num_bess + unit_b]
                        pdis = x[idx_pdis + time_step * num_bess + unit_b]
                        soc_curr = x[idx_soc + time_step * num_bess + unit_b]
                        
                        if time_step == 0:
                            soc_prev = self.bess_cfg.soc_initial
                        else:
                            soc_prev = x[idx_soc + (time_step - 1) * num_bess + unit_b]

                        delta_soc = ((self.bess_cfg.eta_charge * pch) - (pdis / self.bess_cfg.eta_discharge)) * dt / E_cap
                        return float(soc_curr - (soc_prev + delta_soc))
                    return soc_rule

                constraints.append({'type': 'eq', 'fun': make_soc_rule(b, t, cap_b)})

        # 3. Generator Ramping Limits (Eq. 13)
        for t in range(1, T):
            for g in range(num_gen):
                def make_ramp_up_rule(time_step, gen_unit):
                    def ramp_up(x: np.ndarray) -> float:
                        pg_curr = x[idx_pg + time_step * num_gen + gen_unit]
                        pg_prev = x[idx_pg + (time_step - 1) * num_gen + gen_unit]
                        return float(self.sys.ramp_limits[gen_unit] - (pg_curr - pg_prev))
                    return ramp_up

                def make_ramp_down_rule(time_step, gen_unit):
                    def ramp_down(x: np.ndarray) -> float:
                        pg_curr = x[idx_pg + time_step * num_gen + gen_unit]
                        pg_prev = x[idx_pg + (time_step - 1) * num_gen + gen_unit]
                        return float(self.sys.ramp_limits[gen_unit] - (pg_prev - pg_curr))
                    return ramp_down

                constraints.append({'type': 'ineq', 'fun': make_ramp_up_rule(t, g)})
                constraints.append({'type': 'ineq', 'fun': make_ramp_down_rule(t, g)})

        # Bounds Setup
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

        # Multi-Processing evaluation for Network Power Flow & Losses across timesteps
        worker_args = [
            (t, x_opt, P_D, P_RES[t], P_curt_opt[t], bess_placements, self.sys)
            for t in range(T)
        ]

        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                flow_results = list(executor.map(_eval_timestep_flow_worker, worker_args))
        else:
            flow_results = [_eval_timestep_flow_worker(arg) for arg in worker_args]

        v_dev_total = sum(r[0] for r in flow_results)
        l_loss_total = sum(r[1] for r in flow_results)

        v_dev_avg = float(v_dev_total / T)
        l_loss_total = float(l_loss_total)
        curtailment_pct = float((np.sum(P_curt_opt) / max(1e-5, np.sum(P_RES))) * 100.0)

        commitment_matrix = (P_g_opt > (self.sys.p_min * 0.1)).astype(int).T

        return c_op, c_inv, v_dev_avg, l_loss_total, curtailment_pct, commitment_matrix