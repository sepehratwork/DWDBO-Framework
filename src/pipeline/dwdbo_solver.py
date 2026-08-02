"""
Integrated DWDBO Master Pipeline Coordinator.
Executes Algorithm 1 bi-level iteration logic, manages checkpoint persistence,
and displays progress bars for every stage of the pipeline.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    DatasetConfig, TFTConfig, BESSConfig, AOAConfig, CVaRConfig, 
    ConvergenceConfig, ParallelConfig, CacheConfig, OutputConfig
)
from src.utils.cache import CacheManager
from src.utils.results import PaperResultsGenerator
from src.data_processing import TimeSeriesKNNImputer, DiscreteWaveletDecomposer
from src.models import TFTTrainerEngine
from src.power_system import IEEE30BusData, MultiPeriodOPFSolver
from src.optimization import AdaptiveAOASolver, CVaRRealTimeOptimizer


class DWDBOMasterFramework:
    """Master controller coordinating the full DWDBO optimization framework."""

    def __init__(self,
                 parallel_cfg: Optional[ParallelConfig] = None,
                 cache_cfg: Optional[CacheConfig] = None,
                 output_cfg: Optional[OutputConfig] = None):
        
        self.parallel_cfg = parallel_cfg or ParallelConfig()
        self.cache_cfg = cache_cfg or CacheConfig()
        self.output_cfg = output_cfg or OutputConfig()

        self.cache = CacheManager(self.cache_cfg)
        self.results_gen = PaperResultsGenerator(self.output_cfg)

        self.dataset_cfg = DatasetConfig()
        self.tft_cfg = TFTConfig()
        self.bess_cfg = BESSConfig()
        self.aoa_cfg = AOAConfig()
        self.cvar_cfg = CVaRConfig()
        self.conv_cfg = ConvergenceConfig()

        self.imputer = TimeSeriesKNNImputer()
        self.decomposer = DiscreteWaveletDecomposer()
        self.tft_engine = TFTTrainerEngine(self.tft_cfg)
        self.sys_data = IEEE30BusData()
        self.opf_solver = MultiPeriodOPFSolver(self.sys_data, self.bess_cfg, parallel_config=self.parallel_cfg)
        self.aoa_solver = AdaptiveAOASolver(self.aoa_cfg, self.bess_cfg, parallel_config=self.parallel_cfg)
        self.cvar_optimizer = CVaRRealTimeOptimizer(self.cvar_cfg, parallel_config=self.parallel_cfg)

    def execute_framework(self, df_raw: pd.DataFrame, scheduling_horizon_hours: int = 24) -> Dict[str, Any]:
        """
        Executes complete paper pipeline with progress bars and caching.
        """
        print(f"\n==========================================================================")
        print(f" EXECUTING DWDBO PIPELINE ({scheduling_horizon_hours}-HOUR SCHEDULING HORIZON)")
        print(f"==========================================================================\n")

        # Step 1: Missing Data Imputation Progress Bar
        with tqdm(total=100, desc="[Step 1/5] KNN Multivariate Data Imputation", unit="%") as pbar1:
            cache_key_imp = "step1_imputed_dataframe"
            if self.cache.exists(cache_key_imp):
                df_clean = self.cache.load(cache_key_imp)
            else:
                df_clean = self.imputer.impute_missing_data(df_raw)
                self.cache.save(cache_key_imp, df_clean)
            pbar1.update(100)

        self.results_gen.plot_fig3_knn_imputation(df_raw, df_clean)

        pv_signal = df_clean["solar_power"].fillna(0).to_numpy() if "solar_power" in df_clean.columns else df_clean.iloc[:, 0].to_numpy()
        wind_signal = df_clean["wind_power"].fillna(0).to_numpy() if "wind_power" in df_clean.columns else df_clean.iloc[:, 0].to_numpy()
        res_signal = pv_signal + wind_signal

        # Step 2: DWT Decomposition Progress Bar
        with tqdm(total=100, desc="[Step 2/5] Discrete Wavelet Decomposition (db4)", unit="%") as pbar2:
            cache_key_dwt = "step2_dwt_decomposed_signals"
            if self.cache.exists(cache_key_dwt):
                p_long, p_short, depth_J = self.cache.load(cache_key_dwt)
            else:
                p_long, p_short, depth_J = self.decomposer.decompose_signal(res_signal)
                self.cache.save(cache_key_dwt, (p_long, p_short, depth_J))
            pbar2.update(100)

        pv_long, pv_short, _ = self.decomposer.decompose_signal(pv_signal)
        wind_long, wind_short, _ = self.decomposer.decompose_signal(wind_signal)

        # Step 3: Dual-Path TFT Forecasting Progress Bar
        with tqdm(total=100, desc="[Step 3/5] Dual-Path TFT Model Training & Forecasting", unit="%") as pbar3:
            cache_key_tft = "step3_tft_forecasts_metrics"
            if self.cache.exists(cache_key_tft):
                metrics, pred_long, pred_short, history_pv, history_wind, eval_pv, eval_wind = self.cache.load(cache_key_tft)
            else:
                metrics, pred_long, pred_short, history_pv, eval_pv = self.tft_engine.train_and_forecast_single_source(pv_long, pv_short)
                m_wind, p_wind_l, p_wind_s, history_wind, eval_wind = self.tft_engine.train_and_forecast_single_source(wind_long, wind_short)
                
                metrics["MAE"] = float((metrics["MAE"] + m_wind["MAE"]) / 2.0)
                metrics["RMSE"] = float((metrics["RMSE"] + m_wind["RMSE"]) / 2.0)
                metrics["R2"] = float((metrics["R2"] + m_wind["R2"]) / 2.0)

                self.cache.save(cache_key_tft, (metrics, pred_long, pred_short, history_pv, history_wind, eval_pv, eval_wind))
            pbar3.update(100)

        self.results_gen.print_and_export_table3(metrics)
        self.results_gen.plot_fig4_tft_losses_and_correlation(history_pv, history_wind, eval_pv, eval_wind)
        self.results_gen.plot_fig5_actual_vs_predicted(eval_pv["y_actual"], eval_pv["y_pred"], eval_wind["y_actual"], eval_wind["y_pred"])

        T = scheduling_horizon_hours
        demand_h = df_clean["load_demand"].to_numpy()[:T] if "load_demand" in df_clean.columns else np.full(T, self.sys_data.base_demand)
        p_long_h = pred_long[:T]
        p_short_h = pred_short[:T]
        num_units = self.bess_cfg.num_units

        # Step 4: Adaptive AOA Optimization (Progress bar built into solver)
        cache_key_aoa = f"step4_aoa_opt_horizon_{T}"
        if self.cache.exists(cache_key_aoa):
            best_X, best_fitness, conv_curve, pop_fitness_dist = self.cache.load(cache_key_aoa)
        else:
            def multi_objective_fitness(X: np.ndarray) -> float:
                buses = X[: num_units].astype(int)
                capacities = X[num_units :]
                c_op, c_inv, v_dev, l_loss, curt_pct, _ = self.opf_solver.solve_multi_period_dispatch(
                    T, demand_h, p_long_h, buses, capacities
                )
                w = self.aoa_cfg.weights
                return float(w[0] * c_op + w[1] * c_inv + w[2] * v_dev * 100.0 + w[3] * l_loss * 100.0)

            best_X, best_fitness, conv_curve, pop_fitness_dist = self.aoa_solver.optimize(multi_objective_fitness)
            self.cache.save(cache_key_aoa, (best_X, best_fitness, conv_curve, pop_fitness_dist))

        opt_buses = best_X[: num_units].astype(int)
        opt_capacities = best_X[num_units :]
        opt_power_ratings = np.clip(opt_capacities * 0.25, self.bess_cfg.power_min_mw, self.bess_cfg.power_max_mw)

        self.results_gen.print_and_export_table4(opt_buses, opt_capacities, opt_power_ratings)
        self.results_gen.plot_fig6_aoa_convergence(conv_curve)

        # Comparative OPF Evaluations
        c_op_24, c_inv_24, v_dev_24, l_loss_24, curt_24, commit_24 = self.opf_solver.solve_multi_period_dispatch(
            24, demand_h[:24], p_long_h[:24], opt_buses, opt_capacities
        )
        c_op_24_wo, _, v_dev_24_wo, l_loss_24_wo, curt_24_wo, _ = self.opf_solver.solve_multi_period_dispatch(
            24, demand_h[:24], p_long_h[:24], np.array([]), np.array([])
        )

        m24_with = {"C_op": c_op_24, "C_inv": c_inv_24, "V_dev": v_dev_24, "L_loss": l_loss_24, "Curtailment": curt_24}
        m24_wo = {"C_op": c_op_24_wo, "C_inv": 0.0, "V_dev": v_dev_24_wo, "L_loss": l_loss_24_wo, "Curtailment": curt_24_wo}

        demand_48 = df_clean["load_demand"].to_numpy()[:48] if len(df_clean) >= 48 else np.pad(df_clean["load_demand"].to_numpy(), (0, 48 - len(df_clean)), mode='edge')
        p_long_48 = pred_long[:48] if len(pred_long) >= 48 else np.pad(pred_long, (0, 48 - len(pred_long)), mode='edge')

        c_op_48, c_inv_48, v_dev_48, l_loss_48, curt_48, _ = self.opf_solver.solve_multi_period_dispatch(
            48, demand_48, p_long_48, opt_buses, opt_capacities
        )
        c_op_48_wo, _, v_dev_48_wo, l_loss_48_wo, curt_48_wo, _ = self.opf_solver.solve_multi_period_dispatch(
            48, demand_48, p_long_48, np.array([]), np.array([])
        )

        m48_with = {"C_op": c_op_48, "C_inv": c_inv_48, "V_dev": v_dev_48, "L_loss": l_loss_48, "Curtailment": curt_48}
        m48_wo = {"C_op": c_op_48_wo, "C_inv": 0.0, "V_dev": v_dev_48_wo, "L_loss": l_loss_48_wo, "Curtailment": curt_48_wo}

        self.results_gen.print_and_export_table5(m24_with, m24_wo)
        self.results_gen.plot_fig7_performance_comparison(m24_with, m24_wo, m48_with, m48_wo)
        self.results_gen.plot_fig8_generator_commitment(commit_24)

        # Step 5: Lower-Level CVaR & Sensitivity Analysis Progress Bar
        with tqdm(total=100, desc="[Step 5/5] Lower-Level CVaR Risk Optimization & Sensitivity", unit="%") as pbar5:
            cache_key_cvar = f"step5_cvar_results_{T}"
            if self.cache.exists(cache_key_cvar):
                cvar_cost, exp_cost, zeta, cvar_sensitivity = self.cache.load(cache_key_cvar)
            else:
                error_scenarios = self.cvar_optimizer.sample_forecast_error_scenarios(float(np.mean(p_short_h)))
                cvar_cost, exp_cost, zeta = self.cvar_optimizer.optimize_cvar_risk(c_op_24, error_scenarios)
                cvar_sensitivity = self.cvar_optimizer.run_alpha_sensitivity_analysis(c_op_24, error_scenarios)
                self.cache.save(cache_key_cvar, (cvar_cost, exp_cost, zeta, cvar_sensitivity))
            pbar5.update(100)

        self.results_gen.print_and_export_table6(cvar_sensitivity)
        self.results_gen.plot_fig9_expected_vs_cvar(cvar_sensitivity)
        self.results_gen.plot_fig10_objective_distribution(pop_fitness_dist)

        return {
            "forecast_metrics": metrics,
            "optimal_bess_buses": opt_buses,
            "optimal_bess_capacities_mwh": opt_capacities,
            "operating_cost": c_op_24,
            "expected_cost": exp_cost,
            "cvar_cost": cvar_cost,
            "var_threshold_zeta": zeta,
            "convergence_history": conv_curve
        }