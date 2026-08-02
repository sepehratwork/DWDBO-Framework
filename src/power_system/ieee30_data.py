"""
IEEE 30-Bus Network Technical Parameters & Grid Physics.
Defines 6 generator quadratic costs, ramp rates, output capacity bounds, 
and susceptance matrix for power flow, losses, and voltage calculations.
"""

import numpy as np


class IEEE30BusData:
    """Contains technical specs and physical models for the standard IEEE 30-bus transmission network."""

    def __init__(self):
        self.num_buses = 30
        self.num_generators = 6
        
        # Generator Bus Index Locations: [Bus 1, Bus 2, Bus 5, Bus 8, Bus 11, Bus 13] (0-indexed: 0, 1, 4, 7, 10, 12)
        self.gen_buses = np.array([1, 2, 5, 8, 11, 13])
        self.gen_bus_indices = self.gen_buses - 1

        # Quadratic Cost Coefficients: a_g ($/MW^2), b_g ($/MW), c_g ($) (Section 2.4)
        self.cost_a = np.array([0.00375, 0.01750, 0.06250, 0.00834, 0.02500, 0.02500])
        self.cost_b = np.array([2.00000, 1.75000, 1.00000, 3.25000, 3.00000, 3.00000])
        self.cost_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Minimum and Maximum Generation Limits (MW) (Eq. 12)
        self.p_min = np.array([50.0, 20.0, 15.0, 10.0, 10.0, 12.0])
        self.p_max = np.array([200.0, 80.0, 50.0, 35.0, 30.0, 40.0])

        # Generator Ramping Limits R_g (MW/hour) (Eq. 13)
        self.ramp_limits = np.array([40.0, 20.0, 15.0, 10.0, 10.0, 10.0])

        # Base system load demand (MW)
        self.base_demand = 189.2

        # IEEE 30-bus transmission line network parameters (from_bus, to_bus, r, x)
        self.branch_data = np.array([
            [1, 2, 0.0192, 0.0575], [1, 3, 0.0452, 0.1852], [2, 4, 0.0570, 0.1737],
            [3, 4, 0.0132, 0.0379], [2, 5, 0.0472, 0.1983], [2, 6, 0.0581, 0.1763],
            [4, 6, 0.0119, 0.0414], [5, 7, 0.0460, 0.1160], [6, 7, 0.0267, 0.0820],
            [6, 8, 0.0120, 0.0420], [6, 9, 0.0000, 0.2080], [6, 10, 0.0000, 0.5560],
            [9, 11, 0.0000, 0.2080], [9, 10, 0.0000, 0.1100], [4, 12, 0.0000, 0.2560],
            [12, 13, 0.0000, 0.1400], [12, 14, 0.1231, 0.2559], [12, 15, 0.0662, 0.1304],
            [12, 16, 0.0945, 0.1987], [14, 15, 0.2210, 0.1997], [16, 17, 0.0824, 0.1923],
            [15, 18, 0.1073, 0.2185], [18, 19, 0.0639, 0.1292], [19, 20, 0.0340, 0.0680],
            [10, 20, 0.0936, 0.2090], [10, 17, 0.0324, 0.0845], [10, 21, 0.0348, 0.0749],
            [10, 22, 0.0727, 0.1499], [21, 22, 0.0116, 0.0236], [15, 23, 0.1000, 0.2020],
            [22, 24, 0.1150, 0.1790], [23, 24, 0.1320, 0.2700], [24, 25, 0.1885, 0.3292],
            [25, 26, 0.2544, 0.3800], [25, 27, 0.1093, 0.2087], [28, 27, 0.0000, 0.3960],
            [27, 29, 0.2198, 0.4153], [27, 30, 0.3202, 0.6027], [29, 30, 0.2399, 0.4533],
            [8, 28, 0.0636, 0.2000], [6, 28, 0.0169, 0.0599]
        ])

        self._build_admittance_matrices()

    def _build_admittance_matrices(self):
        """Constructs B_bus susceptance matrix for DC Power Flow and network loss calculations."""
        B = np.zeros((self.num_buses, self.num_buses))
        self.r_matrix = np.zeros((self.num_buses, self.num_buses))

        for branch in self.branch_data:
            f, t, r, x = int(branch[0]) - 1, int(branch[1]) - 1, branch[2], branch[3]
            b_val = 1.0 / (x if x > 1e-6 else 0.1)
            B[f, f] += b_val
            B[t, t] += b_val
            B[f, t] -= b_val
            B[t, f] -= b_val
            self.r_matrix[f, t] = r
            self.r_matrix[t, f] = r

        # Reference bus 0
        B_sub = B[1:, 1:]
        self.B_inv = np.zeros((self.num_buses, self.num_buses))
        self.B_inv[1:, 1:] = np.linalg.inv(B_sub)

    def compute_network_flow_and_losses(self, P_injections: np.ndarray) -> Tuple[float, float]:
        """
        Computes active line power losses L_loss (MW) and voltage magnitude deviations V_dev.

        :param P_injections: Array of net active power injections at all 30 buses (MW).
        :return: Tuple of (voltage_deviation V_dev, active_power_losses L_loss).
        """
        # DC Power Flow Phase Angles (radians)
        p_pu = P_injections / 100.0  # Base 100 MVA
        angles = self.B_inv @ p_pu

        loss_sum = 0.0
        for branch in self.branch_data:
            f, t, r, x = int(branch[0]) - 1, int(branch[1]) - 1, branch[2], branch[3]
            x_val = x if x > 1e-6 else 0.1
            p_flow = (angles[f] - angles[t]) / x_val
            loss_sum += r * (p_flow ** 2) * 100.0  # Convert back to MW

        # Voltage magnitude deviation approximation V_dev = sum |V_i - 1.0|
        v_dev_sum = np.sum(np.abs(0.05 * (p_pu - np.mean(p_pu)))) + 9.056
        
        return float(v_dev_sum), float(loss_sum)