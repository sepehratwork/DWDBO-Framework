"""
IEEE 30-Bus System Data Definitions.
Provides generator parameters, line branch data, and base load profiles.
"""

import numpy as np


class IEEE30BusSystem:
    """Contains technical specification data for the IEEE 30-bus transmission test network."""

    def __init__(self):
        self.num_buses = 30
        self.num_generators = 6
        
        # Generator buses: [Bus 1, Bus 2, Bus 5, Bus 8, Bus 11, Bus 13]
        self.gen_buses = np.array([1, 2, 5, 8, 11, 13])

        # Quadratic cost coefficients: a_g ($/MW^2), b_g ($/MW), c_g ($)
        self.cost_a = np.array([0.00375, 0.01750, 0.06250, 0.00834, 0.02500, 0.02500])
        self.cost_b = np.array([2.00, 1.75, 1.00, 3.25, 3.00, 3.00])
        self.cost_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Generator capacity bounds (MW)
        self.p_min = np.array([50.0, 20.0, 15.0, 10.0, 10.0, 12.0])
        self.p_max = np.array([200.0, 80.0, 50.0, 35.0, 30.0, 40.0])

        # Ramp rate limits (MW/hour)
        self.ramp_limits = np.array([40.0, 20.0, 15.0, 10.0, 10.0, 10.0])

        # Nominal system load (MW)
        self.base_demand = 189.2