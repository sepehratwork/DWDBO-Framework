"""
IEEE 30-Bus Network Technical Parameters.
Defines 6 generator quadratic costs, ramp rates, output capacity bounds, 
and transmission bus data.
"""

import numpy as np


class IEEE30BusData:
    """Contains technical specs and constraints for the standard IEEE 30-bus transmission network."""

    def __init__(self):
        self.num_buses = 30
        self.num_generators = 6
        
        # Generator Bus Index Locations: [Bus 1, Bus 2, Bus 5, Bus 8, Bus 11, Bus 13]
        self.gen_buses = np.array([1, 2, 5, 8, 11, 13])

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