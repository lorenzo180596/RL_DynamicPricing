"""

This module contain all the functionality to compute the simulated demand of customer in the selected period.

Classes
--------------------
In that file the following classes are defined:

1. Demand
   - compute the number of customer arrival for each day of the simulation

"""

import numpy as np

class Demand():
    def __init__(self, time_horizon, total_customers, method):
        self.time_horizon = time_horizon
        self.total_customers = total_customers
        self.method = method

    def compute_customer_arrivals(self):
        if self.method == 'poisson':
            self.daily_arrivals = self._get_poisson_distribution()

    def _get_poisson_distribution(self):
        arrivals_rate_mean = self.total_customers / self.time_horizon
        daily_arrivals = np.random.poisson(arrivals_rate_mean, self.time_horizon)
        return daily_arrivals 
