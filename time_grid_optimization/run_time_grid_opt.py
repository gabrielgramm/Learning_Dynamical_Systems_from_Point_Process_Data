import time
import numpy as np
import torch
from helpers.kernel_Helper import KernelHelper
from helpers.gp_Helper import GP_Helper
from helpers.helpers import Helper
from time_series.ts_Helper import TS_Helper
from ELBO.ELBO import ELBO
from time_grid_optimization.time_grid_optimization import Time_grid_optimization
from helpers.plotter import plot_results, plot_rates, plot_just_post_rate, plot_time_grid

class Global_Optimizer:

    def __init__(self, ts_helper, time_grid_parameters, global_full_stack_params=None):
        self.time_grid_parameters = time_grid_parameters
        if isinstance(time_grid_parameters['tau_list'], torch.Tensor):
            self.tau_list = time_grid_parameters['tau_list']
        else:
            self.tau_list = torch.tensor(time_grid_parameters['tau_list'], dtype=torch.float64)
        self.couplings = time_grid_parameters['couplings']
        self.ts_helper = ts_helper
        if global_full_stack_params is None:
            self.global_full_stack_params = None
        else:
            self.global_full_stack_params = global_full_stack_params

    def set_time_grid_params(self, ts_helper, time_grid_parameters, global_full_stack_params):
        self.time_grid_parameters = time_grid_parameters
        if isinstance(time_grid_parameters['tau_list'], torch.Tensor):
            self.tau_list = time_grid_parameters['tau_list']
        else:
            self.tau_list = torch.tensor(time_grid_parameters['tau_list'], dtype=torch.float64)
        self.couplings = time_grid_parameters['couplings']
        self.ts_helper = ts_helper
        self.global_full_stack_params = global_full_stack_params

    def optimize_time_grid(self, epochs_time_grid = 5, learning_rate_time_grid=0.1):
        print(f"\n############ time grid optimization ({epochs_time_grid} Epochs)  ############")
        start_time_opt_time_grid = time.perf_counter()
        print("   @@ old tau and couplings: @@")
        print("   tau_list:", [f"{x:.2f}" for x in self.tau_list])
        #print("   couplings:", np.around(self.couplings.numpy(), 2))
        print("   couplings:", np.array2string(np.around(self.couplings.numpy(), 2), separator=', '))
        print("\n")

        self.time_grid_optimizer = Time_grid_optimization(self.global_full_stack_params, self.ts_helper, self.couplings, self.tau_list, learning_rate_time_grid)
        new_tau_list, new_couplings = self.time_grid_optimizer.optimize_time_grid(epochs_time_grid)

        print("\n   @@ new tau and couplings: @@")
        print("   tau_list:", [f"{x:.2f}" for x in new_tau_list])
        #print("   couplings:", np.around(new_couplings.numpy(), 2))
        print("   couplings:", np.array2string(np.around(new_couplings.numpy(), 2), separator=', '))

        self.time_grid_parameters['tau_list'] = new_tau_list
        self.time_grid_parameters['couplings'] = new_couplings

        end_time_opt_time_grid = time.perf_counter()
        print(f"   time for time grid opt: {end_time_opt_time_grid-start_time_opt_time_grid:.2f}s\n")

        return self.time_grid_parameters