import time
import numpy as np
import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
import gpytorch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from helpers.kernel_Helper import KernelHelper
from helpers.mp_Helper import MP_Helper
from helpers.gp_Helper import GP_Helper
from helpers.pg_Helper import pg_Helper
from helpers.helpers import Helper
from helpers.helpers import no_grad_method
from ELBO.kl_Helper import Kullback_Leibler
from time_series.ts_Helper import TS_Helper
from ELBO.ELBO import ELBO
from ELBO.ELBO import opt_ELBO
from helpers.plotter import plot_results, plot_rates, plot_just_post_rate, plot_time_grid, plot_post_rate_minimal, plot_surface, plot_posterior_GP

class VariationalInference(nn.Module):

    def __init__(self, optimality_parameters, hyperparameters, time_grid_parameters, ts_helper, full_thinned_process, 
                 num_inducing_points_per_dim, learning_rate, kernel_name):
        
        super(VariationalInference, self).__init__()

        # Initialize time seris helper
        self.ts_helper = ts_helper

        # Initialize the time axis and phase space
        self.end_time = self.ts_helper.end_time
        self.time_discretization = self.ts_helper.time_discretization
        self.n_timesteps = int(self.ts_helper.end_time * self.ts_helper.time_discretization)
        self.dim_phase_space = self.ts_helper.dim_phase_space
        self.phase_space_scale = 0.2
        self.num_processes = self.ts_helper.num_processes
        if isinstance(time_grid_parameters['tau_list'], torch.Tensor):
            self.tau_list = time_grid_parameters['tau_list']
        else:
            self.tau_list = torch.tensor(time_grid_parameters['tau_list'], dtype=torch.float64)
        self.couplings = time_grid_parameters['couplings']

        # Initialize time grid form ts helper
        self.time_grid = self.ts_helper.get_time_grid(self.couplings, self.tau_list)
        self.full_thinned_process = full_thinned_process #one Process
        self.thinned_process = full_thinned_process[:self.end_time * self.time_discretization]
        self.test_thinned_process = full_thinned_process[self.end_time * self.time_discretization:]
        self.cardinality_D = torch.sum(self.thinned_process)
        self.thinned_indices = Helper.get_indices_1d(self.thinned_process)
        self.thinned_shifted_indices = Helper.shift_indices(self.thinned_indices)
        self.full_thinned_indices = Helper.get_indices_1d(self.full_thinned_process)
        self.test_thinned_indices = Helper.get_indices_1d(self.test_thinned_process)
        self.sub_time_grid = torch.index_select(self.time_grid, 0, index=self.thinned_shifted_indices)
        #self.sub_time_grid = self.ts_helper.get_sub_time_grid_one_process(self.thinned_shifted_indices, self.couplings, self.tau_list)

        # Initialize optimizer
        self.optimizer = None
        self.learning_rate = learning_rate

        # Initialize inducing points
        self.num_inducing_points_per_dim = num_inducing_points_per_dim
        self.num_ind_points = self.num_inducing_points_per_dim ** self.dim_phase_space
        self.num_k_means_inducting_points = self.num_inducing_points_per_dim ** self.dim_phase_space
        if hyperparameters['inducing_points_s'] is None:
            self.inducing_points_s = Helper.get_meshgrid_scaled(self.num_inducing_points_per_dim, self.time_grid, self.dim_phase_space)
            self.SGP_prior_mean = hyperparameters['GP_prior_mean'] * torch.ones(self.num_inducing_points_per_dim ** self.dim_phase_space, dtype=torch.float64)
            #self.inducing_points_s = Helper.create_inducing_points_k_means(self.time_grid, self.num_k_means_inducting_points)
            #self.SGP_prior_mean = hyperparameters['GP_prior_mean'] * torch.ones(self.num_inducing_points_per_dim ** self.dim_phase_space, dtype=torch.float64)
        else:
            self.inducing_points_s = hyperparameters['inducing_points_s'] 

        # Initialize the kernel parameters
        self.kernel = kernel_name
        self.kernel_outputscale = hyperparameters['kernel_outputscale']
        self.kernel_lengthscale = hyperparameters['kernel_lengthscale']
        self.kernel_period_length = None
        self.kernel_params = [self.kernel_lengthscale, self.kernel_outputscale]

        # Initialize the KernelHelper
        self.kernel_helper = KernelHelper(self.kernel, self.kernel_params)

        # Initialize q2(GP)
        self.GP_prior_mean = hyperparameters['GP_prior_mean']
        self.GP_prior_mean_D = self.GP_prior_mean * torch.ones(self.sub_time_grid.shape[0], dtype=torch.float64)
        self.GP_prior_mean_extended = self.GP_prior_mean * torch.ones(self.time_grid.shape[0], dtype=torch.float64)
        self.GP_prior_cov = KernelHelper.get_kernel_matrix(self.kernel_helper, self.sub_time_grid, self.sub_time_grid)
        self.SGP_prior_cov ,self.inv_Kss = KernelHelper.get_inv_K_ss(self.kernel_helper,self.inducing_points_s)
        if optimality_parameters['mu_post'] is None:
            self.SGP_post_mean = self.SGP_prior_mean
            self.SGP_post_cov = self.SGP_prior_cov
        else:
            self.SGP_post_mean = optimality_parameters['mu_post']
            self.SGP_post_cov = optimality_parameters['cov_post']

        # Initialize q2(lmbda)
        self.alpha_0 = hyperparameters['alpha_0']
        self.beta_0 = hyperparameters['beta_0']
        self.alpha_post = optimality_parameters['alpha_post']
        self.beta_post = optimality_parameters['beta_post']
        if self.alpha_post is None:
            self.alpha_post = self.alpha_0
            self.beta_post = self.beta_0       
            self.E_lmbda = self.alpha_0 / self.beta_0
            self.E_ln_lmbda = torch.digamma(self.alpha_0) - torch.log(self.beta_0)
        else:
            self.E_lmbda = self.alpha_post / self.beta_post
            self.E_ln_lmbda = torch.digamma(self.alpha_post) - torch.log(self.beta_post)

        # Initialize q1(PG)
        self.c_complete = None
        self.c_complete_squared = None
        self.c_n = None
        self.c_n_squared = None
        self.E_omega_N = None
        self.E_omega_complete = None

        # Initialize q1(MP)
        self.marked_process_intensity_t = None
        self.E_cardinality_MP = None

        # Initialize posterior rate
        self.posterior_rate = None
        self.test_posterior_rate = None
        self.E_f_full_domain = None
        self.E_f_full_domain_on_sub_grid = None

        # for dynamical system
        self.posterior_rate_dynamical_system = None
        self.unstacked_post_rate = None

        # plotting couter
        self.loss_tracker = []
        self.h_time_tracker = 0
        self.opt_time_tracker = 0
        self.oss_tracker_all = None
        self.plot_counter = 0
        self.num_iterations = 0

    @no_grad_method
    def cal_posterior_lmbda(self):
        self.alpha_post = self.alpha_0 + self.cardinality_D + self.E_cardinality_MP
        self.beta_post = self.beta_0 + self.end_time
        self.E_lmbda = self.alpha_post / self.beta_post
        self.E_ln_lmbda = torch.digamma(self.alpha_post) - torch.log(self.beta_post)
        return self.alpha_post, self.beta_post

    @no_grad_method
    def cal_posterior_GP(self):
        mu_post, cov_post = GP_Helper.get_GP_posterior(self, self.inducing_points_s)
        self.SGP_post_cov = cov_post
        self.SGP_post_mean = mu_post
        return mu_post, cov_post

    @no_grad_method
    def cal_posterior_PG(self):
        self.c_complete, self.c_complete_squared = pg_Helper.get_c_complete_posterior(self, self.time_grid)
        self.c_n, self.c_n_squared = pg_Helper.get_posterior_c_n(self.c_complete, self.c_complete_squared, self.thinned_process, self.thinned_shifted_indices)
        self.E_omega_N = pg_Helper.get_E_omega(self.c_n)
        self.E_omega_complete = pg_Helper.get_E_omega(self.c_complete)
        #return self.c_n, self.c_complete, self.E_omega_N, self.E_omega_complete

    @no_grad_method
    def cal_posteriro_marked_rate(self):
        self.marked_process_intensity_t = MP_Helper.get_posterior_marked_rate(self, self.time_grid, self.c_complete)
        self.E_cardinality_MP = MP_Helper.integrage_marked_intensity(self.end_time, self.time_discretization, self.marked_process_intensity_t)
        #print("E_cardinality_MP:", self.E_cardinality_MP.item())

    @no_grad_method
    def cal_q1(self):
        self.cal_posterior_PG()
        self.cal_posteriro_marked_rate()

    @no_grad_method
    def cal_q2(self):
        self.cal_posterior_lmbda()
        self.cal_posterior_GP()

        optimality_parameters = {
            "alpha_post": self.alpha_post,
            "beta_post": self.beta_post,
            "mu_post": self.SGP_post_mean,
            "cov_post": self.SGP_post_cov,
        }    
        return optimality_parameters

    def run_optimization_single_process(self, hyperparameter_optimization, termination_threshold_opt, iterations_per_process=2, epochs_per_hyp_tuning=3):
        previous_L = None
        tolerance_counter = 0
        tolerance_limit = 6
        self.opt_time_tracker = 0
        for i in range(iterations_per_process):
            with torch.no_grad():
                #print("### Iteration: ", i+1, " ###")
                start_time_opt = time.perf_counter()
                self.cal_q1()
                opt_params_not_used = self.cal_q2()
                end_time_opt = time.perf_counter()
                L, L_E_U_s, kl_lmbda, kl_omega_N, kl_marked_process, kl_gp, z_plus, z_minus, E_ln_lmbda = self.cal_lower_evidence_bound()
                self.loss_tracker.append(L.item())
                #print("L:", L.item())
                self.opt_time_tracker += end_time_opt - start_time_opt
                print(f"   ## iteration: {i+1}, L: {L.item():.2f}, time for iteration: {end_time_opt - start_time_opt:.2f}s ##")

                if previous_L is not None:
                    loss_difference = abs(previous_L - L)
                    if loss_difference < termination_threshold_opt:
                        tolerance_counter += 1
                        if tolerance_counter >= tolerance_limit:
                            print(f"   !! terminating at iteration {i+1} due to small loss difference. !!")
                            full_stack_params = Helper.collect_full_stack_params(self)
                            return self.loss_tracker, full_stack_params, self.h_time_tracker, self.opt_time_tracker
                    else:
                        tolerance_counter = 0
                previous_L = L

            if i+1 in hyperparameter_optimization:
                self.h_time_tracker = 0
                start_time_hyp = time.perf_counter()
                print(f"\n   ######  hyperparameter optimization ({epochs_per_hyp_tuning} Epochs)  ######")
                hyperparameters = Helper.collect_hyperparamters(self)
                #print("@@ old hyperparameters: @@")
                #Helper.print_parameters(hyperparameters)
                #self.optimizer = opt_ELBO(self.learning_rate, hyperparameters)
                self.optimizer = opt_ELBO(self.learning_rate, hyperparameters)
                new_hyperparameters, post_mean, post_cov = self.optimizer.optimize(self, epochs_per_hyp_tuning, hyperparameters)
                Helper.set_hyperparameters_ELBO(self, new_hyperparameters)

                # update the vi object 
                self.kernel_params = [self.kernel_lengthscale, self.kernel_outputscale]
                self.kernel_helper = KernelHelper(self.kernel, self.kernel_params)  
                self.GP_prior_mean_D = self.GP_prior_mean * torch.ones(self.sub_time_grid.shape[0], dtype=torch.float64)
                self.GP_prior_mean_extended = self.GP_prior_mean * torch.ones(self.time_grid.shape[0], dtype=torch.float64)
                self.GP_prior_cov = KernelHelper.get_kernel_matrix(self.kernel_helper, self.sub_time_grid, self.sub_time_grid)
                self.SGP_prior_mean = self.GP_prior_mean * torch.ones(self.num_inducing_points_per_dim ** self.dim_phase_space, dtype=torch.float64)
                self.SGP_prior_cov, self.inv_Kss = KernelHelper.get_inv_K_ss(self.kernel_helper, self.inducing_points_s)      
                self.SGP_post_cov = post_cov
                self.SGP_post_mean = post_mean

                L, L_E_U_s, kl_lmbda, kl_omega_N, kl_marked_process, kl_gp, z_plus, z_minus, E_ln_lmbda = self.cal_lower_evidence_bound()
                self.loss_tracker.append(L.item())
                end_time_hyp = time.perf_counter()
                self.h_time_tracker += end_time_hyp - start_time_hyp
                print(f"   ## L after hyper opt: {L.item():.2f}, time for hyper opt: {end_time_hyp - start_time_hyp:.2f}s ##\n")
                #print("\n")
                #print("###  hyperparameter optimization finished  ###\n")
                #print("@@ new hyperparameters: @@")
                #Helper.print_parameters(new_hyperparameters)
                #print("\n")
        full_stack_params = Helper.collect_full_stack_params(self)
        return self.loss_tracker, full_stack_params, self.h_time_tracker, self.opt_time_tracker

    @no_grad_method
    def cal_predictive_rate(self):
        self.time_grid = self.ts_helper.get_time_grid(self.couplings, self.tau_list)
        self.E_f_full_domain = self.kernel_helper.get_E_f_full_domain(self.time_grid,  self.inducing_points_s, self.GP_prior_mean, self.SGP_prior_mean, self.SGP_post_mean)
        self.posterior_rate = self.E_lmbda * torch.sigmoid(self.E_f_full_domain)
        self.E_f_full_domain_on_sub_grid = self.kernel_helper.get_E_f_full_domain(self.sub_time_grid,  self.inducing_points_s, self.GP_prior_mean, self.SGP_prior_mean, self.SGP_post_mean)

    @no_grad_method
    def cal_test_predictive(self, test_time_gird):
        E_test_posterior_rate = self.kernel_helper.get_E_f_full_domain(test_time_gird, self.inducing_points_s, self.GP_prior_mean, self.SGP_prior_mean, self.SGP_post_mean)
        self.test_posterior_rate = self.E_lmbda * torch.sigmoid(E_test_posterior_rate)
        return self.test_posterior_rate
    
    @no_grad_method
    def cal_predicted_time_grid_rate(self, predicted_time_grid):
        predicted_time_grid = torch.tensor(predicted_time_grid, dtype=torch.float64)
        E_predicted_time_grid = self.kernel_helper.get_E_f_full_domain(predicted_time_grid, self.inducing_points_s, self.GP_prior_mean, self.SGP_prior_mean, self.SGP_post_mean)
        post_rate = self.E_lmbda * torch.sigmoid(E_predicted_time_grid)
        post_rate = post_rate[1:]
        return post_rate

    @no_grad_method
    def cal_predicitve_rate_for_dynamical_system(self, mesh):
        ds_mesh = torch.stack([torch.tensor(m).flatten() for m in mesh], dim=1)
        ds_mesh = torch.tensor(ds_mesh, dtype=torch.float64)
        E_dynamical_system = self.kernel_helper.get_E_f_full_domain(ds_mesh, self.inducing_points_s, self.GP_prior_mean, self.SGP_prior_mean, self.SGP_post_mean)
        self.posterior_rate_dynamical_system = self.E_lmbda * torch.sigmoid(E_dynamical_system)

    def eval_post_gp(self, x1, x2, x3=None):
        if x3==None:
            point = torch.tensor([[x1, x2]], dtype=torch.float64)
            E_at_x1_x2 = self.kernel_helper.get_E_f_full_domain(point, self.inducing_points_s, self.GP_prior_mean, self.SGP_prior_mean, self.SGP_post_mean)
            sig_E_post_rate = torch.sigmoid(E_at_x1_x2)
            post_gp_rate = self.E_lmbda * sig_E_post_rate
            return post_gp_rate, sig_E_post_rate
        else:
            point = torch.tensor([[x1, x2, x3]], dtype=torch.float64)
            E_at_x1_x2 = self.kernel_helper.get_E_f_full_domain(point, self.inducing_points_s, self.GP_prior_mean, self.SGP_prior_mean, self.SGP_post_mean)
            sig_E_post_rate = torch.sigmoid(E_at_x1_x2)
            post_gp_rate = self.E_lmbda * sig_E_post_rate
            return post_gp_rate, sig_E_post_rate
    
    '''def optimize_ELBO(self, steps=2):
        hyperparameters = Helper.collect_hyperparamters(self)
        #print("@@ old hyperparameters: @@")
        #Helper.print_parameters(hyperparameters)
        #self.optimizer = opt_ELBO(self.learning_rate, hyperparameters)
        self.optimizer = opt_ELBO(self.learning_rate, hyperparameters)
        new_hyperparameters, updated_mean, updated_cov = self.optimizer.optimize(self, steps, hyperparameters)
        return new_hyperparameters, updated_mean, updated_cov'''

    @no_grad_method
    def cal_lower_evidence_bound(self):
        L_E_U_s, z_plus, z_minus, sum_ln_lmbda = ELBO.E_U_s(self)
        kl_lmbda = Kullback_Leibler.kl_lmbda(self.alpha_0, self.beta_0, self.alpha_post, self.beta_post)
        kl_omega_N = Kullback_Leibler.kl_omega_N(self.time_discretization, self.c_n, self.c_n_squared, self.E_omega_N)
        kl_marked_process = Kullback_Leibler.kl_marked_process(self.end_time, self.time_discretization, self.marked_process_intensity_t, 
                                                               self.E_lmbda, self.E_ln_lmbda, self.c_complete, self.c_complete_squared, 
                                                               self.E_omega_complete)
        kl_gp = Kullback_Leibler.kl_gp(self.SGP_prior_mean, self.SGP_post_mean, self.SGP_prior_cov, self.SGP_post_cov, self.SGP_prior_cov, self.inv_Kss)
        
        '''print("\nsum_ln_lmbda:", sum_ln_lmbda)
        print("z_plus:", z_plus)
        print("z_minus:", z_minus)
        print("L_E_U_s:", L_E_U_s)
        print("kl_PG:", kl_omega_N)
        print("kl_lmbda:", kl_lmbda)
        print("kl_MP:", kl_marked_process)
        print("kl_GP:", kl_gp)
        print("L:", L_E_U_s - kl_lmbda - kl_omega_N - kl_marked_process - kl_gp, "\n")'''

        if kl_lmbda < 0 or kl_omega_N < 0 or kl_marked_process < 0 or kl_gp < -1e-8:
            raise ValueError("KL divergences must be positive")
        L =  L_E_U_s - kl_lmbda - kl_omega_N - kl_marked_process - kl_gp
        return L, L_E_U_s.item(), kl_lmbda.item(), kl_omega_N.item(), kl_marked_process.item(), kl_gp.item(), z_plus.item(), z_minus.item(), sum_ln_lmbda.item()

    @no_grad_method    
    def to_device(self, device):
        self.inducing_points_s = self.inducing_points_s.to(device)
        self.thinned_shifted_indices = self.thinned_shifted_indices.to(device)
        self.SGP_post_mean = self.SGP_post_mean.to(device)
        self.SGP_post_cov = self.SGP_post_cov.to(device)
        self.kernel_outputscale = self.kernel_outputscale.to(device)
        self.kernel_lengthscale = self.kernel_lengthscale.to(device)
        self.E_omega_N = self.E_omega_N.to(device)
        self.E_omega_complete = self.E_omega_complete.to(device)
        self.marked_process_intensity_t = self.marked_process_intensity_t.to(device)

    @no_grad_method   
    def plot_results(self, gp_sample, c_map='bone', start=200, xlim=6000):
        plot_results(self, gp_sample, self.posterior_rate, self.E_f_full_domain_on_sub_grid, self.E_f_full_domain, self.couplings, self.tau_list, c_map, start, xlim)
        #plot_rates(self, gp_sample, self.posterior_rate, start, xlim=1000)

    @no_grad_method   
    def plot_surface(self, i, mesh, colormap, grid_padding, start=200, xlim=3000, elev=30, azim=45):
        plot_posterior_GP(self, i, mesh, self.posterior_rate_dynamical_system, colormap, grid_padding, start, xlim)
        plot_surface(self, i, mesh, self.posterior_rate_dynamical_system, colormap, grid_padding, start, xlim, elev, azim)

    @no_grad_method
    def plot_minimal(self, start=0, xlim=2500):
        self.cal_predictive_rate()
        #plot_just_post_rate(gp_sample, self.posterior_rate, start, xlim, plot_counter)
        plot_post_rate_minimal(self, start, xlim)

    @no_grad_method
    def plot_time_grid(self, size=3.5, start=0, end=2000):
        plot_time_grid(self.ts_helper, self.time_grid, self.sub_time_grid, size, start, end)

    @no_grad_method
    def plot_rates(self, i, start=0, end=4000, show_points=False):
        true_rate = self.ts_helper.true_rates[i]
        plot_rates(self, true_rate, start, end, show_points)
        
    '''def optimize_time_grid(self, epochs_time_grid=3, learning_rate_time_grid=0.1):
        print("\n##### time grid optimization #####")
        start_time_opt_time_grid = time.perf_counter()
        print("@@ old tau and couplings: @@")
        print("tau_list:", self.tau_list.tolist())
        print("couplings:", self.couplings.tolist())
        print("\n")

        hyperparameters = Helper.collect_hyperparamters(self)
        self.time_grid_optimizer = Time_grid_optimization(hyperparameters, learning_rate_time_grid)
        tau_list, couplings = self.time_grid_optimizer.optimize_time_grid(self, epochs_time_grid)

        print("\n@@ new tau and couplings: @@")
        print("tau_list:", tau_list.tolist())
        print("couplings:", couplings.tolist())
        
        # update the vi object 
        self.tau_list = tau_list
        self.couplings = couplings
        self.time_grid = self.ts_helper.get_time_grid(self.couplings, self.tau_list)
        self.sub_time_grid = self.ts_helper.get_sub_time_grid_one_process(self.thinned_shifted_indices, self.couplings, self.tau_list)    

        L, L_E_U_s, kl_lmbda, kl_omega_N, kl_marked_process, kl_gp, z_plus, z_minus, E_ln_lmbda = self.cal_lower_evidence_bound()
        self.loss_tracker.append(L.item())
        end_time_opt_time_grid = time.perf_counter()
        print(f"\nL after hyper opt: {L.item()}, time for hyper opt: {end_time_opt_time_grid-start_time_opt_time_grid}s \n")

        full_stack_params = Helper.collect_full_stack_params(self)
        return self.loss_tracker, full_stack_params'''