import numpy as np
import torch
import gpytorch
from helpers.kernel_Helper import KernelHelper
from helpers.helpers import Helper

class GP_Helper():

    def get_GP_posterior(vi, inducing_points_s):
        time_grid = vi.time_grid
        sub_time_grid = vi.sub_time_grid
        time_discretization = vi.time_discretization
        mu_s_0 = vi.SGP_prior_mean
        mu_0 = vi.GP_prior_mean_D
        mu_0_extended = vi.GP_prior_mean_extended
        E_omega_complete = vi.E_omega_complete
        E_omega_N = vi.E_omega_N
        marked_rate = vi.marked_process_intensity_t

        v_plus = vi.kernel_helper.get_linear_term_v(sub_time_grid, inducing_points_s, E_omega_N,  mu_s_0, mu_0, marked_rate, v_plus=True)
        v_minus = vi.kernel_helper.get_linear_term_v(time_grid, inducing_points_s, E_omega_complete,  mu_s_0, mu_0_extended, marked_rate, v_plus=False)
        v_minus /= time_discretization
        v_s = v_plus + v_minus

        sigma_s = vi.kernel_helper.get_sigma_s(sub_time_grid, inducing_points_s, E_omega_N)
        sigma_s_complete = vi.kernel_helper.get_sigma_s_complete(time_grid, inducing_points_s, E_omega_complete, marked_rate)
        sigma_s_complete /= time_discretization
        bar_sigma_s = sigma_s + sigma_s_complete

        K_ss, inv_K_ss = vi.kernel_helper.get_inv_K_ss(inducing_points_s)
        cov_post = torch.inverse(inv_K_ss + bar_sigma_s)
        Helper.test_inverse(inv_K_ss + bar_sigma_s, cov_post)

        temp = torch.matmul(inv_K_ss, mu_s_0)

        mu_post = v_s + temp
        mu_post = torch.matmul(mu_post, cov_post)
        return mu_post, cov_post    
        
    ### old ###

    '''
    def get_Sigma_s(E_omega_N):
        # E_omega_N has dimension ( 1 x T )
        Sigma_s = torch.matmul(E_omega_N, torch.matmul(KernelHelper.get_kappa_forward(), KernelHelper.get_kappa_backward()))
        # Sigma_s has dimension ( T x s )
        return Sigma_s
    
    def get_v_s_plus(time_grid, E_omega_N, GP_prior_mean, SGP_prior_mean):
        # v_s_plus has dimension ( T x s )
        kappa_forward = KernelHelper.get_kappa_forward()
        kappa_backward = KernelHelper.get_kappa_backward()
        # now is dim ( T x T )
        v_s_plus = torch.full((time_grid.shape[0], time_grid.shape[0]), 0.5) - torch.matmul(E_omega_N , (GP_prior_mean - torch.matmul(SGP_prior_mean, kappa_backward)))
        # now is dim ( T x s )
        v_s_plus = torch.matmul(v_s_plus, kappa_forward)
        return v_s_plus
    
    def v_s_minus(time_grid, E_omega_N, GP_prior_mean, SGP_prior_mean):
        # v_s_plus has dimension ( T x s )
        kappa_forward = KernelHelper.get_kappa_forward()
        kappa_backward = KernelHelper.get_kappa_backward()
        # now is dim ( T x T )
        v_s_minus = torch.full((time_grid.shape[0], time_grid.shape[0]), -0.5) - torch.matmul(E_omega_N.t() , (GP_prior_mean - torch.matmul(SGP_prior_mean, kappa_backward)))
        # now is dim ( T x s )
        v_s_minus = torch.matmul(v_s_minus, kappa_forward)
        return v_s_minus
    
    def get_bar_Sigma_s(Sigma_s, marked_process_intensity_t):
        bar_Sigma_s = (torch.sum(Sigma_s, dim=0))  #  + int(Sigma_s * marked_process_intensity_t) 
        print("Hello, Im from get_bar_Sigma_s, and I want to say that the integral over the marked process intensity is not implemented yet.")

    def get_bar_v_s(v_s_plus, v_s_minus, marked_process_intensity_t):
        bar_v_s = (torch.sum(v_s_plus, dim=0))   #  + int(v_s_minus * marked_process_intensity_t)  ^T
        print("Hello, Im from get_bar_v_s, and I want to say that the integral over the marked process intensity is not implemented yet.")
        return bar_v_s
    
    def get_Sigma_posterior(bar_Sigma_s, SGP_prior_cov):
        Sigma_posterior = torch.inverse(torch.inverse(SGP_prior_cov) + bar_Sigma_s)
        return Sigma_posterior
    
    def get_mean_posterior(bar_v_s, Sigma_posterior, SGP_prior_cov, SGP_prior_mean):
        mean_posterior = torch.matmul(Sigma_posterior, torch.matmul(torch.inverse(SGP_prior_cov), SGP_prior_mean) + bar_v_s)
        return mean_posterior
''' 
'''
    def __init__(self, ts_Helper, KernelHelper, gp_prior_mean, sgp_prior_mean, gp_prior_cov, sgp_prior_cov):
        self.KernelHelper = KernelHelper
        self.ts_Helper = ts_Helper
        self.GP_prior_mean = gp_prior_mean
        self.GP_prior_cov = gp_prior_cov
        self.SGP_prior_mean = sgp_prior_mean
        self.SGP_prior_cov = sgp_prior_cov
'''