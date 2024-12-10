import numpy as np
import torch
from polyagamma import random_polyagamma
from helpers.helpers import Helper
from helpers.kernel_Helper import KernelHelper

class pg_Helper:

    def get_c_complete_posterior(vi, time_grid):
        #time_grid = vi.time_grid
        inducing_points_s = vi.inducing_points_s
        mu_0 = vi.GP_prior_mean_D
        mu_0_extended = vi.GP_prior_mean_extended
        mu_s_0 = vi.SGP_prior_mean
        E_fs = vi.SGP_post_mean
        cov_s = vi.SGP_post_cov

        quadratic_term = vi.kernel_helper.get_quadratic_term(time_grid, vi.inducing_points_s, vi.SGP_post_cov, vi.SGP_post_mean)
        linear_term_omega = vi.kernel_helper.get_linear_term_omega(time_grid, vi.inducing_points_s,   vi.SGP_post_mean, vi.SGP_prior_mean, vi.GP_prior_mean_extended)
        sig_t_given_fs = vi.kernel_helper.get_sigma_t_given_fs(time_grid, vi.inducing_points_s)
        bracket_term = vi.kernel_helper.get_bracket_term_of_const(time_grid, vi.inducing_points_s, vi.SGP_prior_mean, vi.GP_prior_mean_extended, sig_t_given_fs)
        constant_term = bracket_term
        second_momnet = quadratic_term + 2 * linear_term_omega + constant_term
        c_complete_squared = second_momnet
        c_complete = torch.sqrt(second_momnet)
        return c_complete, c_complete_squared

    def get_posterior_c_n(c_complete, c_complete_squared, data, thinned_shifted_indices):
        #sub_time_grid = vi.sub_time_grid
        #c_complete = vi.c_complete 
        #c_complete_squared = vi.c_complete_squared
        #thinned_shifted_indices = vi.thinned_shifted_indices

        c_n = c_complete[thinned_shifted_indices]                  #old
        c_n_squared = c_complete_squared[thinned_shifted_indices]  #old

        c_n_marked = c_complete * data
        c_n_marked_squared = c_complete_squared * data
        #return c_n, c_n_squared
        return c_n_marked, c_n_marked_squared

    def get_pg_1_0(n):
        return torch.tensor(random_polyagamma(size=n))
    
    def get_pg_1_c(n, c):
        return random_polyagamma(n, c)
    
    def get_E_omega(c_n):
        E_omega_N = torch.zeros(c_n.shape)
        for i in range(len(c_n)):
            if c_n[i] == 0:
                continue
            else:
                E_omega_N[i] = 1 / (2 * c_n[i])
                E_omega_N[i] *= torch.tanh(c_n[i] / 2)
        return E_omega_N
    