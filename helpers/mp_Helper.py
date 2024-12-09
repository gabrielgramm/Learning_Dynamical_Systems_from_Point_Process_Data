import numpy as np
import torch
import gpytorch
from helpers.kernel_Helper import KernelHelper
from helpers.helpers import Helper

class MP_Helper():

    def integrage_marked_intensity(end_time, time_discretization, marked_intensity):
        return torch.sum(marked_intensity)/ time_discretization
    
    def get_posterior_marked_rate(vi, time_grid, c_complete):
        #time_grid = vi.time_grid
        sub_time_grid = vi.sub_time_grid
        inducing_points_s = vi.inducing_points_s
        mu_0 = vi.GP_prior_mean_D
        mu_0_extended = vi.GP_prior_mean_extended
        mu_s_0 = vi.SGP_prior_mean
        E_fs = vi.SGP_post_mean
        E_ln_lmbda = vi.E_ln_lmbda
        #c_complete = vi.c_complete

        E_f_full_domain = vi.kernel_helper.get_E_f_full_domain(time_grid, inducing_points_s, mu_0_extended, mu_s_0, E_fs)
        post_rate_t = 0.5 * torch.exp(E_ln_lmbda - 0.5 * E_f_full_domain)
        post_rate_t *= 1 / torch.cosh(c_complete/2)
        return post_rate_t
        