import numpy as np
import torch
from helpers.kernel_Helper import KernelHelper
from helpers.helpers import Helper
from time_series.ts_Helper import TS_Helper

def get_z_plus(vi, time_grid, inducing_points_s, E_omega_N, E_fs, cov_s,  mu_0, mu_s_0):
    quadratic_term = vi.kernel_helper.get_quadratic_term(time_grid, inducing_points_s, cov_s, E_fs)
    #print("quadratic_term plus", sum(quadratic_term))
    quadratic_term = quadratic_term * E_omega_N
    linear_term_zplus = vi.kernel_helper.get_linear_term_z(time_grid, inducing_points_s, E_omega_N,  E_fs, mu_s_0, mu_0, z_plus=1)
    #print("linear_term_zplus", sum(linear_term_zplus))
    sig_t_given_fs = vi.kernel_helper.get_sigma_t_given_fs(time_grid, inducing_points_s)
    bracket_term = vi.kernel_helper.get_bracket_term_of_const(time_grid, inducing_points_s, mu_s_0, mu_0, sig_t_given_fs)
    bracket_term = bracket_term * 0.5 * E_omega_N
    constant_term = vi.kernel_helper.get_constant_term_z(time_grid, inducing_points_s, bracket_term,  mu_s_0, mu_0,  z_plus=1)
    #print("constant_term zplus", sum(constant_term))
    z_plus = - 0.5 * quadratic_term + linear_term_zplus + constant_term
    return z_plus

def get_z_plus_marked(vi, time_grid, inducing_points_s, data, E_omega_N, E_fs, cov_s,  mu_0, mu_s_0):
    quadratic_term = vi.kernel_helper.get_quadratic_term(time_grid, inducing_points_s, cov_s, E_fs)
    #print("quadratic_term plus", sum(quadratic_term))
    quadratic_term = quadratic_term * E_omega_N * data
    linear_term_zplus = vi.kernel_helper.get_linear_term_z_marked(time_grid, inducing_points_s, data, E_omega_N,  E_fs, mu_s_0, mu_0, z_plus=1)
    #print("linear_term_zplus", sum(linear_term_zplus))
    sig_t_given_fs = vi.kernel_helper.get_sigma_t_given_fs(time_grid, inducing_points_s)
    bracket_term = vi.kernel_helper.get_bracket_term_of_const(time_grid, inducing_points_s, mu_s_0, mu_0, sig_t_given_fs)
    bracket_term = bracket_term * 0.5 * E_omega_N
    constant_term = vi.kernel_helper.get_constant_term_z(time_grid, inducing_points_s, bracket_term,  mu_s_0, mu_0,  z_plus=1)
    constant_term = constant_term * data
    #print("constant_term zplus", sum(constant_term))
    z_plus = - 0.5 * quadratic_term + linear_term_zplus + constant_term
    return z_plus

def get_z_minus(vi, time_grid, inducing_points_s, E_omega_complete, E_fs, cov_s,  mu_0, mu_s_0):
    quadratic_term = vi.kernel_helper.get_quadratic_term(time_grid, inducing_points_s, cov_s, E_fs)
    #print("quadratic_term minus", sum(quadratic_term))
    quadratic_term = quadratic_term * E_omega_complete
    linear_term_zminus = vi.kernel_helper.get_linear_term_z(time_grid, inducing_points_s, E_omega_complete,  E_fs, mu_s_0, mu_0, z_plus=0)
    #print("linear_term_zminus", sum(linear_term_zminus))
    sig_t_given_fs = vi.kernel_helper.get_sigma_t_given_fs(time_grid, inducing_points_s)
    bracket_term = vi.kernel_helper.get_bracket_term_of_const(time_grid, inducing_points_s, mu_s_0, mu_0, sig_t_given_fs)
    #print("bracket_term zminus", sum(bracket_term))
    bracket_term = bracket_term * 0.5 * E_omega_complete
    constant_term = vi.kernel_helper.get_constant_term_z(time_grid, inducing_points_s, bracket_term,  mu_s_0, mu_0,  z_plus=0)
    #print("constant_term zminus", sum(constant_term))
    z_minus = - 0.5 * quadratic_term + linear_term_zminus + constant_term
    return z_minus