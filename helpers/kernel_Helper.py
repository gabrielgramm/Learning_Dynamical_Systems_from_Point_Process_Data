import numpy as np
import torch
import gpytorch
import time
from helpers.helpers import Helper

class KernelHelper:

    def __init__(self, kernel, kernel_params):
        self.kernel_name = kernel
        self.kernel_lenthscale = kernel_params[0]
        self.kernel_variance = kernel_params[1]

    # if adding a kernel you also need to add it in ELBO.py
    def get_kernel(self):
        if self.kernel_name == 'RBF':
            rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            rbf_kernel.base_kernel.lengthscale = self.kernel_lenthscale
            rbf_kernel.outputscale = self.kernel_variance
            return rbf_kernel
    
        else:
            raise ValueError("Kernel not implemented.")
        
    def get_kernel(self):
        if self.kernel_name == 'RBF':
            rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            rbf_kernel.base_kernel.lengthscale = self.kernel_lenthscale
            rbf_kernel.outputscale = self.kernel_variance
            return rbf_kernel
        
        elif self.kernel_name == 'Periodic':
            periodic_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
            periodic_kernel.base_kernel.lengthscale = self.kernel_lenthscale
            periodic_kernel.base_kernel.period_length = self.kernel_period_length  # You should define this attribute for period length
            periodic_kernel.outputscale = self.kernel_variance
            return periodic_kernel
        
        elif self.kernel_name == 'Matern':
            matern_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=self.matern_nu))
            matern_kernel.base_kernel.lengthscale = self.kernel_lenthscale
            matern_kernel.outputscale = self.kernel_variance
            return matern_kernel
        
        elif self.kernel_name == 'Polynomial':
            poly_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=self.poly_degree))
            poly_kernel.base_kernel.lengthscale = self.kernel_lenthscale
            poly_kernel.outputscale = self.kernel_variance
            return poly_kernel
        
        else:
            raise ValueError("Kernel not implemented.")
            
    
    def get_kernel_matrix(self, loc_1, loc_2):
        kernel = self.get_kernel()
        # locations_x_t has dimension ( T x dim_phase_space )
        # locations_x_s has dimension ( s x dim_phase_space )
        kernel_matrix = kernel(loc_1, loc_2)
        k = kernel_matrix.evaluate()
        return k.to(torch.float64)
    
    def get_kernel_value(self, point1, point2):
        kernel = self.get_kernel()
        point1 = point1.unsqueeze(0)  # Shape: (1, n)
        point2 = point2.unsqueeze(0)  # Shape: (1, n)
        kernel_value = kernel(point1, point2)
        scalar_kernel_value = kernel_value.evaluate().item()
        return scalar_kernel_value

    def get_K_ss(self, inducing_points_s):
        kernel = self.get_kernel()
        # locations_x_s has dimension ( T x dim_phase_space )
        kernel_matrix = kernel(inducing_points_s)
        # K_ss has dimensiton ( s , s )
        K_ss = kernel_matrix.evaluate()
        return K_ss.to(torch.float64)
    
    def get_inv_K_ss(self, inducing_points_s):
        kernel = self.get_kernel()
        kernel_matrix = kernel(inducing_points_s)
        K_ss = kernel_matrix.evaluate()
        #Helper.test_positive_semi_definite(K_ss)
        #K_ss = (K_ss + K_ss.t()) / 2
        jitter = 1e-6 * torch.eye(K_ss.size(0), dtype=K_ss.dtype)
        K_ss += jitter
        K_ss = K_ss.to(torch.float64)
        inv_K_ss = torch.inverse(K_ss)
        I = torch.matmul(K_ss, inv_K_ss)
        #Helper.test_positive_semi_definite(K_ss)
        #Helper.test_inverse(K_ss, _inv_K_ss)
        return K_ss, inv_K_ss
    
    '''
    def get_inv_K_ss(self, inducing_points_s):
        kernel = self.get_kernel()
        # locations_x_s has dimension ( T x dim_phase_space )
        kernel_matrix = kernel(inducing_points_s)
        # K_ss has dimensiton ( s , s )
        K_ss = kernel_matrix.evaluate()
        K_ss = K_ss.to(torch.float64)
        _inv_K_ss = torch.inverse(K_ss)
        #_inv_K_ss = torch.linalg.pinv(K_ss)
        #_inv_K_ss = _inv_K_ss.to(torch.float64)
        I = torch.matmul(K_ss, _inv_K_ss)
        return K_ss, _inv_K_ss
    '''

    def get_inv_K_ss_cholesky(self, inducing_points_s):
        kernel = self.get_kernel()
        # locations_x_s has dimension (T x dim_phase_space)
        kernel_matrix = kernel(inducing_points_s)
        # K_ss has dimension (s, s)
        K_ss = kernel_matrix.evaluate()
        K_ss = K_ss.to(torch.float64)
        L = torch.linalg.cholesky(K_ss)
        L_inv = torch.inverse(L)
        _inv_K_ss = L_inv.T @ L_inv
        _inv_K_ss = _inv_K_ss.to(torch.float64)
        I = torch.matmul(K_ss, _inv_K_ss)
        return K_ss, _inv_K_ss
    
    def get_kappa_forward(self, time_grid, inducing_points_s):
        # k_x_t__x_s has dimension (T x s )
        k_x_t__x_s = self.get_kernel_matrix(time_grid, inducing_points_s)
        K_ss, _inv_K_ss = self.get_inv_K_ss(inducing_points_s)
        #Helper.test_positive_semi_definite(K_ss)
        #Helper.test_inverse(K_ss, _inv_K_ss)
        kappa = torch.matmul(k_x_t__x_s, _inv_K_ss)
        return kappa.to(torch.float64)
    
    def get_kappa_backward(self, time_grid, inducing_points_s):
        # k_x_t__x_s has dimension ( s x T)
        k_x_s__x_t = self.get_kernel_matrix(inducing_points_s, time_grid)
        K_ss, _inv_K_ss = self.get_inv_K_ss(inducing_points_s)
        #Helper.test_positive_semi_definite(K_ss)
        kappa = torch.matmul(_inv_K_ss, k_x_s__x_t)
        return kappa.to(torch.float64)
    
    ############################################################################################
    #############  the following functions are for calculating z_plus and z_minus  #############
    ############################################################################################
    
    def get_sigma_t_given_fs(self, time_grid, inducing_points_s):
        k_x_s__x_t = self.get_kernel_matrix(inducing_points_s, time_grid)
        if self.kernel_name == 'RBF':
            #print("double check K_t_t in sigma_t_given_fs for the RBF kernel")
            K_t_t = self.kernel_variance * torch.ones(time_grid.shape[0], dtype=torch.float64)
            '''
            K_t_t2 = torch.tensor([self.get_kernel_value(t, t) for t in time_grid], dtype=torch.float64)
            if not torch.allclose(K_t_t, K_t_t2):
                raise ValueError("K_t_t is not equal to K_t_t2")
            '''
        else:
            print("implement kernel", self.kernel_name)
            start = time.perf_counter()
            K_t_t = torch.tensor([self.get_kernel_value(t, t) for t in time_grid], dtype=torch.float64)
            end = time.perf_counter()
            print("slow computation of k(t,t) in kernel helper, sec:", end-start)
        kappa_forward = self.get_kappa_forward(time_grid, inducing_points_s) 
        sigma_t_given_fs = K_t_t - torch.sum(kappa_forward * k_x_s__x_t.transpose(0,1)  , dim=1) 
        return sigma_t_given_fs
    
    def get_quadratic_term(self, time_grid, inducing_points_s, cov_s, mean_s):
        sec_mom = cov_s + torch.ger(mean_s, mean_s)
        kappa_f = self.get_kappa_forward(time_grid, inducing_points_s)
        kappa_b = self.get_kappa_backward(time_grid, inducing_points_s)
        x = torch.matmul(kappa_f, sec_mom)
        y = kappa_b.transpose(0,1)
        z = torch.sum(x * y, dim=1)
        return z
    
    ''' was needed for testing
    def get_quadratic_term2(self, time_grid, inducing_points_s, cov_s, mean_s, E_omega_N):
        quadratic_term = torch.zeros(time_grid.shape[0], dtype=torch.float64)
        sec_mom = cov_s + torch.ger(mean_s, mean_s)
        kappa_f = self.get_kappa_forward(time_grid, inducing_points_s)
        kappa_b = self.get_kappa_backward(time_grid, inducing_points_s)
        for i in range(time_grid.shape[0]):
            x = torch.ger(kappa_f[i], kappa_b.mT[i])
            y = torch.matmul(x, sec_mom)
            quadratic_term[i] = E_omega_N[i] * torch.trace(y)
        return quadratic_term
    '''

    def get_linear_term_z(self, time_grid, inducing_points_s, E_omega, E_fs, mu_s_0, mu_0, z_plus):
        # mu * kappa backward
        #mu_0_extended = torch.full((time_grid.shape[0],), mu_0[0], dtype=torch.float64)
        z = torch.matmul(mu_s_0, self.get_kappa_backward(time_grid, inducing_points_s))
        z = mu_0 - z
        z *= E_omega
        one_half = torch.full((time_grid.shape[0],), 0.5)
        if z_plus == True:
            z = one_half - z
        else:
            z = -one_half - z
        # instead of z times kappa forward, we do first kappa forward times fs
        temp = torch.matmul(self.get_kappa_forward(time_grid, inducing_points_s), E_fs)
        z *= temp
        return z
    
    def get_linear_term_z_marked(self, time_grid, inducing_points_s, data, E_omega, E_fs, mu_s_0, mu_0, z_plus):
        # mu * kappa backward
        #mu_0_extended = torch.full((time_grid.shape[0],), mu_0[0], dtype=torch.float64)
        z = torch.matmul(mu_s_0, self.get_kappa_backward(time_grid, inducing_points_s))
        z = mu_0 - z
        z *= E_omega
        one_half = torch.full((time_grid.shape[0],), 0.5)
        if z_plus == True:
            z = one_half - z
            z *= data
        else:
            z = -one_half - z
        # instead of z times kappa forward, we do first kappa forward times fs
        temp = torch.matmul(self.get_kappa_forward(time_grid, inducing_points_s), E_fs)
        z *= temp
        return z
    
    def get_bracket_term_of_const(self, time_grid, inducing_points_s, mu_s_0, mu_0, sig_t_given_fs):
        #mu_extended = torch.full((time_grid.shape[0],), mu_0[0], dtype=torch.float64)
        z = sig_t_given_fs + torch.pow(mu_0,2)
        #print("\n")
        #print(sum(sig_t_given_fs))
        #print(sum(torch.pow(mu_0,2)))
        z -= 2 * mu_0 * torch.matmul(self.get_kappa_forward(time_grid, inducing_points_s), mu_s_0)
        #print(sum(mu_0 * torch.matmul(self.get_kappa_forward(time_grid, inducing_points_s), mu_s_0)))
        y = torch.matmul(mu_s_0, self.get_kappa_backward(time_grid, inducing_points_s))
        z += torch.pow(y,2)
        #print(sum(torch.pow(y,2)))
        #print("\n")
        return z

    def get_constant_term_z(self, time_grid, inducing_points_s,  bracket_term, mu_s_0, mu_0, z_plus):
        #mu_extended = torch.full((time_grid.shape[0],), mu_0[0], dtype=torch.float64)
        if z_plus == True:
            z = 0.5 * mu_0
            z -= 0.5 * torch.matmul(self.get_kappa_forward(time_grid, inducing_points_s), mu_s_0)
        else:
            z = - 0.5 * mu_0
            z += 0.5 * torch.matmul(self.get_kappa_forward(time_grid, inducing_points_s), mu_s_0)
        z -= bracket_term 
        z -= torch.full((time_grid.shape[0],), torch.log(torch.tensor(2.0)))
        return z
    
    ############################################################################################
    ##############  the following functions are for calculating omega posterior  ###############
    ############################################################################################

    def get_linear_term_omega(self, time_grid, inducing_points_s, mean_s, mu_s_0, mu_0):
        #mu_0_extended = torch.full((time_grid.shape[0],), mu_0[0], dtype=torch.float64)
        kappa_f = self.get_kappa_forward(time_grid, inducing_points_s)
        kappa_b = self.get_kappa_backward(time_grid, inducing_points_s)
        # mu * kappa backward
        z = torch.matmul(mu_s_0, kappa_b)
        z = mu_0 - z
        # instead of z times kappa forward, we do first kappa forward times fs
        temp = torch.matmul(kappa_f, mean_s)
        z *= temp
        return z

    ############################################################################################
    ############  the following functions are for calculating marked process rate  #############
    ############################################################################################
    
    def get_E_f_full_domain(self, time_grid, inducing_points_s, mu_0, mu_s_0, E_fs):
        #mu_0_extended = torch.full((time_grid.shape[0],), mu_0[0], dtype=torch.float64)
        diff = E_fs - mu_s_0
        e = mu_0 + torch.matmul(self.get_kappa_forward(time_grid, inducing_points_s), diff)
        return e
    
    ############################################################################################
    ################  the following functions are for calculating gp posterior  ################
    ############################################################################################

    def get_sigma_s(self, time_gird, inducing_points_s, E_omega_N):
        kappa_b = self.get_kappa_backward(time_gird, inducing_points_s)
        kappa_f = self.get_kappa_forward(time_gird, inducing_points_s)
        temp = E_omega_N.t() * kappa_b
        sigma_s = torch.matmul(temp, kappa_f)
        return sigma_s
    
    def get_sigma_s_marked(self, time_gird, inducing_points_s, E_omega_N, data):
        kappa_b = self.get_kappa_backward(time_gird, inducing_points_s)
        kappa_f = self.get_kappa_forward(time_gird, inducing_points_s)
        test2 = E_omega_N.t() * data
        temp = test2 * kappa_b
        sigma_s = torch.matmul(temp, kappa_f)
        return sigma_s

    def get_sigma_s_complete(self, time_grid, inducing_points_s, E_omega_complete, marked_rate):
        kappa_b = self.get_kappa_backward(time_grid, inducing_points_s)
        kappa_f = self.get_kappa_forward(time_grid, inducing_points_s)
        temp = E_omega_complete.t() * kappa_b
        temp2 = kappa_f * marked_rate.view(marked_rate.shape[0], 1)
        sigma_s_complete = torch.matmul(temp, temp2)
        return sigma_s_complete
    
    def get_linear_term_v(self, time_grid, inducing_points_s, E_omega, mu_s_0, mu_0, marked_rate, v_plus):
        # mu * kappa backward
        x = torch.matmul(mu_s_0, self.get_kappa_backward(time_grid, inducing_points_s))
        v = mu_0 - x
        v *= E_omega
        one_half = torch.full((time_grid.shape[0],), 0.5)
        if v_plus == True:
            y = one_half - v
            v = torch.matmul(y, self.get_kappa_forward(time_grid, inducing_points_s))
            return v
        else:
            y = -one_half - v
            v = y * marked_rate
            v = torch.matmul(v, self.get_kappa_forward(time_grid, inducing_points_s))
            return v

    def get_linear_term_v_marked(self, time_grid, inducing_points_s, E_omega, mu_s_0, mu_0, data, marked_rate, v_plus):
        # mu * kappa backward
        x = torch.matmul(mu_s_0, self.get_kappa_backward(time_grid, inducing_points_s))
        v = mu_0 - x
        v *= E_omega
        one_half = torch.full((time_grid.shape[0],), 0.5)
        if v_plus == True:
            y = one_half - v
            v = y * data
            v = torch.matmul(v, self.get_kappa_forward(time_grid, inducing_points_s))
            return v
        else:
            y = -one_half - v
            v = y * marked_rate
            v = torch.matmul(v, self.get_kappa_forward(time_grid, inducing_points_s))
            return v

        

        


