import torch
import time
torch.autograd.set_detect_anomaly(True)
import gpytorch
import torch.optim as optim
from helpers.kernel_Helper import KernelHelper
from helpers.gp_Helper import GP_Helper
from helpers.helpers import Helper
from helpers.helpers import no_grad_method
from ELBO.ELBO_helper import get_z_plus, get_z_minus, get_z_plus_marked
from time_series.ts_Helper import TS_Helper

class opt_ELBO:

    def __init__(self, learning_rate, hyperparameters):    

        self.positive_constraint = gpytorch.constraints.Positive()
        self.GP_prior_mean = torch.nn.Parameter(hyperparameters['GP_prior_mean'].clone().detach(), requires_grad=True)
        self.inducing_points_s = torch.nn.Parameter(hyperparameters['inducing_points_s'].clone().detach(), requires_grad=True)
        alpha_0 = hyperparameters['alpha_0'].clone().detach()
        beta_0 = hyperparameters['beta_0'].clone().detach()
        self.raw_alpha_0 = torch.nn.Parameter(self.positive_constraint.inverse_transform(alpha_0), requires_grad=True)
        self.raw_beta_0 = torch.nn.Parameter(self.positive_constraint.inverse_transform(beta_0), requires_grad=True)
    
        self.kernel_lengthscale = hyperparameters['kernel_lengthscale'].clone().detach()
        self.kernel_outputscale = hyperparameters['kernel_outputscale'].clone().detach()
        self.raw_lengthscale = torch.nn.Parameter(self.positive_constraint.inverse_transform(self.kernel_lengthscale), requires_grad=True)
        self.raw_outputscale = torch.nn.Parameter(self.positive_constraint.inverse_transform(self.kernel_outputscale), requires_grad=True)

        param_groups = [
            {'params': [self.GP_prior_mean], 'lr': learning_rate},
            {'params': [self.raw_alpha_0], 'lr': learning_rate},
            {'params': [self.raw_beta_0], 'lr': learning_rate},
            {'params': [self.raw_lengthscale], 'lr': learning_rate*2},
            {'params': [self.raw_outputscale], 'lr': learning_rate*2},
            {'params': [self.inducing_points_s], 'lr': learning_rate}
        ]
        self.optimizer = optim.Adam(param_groups)

    def optimize(self, vi, steps, hyperparameters):
        for epoch in range(steps):
            start_time=time.perf_counter()
            self.optimizer.zero_grad()
            ELBO = self.elbo(vi)
            loss = - ELBO
            loss.backward(retain_graph=True)         
            self.optimizer.step()
            with torch.no_grad():
                vi.SGP_post_mean, vi.SGP_post_cov = GP_Helper.get_GP_posterior(vi, self.inducing_points_s)
            end_time=time.perf_counter()
            #print(f"   ## Epoch {epoch+1}, ELBO: {ELBO.item():.2f}, time for epoch: {end_time - start_time:.2f}s @@")
            #vi.loss_tracker.append(-loss.item())

        hyperparameters['GP_prior_mean'] = self.GP_prior_mean.clone().detach().requires_grad_(False)
        hyperparameters['inducing_points_s'] = self.inducing_points_s.clone().detach().requires_grad_(False)

        raw_alpha_0 = self.raw_alpha_0.clone().detach().requires_grad_(False)
        raw_beta_0 = self.raw_beta_0.clone().detach().requires_grad_(False)
        hyperparameters['alpha_0'] = self.positive_constraint.transform(raw_alpha_0)
        hyperparameters['beta_0'] = self.positive_constraint.transform(raw_beta_0)
        
        raw_lengthscale = self.raw_lengthscale.clone().detach().requires_grad_(False)
        raw_outputscale = self.raw_outputscale.clone().detach().requires_grad_(False)
        hyperparameters['kernel_lengthscale'] = self.positive_constraint.transform(raw_lengthscale)
        hyperparameters['kernel_outputscale'] = self.positive_constraint.transform(raw_outputscale)
        print("   new kernel lengthscale", hyperparameters['kernel_lengthscale'].item())
        return hyperparameters, vi.SGP_post_mean, vi.SGP_post_cov

    # We cal negative ELBO and do descent instead of ascent
    def elbo(self, vi):

        ''' &&&&& initialization &&&&& '''
        time_grid = vi.ts_helper.get_time_grid(vi.couplings, vi.tau_list)
        sub_time_grid = vi.ts_helper.get_sub_time_grid_one_process(vi.thinned_shifted_indices, vi.couplings, vi.tau_list) 
        mu_0_sub = self.GP_prior_mean * torch.ones(sub_time_grid.shape[0], dtype=torch.float64)
        mu_0_extended = self.GP_prior_mean * torch.ones(time_grid.shape[0], dtype=torch.float64)
        mu_s_0 = self.GP_prior_mean * torch.ones(self.inducing_points_s.shape[0], dtype=torch.float64)

        alpha_0 = self.positive_constraint.transform(self.raw_alpha_0)
        beta_0 = self.positive_constraint.transform(self.raw_beta_0)
        
        #kernel initialization     
        if vi.kernel == 'RBF':
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            kernel.raw_outputscale = self.raw_outputscale
            kernel.base_kernel.raw_lengthscale = self.raw_lengthscale
        elif self.kernel_name == 'Periodic':
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
            kernel.base_kernel.raw_lengthscale = self.raw_lengthscale
            #kernel.base_kernel.raw_period_length = self.raw_period_length
            kernel.raw_outputscale = self.raw_outputscale  
        elif self.kernel_name == 'Matern':
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=self.matern_nu))
            kernel.base_kernel.raw_lengthscale = self.raw_lengthscale
            kernel.raw_outputscale = self.raw_outputscale
        else:
            raise ValueError("Kernel not implemented")

        #cal K_ss and inv_K_ss
        kernel_matrix = kernel(self.inducing_points_s)
        K_ss_ELBO = kernel_matrix.evaluate()
        jitter = 1e-6 * torch.eye(K_ss_ELBO.size(0), dtype=K_ss_ELBO.dtype)
        K_ss_ELBO += jitter
        inv_K_ss_ELBO = torch.inverse(K_ss_ELBO)

        #kappafull
        kernel_matrix_full = kernel(vi.time_grid, self.inducing_points_s)
        k_x_t__x_s_full = kernel_matrix_full.evaluate()
        kappa_f_full = torch.matmul(k_x_t__x_s_full, inv_K_ss_ELBO)
        kappa_b_full = torch.transpose(kappa_f_full, 0, 1)

        #kappa_sub
        kernel_matrix_sub = kernel(vi.sub_time_grid, self.inducing_points_s)
        k_x_t__x_s_sub = kernel_matrix_sub.evaluate()
        kappa_f_sub = torch.matmul(k_x_t__x_s_sub, inv_K_ss_ELBO)
        kappa_b_sub = torch.transpose(kappa_f_sub, 0, 1)


        ''' &&&&& z_plus &&&&& '''
        #cal quadratic term
        sec_mom = vi.SGP_post_cov + torch.ger(vi.SGP_post_mean, vi.SGP_post_mean)
        quadratic_term = torch.sum((torch.matmul(kappa_f_sub, sec_mom)) * (kappa_b_sub.transpose(0,1)), dim=1)
        quadratic_term *= vi.E_omega_N  

        #cal linear term
        mu_s_0_kappa_sub = torch.matmul(mu_s_0, kappa_b_sub)
        lin_term_plus = mu_0_sub - mu_s_0_kappa_sub
        lin_term_plus *= vi.E_omega_N
        lin_term_plus = 0.5 - lin_term_plus
        temp2 = torch.matmul(kappa_f_sub, vi.SGP_post_mean)
        lin_term_plus *= temp2

        #cal sig_t_given_fs_sub
        if vi.kernel == 'RBF':
            #k_t_t = self.kernel_outputscale * torch.ones(sub_time_grid.shape[0], dtype=torch.float64)
            k_t_t = self.positive_constraint.transform(self.raw_outputscale) * torch.ones(sub_time_grid.shape[0], dtype=torch.float64)
        else:
            raise ValueError("Kernel not implemented")
        sigma_t_given_fs_sub = k_t_t - torch.sum(kappa_f_sub * k_x_t__x_s_sub, dim=1)

        #cal bracket_term
        bracket_term = sigma_t_given_fs_sub + torch.pow(mu_0_sub, 2)
        bracket_term -= 2 * mu_0_sub * mu_s_0_kappa_sub
        bracket_term += torch.pow(mu_s_0_kappa_sub, 2)

        #cal constant term
        const_term = 0.5 * mu_0_sub
        const_term -= 0.5 * mu_s_0_kappa_sub
        const_term -= bracket_term * 0.5 * vi.E_omega_N
        const_term -= torch.log(torch.tensor(2.0))

        z_plus = - 0.5 * quadratic_term + lin_term_plus + const_term
        sum_z_plus = torch.sum(z_plus)


        ''' &&&&& z_minus &&&&& '''
        #cal quadratic term
        sec_mom = vi.SGP_post_cov + torch.ger(vi.SGP_post_mean, vi.SGP_post_mean)
        quadratic_term = torch.sum((torch.matmul(kappa_f_full, sec_mom)) * (kappa_f_full), dim=1)
        quadratic_term *= vi.E_omega_complete

        #cal linear term
        mu_s_0_kappa_b_full = torch.matmul(mu_s_0, kappa_b_full)
        lin_term_minus = mu_0_extended - mu_s_0_kappa_b_full
        lin_term_minus *= vi.E_omega_complete
        lin_term_minus = - 0.5 - lin_term_minus
        temp5 = torch.matmul(kappa_f_full, vi.SGP_post_mean)
        lin_term_minus *= temp5

        #cal sig_t_given_fs_sub
        if vi.kernel == 'RBF':
            #k_t_t = self.kernel_outputscale * torch.ones(time_grid.shape[0], dtype=torch.float64)
            k_t_t = self.positive_constraint.transform(self.raw_outputscale) * torch.ones(time_grid.shape[0], dtype=torch.float64)
        else:
            raise ValueError("Kernel not implemented")
        sigma_t_given_fs_full = k_t_t - torch.sum(kappa_f_full * k_x_t__x_s_full, dim=1)

        #cal bracket_term
        bracket_term_full = sigma_t_given_fs_full + torch.pow(mu_0_extended,2)
        bracket_term_full -= 2 * mu_0_extended * mu_s_0_kappa_b_full
        bracket_term_full += torch.pow(mu_s_0_kappa_b_full,2)

        #cal constant term
        const_term_minus = - 0.5 * mu_0_extended
        const_term_minus += 0.5 * mu_s_0_kappa_b_full
        const_term_minus -= bracket_term_full * 0.5 * vi.E_omega_complete
        const_term_minus -= torch.log(torch.tensor(2.0))

        #cal L_E_U_s
        z_minus = - 0.5 * quadratic_term + lin_term_minus + const_term_minus
        z_minus *= vi.marked_process_intensity_t
        sum_z_minus = torch.sum(z_minus) / vi.time_discretization
        sum_ln_lmbda = vi.E_ln_lmbda * sub_time_grid.shape[0]
        L_E_U_s = sum_z_plus + sum_z_minus + sum_ln_lmbda
        

        ''' &&&&& PG kl divergence &&&&& '''
        kl_PG = sum(torch.log(torch.cosh(vi.c_n / 2)) - (vi.c_n_squared / 2) * vi.E_omega_N)


        ''' &&&&& MP kl divergence &&&&& '''
        kl_MP = - torch.sum(vi.marked_process_intensity_t * (1 + vi.E_ln_lmbda)) / vi.time_discretization
        kl_MP += vi.end_time * vi.E_lmbda
        kl_MP += torch.sum((torch.log(vi.marked_process_intensity_t)) * vi.marked_process_intensity_t) / vi.time_discretization
        kl_MP += torch.sum(torch.log(torch.cosh(vi.c_complete / 2)) * vi.marked_process_intensity_t) / vi.time_discretization
        kl_MP -= torch.sum(vi.c_complete_squared/2 * vi.E_omega_complete * vi.marked_process_intensity_t) / vi.time_discretization


        ''' &&&&& lmbda kl divergence &&&&& '''
        kl_lmbda = (vi.alpha_post - alpha_0) * torch.digamma(vi.alpha_post)
        kl_lmbda -= torch.lgamma(vi.alpha_post) - torch.lgamma(alpha_0)
        kl_lmbda += alpha_0 * (torch.log(vi.beta_post) - torch.log(beta_0))
        kl_lmbda -= (vi.beta_post - beta_0) * (vi.alpha_post / vi.beta_post)


        ''' &&&&& GP kl divergence &&&&& '''
        D = mu_s_0.shape[0]
        tr_term   = torch.sum(inv_K_ss_ELBO * torch.transpose(vi.SGP_post_cov, 0, 1))
        diff = mu_s_0 - vi.SGP_post_mean
        quad_term = torch.dot(diff, torch.matmul(inv_K_ss_ELBO , diff))
        # det term 
        chol_prior = torch.linalg.cholesky(K_ss_ELBO)
        log_det_cov_prior =  2 * torch.sum(torch.log(torch.diagonal(chol_prior)))
        chol_post = torch.linalg.cholesky(vi.SGP_post_cov)
        log_det_cov_post =  2 * torch.sum(torch.log(torch.diagonal(chol_post)))
        det_term = log_det_cov_prior - log_det_cov_post
        kl_GP = 0.5 * (tr_term - D + quad_term + det_term)

        ''' &&&&& ELBO &&&&& '''
        L =  L_E_U_s - kl_lmbda - kl_PG - kl_MP - kl_GP  
        
        '''print("\n")
        print("sum_ln_lmbda", sum_ln_lmbda)
        print("z plus", sum_z_plus)
        print("z minus", sum_z_minus)
        print("L_E_U_s in ELBO", L_E_U_s)
        print("kl_PG", kl_PG)
        print("kl_lmbda", kl_lmbda)
        print("kl_MP", kl_MP)
        print("kl_GP", kl_GP)
        print("L", - L)
        print("\n")'''

        return L
    
class ELBO:
    @no_grad_method
    def E_U_s(vi):
        time_grid = vi.time_grid
        inducing_points_s = vi.inducing_points_s
        time_discretization = vi.time_discretization
        sub_time_grid = vi.sub_time_grid

        E_omega_N = vi.E_omega_N
        E_omega_complete = vi.E_omega_complete
        mu_0 = vi.GP_prior_mean_D
        mu_0_extended = vi.GP_prior_mean_extended
        mu_s_0 = vi.SGP_prior_mean
        E_fs = vi.SGP_post_mean
        cov_s = vi.SGP_post_cov
        E_ln_lmbda = vi.E_ln_lmbda
        marked_rate = vi.marked_process_intensity_t
        data = torch.roll(vi.thinned_process, shifts=-1)
        data[-1] = 0

        #z_plus = get_z_plus(vi, sub_time_grid, inducing_points_s, E_omega_N, E_fs, cov_s,  mu_0, mu_s_0)
        z_plus_marked = get_z_plus_marked(vi, time_grid, inducing_points_s, data, E_omega_N, E_fs, cov_s,  mu_0_extended, mu_s_0)
        z_minus = get_z_minus(vi, time_grid, inducing_points_s, E_omega_complete, E_fs, cov_s,  mu_0_extended, mu_s_0)

        sum_z_plus = torch.sum(z_plus_marked, dim=0)
        integrand = z_minus * marked_rate
        z_minus = torch.sum(integrand) / time_discretization

        sum_ln_lmbda = E_ln_lmbda * sub_time_grid.shape[0]
        '''
        print("sum_ln_lmbda", sum_ln_lmbda)
        print("z plus", sum_z_plus)
        print("z minus", z_minus)
        '''
        L_EU_s = sum_z_plus + z_minus + sum_ln_lmbda
        return L_EU_s, sum_z_plus , z_minus, sum_ln_lmbda
    
''' &&&&& backwards GP kl divergence &&&&& '''
'''
D = mu_s_0.shape[0]
tr_term2   = torch.sum(torch.transpose(vi.SGP_post_cov, 0, 1) * inv_K_ss_ELBO)
diff2 =  vi.SGP_post_mean - mu_s_0
quad_term2 = torch.dot(diff2, torch.matmul(vi.SGP_post_cov , diff2))
# det term 
chol_prior2 = torch.linalg.cholesky(vi.SGP_prior_cov)
log_det_cov_prior2 =  2 * torch.sum(torch.log(torch.diagonal(chol_prior2)))
chol_post2 = torch.linalg.cholesky(vi.SGP_post_cov)
log_det_cov_post2 =  2 * torch.sum(torch.log(torch.diagonal(chol_post2)))
det_term2 =  log_det_cov_post2 - log_det_cov_prior2
kl_GP_backwards = 0.5 * (tr_term2 - D + quad_term2 + det_term2)
'''



