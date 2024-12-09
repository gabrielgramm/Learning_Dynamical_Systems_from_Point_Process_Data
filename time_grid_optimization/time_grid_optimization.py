import time
import torch
torch.autograd.set_detect_anomaly(True)
import gpytorch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from helpers.kernel_Helper import KernelHelper
from helpers.gp_Helper import GP_Helper
from helpers.helpers import Helper
from helpers.plotter import plot_time_grid

class Time_grid_optimization:

    def __init__(self, global_full_stack_params, ts_helper, couplings, tau_list, learning_rate_time_grid):    

        self.global_full_stack_params = global_full_stack_params
        self.ts_helper = ts_helper
        self.positive_constraint = gpytorch.constraints.Positive()
        self.couplings = torch.nn.Parameter(couplings.clone().detach(), requires_grad=True)
        self.tau_list = tau_list.clone().detach()
        self.raw_tau_list = torch.nn.Parameter(self.positive_constraint.inverse_transform(self.tau_list), requires_grad=True)

        param_groups = [
            {'params': [self.couplings], 'lr': learning_rate_time_grid},
            {'params': [self.raw_tau_list], 'lr': learning_rate_time_grid},
        ]
        self.optimizer = optim.Adam(param_groups)

    def optimize_time_grid(self, steps):
        for epoch in range(steps):
            start_time=time.perf_counter()
            self.optimizer.zero_grad()
            loss = self.elbo_all_processes()
            loss.backward(retain_graph=True)       
            self.optimizer.step()
            end_time=time.perf_counter()
            #print(f'couplings: {self.couplings}')
            print("\n")
            print(f"   Epoch {epoch+1}, Global_ELBO: {-loss.item():.2f}, time for epoch: {end_time - start_time:.2f}s")
            print("   raw_tau_list:", [f"{x:.2f}" for x in self.raw_tau_list])

        self.tau_list = self.positive_constraint.transform(self.raw_tau_list.clone().detach().requires_grad_(False))
        time_grid = self.ts_helper.get_time_grid(self.couplings.detach(), self.tau_list)
        plot_time_grid(self.ts_helper, time_grid, size=5, start=200, end=3000, show_sub_time_grid=True, process=None)
        return self.tau_list, self.couplings.detach().requires_grad_(False)
    

    ####### functions for time grid optimization #######

    def get_time_grid_kernel(self, x, tau):
        exp_function = torch.exp(-x/tau) /tau
        exp_function[x < 0] = 0
        kernel = exp_function / (torch.sum(exp_function) / self.ts_helper.time_discretization)   #normalizeation
        #plt.figure(figsize=(3, 3))
        #plt.plot(exp_function.clone().detach())
        #plt.plot(kernel.clone().detach())
        #plt.show()
        return kernel       

    def convolve(self, data, tau, kernel_effect_length, time_discretization):
        if tau < 0:
            raise ValueError("Tau must be greater than 0")
        elif tau < 1:
            opt_kernel_length = kernel_effect_length
            opt_kernel_length = torch.ceil(opt_kernel_length)
        else:
            opt_kernel_length = kernel_effect_length# * torch.log(tau)
            opt_kernel_length = torch.ceil(opt_kernel_length)
                
        temp_time_grid = torch.linspace(0, int(opt_kernel_length.item()), int(opt_kernel_length.item()) * time_discretization)
        time_grid_kernel = self.get_time_grid_kernel(temp_time_grid, tau)
        kernel_tensor = time_grid_kernel.unsqueeze(0).unsqueeze(0).double()
        kernel_tensor = kernel_tensor.flip(2)
        padding = kernel_tensor.shape[2] - 1  # This will pad equally on both sides
        conv = torch.zeros(data.shape[0], data.shape[1] + time_grid_kernel.shape[0] - 1)
        for i in range(0, data.shape[0]):
            x_tensor = data[i].clone().unsqueeze(0).unsqueeze(0).double()  # Shape: (num_processes, 1, len(x))
            x_padded = F.pad(x_tensor, (padding, padding))
            _1dconv = F.conv1d(x_padded, kernel_tensor)
            conv[i] = _1dconv.squeeze()
        add_zeros = torch.zeros(conv.shape[0], 1)
        conv = torch.cat((conv, add_zeros), dim=1)
        cut = int(conv.shape[1] - time_grid_kernel.shape[0])
        conv = conv[:, :cut]
        return conv

    ######## We cal negative ELBO and do descent instead of ascent ########

    def elbo_all_processes(self):

        global_loss = .0

        ### &&&&& initialization &&&&& ###
        if len(self.raw_tau_list) != self.ts_helper.dim_phase_space:
            raise ValueError("The number of tau values should be equal to the dimension of the phase space.")

        tau_list = self.positive_constraint.transform(self.raw_tau_list)

        time_grid = torch.empty(self.ts_helper.data.shape[1], self.ts_helper.dim_phase_space, dtype=torch.float64)
        for axis in range(0, self.ts_helper.dim_phase_space):
            convolved = self.convolve(self.ts_helper.data, tau_list[axis], self.ts_helper.kernel_effect_length, self.ts_helper.time_discretization)
            axis_value = torch.sum(self.couplings[:, axis].unsqueeze(1) * convolved, dim=0)
            #axis_value = self.get_axis_of_phase_space(self.ts_helper.data, self.tau_list[axis], axis, self.couplings, self.ts_helper.kernel_effect_length, self.ts_helper.time_discretization)
            time_grid[:, axis] = axis_value
        
        time_grid = time_grid.to(torch.float64)

        #print("time_grid_in_global_elbo_calculation")
        #plot_time_grid(self.ts_helper, time_grid.detach(), size=3, start=200, end=3000, show_sub_time_grid=True, process=None)       

        for process, full_stack_params in enumerate(self.global_full_stack_params):

            sub_time_grid = torch.index_select(time_grid, 0, full_stack_params['thinned_shifted_indices'])

            ### time gird is not float 64 ###
            #plot_time_grid(self.ts_helper, time_grid.detach(), size=4, show_sub_time_grid=True)
            
            mu_0_sub = full_stack_params['GP_prior_mean'] * torch.ones(sub_time_grid.shape[0], dtype=torch.float64)
            mu_0_extended = full_stack_params['GP_prior_mean'] * torch.ones(time_grid.shape[0], dtype=torch.float64)
            mu_s_0 = full_stack_params['GP_prior_mean'] * torch.ones(full_stack_params['inducing_points_s'].shape[0], dtype=torch.float64)
            
            #kernel initialization     
            if full_stack_params['kernel'] == 'RBF':
                kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                kernel.outputscale = full_stack_params['kernel_outputscale']
                kernel.base_kernel.lengthscale = full_stack_params['kernel_lengthscale']
            else:
                raise ValueError("Kernel not implemented")

            #cal K_ss and inv_K_ss
            kernel_matrix = kernel(full_stack_params['inducing_points_s'])
            K_ss_ELBO = kernel_matrix.evaluate()
            jitter = 1e-6 * torch.eye(K_ss_ELBO.size(0), dtype=K_ss_ELBO.dtype)
            K_ss_ELBO += jitter
            inv_K_ss_ELBO = torch.inverse(K_ss_ELBO)

            #kappafull
            kernel_matrix_full = kernel(time_grid, full_stack_params['inducing_points_s'])
            k_x_t__x_s_full = kernel_matrix_full.evaluate()
            kappa_f_full = torch.matmul(k_x_t__x_s_full, inv_K_ss_ELBO)
            kappa_b_full = torch.transpose(kappa_f_full, 0, 1)

            #kappa_sub
            kernel_matrix_sub = kernel(sub_time_grid, full_stack_params['inducing_points_s'])
            k_x_t__x_s_sub = kernel_matrix_sub.evaluate()
            kappa_f_sub = torch.matmul(k_x_t__x_s_sub, inv_K_ss_ELBO)
            kappa_b_sub = torch.transpose(kappa_f_sub, 0, 1)


            ### &&&&& z_plus &&&&& ###
            #cal quadratic term
            sec_mom = full_stack_params['cov_post'] + torch.ger(full_stack_params['mu_post'], full_stack_params['mu_post'])
            quadratic_term = torch.sum((torch.matmul(kappa_f_sub, sec_mom)) * (kappa_b_sub.transpose(0,1)), dim=1)
            quadratic_term *= full_stack_params['E_omega_N']

            #cal linear term
            mu_s_0_kappa_sub = torch.matmul(mu_s_0, kappa_b_sub)
            lin_term_plus = mu_0_sub - mu_s_0_kappa_sub
            lin_term_plus *= full_stack_params['E_omega_N']
            lin_term_plus = 0.5 - lin_term_plus
            temp2 = torch.matmul(kappa_f_sub, full_stack_params['mu_post'])
            lin_term_plus *= temp2

            #cal sig_t_given_fs_sub
            if full_stack_params['kernel'] == 'RBF':
                k_t_t = full_stack_params['kernel_outputscale'] * torch.ones(sub_time_grid.shape[0], dtype=torch.float64)# * sub_time_grid[:,0] #this was a test
            else:
                raise ValueError("Kernel not implemented")
            sigma_t_given_fs_sub = k_t_t - torch.sum(kappa_f_sub * k_x_t__x_s_sub, dim=1)

            #cal bracket_term
            bracket_term = sigma_t_given_fs_sub + torch.pow(mu_0_sub, 2)
            bracket_term -= 2 * mu_0_sub * mu_s_0_kappa_sub
            bracket_term += torch.pow(mu_s_0_kappa_sub, 2)
            
            '''print("\n")
            print(sum(sigma_t_given_fs_sub))
            print(sum(torch.pow(mu_0_sub, 2)))
            print(sum(2 * mu_0_sub * mu_s_0_kappa_sub))
            print(sum(torch.pow(mu_s_0_kappa_sub, 2)))
            print("\n")'''

            #cal constant term
            const_term = 0.5 * mu_0_sub
            const_term -= 0.5 * mu_s_0_kappa_sub
            const_term -= bracket_term * 0.5 * full_stack_params['E_omega_N']
            const_term -= torch.log(torch.tensor(2.0))

            z_plus = - 0.5 * quadratic_term + lin_term_plus + const_term
            sum_z_plus = torch.sum(z_plus)


            ### &&&&& z_minus &&&&& ###
            #cal quadratic term
            sec_mom = full_stack_params['cov_post'] + torch.ger(full_stack_params['mu_post'], full_stack_params['mu_post'])
            quadratic_term = torch.sum((torch.matmul(kappa_f_full, sec_mom)) * (kappa_f_full), dim=1)
            quadratic_term *= full_stack_params['E_omega_complete']

            #cal linear term
            mu_s_0_kappa_b_full = torch.matmul(mu_s_0, kappa_b_full)
            lin_term_minus = mu_0_extended - mu_s_0_kappa_b_full
            lin_term_minus *= full_stack_params['E_omega_complete']
            lin_term_minus = - 0.5 - lin_term_minus
            temp5 = torch.matmul(kappa_f_full, full_stack_params['mu_post'])
            lin_term_minus *= temp5

            #cal sig_t_given_fs_sub
            if full_stack_params['kernel'] == 'RBF':
                k_t_t = full_stack_params['kernel_outputscale'] * torch.ones(time_grid.shape[0], dtype=torch.float64)
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
            const_term_minus -= bracket_term_full * 0.5 * full_stack_params['E_omega_complete']
            const_term_minus -= torch.log(torch.tensor(2.0))

            #cal L_E_U_s
            z_minus = - 0.5 * quadratic_term + lin_term_minus + const_term_minus
            z_minus *= full_stack_params['marked_process_intensity_t']
            sum_z_minus = torch.sum(z_minus) / self.ts_helper.time_discretization
            sum_ln_lmbda = full_stack_params['E_ln_lmbda'] * sub_time_grid.shape[0]
            L_E_U_s = sum_z_plus + sum_z_minus + sum_ln_lmbda
            
            
            ### &&&&& PG kl divergence &&&&& ###
            kl_PG = sum(torch.log(torch.cosh(full_stack_params['c_n'] / 2)) - (full_stack_params['c_n_squared'] / 2) * full_stack_params['E_omega_N'])


            ### &&&&& MP kl divergence &&&&& ###
            kl_MP = - torch.sum(full_stack_params['marked_process_intensity_t'] * (1 + full_stack_params['E_ln_lmbda'])) / self.ts_helper.time_discretization
            kl_MP += self.ts_helper.end_time * full_stack_params['E_lmbda']
            kl_MP += torch.sum((torch.log(full_stack_params['marked_process_intensity_t'])) * full_stack_params['marked_process_intensity_t']) / self.ts_helper.time_discretization
            kl_MP += torch.sum(torch.log(torch.cosh(full_stack_params['c_complete'] / 2)) * full_stack_params['marked_process_intensity_t']) / self.ts_helper.time_discretization
            kl_MP -= torch.sum(full_stack_params['c_complete_squared']/2 * full_stack_params['E_omega_complete'] * full_stack_params['marked_process_intensity_t']) / self.ts_helper.time_discretization


            ### &&&&& lmbda kl divergence &&&&& ###
            kl_lmbda = (full_stack_params['alpha_post'] - full_stack_params['alpha_0']) * torch.digamma(full_stack_params['alpha_post'])
            kl_lmbda -= torch.lgamma(full_stack_params['alpha_post']) - torch.lgamma(full_stack_params['alpha_0'])
            kl_lmbda += full_stack_params['alpha_0'] * (torch.log(full_stack_params['beta_post']) - torch.log(full_stack_params['beta_0']))
            kl_lmbda -= (full_stack_params['beta_post'] - full_stack_params['beta_0']) * (full_stack_params['alpha_post'] / full_stack_params['beta_post'])


            ### &&&&& GP kl divergence &&&&& ###
            D = mu_s_0.shape[0]
            tr_term   = torch.sum(inv_K_ss_ELBO * torch.transpose(full_stack_params['cov_post'], 0, 1))
            diff = mu_s_0 - full_stack_params['mu_post']
            quad_term = torch.dot(diff, torch.matmul(inv_K_ss_ELBO , diff))
            # det term 
            chol_prior = torch.linalg.cholesky(K_ss_ELBO)
            log_det_cov_prior =  2 * torch.sum(torch.log(torch.diagonal(chol_prior)))
            chol_post = torch.linalg.cholesky(full_stack_params['cov_post'])
            log_det_cov_post =  2 * torch.sum(torch.log(torch.diagonal(chol_post)))
            det_term = log_det_cov_prior - log_det_cov_post
            kl_GP = 0.5 * (tr_term - D + quad_term + det_term)

            print("\n")
            print("sum_ln_lmbda", sum_ln_lmbda)
            print("z plus", sum_z_plus)
            print("z minus", sum_z_minus)
            print("L_E_U_s in ELBO", L_E_U_s)
            #print("kl_PG", kl_PG)
            #print("kl_lmbda", kl_lmbda)
            #print("kl_MP", kl_MP)
            #print("kl_GP", kl_GP)  

            ### &&&&& ELBO &&&&& ###
            L =  L_E_U_s - kl_lmbda - kl_PG - kl_MP - kl_GP
            global_loss += L

            print(f'L: {L} and the global {global_loss}')

            #print(f'test the globel elbo {global_loss}     {L}')
            
        return - global_loss
    



