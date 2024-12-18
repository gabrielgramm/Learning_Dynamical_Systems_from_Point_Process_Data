import time
import torch
import numpy as np
torch.autograd.set_detect_anomaly(True)
import gpytorch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from helpers.helpers import Helper
from helpers.plotter import plot_time_grid

class Time_grid_optimizer2(nn.Module):

    def __init__(self, device, list_vi_objects, ts_helper, time_grid_parameters, 
                 learning_rate_tau, learning_rate_couplings, epochs_time_grid, 
                 gradient_clip_norm, termination_threshold_tg):
        super(Time_grid_optimizer2, self).__init__()
        self.device = device
        self.positive_constraint = gpytorch.constraints.Positive()
        self.list_vi_objects = list_vi_objects
        self.time_grid_parameters = time_grid_parameters
        self.ts_helper = ts_helper.to(device)
        self.epochs_time_grid = torch.tensor(epochs_time_grid, device=device)
        self.learning_rate_tau = torch.tensor(learning_rate_tau, device=device)
        self.learning_rate_couplings = torch.tensor(learning_rate_couplings, device=device)
        self.max_grad_norm = torch.tensor(gradient_clip_norm, device=device)
        self.threshold = torch.tensor(termination_threshold_tg, device=device)
        if isinstance(time_grid_parameters['tau_list'], torch.Tensor):
            self.tau_list = time_grid_parameters['tau_list'].to(device)
        else:
            self.tau_list = torch.tensor(time_grid_parameters['tau_list'], device=device, dtype=torch.float64)
        self.couplings = time_grid_parameters['couplings'].to(device)
        self.couplings = self.couplings.to(torch.float64)
        self.opt_couplings = torch.nn.Parameter(self.couplings, requires_grad=True)
        self.raw_tau_list = torch.nn.Parameter(self.positive_constraint.inverse_transform(self.tau_list), requires_grad=True)

        param_groups = [
            {'params': [self.opt_couplings], 'lr': self.learning_rate_couplings},
            {'params': [self.raw_tau_list], 'lr': self.learning_rate_tau},
        ]
        self.optimizer = optim.Adam(param_groups, foreach=False)

    def set_vi_objects(self, list_vi_objects):
        self.list_vi_objects = list_vi_objects
        #assert isinstance(self.couplings, torch.Tensor), "couplings must be a torch.Tensor"
        #assert isinstance(self.tau_list, torch.Tensor), "tau_list must be a torch.Tensor"
        #self.opt_couplings = torch.nn.Parameter(self.couplings, requires_grad=True)
        #self.raw_tau_list = torch.nn.Parameter(self.positive_constraint.inverse_transform(self.tau_list), requires_grad=True)

    def optimize_time_grid(self, list_vi_objects):

        start_time_opt_time_grid = time.perf_counter()
        previous_loss = 0.
        tolerance_counter = 0
        tolerance_limit = 3

        print(f"\n@@@@@@@@@@@@ time grid optimization ({self.epochs_time_grid} Epochs)  @@@@@@@@@@@@\n")
        #print("     @@ old tau and couplings: @@")
        #print("     tau_list:", [f"{x:.2f}" for x in self.tau_list])
        #print(f"     couplings: [{', '.join([str(list(row)) for row in self.couplings.numpy().round(2)])}]")
        #print(f"     couplings: {self.couplings}")
        #print("\n")

        start_time=time.perf_counter()
        print_first_global_elbo = 0
        for epoch in range(self.epochs_time_grid):
            epoch_start_time=time.perf_counter()
            self.optimizer.zero_grad()
            #global_elbo = self.elbo_all_processes(self.list_vi_objects)
            global_elbo = self.elbo_all_processes()
            if epoch == 0:
                print_first_global_elbo = global_elbo.detach()
            loss = - global_elbo
            loss.backward(retain_graph=True)
            #torch.nn.utils.clip_grad_norm_([self.raw_tau_list], self.max_grad_norm)
            #torch.nn.utils.clip_grad_norm_([self.opt_couplings], self.max_grad_norm)
            #print(f"   cliped gradient couplings: {self.opt_couplings.grad}")
            #print(f"   cliped gradient tau: {self.raw_tau_list.grad}")
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            '''with torch.no_grad():
                self.opt_couplings -= 0.0001 * self.opt_couplings.grad
            self.opt_couplings = self.opt_couplings.detach().requires_grad_()'''

            #update vi
            '''updated_time_grid = self.ts_helper.get_time_grid(self.opt_couplings.clone().detach(), self.raw_tau_list.clone().detach())
            for vi in self.list_vi_objects:
                vi.time_grid = updated_time_grid
                vi.sub_time_grid = torch.index_select(updated_time_grid, 0, vi.thinned_shifted_indices)
                vi.c_complete, vi.c_complete_squared = pg_Helper.get_c_complete_posterior(vi, updated_time_grid)
                vi.c_n, vi.c_n_squared = pg_Helper.get_posterior_c_n(vi.c_complete, vi.c_complete_squared, vi.thinned_shifted_indices)
                vi.E_omega_N = pg_Helper.get_E_omega(vi.c_n)
                vi.E_omega_complete = pg_Helper.get_E_omega(vi.c_complete)
                vi.marked_process_intensity_t = MP_Helper.get_posterior_marked_rate(vi, updated_time_grid, vi.c_complete)
                vi.E_cardinality_MP = MP_Helper.integrage_marked_intensity(vi.end_time, vi.time_discretization, vi.marked_process_intensity_t)
                vi.alpha_post = vi.alpha_0 + vi.cardinality_D + vi.E_cardinality_MP
                vi.beta_post = vi.beta_0 + vi.end_time
                vi.E_lmbda = vi.alpha_post / vi.beta_post
                vi.E_ln_lmbda = torch.digamma(vi.alpha_post) - torch.log(vi.beta_post)
                vi.SGP_post_mean, vi.SGP_post_cov = GP_Helper.get_GP_posterior(vi, vi.inducing_points_s)'''
            loss_change = abs(previous_loss - -loss.item())
            if loss_change < self.threshold:
                tolerance_counter += 1
                if tolerance_counter >= tolerance_limit:
                    if epoch > 5:
                        epoch_end_time=time.perf_counter()
                        #print(f"   @@ Epoch {epoch+1}, Global_ELBO: {global_elbo.item():.2f}, time for epoch: {epoch_end_time - epoch_start_time:.2f}s @@")
                        print(f"   !! Stopping early at epoch {epoch+1} due to minimal change in loss !!")
                        break
            else:
                tolerance_counter = 0
            previous_loss = -loss.item()
            epoch_end_time=time.perf_counter()
            #print(f"   @@ Epoch {epoch+1}, Global_ELBO: {global_elbo.item():.2f}, time for epoch: {epoch_end_time - epoch_start_time:.2f}s @@")
        
        end_time = time.perf_counter()
        print(f"   @@ Global_ELBO from: {print_first_global_elbo.item():.2f} to: {global_elbo.item():.2f}, time used: {end_time - start_time:.2f}s @@\n")
        self.optimizer.zero_grad()
        self.tau_list = self.positive_constraint.transform(self.raw_tau_list.clone().detach().requires_grad_(False))
        self.couplings = self.opt_couplings.clone().detach().requires_grad_(False)
        time_grid = self.ts_helper.get_time_grid(self.couplings.to('cpu'), self.tau_list.to('cpu'))
        
        #print("\n     @@ new tau and couplings: @@")
        #print("     tau_list:", [f"{x:.2f}" for x in self.tau_list])
        #print(f"     couplings: {self.couplings}")
        plot_time_grid(self.ts_helper, time_grid, size=5, start=800, end=6000, show_sub_time_grid=True, process=None)
        print("   the new taus are:", self.tau_list.cpu().numpy())
        print("   the new couplings are:", self.couplings.cpu().numpy())
        print("\n")

        self.time_grid_parameters['tau_list'] = self.tau_list.to('cpu')
        self.time_grid_parameters['couplings'] = self.couplings.to('cpu')

        end_time_opt_time_grid = time.perf_counter()
        #print(f"     time for time grid opt: {end_time_opt_time_grid-start_time_opt_time_grid:.2f}s\n")
        return self.time_grid_parameters, end_time - start_time

    def to_cpu(self):
        for vi in self.list_vi_objects:
            vi.to_device('cpu')
    

    ####### functions for time grid optimization #######

    def get_time_grid_kernel(self, x, tau):
        exp_function = torch.exp(-x / tau)
        exp_function = exp_function * (x >= 0).float()
        ########################################################################################################
        kernel = exp_function
        #kernel = exp_function / (torch.sum(exp_function) / self.ts_helper.time_discretization)   #normalizeation
        ########################################################################################################
        return kernel       

    def convolve(self, data, tau, kernel_effect_length, time_discretization):
        '''if tau < 0:
            raise ValueError("Tau must be greater than 0")
        elif tau < 1:
            opt_kernel_length = kernel_effect_length
            opt_kernel_length = torch.ceil(opt_kernel_length)
        else:
            opt_kernel_length = kernel_effect_length# * torch.log(tau)
            opt_kernel_length = torch.ceil(opt_kernel_length)               
        temp_time_grid = torch.linspace(0, int(opt_kernel_length.item()), int(opt_kernel_length.item()) * time_discretization)'''
        temp_time_grid = torch.linspace(0, kernel_effect_length, kernel_effect_length * time_discretization, device=self.device)
        time_grid_kernel = self.get_time_grid_kernel(temp_time_grid, tau)
        kernel_tensor = time_grid_kernel.unsqueeze(0).unsqueeze(0).double()
        kernel_tensor = kernel_tensor.flip(2)
        sum_exp_function = torch.sum(kernel_tensor)     
        padding = kernel_tensor.shape[2] - 1  # This will pad equally on both sides
        conv = torch.zeros(data.shape[0], data.shape[1] + time_grid_kernel.shape[0] - 1, device=self.device, dtype=torch.float64)
        for i in range(0, data.shape[0]):
            x_tensor = data[i].clone().unsqueeze(0).unsqueeze(0).double().to(self.device)  # Shape: (num_processes, 1, len(x))
            x_padded = F.pad(x_tensor, (padding, padding))
            _1dconv = F.conv1d(x_padded, kernel_tensor)
            conv[i] = _1dconv.squeeze()
        add_zeros = torch.zeros(conv.shape[0], 1, device=self.device, dtype=torch.float64)
        conv = torch.cat((conv, add_zeros), dim=1)
        conv = conv[:, :data.shape[1]]
        return conv, sum_exp_function

    ######## We cal negative ELBO and do descent instead of ascent ########

    def elbo_all_processes(self):

        global_elbo = .0

        ### &&&&& initialization &&&&& ###
        if len(self.raw_tau_list) != self.ts_helper.dim_phase_space:
            raise ValueError("Length of raw_tau_list does not match dim_phase_space.")

        tau_list = self.positive_constraint.transform(self.raw_tau_list)

        time_grid_old = torch.zeros(self.ts_helper.data.shape[1], self.ts_helper.dim_phase_space, dtype=torch.float64, device=self.device)

        for axis in range(0, self.ts_helper.dim_phase_space):
            convolved, sum_exp_function = self.convolve(self.ts_helper.data, tau_list[axis], self.ts_helper.kernel_effect_length, self.ts_helper.time_discretization)
            scaled_coupling = self.opt_couplings[:, axis] #/ (sum_exp_function / self.ts_helper.time_discretization)
            axis_value_old = torch.sum(scaled_coupling.unsqueeze(1) * convolved, dim=0)
            time_grid_old[:, axis] = axis_value_old
        time_grid = time_grid_old
        
        #plot_time_grid(self.ts_helper, time_grid.detach().to('cpu'), size=5, start=200, end=6000, show_sub_time_grid=True, process=None)       

        for process, vi in enumerate(self.list_vi_objects):

            vi.to_device(self.device)

            sub_time_grid = torch.index_select(time_grid, 0, vi.thinned_shifted_indices)

            ### time gird is not float 64 ###
            #plot_time_grid(self.ts_helper, time_grid.detach(), size=4, show_sub_time_grid=True)

            data = torch.roll(vi.thinned_process, shifts=-1).to(self.device)
            data[-1] = 0
            
            mu_0_sub = vi.GP_prior_mean * torch.ones(sub_time_grid.shape[0], dtype=torch.float64, device=self.device)
            mu_0_extended = vi.GP_prior_mean * torch.ones(time_grid.shape[0], dtype=torch.float64, device=self.device)
            mu_s_0 = vi.GP_prior_mean * torch.ones(vi.inducing_points_s.shape[0], dtype=torch.float64, device=self.device)
            
            #kernel initialization     
            if vi.kernel == 'RBF':
                kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(self.device)
                kernel.outputscale = vi.kernel_outputscale
                kernel.base_kernel.lengthscale = vi.kernel_lengthscale
            else:
                raise ValueError("Kernel not implemented")

            #cal K_ss and inv_K_ss
            kernel_matrix = kernel(vi.inducing_points_s)
            K_ss_ELBO = kernel_matrix.evaluate()

            #with jitter
            K_ss_ELBO = K_ss_ELBO + torch.eye(K_ss_ELBO.size(0), dtype=K_ss_ELBO.dtype, device=K_ss_ELBO.device) * 1e-6
            inv_K_ss_ELBO = torch.inverse(K_ss_ELBO)
            if torch.isnan(inv_K_ss_ELBO).any():
                raise ValueError("The calculated Inverse of the inducing points contains NaN values")
        
            #with cholensky
            '''
            LL = torch.linalg.cholesky(K_ss_ELBO)
            identity = torch.eye(K_ss_ELBO.size(0), dtype=K_ss_ELBO.dtype, device=K_ss_ELBO.device)
            inv_K_ss_ELBO = torch.cholesky_solve(identity, LL)
            if torch.isnan(inv_K_ss_ELBO).any():
                raise ValueError("The calculated Inverse of the inducing points contains NaN values")
            log_det = 2 * torch.sum(torch.log(torch.diag(LL)))
            '''

            #kappafull
            kernel_matrix_full = kernel(time_grid, vi.inducing_points_s)
            k_x_t__x_s_full = kernel_matrix_full.evaluate()
            #kappa_f_full = k_x_t__x_s_full
            kappa_f_full = torch.matmul(k_x_t__x_s_full, inv_K_ss_ELBO)
            kappa_b_full = torch.transpose(kappa_f_full, 0, 1)

            #kappa_sub
            kernel_matrix_sub = kernel(sub_time_grid, vi.inducing_points_s)
            k_x_t__x_s_sub = kernel_matrix_sub.evaluate()
            #kappa_f_sub = k_x_t__x_s_sub
            kappa_f_sub = torch.matmul(k_x_t__x_s_sub, inv_K_ss_ELBO)
            kappa_b_sub = torch.transpose(kappa_f_sub, 0, 1)


            ''' &&&&& z_plus &&&&& '''
            #cal quadratic term
            sec_mom = vi.SGP_post_cov + torch.ger(vi.SGP_post_mean, vi.SGP_post_mean)
            #quadratic_term_plus = torch.sum((torch.matmul(kappa_f_sub, sec_mom)) * (kappa_b_sub.transpose(0,1)), dim=1)
            quadratic_term_plus = torch.sum((torch.matmul(kappa_f_full, sec_mom)) * (kappa_b_full.transpose(0,1)), dim=1)
            quadratic_term_plus *= vi.E_omega_N
            quadratic_term_plus *= data

            #cal linear term
            test = torch.matmul(kappa_f_sub, vi.SGP_post_mean)
            #mu_s_0_kappa_sub = torch.matmul(mu_s_0, kappa_b_sub)
            #lin_term_plus = mu_0_sub - mu_s_0_kappa_sub
            mu_s_0_kappa_full = torch.matmul(mu_s_0, kappa_b_full)
            lin_term_plus = mu_0_extended - mu_s_0_kappa_full
            lin_term_plus = lin_term_plus * vi.E_omega_N
            lin_term_plus = 0.5 - lin_term_plus
            #temp2 = torch.matmul(kappa_f_sub, vi.SGP_post_mean)
            temp2 = torch.matmul(kappa_f_full, vi.SGP_post_mean)
            lin_term_plus *= temp2
            lin_term_plus *= data

            #cal sig_t_given_fs_sub
            if vi.kernel == 'RBF':
                #k_t_t = vi.kernel_outputscale * torch.ones(sub_time_grid.shape[0], dtype=torch.float64, device=self.device)# * sub_time_grid[:,0] #this was a test
                k_t_t_full = vi.kernel_outputscale * torch.ones(time_grid.shape[0], dtype=torch.float64, device=self.device)
            else:
                raise ValueError("Kernel not implemented")
            #sigma_t_given_fs_sub = k_t_t - torch.sum(kappa_f_sub * k_x_t__x_s_sub, dim=1)
            sigma_t_given_fs_full = k_t_t_full - torch.sum(kappa_f_full * k_x_t__x_s_full, dim=1)
            '''
            #cal bracket_term
            bracket_term = (sigma_t_given_fs_sub 
                        + torch.pow(mu_0_sub, 2) 
                        - 2 * mu_0_sub * mu_s_0_kappa_sub 
                        + torch.pow(mu_s_0_kappa_sub, 2))


            #cal constant term
            const_term_plus = (0.5 * mu_0_sub 
                        - 0.5 * mu_s_0_kappa_sub 
                        - bracket_term * 0.5 * vi.E_omega_N 
                        - torch.log(torch.tensor(2.0, device=self.device)))
            '''
                        #cal bracket_term
            bracket_term = (sigma_t_given_fs_full 
                        + torch.pow(mu_0_extended, 2) 
                        - 2 * mu_0_extended * mu_s_0_kappa_full 
                        + torch.pow(mu_s_0_kappa_full, 2))


            #cal constant term
            const_term_plus = (0.5 * mu_0_extended 
                        - 0.5 * mu_s_0_kappa_full 
                        - bracket_term * 0.5 * vi.E_omega_N 
                        - torch.log(torch.tensor(2.0, device=self.device)))
            const_term_plus *= data

            z_plus = - 0.5 * quadratic_term_plus + lin_term_plus + const_term_plus
            sum_z_plus = torch.sum(z_plus)


            ''' &&&&& z_minus &&&&& '''
            #cal quadratic term
            sec_mom = vi.SGP_post_cov + torch.ger(vi.SGP_post_mean, vi.SGP_post_mean)
            quadratic_term_minus = torch.sum((torch.matmul(kappa_f_full, sec_mom)) * (kappa_f_full), dim=1)
            quadratic_term_minus = quadratic_term_minus * vi.E_omega_complete

            #cal linear term
            mu_s_0_kappa_b_full = torch.matmul(mu_s_0, kappa_b_full)
            lin_term_minus = mu_0_extended - mu_s_0_kappa_b_full
            lin_term_minus = lin_term_minus * vi.E_omega_complete
            lin_term_minus = - 0.5 - lin_term_minus
            temp5 = torch.matmul(kappa_f_full, vi.SGP_post_mean)
            lin_term_minus = lin_term_minus * temp5

            #cal sig_t_given_fs_sub
            if vi.kernel == 'RBF':
                k_t_t = vi.kernel_outputscale * torch.ones(time_grid.shape[0], dtype=torch.float64, device=self.device)
            else:
                raise ValueError("Kernel not implemented")
            sigma_t_given_fs_full = k_t_t - torch.sum(kappa_f_full * k_x_t__x_s_full, dim=1)

            #cal bracket_term
            bracket_term_full = (sigma_t_given_fs_full 
                                + torch.pow(mu_0_extended, 2) 
                                - 2 * mu_0_extended * mu_s_0_kappa_b_full 
                                + torch.pow(mu_s_0_kappa_b_full, 2))

            #cal constant term
            const_term_minus = (0.5 * mu_0_extended 
                - 0.5 * mu_s_0_kappa_b_full 
                - bracket_term_full * 0.5 * vi.E_omega_complete 
                - torch.log(torch.tensor(2.0, device=self.device)))

            #cal L_E_U_s
            pre_z_minus = - 0.5 * quadratic_term_minus + lin_term_minus + const_term_minus
            z_minus = pre_z_minus * vi.marked_process_intensity_t
            sum_z_minus = torch.sum(z_minus) / self.ts_helper.time_discretization

            #sum_ln_lmbda = vi.E_ln_lmbda * sub_time_grid.shape[0]
            sum_ln_lmbda = vi.E_ln_lmbda * torch.sum(data)

            L_E_U_s = sum_z_plus + sum_z_minus + sum_ln_lmbda            
            
            ### &&&&& PG kl divergence &&&&& ###
            '''kl_PG = sum(torch.log(torch.cosh(vi.c_n / 2)) - (vi.c_n_squared / 2) * vi.E_omega_N)


            ### &&&&& MP kl divergence &&&&& ###
            kl_MP = - torch.sum(vi.marked_process_intensity_t * (1 + vi.E_ln_lmbda)) / self.ts_helper.time_discretization
            kl_MP += self.ts_helper.end_time * vi.E_lmbda
            kl_MP += torch.sum((torch.log(vi.marked_process_intensity_t)) * vi.marked_process_intensity_t) / self.ts_helper.time_discretization
            kl_MP += torch.sum(torch.log(torch.cosh(vi.c_complete / 2)) * vi.marked_process_intensity_t) / self.ts_helper.time_discretization
            kl_MP -= torch.sum(vi.c_complete_squared/2 * vi.E_omega_complete * vi.marked_process_intensity_t) / self.ts_helper.time_discretization


            ### &&&&& lmbda kl divergence &&&&& ###
            kl_lmbda = (vi.alpha_post - vi.alpha_0) * torch.digamma(vi.alpha_post)
            kl_lmbda -= torch.lgamma(vi.alpha_post) - torch.lgamma(vi.alpha_0)
            kl_lmbda += vi.alpha_0 * (torch.log(vi.beta_post) - torch.log(vi.beta_0))
            kl_lmbda -= (vi.beta_post - vi.beta_0) * (vi.alpha_post / vi.beta_post)


            ### &&&&& GP kl divergence &&&&& ###
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
            kl_GP = 0.5 * (tr_term - D + quad_term + det_term)'''

            '''print("\n")
            print("sum_ln_lmbda", sum_ln_lmbda)
            print("z plus", sum_z_plus)
            print("z minus", sum_z_minus)
            print("L_E_U_s in ELBO", L_E_U_s)
            #print("kl_PG", kl_PG)
            #print("kl_lmbda", kl_lmbda)
            #print("kl_MP", kl_MP)
            #print("kl_GP", kl_GP)'''

            ### &&&&& ELBO &&&&& ###
            L =  L_E_U_s #- kl_lmbda - kl_PG - kl_MP - kl_GP
            #print(f'time_grid opt L: {L}')
            global_elbo = global_elbo + L

            #loss = loss + torch.sum(quadratic_term_plus) + torch.sum(quadratic_term_minus)

            #print(f'L: {L} and the global {global_loss}')
            #torch.sum(kappa_b_full) + torch.sum(kappa_b_sub)
        return global_elbo
    





