import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from helpers.helpers import Helper
from helpers.helpers import no_grad_method

class TS_Helper(nn.Module):

    def __init__(self, full_data, full_true_rates, dim_phase_space, num_processes, end_time, kernel_effect_length=1, time_discretization=100):
        super(TS_Helper, self).__init__()        
        self.full_true_rates = full_true_rates
        self.time_discretization = time_discretization
        self.kernel_effect_length = torch.tensor(kernel_effect_length)
        self.num_processes = num_processes
        self.dim_phase_space = dim_phase_space
        self.end_time = end_time

        # cut data as test set
        self.full_data = full_data
        self.data = self.full_data[:,:self.end_time * self.time_discretization]
        self.test_data = self.full_data[:,self.end_time * self.time_discretization:]
        self.full_true_rates = full_true_rates
        self.true_rates = self.full_true_rates[:,:self.end_time * self.time_discretization]
        self.test_true_rates = self.full_true_rates[:,self.end_time * self.time_discretization:]

    @no_grad_method
    def get_h(self, x, tau):
        exp_function = torch.exp(-x/tau)# / tau
        exp_function = exp_function * (x >= 0).float()
        kernel = exp_function

        ''' 
        if you want to normalize/unnormlaize the kernel, you also 
            need to change in time_grid_optimizer2 and there not only in the get_h function
            but also in the for axis loop there is a scaling of the couplings  
        '''
        ###############################################################################################
        #kernel = exp_function / (torch.sum(exp_function) / self.time_discretization)   #normalization
        ###############################################################################################

        #plt.figure(figsize=(3, 3))
        #plt.plot(exp_function)
        #plt.plot(kernel)
        #plt.show()
        return kernel       

    @no_grad_method
    def convolve(self, data, tau):
        '''if tau < 0:
            raise ValueError("Tau must be greater than 0")
        elif tau < 1:
            l = self.kernel_effect_length
            l = torch.ceil(l)
        else:
            l = self.kernel_effect_length# + torch.log(tau)
            l = torch.ceil(l)'''
        temp_time_grid = torch.linspace(0, self.kernel_effect_length, self.kernel_effect_length * self.time_discretization)
        h = self.get_h(temp_time_grid, tau)
        #h_tensor = torch.tensor(h, dtype=torch.float64).clone().detach().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, len(h))
        h_tensor = h.unsqueeze(0).unsqueeze(0).double()
        h_tensor = h_tensor.flip(2)
        padding = h_tensor.shape[2] - 1  # This will pad equally on both sides
        conv = torch.zeros(data.shape[0], data.shape[1] + h.shape[0] - 1, dtype=torch.float64)
        for i in range(0, data.shape[0]):
            x_tensor = data[i].clone().detach().unsqueeze(0).unsqueeze(0).double()  # Shape: (num_processes, 1, len(x))
            x_padded = F.pad(x_tensor, (padding, padding))
            _1dconv = F.conv1d(x_padded, h_tensor)
            conv[i] = _1dconv.squeeze()
        add_zeros = torch.zeros(conv.shape[0], 1)
        conv = torch.cat((conv, add_zeros), dim=1)
        cut = int(conv.shape[1] - h.shape[0])
        conv = conv[:, :cut]
        '''
        #plot covolution
        fig, axs = plt.subplots(conv.shape[0], 1, figsize=(10, 4))
        for i in range(conv.shape[0]):
            axs[i].plot(conv[i], marker='o', color='black', linestyle='-', markersize=3)
            axs[i].grid(True)
            axs[i].set_xlim([0, 400])
        plt.show()
        '''
        return conv
    
    @no_grad_method
    def get_axis_of_phase_space(self, data, tau, axis, couplings):
        convolved = self.convolve(data, tau)
        axis_value = torch.sum(couplings[:, axis].unsqueeze(1) * convolved, dim=0)
        return axis_value
    
    @no_grad_method
    def get_time_grid(self, couplings, tau_list):
        time_grid = torch.zeros(self.data.shape[1], self.dim_phase_space, dtype=torch.float64)
        for axis in range(0, self.dim_phase_space):
            if axis < len(tau_list):
                axis_value = self.get_axis_of_phase_space(self.data, tau_list[axis], axis, couplings)
                time_grid[:, axis] = axis_value
            else:
                raise ValueError("The number of tau values should be equal to the dimension of the phase space")
        time_grid.to(torch.float64)
        return time_grid

    @no_grad_method
    def get_test_time_grid(self, couplings, tau_list):
        test_time_grid = torch.zeros(self.full_data.shape[1], self.dim_phase_space, dtype=torch.float64)
        for axis in range(0, self.dim_phase_space):
            if axis < len(tau_list):
                axis_value = self.get_axis_of_phase_space(self.full_data, tau_list[axis], axis, couplings)
                test_time_grid[:, axis] = axis_value
            else:
                raise ValueError("The number of tau values should be equal to the dimension of the phase space")
        test_time_grid.to(torch.float64)
        test_time_grid = test_time_grid[self.end_time * self.time_discretization:]
        return test_time_grid
    
    @no_grad_method
    def get_sub_time_grid_one_process(self, indices, couplings, tau_list):
        time_grid = self.get_time_grid(couplings, tau_list)
        subset_time_grid = torch.index_select(time_grid, 0, index=indices)
        return subset_time_grid    

    # plotter
    @no_grad_method
    def plot_time_grid(self, time_grid, sub_time_grid, start, end):
        plt.figure(figsize=(8, 8))
        plt.title('Points in Phase Space')
        plt.plot(time_grid[start:end, 0], time_grid[start:end, 1], marker='o', color='black', linestyle=':', markersize=3, alpha=0.3)  
        if sub_time_grid is not None:
            #plt.plot(sub_time_grid[start:end, 0], sub_time_grid[start:end, 1], marker='o', color='red', markersize=4)
            plt.scatter(sub_time_grid[start:end, 0], sub_time_grid[start:end, 1], color='red', s=40)
        plt.grid(True)
        plt.show()

    @no_grad_method
    def plot_time_axis(self, true_rate, start, end):
        plt.figure(figsize=(10, 2))
        plt.title('Points in Time Space')
        time_grid = self.get_time_grid()
        data = self.data

        # Generate a color for each iteration
        cmap = plt.get_cmap('Paired')
        colors = [cmap((i + 0.5) / data.shape[0]) for i in range(data.shape[0])]
        s = [30 - 10 * i for i in range(data.shape[0])]
        for i in range (0, data.shape[0]):
            thinned_indices = Helper.get_indices_1d(data[i])
            thinned_shifted_indices = Helper.shift_indices(thinned_indices)
            thinned_shifted_indices = thinned_shifted_indices[thinned_shifted_indices < end]
            helper_zeros = np.zeros(len(thinned_shifted_indices))
            plt.plot(true_rate[i, start:end], color='black')
            plt.scatter(thinned_shifted_indices[start:end], helper_zeros[start:end], color=colors[i], s=s[i], alpha=0.5)

        plt.grid(True)
        plt.show()

        true_rate = torch.sigmoid(true_rate * 1.5) - 0.1
        fig, axs = plt.subplots(data.shape[0], sharex=True, sharey=True, figsize=(10, 1*data.shape[0]))
        fig.suptitle('True Rate and Measured Points in Time Space')
        plt.subplots_adjust(hspace=0.3)
        for i in range(data.shape[0]):
            axs[i].plot(true_rate[i], color='black')
            thinned_indices = Helper.get_indices_1d(data[i])
            thinned_shifted_indices = Helper.shift_indices(thinned_indices)
            thinned_shifted_indices = thinned_shifted_indices[thinned_shifted_indices < end]
            helper_zeros = np.zeros(len(thinned_shifted_indices))
            axs[i].scatter(thinned_shifted_indices[start:end], helper_zeros[start:end], marker=".", alpha=.5,color=colors[i])
        axs[i].set_xlabel("time axis")
        plt.xlim(start,end)
        plt.show()

    @no_grad_method
    def plot_complete_time_grid(self, start, end):
        plt.figure(figsize=(8, 8))
        plt.title('Points in Phase Space')
        time_grid = self.get_time_grid()
        data = self.data
        plt.plot(time_grid[start:end, 0], time_grid[start:end, 1], marker='o', color='black', linestyle=':', markersize=3, alpha=0.3)  

        # Generate a color for each iteration
        cmap = plt.get_cmap('Paired')
        colors = [cmap((i + 0.5 ) / data.shape[0]) for i in range(data.shape[0])]
        s = [50 - 20 * i for i in range(data.shape[0])]
        for i in range (0, data.shape[0]):
            thinned_indices = Helper.get_indices_1d(data[i])
            thinned_shifted_indices = Helper.shift_indices(thinned_indices)
            thinned_shifted_indices = thinned_shifted_indices[thinned_shifted_indices < end]
            sub_time_grid = self.get_sub_time_grid_one_process(thinned_shifted_indices)
            plt.scatter(sub_time_grid[start:end, 0], sub_time_grid[start:end, 1], color=colors[i], s=s[i], alpha=0.5)
        plt.grid(True)
        plt.show()
