from helpers import Helper
import numpy as np
import random
import torch
import gpytorch

class Generator:

    ########## gp prior ##########

    def generate_gp_prior(locations, n):
        locations = torch.tensor(locations[:, 0], dtype=torch.float64)  # Ensure locations is a PyTorch tensor
        gp = torch.zeros((n, locations.shape[0]))  # Initialize gp as a PyTorch tensor

        for i in range(n):
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        ############################################################
            kernel.outputscale = 300.0
            kernel.base_kernel.lengthscale = 6.
        ############################################################

            with torch.no_grad():
                cov_matrix = kernel(locations, locations).evaluate()
                jitter = 1e-6 * torch.eye(locations.size(0))  # Adding a small jitter term
                cov_matrix += jitter
                
            mean = torch.zeros(locations.shape[0])
            mvn = gpytorch.distributions.MultivariateNormal(mean, cov_matrix)
            samples = mvn.sample(sample_shape=torch.Size([1]))
            samples -= torch.mean(samples)
            samples -= 8
            samples = samples.reshape(1, -1)
            scaled_temp = torch.sigmoid(samples)     
            gp[i, :] = scaled_temp[0]
            
        return gp

    ########## poisson process ##########

    def create_poisson_process(input, rate):
        temp_poisson = np.zeros(input.shape)
        sum_lost_events_list = []
        for i in range(0,input.shape[0]):
            timestamps = Helper.poisson_process2(input.shape[1], rate[i])
            temp = Helper.timestamps_to_array(timestamps, input.shape[1])
            temp_poisson[i,:] = temp
            sum_lost_events = int(len(timestamps) - temp.sum())
            sum_lost_events_list.append(sum_lost_events)
            print("We lost", sum_lost_events, "events in diskretization.")
        return temp_poisson, sum_lost_events_list
    
    def create_poisson_timestamps(number_of_processes, rate, T):
        temp = []
        for i in range(number_of_processes):
            timestamps = Helper.generate_poisson_timestamps(rate[i], T)
            temp.append(timestamps)
        return temp


    ######### thinning ##########

    def thinning_process(gp_sample, pois):
        temp = np.zeros(pois.shape)
        sum = 0
        for j in range(pois.shape[1]):
            for i in range(0, pois.shape[0]):
                if pois[i][j] == 1:
                    if random.random() < gp_sample[i][j]:
                        temp[i][j] = 1
                        sum += 1
        return temp, sum


def squared_exponential_kernel(x1, x2, lengthscale=0.5, variance=0.5):
    pairwise_sq_dists = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return variance * np.exp(-0.5 * pairwise_sq_dists / lengthscale**2)

def periodic_kernel(x1, x2, period=2.0, lengthscale=1.0):
    sq_dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return np.exp(-2 * np.sin(np.pi / period * np.sqrt(sq_dist))**2 / lengthscale**2)

def laplacian_kernel(x1, x2, lengthscale=0.5):
    sq_dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return np.exp(-np.sqrt(sq_dist) / lengthscale)


