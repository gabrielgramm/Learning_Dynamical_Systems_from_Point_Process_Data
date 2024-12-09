import torch
import matplotlib.pyplot as plt
from data_generation.generative_model import Generator
from data_generation.van_der_pol import *
from helpers.helpers import Helper
from helpers.helpers import no_grad_method
'''
@no_grad_method
def generate_data_3(self):  
    sample_locations = self.get_sample_locations_2d()
    gp_samples = Generator.generate_gp_prior(sample_locations, self.number_of_processes)
    poisson_process, sum_lost_events = Generator.create_poisson_process(gp_samples, self.poisson_rate)
    thinned_process, sum = Generator.thinning_process(gp_samples, poisson_process)
'''
class DataGenerator:

    def __init__(self, num_gp_samples, end_time, number_of_processes, poisson_rate = 0.1, time_discretization=100,  mu_vdp=0.5, start_point = [0.5, 2]):
        self.generator = Generator()
        self.number_of_processes = number_of_processes
        self.end_time = end_time
        self.time_discretization = time_discretization
        self.n_timesteps = self.end_time * self.time_discretization
        self.num_gp_samples = num_gp_samples

        self.poisson_rate = [poisson_rate] * number_of_processes
        self.mu_vdp = mu_vdp
        self.start_point = start_point

    @no_grad_method
    def interpolate_gp_samples(self, gp_samples):
        x_old = np.linspace(0, self.end_time, num=self.num_gp_samples)
        x_new = np.linspace(0, self.end_time, num=self.end_time * self.time_discretization)
        interpolated = []
        for i in range(gp_samples.shape[0]):
            temp = np.interp(x_new, x_old, gp_samples[i])
            interpolated.append(temp)
        return np.array(interpolated)

    @no_grad_method
    def get_sample_locations_2d(self):
        vdp = generate_vdp(self.num_gp_samples, self.end_time, self.start_point, self.mu_vdp)
        x1,x2 = vdp.y[0], vdp.y[1]
        sample_locations = np.vstack([x1, x2]).T
        return sample_locations
    
    @no_grad_method
    def get_sample_locations_lorenz(self):

        def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
            x, y, z = state
            dxdt = sigma * (y - x)
            dydt = x * (rho - z) - y
            dzdt = x * y - beta * z
            return [dxdt, dydt, dzdt]

        initial_state = [1.0, 1.0, 10.0]
        t_span = (0, self.end_time) 
        t_eval = np.linspace(t_span[0], t_span[1], self.num_gp_samples)

        solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
        x = solution.y[0]
        y = solution.y[1]
        z = solution.y[2]
        trajectory = np.vstack([x, y, z]).T
        
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Trajectory")
        plt.show()
        '''

        return trajectory
    
    @no_grad_method
    def generate_data(self):       
        sample_locations = self.get_sample_locations_lorenz()
        gp_samples = Generator.generate_gp_prior(sample_locations, self.number_of_processes)
        poisson_process, sum_lost_events = Generator.create_poisson_process(gp_samples, self.poisson_rate)
        thinned_process, sum = Generator.thinning_process(gp_samples, poisson_process)

        #scaling by lambda_bar
        for i in range(gp_samples.shape[0]):
            lost_events_rate = sum_lost_events[i] / (self.end_time * self.time_discretization)
            lambda_bar = self.time_discretization * (self.poisson_rate[i] - lost_events_rate)
            gp_samples[i] = gp_samples[i] * lambda_bar
        return gp_samples, poisson_process, thinned_process, sum
    
    @no_grad_method
    def generate_data_2(self):       
        sample_locations = self.get_sample_locations_2d()
        gp_samples = Generator.generate_gp_prior(sample_locations, self.number_of_processes)
        timestamps = Generator.create_poisson_timestamps(self.number_of_processes, self.poisson_rate, self.end_time)
        interpolated = self.interpolate_gp_samples(gp_samples)
        thinned_process, sum = Generator.thinning_process(gp_samples, interpolated)
        return timestamps, gp_samples, thinned_process
    
    @no_grad_method
    def generate_data_3(self):  
        sample_locations = self.get_sample_locations_2d()
        gp_samples = Generator.generate_gp_prior(sample_locations, self.number_of_processes)
        poisson_process, sum_lost_events = Generator.create_poisson_process(gp_samples, self.poisson_rate)
        thinned_process, sum = Generator.thinning_process(gp_samples, poisson_process)

        #scaling by lambda_bar
        for i in range(gp_samples.shape[0]):
            lost_events_rate = sum_lost_events[i] / (self.end_time * self.time_discretization)
            lambda_bar = self.time_discretization * (self.poisson_rate[i] - lost_events_rate)
            gp_samples[i] = gp_samples[i] * lambda_bar
        return gp_samples, poisson_process, thinned_process, sum
    
    @no_grad_method
    def print_data_status(gp_samples, poisson_process, thinned_process):
        print("####### Data loaded successfully. ####### ",
              "\n Data Status:",
              "\n  num_processes: ", gp_samples.shape[0], 
              "\n  num_timesteps: ", gp_samples.shape[1],
              "\n sum_of_first_poisson_process: ", poisson_process[0].sum(),
              "\n sum_of_first_thinned_process: ", thinned_process[0].sum())
        
    @no_grad_method
    def plot_generated_data(gp_samples, poisson_process, thinned_process, xlim=1000):
        thinned_indices = []
        for i in range(thinned_process.shape[0]):
            indices = Helper.get_indices_1d(thinned_process[i])
            shifted_indices = Helper.shift_indices(indices)
            thinned_indices.append(shifted_indices)

        if gp_samples.shape[0] == 1:
            axs = [axs]

        fig, axs = plt.subplots(gp_samples.shape[0], 1, figsize=(8, gp_samples.shape[0] * 1))
        for i in range(gp_samples.shape[0]):           
            ax = axs[i]  # Select the current axis
            ax.plot(gp_samples[i], color='black', linewidth=.5)  # Plot the true rate
            # ax.scatter(list_vi_objects[0].test_thinned_indices, np.zeros(len(list_vi_objects[0].test_thinned_indices)), marker=".", alpha=.5, color='black')
            ax.scatter(thinned_indices[i], np.zeros(len(thinned_indices[i])), marker=".", alpha=.5,color='black', s=3)
            ax.tick_params(axis='both', labelsize=7)  # Set tick label size
            ax.set_xlim(0, xlim)  # Set x-axis limit
        plt.tight_layout()
        plt.show()

