import numpy as np
from data_generation.data_generator import DataGenerator
import os

end_time = 140
time_discretization = 100
number_of_processes = 2
poisson_rate = 0.5
number_gp_samples = end_time * time_discretization

Data_Generator = DataGenerator(number_gp_samples, end_time, number_of_processes,  poisson_rate, time_discretization,  mu_vdp=0.5)
gp_samples, poisson_process, thinned_process, sum = Data_Generator.generate_data(type=marked)

data_dict = {
    'gp_samples': gp_samples,
    'poisson_process': poisson_process,
    'thinned_process': thinned_process
}
#DataGenerator.plot_generated_data(gp_samples, poisson_process, thinned_process, xlim=5000)
# store data
file_path = os.path.join('data_generation\\generated_data', f'marked_T{end_time*time_discretization}_n{number_of_processes}_mu{poisson_rate}.npz')
np.savez(file_path, **data_dict)
print(f"Data saved to {file_path}")