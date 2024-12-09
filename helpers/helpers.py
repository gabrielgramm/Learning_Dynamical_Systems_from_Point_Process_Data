import numpy as np
import torch
import gpytorch
from sklearn.cluster import KMeans

def no_grad_method(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper
class Helper:

    def fitted_inhomogeneous_poisson_log_likelihood(vi):
        time_discretization = vi.ts_helper.time_discretization
        thinned_indices = vi.thinned_indices
        fitted_rate = vi.posterior_rate
        
        integral_term = torch.sum(fitted_rate) / time_discretization
        sum_term = torch.sum(torch.log(fitted_rate[thinned_indices]))
        log_llh = sum_term - integral_term   
        return log_llh.item()
    
    def true_fitted_inhomogeneous_poisson_likelihood(vi, true_rate):
        time_discretization = vi.ts_helper.time_discretization
        thinned_indices = vi.thinned_indices

        true_rate = torch.tensor(true_rate) if not isinstance(true_rate, torch.Tensor) else true_rate
        integral_term_true = torch.sum(true_rate) / time_discretization
        true_log_llh = torch.sum(torch.log(true_rate[thinned_indices])) - integral_term_true    
        return true_log_llh.item()

    def test_inhomogeneous_poisson_log_likelihood(vi, the_rate_to_test=None):
        time_discretization = vi.ts_helper.time_discretization
        test_thinned_indices = vi.test_thinned_indices
        test_thinned_indices = test_thinned_indices - 1
        if the_rate_to_test is None:
            test_fitted_rate = vi.test_posterior_rate
        else:
            test_fitted_rate = the_rate_to_test
            test_thinned_indices = test_thinned_indices[test_thinned_indices < len(test_fitted_rate)]
        integral_term = torch.sum(test_fitted_rate) / time_discretization
        sum_term = torch.sum(torch.log(test_fitted_rate[test_thinned_indices]))
        log_llh = sum_term - integral_term   
        return log_llh.item()
    
    def true_test_inhomogeneous_poisson_likelihood(vi, true_rate):
        time_discretization = vi.ts_helper.time_discretization
        test_thinned_indices = vi.test_thinned_indices
        #test_thinned_indices = test_thinned_indices - 1

        true_rate = torch.tensor(true_rate) if not isinstance(true_rate, torch.Tensor) else true_rate
        integral_term_true = torch.sum(true_rate) / time_discretization
        true_log_llh = torch.sum(torch.log(true_rate[test_thinned_indices])) - integral_term_true    
        return true_log_llh.item()
    
    def get_meshgrid(num_inducing_points_per_dim, scale, dim):
        # Generate points along each axis
        axes = [np.linspace(0, scale, num_inducing_points_per_dim) for _ in range(dim)]
        # Create a meshgrid for the specified dimension
        mesh = np.meshgrid(*axes)
        # Flatten the grid arrays to get a list of points
        inducting_points = np.vstack([m.ravel() for m in mesh]).T
        inducting_points = torch.tensor(inducting_points, dtype=torch.float64)
        return inducting_points
    
    def get_meshgrid_scaled(num_inducing_points_per_dim, time_grid, dim):
        min_values, _ = time_grid.min(dim=0)
        min_values_list = min_values.tolist()
        max_values, _ = time_grid.max(dim=0)
        max_values_list = max_values.tolist()
        # Generate points along each axis
        axes = [np.linspace(min_values_list[i], max_values_list[i], num_inducing_points_per_dim) for i in range(dim)]
        # Create a meshgrid for the specified dimension
        mesh = np.meshgrid(*axes)
        # Flatten the grid arrays to get a list of points
        inducting_points = np.vstack([m.ravel() for m in mesh]).T
        inducting_points = torch.tensor(inducting_points, dtype=torch.float64)
        return inducting_points

    def get_indices(poisson):
        indices = []
        for i in range(0, poisson.shape[0]):
            temp_ind = np.where(poisson[i,:] == 1)
            temp_ind = list(temp_ind)
            indices.append(temp_ind[0][:])
        return indices
    
    def get_indices_1d(array):
        indices = [index for index, value in enumerate(array) if value == 1]
        return torch.tensor(indices)
    
    def shift_indices(thinned_indices):
        shifted_indices = thinned_indices - 1
        if shifted_indices[0] < 0:
            shifted_indices[0] = 0
        if len(shifted_indices) > 1 and shifted_indices[0] == 0 and shifted_indices[1] == 0:
            print("       Shifting of indices occurs in two events at index 0.")
        if len(shifted_indices) != len(thinned_indices):
            print("       Warning: Some indices were shifted to negative values and were removed.")
        return shifted_indices

    def timestamps_to_array(timestamps, interval):
        temp = np.zeros(interval)
        for i in range(len(timestamps)):
            temp[int(timestamps[i])] = 1
        return temp

    def array_to_timestamps(array):
        temp = []
        for i in range(len(array)):
            if array[i] == 1:
                temp.append(i)
        return temp

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def get_helper_zeros(indices):
        helper_zeros = []
        for i in range(len(indices)):
            temp = np.zeros(len(indices[i]))
            helper_zeros.append(list(temp))
        return helper_zeros

    def poisson_process(interval, rate):
        rand = np.random.rand(interval)
        temp = np.zeros(interval)
        for i in range(interval):
            if rand[i] < rate:
                temp[i] = 1
        return temp

    def poisson_process2(interval, rate):
        events = rate * interval
        number_of_events = np.random.poisson(events, size=100)
        pick_one_poisson = number_of_events[np.random.randint(0, number_of_events.shape[0])]
        print("We samples", pick_one_poisson, "events from the poisson distribution.")
        timestamps = np.random.rand(pick_one_poisson) * interval
        return timestamps
    
    def generate_poisson_timestamps(rate, T):
        current_time = np.float64(0.0)
        timestamps = []
        while current_time < T:
            inter_arrival_time = np.float64(np.random.exponential(1.0 / rate))
            current_time = np.float64(current_time + inter_arrival_time)
            if current_time < T:
                timestamps.append(current_time)     
        return np.array(timestamps, dtype=np.float64)
    
    def test_inverse(K_ss, inv_K_ss, tolerance=1e-8):        
        identity_approx = torch.matmul(K_ss, inv_K_ss)
        identity = torch.eye(K_ss.size(0), dtype=torch.float64)
        max_deviation = torch.max(torch.abs(identity - identity_approx)).item()    
        if max_deviation > tolerance:
            print(f"       Warning: Maximum deviation from identity matrix is {max_deviation}, which is greater than the tolerance {tolerance}.")

    def test_positive_semi_definite(matrix):
        if matrix.ndim != 2 or matrix.size(0) != matrix.size(1):
            raise ValueError("Input must be a 2D square matrix")   
        # Check if the matrix is symmetric
        if not torch.equal(matrix, matrix.T):
            test = matrix -matrix.T
            print("       !!! kernel is not symmetric !!!")
            print("       max deviation: ", torch.max(torch.abs(test)))
        else:
            print("       is symmetric :)")
        eigenvalues = torch.linalg.eigvalsh(matrix)
        if torch.all(eigenvalues >= 0):
            print("       is positive semi-definite :)")
        else:
            print("       !!! kernel is not positive semi-definite !!!")

    def test_is_same(x, y):
        if torch.equal(x, y):
            print("       The two tensors are the same.")
        else:
            print("       the max dist is: ", torch.max(torch.abs(x - y)))

    def logdet(matrix):
        # Use Cholesky decomposition for numerical stability
        chol = torch.linalg.cholesky(matrix)
        return 2 * torch.sum(torch.log(torch.diagonal(chol)))

    def collect_hyperparamters(vi):
        hyperparameters = {
            'kernel_outputscale': vi.kernel_outputscale,
            'kernel_lengthscale': vi.kernel_lengthscale,
            'alpha_0': vi.alpha_0,
            'beta_0': vi.beta_0,
            'GP_prior_mean': vi.GP_prior_mean,
            'inducing_points_s': vi.inducing_points_s,
            'tau_list': vi.tau_list,
            'couplings': vi.couplings
        }
        return hyperparameters
    
    def collect_full_stack_params(vi):
        full_stack_params = {
            # optimality parameters
            "alpha_post": vi.alpha_post,
            "beta_post": vi.beta_post,

            "mu_post": vi.SGP_post_mean,
            "cov_post": vi.SGP_post_cov,

            'c_n': vi.c_n,
            'c_n_squared': vi.c_n_squared,
            'c_complete': vi.c_complete,
            'c_complete_squared': vi.c_complete_squared,
            'E_omega_N': vi.E_omega_N,
            'E_omega_complete': vi.E_omega_complete,

            'marked_process_intensity_t': vi.marked_process_intensity_t,

            # prior parameters
            "alpha_0": vi.alpha_0,
            "beta_0": vi.beta_0,
            'E_lmbda': vi.E_lmbda,
            'E_ln_lmbda': vi.E_ln_lmbda,
            "GP_prior_mean": vi.GP_prior_mean,

            # hyperparameters
            'kernel_outputscale': vi.kernel_outputscale,
            'kernel_lengthscale': vi.kernel_lengthscale,
            'alpha_0': vi.alpha_0,
            'beta_0': vi.beta_0,
            'GP_prior_mean': vi.GP_prior_mean,
            'inducing_points_s': vi.inducing_points_s,

            #kernel parameters
            'kernel': vi.kernel,
            'kernel_outputscale': vi.kernel_outputscale,
            'kernel_lengthscale': vi.kernel_lengthscale,

            #data params
            'thinned_shifted_indices': vi.thinned_shifted_indices,

            #time grid parameters
            #'tau_list': vi.tau_list,
            #'couplings': vi.couplings
        }
        return full_stack_params
    
    def set_hyperparameters_ELBO(vi, hyperparameters):
        vi.kernel_outputscale = hyperparameters['kernel_outputscale']
        vi.kernel_lengthscale = hyperparameters['kernel_lengthscale']
        vi.alpha_0 = hyperparameters['alpha_0']
        vi.beta_0 = hyperparameters['beta_0']
        vi.GP_prior_mean = hyperparameters['GP_prior_mean']
        vi.inducing_points_s = hyperparameters['inducing_points_s']

    def set_time_grid_params(list_vi_objects, time_grid_params):
        for vi in list_vi_objects:
            vi.tau_list = time_grid_params['tau_list']
            vi.couplings = time_grid_params['couplings']
            vi.time_grid = vi.ts_helper.get_time_grid(vi.couplings, vi.tau_list)
            vi.sub_time_grid = torch.index_select(vi.time_grid, 0, vi.thinned_shifted_indices)
    
    def print_parameters(dictionary):
        for key, value in dictionary.items():
            if key == 'inducing_points_s':
                if value == None:
                    print(f"{key}: None")
                else:
                    print(f"{key}: {value[0:4]}")
            else:
                print(f"{key}: {value}")
        print("\n")

    @no_grad_method
    def get_ds_mesh(time_grid, n_grid_points_per_axis, grid_padding):
        min_values, _ = time_grid.min(dim=0)
        min_values -= grid_padding
        max_values, _ = time_grid.max(dim=0)
        max_values += grid_padding
        min_values_list = min_values.tolist()
        max_values_list = max_values.tolist()
        #min_values_list[0] = 10
        #min_values_list[1] = 4
        #max_values_list[0] = 65
        #max_values_list.reverse()
        axes = [torch.linspace(min_values_list[i], max_values_list[i], n_grid_points_per_axis) for i in range(time_grid.shape[1])]
        X, Y = np.meshgrid(axes[0], axes[1])
        #X, Y = torch.meshgrid(axes[0], axes[1])
        return X, Y
    
    @no_grad_method
    def create_inducing_points_k_means(time_grid, num_inducing_points):
        time_grid = time_grid.numpy()
        kmeans = KMeans(n_clusters=num_inducing_points)
        print("time_grid shape: ", time_grid.shape)
        kmeans.fit(time_grid)
        inducing_points = kmeans.cluster_centers_
        inducing_points_tensor = torch.tensor(inducing_points)
        return inducing_points_tensor
