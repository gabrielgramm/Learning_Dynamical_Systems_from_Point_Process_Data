import matplotlib.pyplot as plt
import numpy as np
from helpers.helpers import Helper
import torch
from matplotlib.colors import Normalize, LinearSegmentedColormap

def plot_results(vi, gp_sample, posterior_rate, E_f_full_domain, full_E_f_full_domain, couplings, tau_list, c_map='bone', start=0, xlim=1500):
    posterior_rate = posterior_rate.detach().numpy()
    test = torch.sigmoid(E_f_full_domain).detach().numpy()
    E_f_full_domain = E_f_full_domain.detach().numpy()
    full_E_f_full_domain = full_E_f_full_domain.detach().numpy()
    SGP_post_mean = vi.SGP_post_mean.detach().numpy()
    helper_zeros = np.zeros(len(vi.thinned_indices))

    cmap = plt.get_cmap('Paired')
    colors = [cmap((i + 0.5)/ vi.ts_helper.data.shape[0]) for i in range(vi.ts_helper.data.shape[0])]
    colors = ["#008080", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]

    '''
    plt.figure(figsize=(14, 2))
    plt.plot(full_E_f_full_domain, color=colors[0])
    plt.title('full_E_f_full_domain')
    plt.xlim(start, xlim)
    plt.show
    '''
    plt.figure(figsize=(12, 1.5))
    plt.plot(gp_sample,'black')
    plt.plot(posterior_rate, color='#008080', linewidth=2)
    thinned_shifted_indices = vi.thinned_shifted_indices[vi.thinned_shifted_indices < xlim]
    plt.scatter(thinned_shifted_indices, np.zeros(len(thinned_shifted_indices)), s=8, alpha=0.5, color='black')
    #plt.title('fitted Rate Process 1', fontsize=9)
    plt.tick_params(axis='both', labelsize=8)
    plt.xlim(start, xlim)
    plt.show
    '''
    #phase space
    plt.figure(figsize=(8, 8)) 
    plt.scatter(vi.time_grid[:, 0], vi.time_grid[:, 1], c='black', marker='o', s=2, alpha=0.2)
    plt.scatter(vi.inducing_points_s[:, 0].detach().numpy(), vi.inducing_points_s[:, 1].detach().numpy(), c = SGP_post_mean,  marker='x', s=10)
    cmap = plt.get_cmap('plasma')
    for i in range(vi.ts_helper.data.shape[0]):
        thinned_indices = Helper.get_indices_1d(vi.ts_helper.data[i])
        thinned_shifted_indices = Helper.shift_indices(thinned_indices)
        thinned_shifted_indices = thinned_shifted_indices[thinned_shifted_indices < xlim]
        sub_time_grid = vi.ts_helper.get_sub_time_grid_one_process(thinned_shifted_indices, couplings, tau_list)
        plt.scatter(sub_time_grid[start:xlim, 0], sub_time_grid[start:xlim, 1], color=colors[i], s=5, alpha=0.5)
    plt.scatter(vi.sub_time_grid[start:xlim, 0], vi.sub_time_grid[start:xlim, 1], c = E_f_full_domain[start:xlim], marker='o', s=10)
    plt.colorbar(label='Sampled Values',aspect=50)
    #plt.plot(oscillator.y[0], oscillator.y[1], "-",color='black')
    plt.grid(True)
    plt.show()
    '''
    if vi.time_grid.shape[1] == 1:
        plt.figure(figsize=(7, 6))
        #plt.scatter(vi.inducing_points_s[:, 0].detach().numpy(), np.zeros(vi.inducing_points_s.shape[0]), c = SGP_post_mean,  marker='x', s=10,  cmap=c_map)
        #plt.scatter(vi.time_grid[start:xlim, 0], vi.time_grid[start:xlim, 1], c = full_E_f_full_domain[start:xlim], marker='o', s=10)
        plt.scatter(vi.time_grid[start:xlim, 0], np.zeros(xlim-start), c = posterior_rate[start:xlim], marker='o', s=10, cmap=c_map)
        #plt.colorbar(label='posterior GP mean',aspect=70)
        #cbar.set_label('posterior GP mean Process 2', fontsize=16)
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=7)
        plt.show()
    else:   
        plt.figure(figsize=(7, 6))
        #plt.scatter(vi.inducing_points_s[:, 0].detach().numpy(), vi.inducing_points_s[:, 1].detach().numpy(), c = SGP_post_mean,  marker='x', s=10,  cmap=c_map)
        #plt.scatter(vi.time_grid[start:xlim, 0], vi.time_grid[start:xlim, 1], c = full_E_f_full_domain[start:xlim], marker='o', s=10)
        plt.scatter(vi.time_grid[start:xlim, 0], vi.time_grid[start:xlim, 1], c = posterior_rate[start:xlim], marker='o', s=10, cmap=c_map)
        plt.colorbar(aspect=70) #plt.colorbar(label='posterior GP mean',aspect=70)
        #cbar.set_label('posterior GP mean Process 2', fontsize=16)
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=7)
        plt.show()

def plot_rates(list_vi_objects, start=0, xlim=1500, show_points=False):
    height_per_subplot = 1.5
    total_height = len(list_vi_objects) * height_per_subplot

    plt.figure(figsize=(8, total_height))  # Set the figure size with dynamic height
    plt.suptitle('True, Posterior and Marked Rate', fontsize=10)

    for i, vi in enumerate(list_vi_objects):

        true_rate = vi.ts_helper.test_true_rates[i].detach().numpy()
        posterior_rate = vi.test_posterior_rate.detach().numpy()

        plt.subplot(len(list_vi_objects), 1, i + 1)  # Create a subplot for each set of rates
        plt.plot(true_rate, color='black', linewidth=0.7, label='True Rate')
        plt.plot(posterior_rate, color='blue', linewidth=1, label='Posterior Rate')
        # plt.plot(marked_rate, color='blue', linewidth=1.5)

        if show_points == True:
            thinned_indices = vi.thinned_indices[(vi.thinned_indices > start) & (vi.thinned_indices < xlim)]
            plt.scatter(thinned_indices, np.zeros(len(thinned_indices)), s=2, alpha=0.5, color='black')

        plt.xlim(start, xlim)
        plt.tick_params(axis='both', labelsize=7)
        plt.grid(True)
    plt.tight_layout()
    plt.show() 

def plot_test_rates(list_vi_objects, key, start=0, xlim=2000, show_points=False):
    height_per_subplot = 1.3
    total_height = len(list_vi_objects) * height_per_subplot
    plt.figure(figsize=(8, total_height))  # Set the figure size with dynamic height
    #plt.suptitle(f'{key}', fontsize=10)
    #plt.suptitle('2 dimensional Phase Space', fontsize=10)
    colors = ["#008080", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]

    for i, vi in enumerate(list_vi_objects):

        test_true_rate = vi.ts_helper.test_true_rates[i].detach().numpy()   
        test_posterior_rate = vi.test_posterior_rate.detach().numpy()
        plt.subplot(len(list_vi_objects), 1, i + 1)  # Create a subplot for each set of rates
        plt.plot(test_true_rate, color='black', linewidth=1, label='True Rate')
        plt.plot(test_posterior_rate, color=colors[i], linewidth=1.2, label='Test Posterior Rate')

        if show_points == True:
            test_thinned_indices = vi.test_thinned_indices[(vi.test_thinned_indices > start) & (vi.test_thinned_indices < xlim)]
            test_thinned_indices = test_thinned_indices.tolist()
            plt.scatter(test_thinned_indices, np.zeros(test_thinned_indices), s=2, alpha=0.5, color='black')

        plt.xlim(start, xlim)
        plt.tick_params(axis='both', labelsize=7)
        plt.grid(True)
        if i == 0:
            plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show() 

def plot_time_grid(ts_helper, time_grid, size, start=0, end=1500, show_sub_time_grid=False, process=0):
    if show_sub_time_grid == False:
        plt.figure(figsize=(size, size))
        plt.title(r'x(t) in Phase Space', fontsize=8)
        plt.plot(time_grid[start:end, 0], time_grid[start:end, 1], marker='o', color='black', linestyle=':', markersize=np.ceil(size/4), alpha=1)
        plt.tick_params(axis='both', labelsize=7)  
        plt.grid(True)
        plt.show()
    elif show_sub_time_grid == True and process != None:
        cmap = plt.get_cmap('plasma')
        colors = [cmap((i + 0.5)/ ts_helper.data.shape[0]) for i in range(ts_helper.data.shape[0])]
        colors = ["#008080", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]
        plt.figure(figsize=(size, size))
        plt.title(r'$x(t)$ in Phase Space', fontsize=8)  
        thinned_indices = Helper.get_indices_1d(ts_helper.data[process])
        thinned_shifted_indices = Helper.shift_indices(thinned_indices)
        thinned_shifted_indices = thinned_shifted_indices[(thinned_shifted_indices > start) & (thinned_shifted_indices < end)]
        sub_time_grid = torch.index_select(time_grid, 0, index=thinned_shifted_indices)

        plt.plot(time_grid[start:end, 0], time_grid[start:end, 1], marker='o', color='black', linestyle=':', markersize=np.ceil(size/4), alpha=0.7)
        plt.plot(sub_time_grid[start:end, 0], sub_time_grid[start:end, 1], marker='o', color=colors[process], markersize=np.ceil(size/2)-1, linestyle='None')
        plt.tick_params(axis='both', labelsize=7)
        plt.grid(True)
        plt.show()
    elif show_sub_time_grid == True and process == None:
        cmap = plt.get_cmap('plasma')
        colors = [cmap((i + 0.5)/ ts_helper.data.shape[0]) for i in range(ts_helper.data.shape[0])]
        colors = ["#008080", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]
        plt.figure(figsize=(size, size))
        plt.title(r'$x(t)$ in Phase Space', fontsize=8)
        plt.plot(time_grid[start:end, 0], time_grid[start:end, 1], marker='o', color='black', linestyle=':', markersize=np.ceil(size/4), alpha=0.7)
        for i in range(ts_helper.data.shape[0]):
            thinned_indices = Helper.get_indices_1d(ts_helper.data[i])
            thinned_shifted_indices = Helper.shift_indices(thinned_indices)
            thinned_shifted_indices = thinned_shifted_indices[(thinned_shifted_indices > start) & (thinned_shifted_indices < end)]
            sub_time_grid = torch.index_select(time_grid, 0, index=thinned_shifted_indices)
            plt.plot(sub_time_grid[:, 0], sub_time_grid[:, 1], marker='o', color=colors[i], markersize=np.ceil(size/2), linestyle='None')
        plt.tick_params(axis='both', labelsize=7)
        plt.grid(True)
        plt.show()

def plot_big_time_grid(ts_helper, time_grid, size, start=0, end=1500, tau_list=None, couplings=None, no_legend=False):
    if time_grid.shape[1] == 1: # for 1d phase space
        plt.figure(figsize=(size, size))
        plt.title(r'x(t) in Phase Space', fontsize=11)
        if no_legend == False:
            if tau_list is not None and couplings is not None:
                formatted_couplings = [[round(val.item(), 3) for val in row] for row in couplings]
                formatted_tau_list = [round(val.item(), 2) for val in tau_list]
                handle1, = plt.plot([], [], ' ', label=f'tau: {formatted_tau_list}', markersize=3)
                handle2, = plt.plot([], [], ' ', label=f'couplings: {formatted_couplings}', markersize=3)
                plt.legend(handles=[handle1, handle2], loc='upper left')
        cmap = plt.get_cmap('plasma')
        colors = [cmap((i + 0.5)/ ts_helper.data.shape[0]) for i in range(ts_helper.data.shape[0])]
        colors = ["#008080", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]
        plt.plot(time_grid[start:end, 0], np.zeros(end-start), marker='o', color='black', linestyle=':', markersize=2, alpha=0.5)
        for i in range(ts_helper.data.shape[0]):   
            thinned_indices = Helper.get_indices_1d(ts_helper.data[i])
            thinned_shifted_indices = Helper.shift_indices(thinned_indices)
            thinned_shifted_indices = thinned_shifted_indices[(thinned_shifted_indices > start) & (thinned_shifted_indices < end)]
            sub_time_grid = torch.index_select(time_grid, 0, index=thinned_shifted_indices)
            plt.plot(sub_time_grid[:, 0], np.zeros(sub_time_grid.shape[0]), marker='o', color=colors[i], markersize=4, linestyle='None',label=f'Process {i+1}')
            if no_legend == False:
                plt.legend(loc='upper left')
        plt.tick_params(axis='both', labelsize=7)
        plt.grid(False)
        plt.show()
    elif time_grid.shape[1] == 2: # for 2d phase space
        plt.figure(figsize=(size, size))
        #plt.title(r'$x(t)$ in Phase Space', fontsize=11)
        if no_legend == False:
            if tau_list is not None and couplings is not None:
                formatted_couplings = [[round(val.item(), 3) for val in row] for row in couplings]
                formatted_tau_list = [round(val.item(), 2) for val in tau_list]
                handle1, = plt.plot([], [], ' ', label = fr'$\tau$: {formatted_tau_list}')
                handle2, = plt.plot([], [], ' ', label=fr'$\theta$: {formatted_couplings}')
                plt.legend(handles=[handle1, handle2], loc='upper left')
        cmap = plt.get_cmap('plasma')
        colors = [cmap((i + 0.5)/ ts_helper.data.shape[0]) for i in range(ts_helper.data.shape[0])]
        colors = ["#008080", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]
        plt.plot(time_grid[start:end, 0], time_grid[start:end, 1], marker='o', color='black', linestyle=':', markersize=2, alpha=0.5)
        for i in range(ts_helper.data.shape[0]):   
            thinned_indices = Helper.get_indices_1d(ts_helper.data[i])
            thinned_shifted_indices = Helper.shift_indices(thinned_indices)
            thinned_shifted_indices = thinned_shifted_indices[(thinned_shifted_indices > start) & (thinned_shifted_indices < end)]
            sub_time_grid = torch.index_select(time_grid, 0, index=thinned_shifted_indices)
            plt.plot(sub_time_grid[:, 0], sub_time_grid[:, 1], marker='o', color=colors[i], markersize=2, linestyle='None', label=f'Process {i+1}')
            if no_legend == False:
                plt.legend(loc='upper left', fontsize=11, markerscale=5)
        plt.tick_params(axis='both', labelsize=8)
        plt.grid(True)
        plt.show()
    elif time_grid.shape[1] == 3:
        print('############# plotting in 3d #############')
        time_grid = time_grid.numpy()
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot(111, projection='3d')
        #plt.title(r'$x(t)$ in Phase Space', fontsize=11)
        cmap = plt.get_cmap('YlGnBu')
        colors = ["#008080", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]   
        print("time_grid", time_grid.shape)   
        ax.plot(time_grid[:, 0], time_grid[:, 1], time_grid[:, 2], marker='o', color='black', linestyle=':', markersize=2, alpha=0.5)
        for i in range(ts_helper.data.shape[0]):
            print("data", ts_helper.data[i].shape)
            thinned_indices = Helper.get_indices_1d(ts_helper.data[i])
            thinned_shifted_indices = Helper.shift_indices(thinned_indices)
            thinned_shifted_indices = thinned_shifted_indices.numpy()
            sub_time_grid = np.take(time_grid, thinned_shifted_indices, axis=0)
            ax.plot(sub_time_grid[:, 0], sub_time_grid[:, 1], sub_time_grid[:, 2], 
                    marker='o', color=colors[i], markersize=2, linestyle='None', 
                    label=f'Process {i+1}')
        
        plt.tick_params(axis='both', labelsize=8)
        plt.grid(True)
        plt.show()


def plot_loss(loss_tracker, size):
    if isinstance(loss_tracker, list):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(2 * size, size))
        max_len = max(len(loss) for loss in loss_tracker)
        lines = []
        labels = []
        colors = plt.cm.viridis(np.linspace(0, 1, len(loss_tracker)))  # Generate a color map
        colors = ["#008080", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]
        
        for i in range(len(loss_tracker)):
            temp = torch.tensor(loss_tracker[i])
            min_val = torch.min(temp)
            max_val = torch.max(temp)
            normalized_temp = (temp - min_val) / (max_val - min_val)
            normalized_temp = normalized_temp.detach().numpy()
            padded_normalized_temp = np.pad(normalized_temp, (0, max_len - len(normalized_temp)), constant_values=np.nan)
            
            # Use the generated color for each line
            line, = ax2.plot(padded_normalized_temp, label=f'process {i+1}', color=colors[i])
            lines.append(line)
            labels.append(f'process {i+1}')
        
        ax2.set_title('Normalized', fontsize=9)
        ax2.set_xlabel('Iteration', fontsize=9)
        ax2.set_ylabel('ELBO', fontsize=9)
        ax2.tick_params(axis='both', labelsize=7)

        for i in range(len(loss_tracker)):
            unnormalized_temp = torch.tensor(loss_tracker[i]).detach().numpy()
            padded_unnormalized_temp = np.pad(unnormalized_temp, (0, max_len - len(unnormalized_temp)), constant_values=np.nan)
            
            # Use the same color as in ax1
            ax1.plot(padded_unnormalized_temp, label=f'process {i+1}', color=colors[i])
        
        ax1.set_title('Unnormalized', fontsize=9)
        ax1.set_xlabel('Iteration', fontsize=9)
        ax1.set_ylabel('Evidence Lower Bound', fontsize=9)
        ax1.tick_params(axis='both', labelsize=7)

        ax2.legend(lines, labels, loc='lower right', fontsize=8, ncol=1)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
        plt.show()
    else:
        plt.figure(figsize=(size, size))
        loss_tracker = torch.tensor(loss_tracker)
        loss_tracker = torch.softmax(torch.tensor(loss_tracker), dim=0).detach().numpy()
        plt.plot(loss_tracker)
        plt.xlabel('Iteration')
        plt.ylabel('Evidence Lower Bound')
        plt.tick_params(axis='both', labelsize=7)
        plt.show()

def plot_just_post_rate(gp_sample, posterior_rate, start=0, xlim=1500, plot_counter=0):
    posterior_rate = posterior_rate.detach().numpy()
    plt.figure(figsize=(10, 1))
    plt.plot(gp_sample,'black')
    plt.plot(posterior_rate, color='green')
    plt.title(f'posterior_rate, iteration: {plot_counter}', fontsize=8) 
    plt.tick_params(axis='both', labelsize=7)
    plt.xlim(start, xlim)
    plt.show()

def plot_post_rate_minimal(vi, start=0, xlim=1500):
    posterior_rate = vi.posterior_rate.detach().numpy()
    plt.figure(figsize=(8, 1))
    plt.plot(posterior_rate, color='black',  linewidth=0.7)
    thinned_shifted_indices = vi.thinned_shifted_indices[vi.thinned_shifted_indices < xlim]
    plt.scatter(thinned_shifted_indices, np.zeros(len(thinned_shifted_indices)), s=2, alpha=0.5, color='black')
    #plt.title(f'posterior_rate', fontsize=8) 
    plt.tick_params(axis='both', labelsize=7)
    plt.xlim(start, xlim)
    plt.show()

def plot_time_grid_kernel(ts_helper, tau_list, size=4):
    tau_list = torch.tensor(tau_list)
    plt.figure(figsize=(size, size))  # Create the figure only once

    colors = plt.cm.viridis(np.linspace(0, 1, tau_list.shape[0]))  # Generate a color map
    colors = ["#008080", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]

    for i in range(tau_list.shape[0]):
        if tau_list[i] < 0:
            raise ValueError("Tau must be greater than 0")
        elif tau_list[i] < 1:
            l = ts_helper.kernel_effect_length
            l = torch.ceil(l)
        else:
            l = ts_helper.kernel_effect_length
            l = torch.ceil(l)

        x = np.linspace(0, int(l), int(l) * ts_helper.time_discretization)
        exp_function = torch.exp(-x/tau_list[i])# / tau_list[i]
        exp_function[x < 0] = 0
        kernel = exp_function # / (torch.sum(exp_function) / ts_helper.time_discretization)   # normalization
        
        # Use a consistent color for each tau
        plt.plot(kernel, label=f'tau: {tau_list[i]:.2f}', color=colors[i])  # Plot each kernel with its own color

    plt.title('Phase Space Kernel', fontsize=8)
    plt.grid(False)
    plt.tick_params(axis='both', labelsize=7)
    plt.legend()  # Add a legend to distinguish different tau values
    plt.show()

def plot_posterior_GP(vi, i, mesh, post_gp_full_domain, colormap, grid_padding, start=0, xlim=1500):
    post_gp = post_gp_full_domain.detach().numpy()
    post_gp_reshape = post_gp_full_domain.reshape(mesh[1].shape).detach().numpy()

    fig = plt.figure(figsize=(7, 6))

    plt.scatter(mesh[0], mesh[1], c=post_gp, cmap=colormap, marker='s')
    #plt.scatter(vi.inducing_points_s[:, 0].detach().numpy(), vi.inducing_points_s[:, 1].detach().numpy(), c='red', marker='x', s=4)
    cbar= plt.colorbar(aspect=50)
    cbar.ax.tick_params(labelsize=7)

    #for the points in the phase space
    cmap = plt.get_cmap('plasma')
    colors = [cmap((i + 0.5)/ vi.ts_helper.data.shape[0]) for i in range(vi.ts_helper.data.shape[0])]
    colors = ["#09b0b0", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]

    vi.time_grid = vi.ts_helper.get_time_grid(couplings= vi.couplings, tau_list=vi.tau_list)
    thinned_shifted_indices = vi.thinned_shifted_indices[(vi.thinned_shifted_indices > start) & (vi.thinned_shifted_indices < xlim)]
    sub_time_grid = torch.index_select(vi.time_grid, 0, index=thinned_shifted_indices)
    plt.plot(vi.time_grid[start:xlim, 0], vi.time_grid[start:xlim, 1], marker='o', color='#48494a', linestyle=':', markersize=1, alpha=1)
    plt.plot(sub_time_grid[:, 0], sub_time_grid[:, 1], marker='o', color=colors[i], markersize=2, linestyle='None')
    plt.tick_params(axis='both', labelsize=7)
    plt.xlabel('x1', fontsize=9)
    plt.ylabel('x2', fontsize=9)
    plt.xlim(min(vi.time_grid[start:xlim,0])-.1, max(vi.time_grid[start:xlim,0])+.1)
    #plt.xlim(7.5, max(vi.time_grid[:,0])+.1)
    plt.ylim(min(vi.time_grid[start:xlim,1])-.2, max(vi.time_grid[start:xlim,1])+.2)
    plt.show()

def plot_surface(vi, i, mesh, post_gp_full_domain, colormap, grid_padding, start=0, xlim=1500, elev=30, azim=45):
    post_gp = post_gp_full_domain.detach().numpy()
    post_gp_reshape = post_gp_full_domain.reshape(mesh[1].shape).detach().numpy()
    print(post_gp_reshape.shape)
    time_grid = vi.ts_helper.get_time_grid(couplings= vi.couplings, tau_list=vi.tau_list)

    min_x, max_x, min_y, max_y = torch.min(time_grid[:, 0]), torch.max(time_grid[:, 0]), torch.min(time_grid[:, 1]), torch.max(time_grid[:, 1])
    X, Y =  mesh[0], mesh[1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, post_gp_reshape, cmap=colormap, edgecolor='none', alpha=0.9)
    
    # For the points in the phase space
    cmap = plt.get_cmap('plasma')
    colors = [cmap((i + 0.5) / vi.ts_helper.data.shape[0]) for i in range(vi.ts_helper.data.shape[0])]
    colors = ["#008080", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]
    thinned_shifted_indices = vi.thinned_shifted_indices[(vi.thinned_shifted_indices > start) & (vi.thinned_shifted_indices < xlim)]
    sub_time_grid = torch.index_select(time_grid, 0, index=thinned_shifted_indices)
    
    plt.plot(time_grid[start:xlim, 0],time_grid[start:xlim, 1], marker='o', color='black', linestyle='', markersize=1, alpha=1)
    plt.plot(sub_time_grid[:, 0], sub_time_grid[:, 1], marker='o', color=colors[i], markersize=1.5, linestyle='None')
    
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.zaxis.set_tick_params(labelsize=7)
    ax.set_zlim(0, 9.8)
    ax.set_xlabel('x1', fontsize=10)
    ax.set_ylabel('x2', fontsize=10)
    ax.set_zlabel('Intensity Rate', fontsize=10)

    # Set the elevation and azimuth angle for the view
    ax.view_init(elev=elev, azim=azim)
    plt.show()
