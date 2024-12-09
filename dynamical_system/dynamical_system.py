import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from helpers.helpers import Helper
from helpers.helpers import no_grad_method
from scipy.integrate import solve_ivp

class ODE:
    def __init__(self, time_grid, list_vi_objects, n_grid_points_per_axis, grid_padding=5):

        self.tau_list = np.array(list_vi_objects[0].tau_list)
        self.couplings = list_vi_objects[0].couplings
        self.time_discretization = list_vi_objects[0].time_discretization
        self.time_grid = time_grid
        self.time_span = time_grid.shape[0]
        self.list_vi_objects = list_vi_objects
        self.n_grid_points_per_axis = n_grid_points_per_axis
        self.grid_padding = grid_padding
        self.out_trajectory = None
        self.gp_of_trajectory = None
        self.list_events = [[] for _ in range(len(self.list_vi_objects))]
        self.list_gp_of_trajectories = [[] for _ in range(len(self.list_vi_objects))]
        self.colors = ["#008080", "#ca91ed", "#de9b64", "#79db7f", "#e86f8a", "#5c62d6", "#ffcc00", "#ff5733", "#33b5e5", "#8e44ad", "#f39c12", "#2ecc71"]

    @no_grad_method
    def plot_streamplot(self, colormap=None , trajectory=None , padding= 4):
        X, Y = Helper.get_ds_mesh(self.time_grid, self.n_grid_points_per_axis, self.grid_padding)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        U_flat = self.x1(X_flat, Y_flat)
        V_flat = self.x2(X_flat, Y_flat)
        U = U_flat.reshape((self.n_grid_points_per_axis, self.n_grid_points_per_axis))
        V = V_flat.reshape((self.n_grid_points_per_axis, self.n_grid_points_per_axis))
    
        magnitude = np.sqrt(U_flat**2 + V_flat**2)
        norm = plt.Normalize(magnitude.min(), magnitude.max())
        cmap = plt.get_cmap(colormap)

        plt.figure(figsize=(8, 7))
        #strem = plt.streamplot(X, Y, U, V, color='steelblue', linewidth=1.5, norm=norm, density=2, arrowstyle='->', arrowsize=1.5)
        strem = plt.streamplot(X, Y, U, V, color=magnitude.reshape(U.shape), cmap=cmap, linewidth=1.5, norm=norm, density=2, arrowstyle='->', arrowsize=1.5)
        if trajectory is not None:
            plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color = 'tomato' , linestyle='-', linewidth=3, markersize=.1, alpha=1)
        cbar = plt.colorbar(strem.lines, label='', aspect=50)
        cbar.ax.tick_params(labelsize=7)
        plt.xlabel('$x_1$', fontsize=9)
        plt.ylabel('$x_2$', fontsize=9)
        #if trajectory is not None:
        #    plt.xlim(min(self.out_trajectory[:, 0]) - padding, max(self.out_trajectory[:, 0]) + padding)
        #    plt.ylim(min(self.out_trajectory[:, 1]) - padding, max(self.out_trajectory[:, 1]) + padding)
        plt.tick_params(axis='both', labelsize=7)
        plt.show()

    @no_grad_method
    def plot_post_gp(self, colormap='plasma'):
        X, Y = Helper.get_ds_mesh(self.time_grid, self.n_grid_points_per_axis, self.grid_padding)          
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        num_plots = len(self.list_vi_objects)
        fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6), constrained_layout=True)
        scatter_plots = []
        for i, (vi, ax) in enumerate(zip(self.list_vi_objects, axs)):
            post_gp = vi.posterior_rate_dynamical_system.detach().numpy()
            plot_gp = ax.scatter(X, Y, c=post_gp, cmap=colormap, marker='s')
            cmap = plt.get_cmap('plasma')
            colors = [cmap((i + 0.5)/ vi.ts_helper.data.shape[0]) for i in range(vi.ts_helper.data.shape[0])]
            colors = ["#008080", "#ca91ed", "#de9b64", '#79db7f' , '#e86f8a', '#5c62d6']
            sub_time_grid = torch.index_select(vi.time_grid, 0, index=vi.thinned_shifted_indices)
            ax.plot(vi.time_grid[800:, 0], vi.time_grid[800:, 1], marker='o', color='gray', linestyle=':', markersize=1, alpha=1)
            ax.plot(sub_time_grid[80:, 0], sub_time_grid[80:, 1], marker='o', color=colors[i], markersize=np.ceil(3), linestyle='None')
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(False)
            cbar = plt.colorbar(plot_gp, ax=ax, orientation='vertical', aspect=50)  
            cbar.ax.tick_params(labelsize=8)
            ax.set_title(f'process {i+1}')
            '''
            post_gp = vi.posterior_rate_dynamical_system.detach().numpy()
            post_gp = ax.scatter(X, Y, c=post_gp, cmap='viridis')
            ax.set_title(f'Plot {i+1}')
            ax.grid(True)
            '''
            #scatter_plots.append(post_gp)
        #plt.tight_layout()
        plt.show()
    

    
    @no_grad_method
    def plot_vectorfield(self, colormap='plasma', time_grid=False, num_arrows=20, dt=0.1, trajectory_steps=1500, start_point=[-10, 0], padding=4):
        X, Y = Helper.get_ds_mesh(self.time_grid, self.n_grid_points_per_axis, self.grid_padding)          
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        U_flat = self.x1(X_flat, Y_flat)
        V_flat = self.x2(X_flat, Y_flat)
        U = U_flat.reshape(X.shape)
        V = V_flat.reshape(Y.shape)
        
        vi = self.list_vi_objects[0]
        plt.figure(figsize=(7, 6))
        #if time_grid:
        #    plt.plot(vi.time_grid[800:, 0], vi.time_grid[800:, 1], marker='o', color='black', linestyle=':', markersize=1, alpha=1)
        scale = int(np.ceil(len(X) / num_arrows))
        x_slice = X[::scale, ::scale]
        y_slice = Y[::scale, ::scale]
        u_slice = U[::scale, ::scale]
        v_slice = V[::scale, ::scale]

        magnitude = np.sqrt(u_slice**2 + v_slice**2)
        norm = plt.Normalize(magnitude.min(), magnitude.max())
        cmap = plt.get_cmap(colormap)

        quiv = plt.quiver(x_slice, y_slice, u_slice, v_slice, magnitude, cmap=cmap, norm=norm, angles="xy", scale_units="xy",  scale=16, width=.004, headwidth=3, headlength=3)       
        self.out_trajectory = self.evolve_trajectory(start_point[0].item(), start_point[1].item(), dt, trajectory_steps)

        plt.plot(self.out_trajectory[:, 0], self.out_trajectory[:, 1], marker='o', color='tomato', linestyle='-', linewidth=2, markersize=1, alpha=1)    # "#ca91ed",
        cbar = plt.colorbar(quiv, label='', aspect=50)
        cbar.ax.tick_params(labelsize=7)
        plt.xlabel('x1', fontsize=9)
        plt.ylabel('x2', fontsize=9)
        #plt.xlim(8, max(self.out_trajectory[:, 0]) +padding)
        #plt.ylim(0, max(self.out_trajectory[:, 1]) +padding)
        #plt.xlim(min(self.out_trajectory[:, 0]) - padding, max(self.out_trajectory[:, 0]) + padding)
        #plt.ylim(min(self.out_trajectory[:, 1]) - padding, max(self.out_trajectory[:, 1]) + padding)
        plt.tick_params(axis='both', labelsize=7)
        plt.show()

        return self.out_trajectory, self.list_gp_of_trajectories

    @no_grad_method
    def evolve_trajectory(self, x0, y0, dt, trajectory_steps):
        trajectory = np.array([(x0, y0)])
        x, y = x0, y0
        for _ in range(trajectory_steps):
            vx, vy = self.posterior_dynamics(dt, (x, y))
            x_new = x + vx * dt
            y_new = y + vy * dt
            new_point = np.array([[x_new.item(), y_new.item()]])
            trajectory = np.append(trajectory, new_point, axis=0)
            x, y = x_new.item(), y_new.item()

        for i in range(len(self.list_gp_of_trajectories)):
            self.list_gp_of_trajectories[i] = np.array(self.list_gp_of_trajectories[i])
        return trajectory 
    
    @no_grad_method
    def posterior_dynamics(self, dt, y):
        x1, x2 = y 
        dx1dt = - x1 / self.tau_list[0]
        dx2dt = - x2 / self.tau_list[1]
        for i, vi in enumerate(self.list_vi_objects):
            post_gp_at_x1_x2, sig_E_post_rate = vi.eval_post_gp(x1, x2)
            self.list_gp_of_trajectories[i].append(post_gp_at_x1_x2.item())  #kill .item() for tensor
            dx1dt += self.couplings[i, 0] * post_gp_at_x1_x2
            dx2dt += self.couplings[i, 1] * post_gp_at_x1_x2
        return dx1dt, dx2dt

    @no_grad_method
    def plot_stream_and_surfaceplot(self, colormap='plasma' , trajectory=None , padding= 4, num_arrows=20, surfacealpha=0.9, elev=30, azim=45, gif=0):
        X, Y = Helper.get_ds_mesh(self.time_grid, self.n_grid_points_per_axis, self.grid_padding)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        U_flat = self.x1(X_flat, Y_flat)
        V_flat = self.x2(X_flat, Y_flat)
        scale = int(np.ceil(X.shape[0]/num_arrows))
        U = U_flat.reshape(X.shape)
        V = V_flat.reshape(Y.shape)
        x_slice = X[::scale, ::scale]
        y_slice = Y[::scale, ::scale]
        u_slice = U[::scale, ::scale]
        v_slice = V[::scale, ::scale]

        U = U_flat.reshape((self.n_grid_points_per_axis, self.n_grid_points_per_axis))
        V = V_flat.reshape((self.n_grid_points_per_axis, self.n_grid_points_per_axis))

        list_post_gp_reshape = []

        ##################################################
        selected_GP = 3
        self.ajusted_list_vi_objects = [self.list_vi_objects[selected_GP]]
        ##################################################

        for i, vi in enumerate(self.ajusted_list_vi_objects):
            post_gp = vi.posterior_rate_dynamical_system
            post_gp_reshape = post_gp.reshape(X.shape).detach().numpy()
            list_post_gp_reshape.append(post_gp_reshape)
    
        for i in range(len(list_post_gp_reshape)):
            fig = plt.figure(figsize=(8, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, list_post_gp_reshape[i], cmap=colormap, edgecolor='none', alpha=surfacealpha)

            magnitude = np.sqrt(u_slice**2 + v_slice**2 + np.zeros_like(u_slice)**2)
            norm = plt.Normalize(magnitude.min(), magnitude.max())
            cmap = plt.get_cmap(colormap)
            colors = cmap(norm(magnitude))
            colors = colors.reshape(-1, 4) 


            
            quiv = ax.quiver(x_slice, y_slice, np.zeros_like(x_slice), u_slice, v_slice, np.zeros_like(u_slice),
                 length=1.5, arrow_length_ratio=0.3, normalize=True, colors=colors)

            
            if trajectory is not None:
                plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color = 'tomato' , linestyle='-', linewidth=2, markersize=.1, alpha=1)
            #cbar = plt.colorbar(strem.lines, label='', aspect=50)
            #cbar.ax.tick_params(labelsize=7)
            
            '''
            magnitude = np.sqrt(U_flat**2 + V_flat**2)
            norm = plt.Normalize(magnitude.min(), magnitude.max())
            cmap = plt.get_cmap(colormap)

            strem = plt.streamplot(X, Y, U, V, color=magnitude.reshape(U.shape), cmap=cmap, linewidth=1.5, norm=norm, density=3, arrowstyle='->', arrowsize=1.5)         
            #ax_2d = fig.add_subplot(111)
            #stream = ax_2d.streamplot(X, Y, U, V, color=magnitude.reshape(U.shape), cmap=cmap,linewidth=1.5, norm=norm, density=3, arrowstyle='->', arrowsize=1.5)
            #ax.set_zlim(bottom=0)
            lines = strem.lines.get_paths()
            for line in lines:
                old_x = line.vertices.T[0]
                old_y = line.vertices.T[1]
                # apply for 2d to 3d transformation here
                new_z = 0
                new_x = old_x
                new_y = old_y
                new_line = line.art3d.line_collection_2d_to_3d()
                ax.plot(new_x, new_y, new_z, 'k')
            '''


            #plane color
            ax.xaxis.set_pane_color((1, 1, 1, 0))  # XZ plane: Light grey with transparency
            ax.yaxis.set_pane_color((1, 1, 1, 0))  # YZ plane: Light grey with transparency
            ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 0.95))  # XY plane: Almost white for better visibility

            # Update grid colors to light grey to keep them visible
            ax.xaxis._axinfo['grid'].update(color=(0.2, 0.2, 0.2, 0.5))  # Light grey grid for XZ plane
            ax.yaxis._axinfo['grid'].update(color=(0.2, 0.2, 0.2, 0.5))  # Light grey grid for YZ plane
            ax.zaxis._axinfo['grid'].update(color=(0.2, 0.2, 0.2, 0.5))  # Solid light grey grid for XY plane

            # Change axis lines to a slightly darker grey for contrast
            ax.xaxis.line.set_color((0.1, 0.1, 0.1, 1.0))  # Dark grey for X axis
            ax.yaxis.line.set_color((0.1, 0.1, 0.1, 1.0))  # Dark grey for Y axis
            ax.zaxis.line.set_color((0.1, 0.1, 0.1, 1.0))  # Dark grey for Z axis

            # Optionally, add ticks to the axes for better visibility
            ax.xaxis.set_tick_params(color=(0.5, 0.5, 0.5, 1.0))  # Tick color for X axis
            ax.yaxis.set_tick_params(color=(0.5, 0.5, 0.5, 1.0))  # Tick color for Y axis
            ax.zaxis.set_tick_params(color=(0.5, 0.5, 0.5, 1.0))  # Tick color for Z axis

            #remove ticks
            #ax.set_xticks([])
            #ax.set_yticks([])
            #ax.set_zticks([])

            ax.xaxis.set_tick_params(labelsize=7)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.zaxis.set_tick_params(labelsize=7)
            
            #labels
            ax.set_xlabel('x1', fontsize=8)
            ax.set_ylabel('x2', fontsize=8)
            ax.set_zlabel('Intensity Rate', fontsize=8)
            

            # Set the elevation and azimuth angle for the view
            ax.view_init(elev=elev, azim=azim)
            plt.show()
            
            image_folder = r'C:\Users\gabri\OneDrive\Dokumente\Studium\Master_Thesis\plots\for_fun\titel_page\gif_img'
            ''' for GIF creation 
            plt.savefig(os.path.join(image_folder, f'plot_{gif:03d}.png'))
            '''
            
    @no_grad_method     
    def plot_post_gp_and_trajectory(self, colormap='plasma', trajectory=None, padding=4):
        X, Y = Helper.get_ds_mesh(self.time_grid, self.n_grid_points_per_axis, self.grid_padding)          
        for i, vi in enumerate(self.list_vi_objects):
            plt.figure(figsize=(7, 6))
            post_gp = vi.posterior_rate_dynamical_system.detach().numpy()
            plot_gp = plt.scatter(X, Y, c=post_gp, cmap=colormap, marker='s')
            sub_time_grid = torch.index_select(vi.time_grid, 0, index=vi.thinned_shifted_indices)
            plt.plot(vi.time_grid[800:, 0], vi.time_grid[800:, 1], marker='o', color='black', linestyle=':', markersize=1, alpha=1)
            plt.plot(sub_time_grid[80:, 0], sub_time_grid[80:, 1], marker='o', color=self.colors[i], markersize=2.5, linestyle='None')
            plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color = 'tomato' , linestyle='-', linewidth=3, markersize=.1, alpha=1)
            plt.tick_params(axis='both', labelsize=8)
            plt.grid(False)
            cbar = plt.colorbar(plot_gp, ax=plt.gca(), orientation='vertical', aspect=40)
            cbar.ax.tick_params(labelsize=7)
            plt.xlabel('x1', fontsize=9)
            plt.ylabel('x2', fontsize=9)
            plt.xlim(min(self.out_trajectory[:, 0]) - padding, max(self.out_trajectory[:, 0]) + padding)
            plt.ylim(min(self.out_trajectory[:, 1]) - padding, max(self.out_trajectory[:, 1]) + padding)
            #plt.xlim(19, 100)
            #plt.ylim(-12, 33)
            plt.show()

























    @no_grad_method
    def BIG_plot_stream_and_surfaceplot(self, colormap='plasma' , trajectory=None, predicted_trajectory=None , padding= 4, num_arrows=25, surfacealpha=0.9, elev=30, azim=45, gif=0):
        X, Y = Helper.get_ds_mesh(self.time_grid, self.n_grid_points_per_axis, self.grid_padding)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        U_flat = self.x1(X_flat, Y_flat)
        V_flat = self.x2(X_flat, Y_flat)
        scale = int(np.ceil(X.shape[0]/num_arrows))
        U = U_flat.reshape(X.shape)
        V = V_flat.reshape(Y.shape)
        x_slice = X[::scale, ::scale]
        y_slice = Y[::scale, ::scale]
        u_slice = U[::scale, ::scale]
        v_slice = V[::scale, ::scale]

        U = U_flat.reshape((self.n_grid_points_per_axis, self.n_grid_points_per_axis))
        V = V_flat.reshape((self.n_grid_points_per_axis, self.n_grid_points_per_axis))

        list_post_gp_reshape = []

        ##################################################
        selected_GP = 3
        self.ajusted_list_vi_objects = [self.list_vi_objects[selected_GP]]
        ##################################################

        for i, vi in enumerate(self.ajusted_list_vi_objects):
            post_gp = vi.posterior_rate_dynamical_system
            post_gp_reshape = post_gp.reshape(X.shape).detach().numpy()
            list_post_gp_reshape.append(post_gp_reshape)
    
        for i in range(len(list_post_gp_reshape)):
            fig = plt.figure(figsize=(12, 13))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, list_post_gp_reshape[i], cmap=colormap, edgecolor='none', alpha=surfacealpha)

            magnitude = np.sqrt(u_slice**2 + v_slice**2 + np.zeros_like(u_slice)**2)
            norm = plt.Normalize(magnitude.min(), magnitude.max())
            cmap = plt.get_cmap(colormap)
            colors = cmap(norm(magnitude))
            colors = colors.reshape(-1, 4) 


            
            quiv = ax.quiver(x_slice, y_slice, np.zeros_like(x_slice), u_slice, v_slice, np.zeros_like(u_slice),
                 length=1.5, arrow_length_ratio=0.3, normalize=True, colors=colors)

            
            if predicted_trajectory is not None:
                plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], color='black', linestyle=':',  linewidth=1, marker='o', markersize=2, alpha=1, zorder=1)
            if trajectory is not None:
                plt.plot(trajectory[:, 0], trajectory[:, 1], color='tomato', linestyle='-',  linewidth=1, marker='o', markersize=1.8, alpha=1, zorder=1)
            #cbar = plt.colorbar(strem.lines, label='', aspect=50)
            #cbar.ax.tick_params(labelsize=7)
            
            '''
            magnitude = np.sqrt(U_flat**2 + V_flat**2)
            norm = plt.Normalize(magnitude.min(), magnitude.max())
            cmap = plt.get_cmap(colormap)

            strem = plt.streamplot(X, Y, U, V, color=magnitude.reshape(U.shape), cmap=cmap, linewidth=1.5, norm=norm, density=3, arrowstyle='->', arrowsize=1.5)         
            #ax_2d = fig.add_subplot(111)
            #stream = ax_2d.streamplot(X, Y, U, V, color=magnitude.reshape(U.shape), cmap=cmap,linewidth=1.5, norm=norm, density=3, arrowstyle='->', arrowsize=1.5)
            #ax.set_zlim(bottom=0)
            lines = strem.lines.get_paths()
            for line in lines:
                old_x = line.vertices.T[0]
                old_y = line.vertices.T[1]
                # apply for 2d to 3d transformation here
                new_z = 0
                new_x = old_x
                new_y = old_y
                new_line = line.art3d.line_collection_2d_to_3d()
                ax.plot(new_x, new_y, new_z, 'k')
            '''


            #plane color
            ax.xaxis.set_pane_color((1, 1, 1, 0))  # XZ plane: Light grey with transparency
            ax.yaxis.set_pane_color((1, 1, 1, 0))  # YZ plane: Light grey with transparency
            ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 0.95))  # XY plane: Almost white for better visibility

            # Update grid colors to light grey to keep them visible
            ax.xaxis._axinfo['grid'].update(color=(0.2, 0.2, 0.2, 0.5))  # Light grey grid for XZ plane
            ax.yaxis._axinfo['grid'].update(color=(0.2, 0.2, 0.2, 0.5))  # Light grey grid for YZ plane
            ax.zaxis._axinfo['grid'].update(color=(0.2, 0.2, 0.2, 0.5))  # Solid light grey grid for XY plane

            # Change axis lines to a slightly darker grey for contrast
            ax.xaxis.line.set_color((0.1, 0.1, 0.1, 1.0))  # Dark grey for X axis
            ax.yaxis.line.set_color((0.1, 0.1, 0.1, 1.0))  # Dark grey for Y axis
            ax.zaxis.line.set_color((0.1, 0.1, 0.1, 1.0))  # Dark grey for Z axis

            # Optionally, add ticks to the axes for better visibility
            ax.xaxis.set_tick_params(color=(0.5, 0.5, 0.5, 1.0))  # Tick color for X axis
            ax.yaxis.set_tick_params(color=(0.5, 0.5, 0.5, 1.0))  # Tick color for Y axis
            ax.zaxis.set_tick_params(color=(0.5, 0.5, 0.5, 1.0))  # Tick color for Z axis

            #remove ticks
            #ax.set_xticks([])
            #ax.set_yticks([])
            #ax.set_zticks([])

            ax.xaxis.set_tick_params(labelsize=7)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.zaxis.set_tick_params(labelsize=7)
            
            #labels
            #ax.set_xlabel('$x_1$', fontsize=8)
            #ax.set_ylabel('x2', fontsize=8)
            #ax.set_zlabel('Intensity Rate', fontsize=8)
            

            # Set the elevation and azimuth angle for the view
            ax.view_init(elev=elev, azim=azim)
            plt.show()
            
            image_folder = r'C:\Users\gabri\OneDrive\Dokumente\Studium\Master_Thesis\plots\for_fun\titel_page\gif_img'
            ''' for GIF creation 
            plt.savefig(os.path.join(image_folder, f'plot_{gif:03d}.png'))
            '''


























    @no_grad_method
    def plot_dynamical_system(self, start=500, stop=2000, x0y0=[0.5,0.5], trajectory_steps=1000, dt=0.1, process_to_plot=0, num_arrows=20, colormap='viridis'):
        X, Y = Helper.get_ds_mesh(self.time_grid, self.n_grid_points_per_axis, self.grid_padding)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        U_flat = self.x1(X_flat, Y_flat)
        V_flat = self.x2(X_flat, Y_flat)
        U = U_flat.reshape((self.n_grid_points_per_axis, self.n_grid_points_per_axis))
        V = V_flat.reshape((self.n_grid_points_per_axis, self.n_grid_points_per_axis))
        
        magnitude = np.sqrt(U_flat**2 + V_flat**2)
        norm = plt.Normalize(magnitude.min(), magnitude.max())
        cmap = plt.get_cmap(colormap)
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        '''  here you can add the streamplot to the plot  '''
        #strm = axs[2].streamplot(X, Y, U, V, color=magnitude.reshape(U.shape), cmap=cmap, linewidth=1, norm=norm, density=2, arrowstyle='->', arrowsize=1.5)
        #axs[0].axhline(0, color='black', linestyle='dashed', linewidth=0.5)
        #axs[0].axvline(0, color='black', linestyle='dashed', linewidth=0.5)
        #axs[0].plot(self.time_grid[start:stop, 0], self.time_grid[start:stop, 1], marker='o', color='black', linestyle='-', markersize=1, alpha=1)
        #fig.colorbar(strm.lines, ax=axs[2], aspect=50)
        #axs[2].tick_params(axis='both', labelsize=8)
        #axs[2].set_title('Streamplot', fontsize=12)

        vi = self.list_vi_objects[process_to_plot]
        post_gp = vi.posterior_rate_dynamical_system.detach().numpy()
        plot_gp = axs[0].scatter(X, Y, c=post_gp, cmap=colormap, marker='s')
        cmap = plt.get_cmap(colormap)
        colors = [cmap((i + 0.5)/ vi.ts_helper.data.shape[0]) for i in range(vi.ts_helper.data.shape[0])]
        colors = ["#008080", "#ca91ed", "#de9b64", '#79db7f' , '#e86f8a', '#5c62d6']
        sub_time_grid = torch.index_select(vi.time_grid, 0, index=vi.thinned_shifted_indices)
        axs[0].plot(vi.time_grid[800:, 0], vi.time_grid[800:, 1], marker='o', color='dimgrey', linestyle=':', markersize=2, alpha=1)
        axs[0].plot(sub_time_grid[80:, 0], sub_time_grid[80:, 1], marker='o', color='lightpink', markersize=5, linestyle='None') #color=colors[process_to_plot]
        fig.colorbar(plot_gp, ax=axs[0], aspect=50)
        axs[0].tick_params(axis='both', labelsize=8)
        axs[0].grid(False)
        axs[0].set_title('Posterior Rate and SDE Trajectory', fontsize=12)

        scale = int(np.ceil(X.shape[0]/num_arrows))
        x_slice = X[::scale, ::scale]
        y_slice = Y[::scale, ::scale]
        u_slice = U[::scale, ::scale]
        v_slice = V[::scale, ::scale]

        magnitude = np.sqrt(u_slice**2 + v_slice**2)
        norm = plt.Normalize(magnitude.min(), magnitude.max())
        plt.get_cmap(colormap)

        print("#####   evolve trajectory   #####")
        #self.out_trajectory = self.evolve_trajectory(x0y0[0], x0y0[1], dt, trajectory_steps)
        print("#####   evolve trajectory done   #####")

        quiv = axs[1].quiver(x_slice, y_slice, u_slice, v_slice, magnitude, cmap=cmap, norm=norm, angles="xy", scale_units="xy", scale=16, width=.004, headwidth=3, headlength=3)
        #axs[1].plot(self.time_grid[start:stop, 0], self.time_grid[start:stop, 1], marker='o', color='black', linestyle='-', markersize=1, alpha=1)
        #axs[1].plot(self.out_trajectory[:, 0], self.out_trajectory[:, 1], marker='o', color='red', linestyle='-', markersize=1, alpha=1)
        axs[1].tick_params(axis='both', labelsize=8)
        axs[1].grid(False)
        axs[1].set_title('Vectorfield and ODE Trajectory', fontsize=12)

        fig.colorbar(quiv, ax=axs[1], aspect=50)
        plt.tight_layout()
        plt.show()

    @no_grad_method
    def euler_integration(self, x0, y0, time_discretization=100, t_final=10):
        dt = 1/time_discretization
        t_values = np.arange(0, t_final + dt, dt)
        n_steps = len(t_values)
        trajectory = np.zeros((n_steps, 2))
        trajectory[0] = [x0, y0]

        for i in range(1, n_steps):
            y_current = trajectory[i - 1]
            dy_dt = self.posterior_dynamics(t_values[i], y_current)
            trajectory[i] = trajectory[i - 1] + dt * np.array(dy_dt)
        return trajectory #, t_values
    
    def solve_ivp_integration(self, x0, y0, time_discretization=100, t_final=10):
        dt = 1/time_discretization
        t_span = (0, t_final)  # Time span
        t_eval = np.arange(0, t_final + dt, dt) 
        sol = solve_ivp(
            self.posterior_dynamics, t_span, [x0, y0],
            method='RK45', t_eval=t_eval
            )
        t_values = sol.t
        trajectory = sol.y.T 
        return trajectory

    

    @no_grad_method
    def x1(self, x1, x2):
        temp = - x1/ self.tau_list[0]
        for i, vi in enumerate(self.list_vi_objects):
            if vi.posterior_rate_dynamical_system is None:
                print('       !!! cal posterior rate dynamical system first !!!')
            temp += np.array(self.couplings[i, 0] * vi.posterior_rate_dynamical_system)
        return temp

    @no_grad_method
    def x2(self, x1, x2):
        temp = - x2/ self.tau_list[1]
        for i, vi in enumerate(self.list_vi_objects):
            if vi.posterior_rate_dynamical_system is None:
                print('       !!! cal posterior rate dynamical system first !!!')
            temp += np.array(self.couplings[i, 1] * vi.posterior_rate_dynamical_system)
        return temp