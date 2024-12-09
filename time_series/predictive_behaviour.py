import torch
import numpy as np
import matplotlib.pyplot as plt
from dynamical_system.dynamical_system import ODE
from helpers.helpers import Helper
from helpers.helpers import no_grad_method


class Predictive_Behaviour(ODE):
    def __init__(self, time_grid, list_vi_objects, grid_padding=5, evolve_steps=2000, x0=0, y0=0, z0=None):
        super().__init__(time_grid, list_vi_objects, grid_padding)
        self.time_discretization = list_vi_objects[0].time_discretization
        self.evolve_steps = evolve_steps
        self.start_point = list_vi_objects[0].time_grid[list_vi_objects[0].end_time * self.time_discretization -1]
        self.samples_events = []
        self.list_events = None
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

    @no_grad_method
    def get_predicted_time_grid(self):
        x0 = self.x0   # self.start_point[0]
        y0 = self.y0  # self.start_point[1]
        print(f'start point {x0} and {y0}')
        x1, x2 = x0, y0
        trajectory = [(x0, y0)]
        gp_of_trajectory = [] 
        self.list_events = [[] for _ in range(len(self.list_vi_objects))]
        for _ in range(self.evolve_steps):
            konvergenzx, konvergenzz, vx, vy, gp_s = self.posterior_dynamics(x1, x2)
            x1_new = x1 + (vx)
            x2_new = x2 + (vy)
            #print(f'ajustments x1 {vx / self.time_discretization} and x2 {vy / self.time_discretization}')
            trajectory.append((x1_new, x2_new)) 
            gp_of_trajectory.append(gp_s)
            x1, x2 = x1_new, x2_new
            #print('-------------------')
        trajectory_np = np.array(trajectory)
        for i in range(len(self.list_events)):
            self.list_events[i] = np.array(self.list_events[i])
        return trajectory_np[:-1], np.array(gp_of_trajectory), self.list_events
        
    @no_grad_method
    def posterior_dynamics(self, x1, x2):
        konvergenz_x1 = (x1 / self.time_discretization) / self.tau_list[0]
        konvergenz_x2 = (x2 / self.time_discretization) /self.tau_list[1]
        dx1dt = -konvergenz_x1
        dx2dt = -konvergenz_x2
        gp_s = np.zeros(len(self.list_vi_objects))
        konv_count = 0
        for i, vi in enumerate(self.list_vi_objects):
            post_gp_at_x1_x2, sig_E_post_rate = vi.eval_post_gp(x1, x2)
            gp_s[i] = post_gp_at_x1_x2
            print(f'   process {i}, location {x1} and {x2}, and post_gp_at_x1_x2 {post_gp_at_x1_x2}')
            probability_for_event = post_gp_at_x1_x2 / (vi.time_discretization)
            if np.random.uniform(0, 1) < probability_for_event:
                print(f'event {i} at {x1} and {x2}')
                print(f'   in x1 {vi.couplings[i, 0]} and in x2 {vi.couplings[i, 1]}')
                print(f'   konv x1 {konvergenz_x1} and konv x2 {konvergenz_x2}')
                self.list_events[i].append((x1, x2))
                dx1dt += np.array(self.couplings[i, 0])  #* post_gp_at_x1_x2)
                dx2dt += np.array(self.couplings[i, 1])  #* post_gp_at_x1_x2)
        #print(f'returned dx1dt {dx1dt} and dx2dt {dx2dt}')
        return konvergenz_x1, konvergenz_x2, dx1dt, dx2dt, gp_s
    
    @no_grad_method
    def get_predicted_time_grid_3d(self):
        x0 = self.x0   # self.start_point[0]
        y0 = self.y0  # self.start_point[1]
        z0 = self.z0
        print(f'start point {x0} and {y0} and {z0}')
        x1, x2, x3 = x0, y0, z0
        trajectory = [(x0, y0, z0)]
        gp_of_trajectory = [] 
        self.list_events = [[] for _ in range(len(self.list_vi_objects))]
        for _ in range(self.evolve_steps):
            vx, vy, vz, gp_s = self.posterior_dynamics_3d(x1, x2, x3)
            x1_new = x1 + (vx)
            x2_new = x2 + (vy)
            x3_new = x3 + (vz)
            #print(f'ajustments x1 {vx / self.time_discretization} and x2 {vy / self.time_discretization}')
            trajectory.append((x1_new, x2_new, x3_new)) 
            gp_of_trajectory.append(gp_s)
            x1, x2, x3 = x1_new, x2_new, x3_new
            #print('-------------------')
        trajectory_np = np.array(trajectory)
        for i in range(len(self.list_events)):
            self.list_events[i] = np.array(self.list_events[i])
        return trajectory_np[:-1], np.array(gp_of_trajectory), self.list_events
    
    @no_grad_method
    def posterior_dynamics_3d(self, x1, x2, x3):
        konvergenz_x1 = (x1 / self.time_discretization) / self.tau_list[0]
        konvergenz_x2 = (x2 / self.time_discretization) /self.tau_list[1]
        konvergenz_x3 = (x3 / self.time_discretization) /self.tau_list[2]
        dx1dt = -konvergenz_x1
        dx2dt = -konvergenz_x2
        dx3dt = -konvergenz_x3
        gp_s = np.zeros(len(self.list_vi_objects))
        for i, vi in enumerate(self.list_vi_objects):
            post_gp_at_x1_x2, sig_E_post_rate = vi.eval_post_gp(x1, x2, x3)
            gp_s[i] = post_gp_at_x1_x2
            print(f'   process {i}, location {x1} and {x2}, and post_gp_at_x1_x2 {post_gp_at_x1_x2}')
            probability_for_event = post_gp_at_x1_x2 / (vi.time_discretization)
            if np.random.uniform(0, 1) < probability_for_event:
                print(f'event {i} at {x1} and {x2}')
                print(f'   in x1 {vi.couplings[i, 0]} and in x2 {vi.couplings[i, 1]}')
                print(f'   konv x1 {konvergenz_x1} and konv x2 {konvergenz_x2}')
                self.list_events[i].append((x1, x2, x3))
                dx1dt += np.array(self.couplings[i, 0])  #* post_gp_at_x1_x2)
                dx2dt += np.array(self.couplings[i, 1])  #* post_gp_at_x1_x2)
                dx3dt += np.array(self.couplings[i, 2])
        #print(f'returned dx1dt {dx1dt} and dx2dt {dx2dt}')
        return dx1dt, dx2dt, dx3dt, gp_s
