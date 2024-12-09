import numpy as np
from scipy.integrate import solve_ivp

########## Van der Pol oscillator ##########

def generate_vdp(n_timesteps, T, x0y0=[1, 0], mu=0.5):
    t_start = 0
    t = np.linspace(t_start, T, n_timesteps)
    
    # Define the vdp function with mu
    vdp = lambda t, z: [z[1], mu * (1 - z[0]**2) * z[1] - z[0]]    
    sol = solve_ivp(vdp, [t_start, T], x0y0, t_eval=t)
    return sol
