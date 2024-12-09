from variational_inference.variational_inference import VariationalInference

def optimization_loop(i, list_full_processes, optimality_parameters, hyperparameters, 
                      time_grid_parameters, ts_helper, num_inducing_points_per_dim, 
                      learning_rate, time_hyp_opt, termination_threshold_opt, 
                      max_iterations, epochs_per_hyp_tuning,
                      list_vi_objects, loss_tracker_all_processes, gp_kernel='RBF',  #, global_full_stack_params
                      hyperparam_time_tracker=0, opt_time_tracker=0):
    
    if i == 0:
        global_full_stack_params = []
        full_opt_time_tracker = 0
        for j in range(len(list_full_processes)):
            print(f'############ Optimization Process: {j+1} ({max_iterations} Iterations) ############')
            vi = VariationalInference(optimality_parameters, hyperparameters, time_grid_parameters, ts_helper, list_full_processes[j], 
                            num_inducing_points_per_dim, learning_rate, kernel_name=gp_kernel)
            loss_tracker, full_stack_params, h_time_tracker, opt_time_tracker = vi.run_optimization_single_process(time_hyp_opt, termination_threshold_opt, max_iterations, epochs_per_hyp_tuning)
            list_vi_objects.append(vi)
            loss_tracker_all_processes.append(loss_tracker)
            global_full_stack_params.append(full_stack_params)
            hyperparam_time_tracker += h_time_tracker
            full_opt_time_tracker += opt_time_tracker
            vi.plot_minimal(xlim=3000)
            print('\n')
    else:
        global_full_stack_params = []
        full_opt_time_tracker = 0
        for j, vi in enumerate(list_vi_objects):
            print(f'############ Optimization Process: {j+1} ({max_iterations} Iterations) ############')
            loss_tracker, full_stack_params, h_time_tracker, opt_time_tracker = vi.run_optimization_single_process(time_hyp_opt, termination_threshold_opt, max_iterations, epochs_per_hyp_tuning)
            loss_tracker_all_processes[j] = loss_tracker           
            global_full_stack_params.append(full_stack_params)
            hyperparam_time_tracker += h_time_tracker
            full_opt_time_tracker += opt_time_tracker
            vi.plot_minimal(xlim=3000)
            print('\n')
    
    return list_vi_objects, loss_tracker_all_processes, global_full_stack_params, hyperparam_time_tracker, full_opt_time_tracker