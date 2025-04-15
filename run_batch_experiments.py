import os

# Base command template
# base_command_1 = (
#     "python ddpg_experiment_v3.py --gpu_id 0 --log_wandb --make_plots --wandb_config.project DA-RL-MF --wandb_config.group {group_idx} --config.seed {seed}"
# )
# base_command = (
#     "python ddpg_experiment_v3.py --gpu_id 0 --log_wandb --wandb_config.project DA-RL-MF --wandb_config.group {group_idx} --config.seed {seed}"
# )

# base_command_1 = (
#     "python ddpg_experiment_v3.py --gpu_id 0 --log_wandb --make_plots --wandb_config.project DA-RL-MF --wandb_config.group {group_idx} --config.seed {seed} --config.enKF.wait_steps {wait_steps}"
# )
# base_command = (
#     "python ddpg_experiment_v3.py --gpu_id 0 --log_wandb --wandb_config.project DA-RL-MF --wandb_config.group {group_idx} --config.seed {seed} --config.enKF.wait_steps {wait_steps}"
# )

# base_command_1 = (
#     "python ddpg_with_enkf_experiment_v3.py --gpu_id 0 --log_wandb --make_plots --wandb_config.project DA-RL-MB-Fo --wandb_config.group {group_idx} --config.seed {seed}"
# )
# base_command = (
#     "python ddpg_with_enkf_experiment_v3.py --gpu_id 0 --log_wandb --wandb_config.project DA-RL-MB-Fo --wandb_config.group {group_idx} --config.seed {seed}"
# )

# base_command_1 = (
#     "python ddpg_with_enkf_experiment_v3.py --gpu_id 1 --log_wandb --make_plots --wandb_config.project DA-RL-MB-Fo --wandb_config.group {group_idx} --config.seed {seed} --config.enKF.low_order_N {low_order_N} --config.enKF.wait_steps {wait_steps}"
# )
# base_command = (
#     "python ddpg_with_enkf_experiment_v3.py --gpu_id 1 --log_wandb --wandb_config.project DA-RL-MB-Fo --wandb_config.group {group_idx} --config.seed {seed} --config.enKF.low_order_N {low_order_N} --config.enKF.wait_steps {wait_steps}"
# )

# base_command_1 = (
#     "python ddpg_with_enkf_esn.py --gpu_id 0 --log_wandb --make_plots --wandb_config.project DA-RL-MB-ESN --wandb_config.group {group_idx} --config.seed {seed}"
# )
# base_command = (
#     "python ddpg_with_enkf_esn.py --gpu_id 0 --log_wandb --wandb_config.project DA-RL-MB-ESN --wandb_config.group {group_idx} --config.seed {seed}"
# )

# base_command_1 = (
#     "python ddpg_with_enkf_esn.py --gpu_id 1 --log_wandb --make_plots --wandb_config.project DA-RL-MB-ESN --wandb_config.group {group_idx} --config.seed {seed} --esn_config.model.reservoir_size {N_reservoir}"
# )
# base_command = (
#     "python ddpg_with_enkf_esn.py --gpu_id 1 --log_wandb --wandb_config.project DA-RL-MB-ESN --wandb_config.group {group_idx} --config.seed {seed} --esn_config.model.reservoir_size {N_reservoir}"
# )

# base_command_1 = (
#     "python ddpg_with_enkf_esn.py --gpu_id 1 --log_wandb --make_plots --wandb_config.project DA-RL-MB-ESN --wandb_config.group {group_idx} --config.seed {seed} --esn_config.model.reservoir_size {N_reservoir} --env_config.sensor_locs {n_sensor}"
# )
# base_command = (
#     "python ddpg_with_enkf_esn.py --gpu_id 1 --log_wandb --wandb_config.project DA-RL-MB-ESN --wandb_config.group {group_idx} --config.seed {seed} --esn_config.model.reservoir_size {N_reservoir} --env_config.sensor_locs {n_sensor}"
# )


base_command_1 = (
    "python ddpg_with_enkf_esn.py --gpu_id 1 --log_wandb --make_plots --wandb_config.project DA-RL-MB-ESN --wandb_config.group {group_idx} --config.seed {seed} --esn_config.model.reservoir_size {N_reservoir} --esn_hyp {hyp_file_name} --env_config.sensor_locs {n_sensor}"
)
base_command = (
    "python ddpg_with_enkf_esn.py --gpu_id 1 --log_wandb --wandb_config.project DA-RL-MB-ESN --wandb_config.group {group_idx} --config.seed {seed} --esn_config.model.reservoir_size {N_reservoir} --esn_hyp {hyp_file_name} --env_config.sensor_locs {n_sensor}"
)

# base_command_1 = (
#     "python ddpg_with_enkf_esn.py --gpu_id 2 --log_wandb --make_plots --wandb_config.project DA-RL-MB-ESN --wandb_config.group {group_idx} --config.seed {seed} --esn_hyp {hyp_file_name}"
# )
# base_command = (
#     "python ddpg_with_enkf_esn.py --gpu_id 2 --log_wandb --wandb_config.project DA-RL-MB-ESN --wandb_config.group {group_idx} --config.seed {seed} --esn_hyp {hyp_file_name}"
# )
# Iterate over 5 seeds
base_idx = 49
noise_level_arr = [0.0]
wait_steps_arr = [100, 250] 
low_order_N_arr = [16, 64]
# N_reservoir_list = [300, 1000, 3000]
N_reservoir_list = [1000]
n_sensor_list = [5,6,8,10,12]
# run_name_list = ["run_20250218_123022",
#                  "run_20250218_125849",
#                  "run_20250218_132755",
#                  "run_20250218_135436",
#                  "run_20250218_142104",
#                  ]
# nu = 0.08
# run_name_list = [#["run_20250316_123912","run_20250316_125739","run_20250316_131140","run_20250316_132536","run_20250316_133945"], # N_reservoir=300
#                  #["run_20250318_142912", "run_20250318_144815", "run_20250318_150318", "run_20250318_151811", "run_20250318_153305"] # N_reservoir=500
#                  #["run_20250316_135350","run_20250316_141618","run_20250316_143407","run_20250316_145159","run_20250316_151003"], # N_reservoir=1000
#                  #["run_20250316_152758","run_20250316_160638","run_20250316_164108","run_20250316_171608","run_20250316_175054"]  # N_reservoir=3000
#                  ]

# nu = 0.05
# run_name_list = [["run_20250409_154246", "run_20250409_161028", "run_20250409_163451", "run_20250409_170014", "run_20250409_172354"]] # N_reservoir = 1000 

# nu = 0.03
run_name_list = [["run_20250410_122336", "run_20250410_124435", "run_20250410_130123", "run_20250410_131814", "run_20250410_133502"]] # N_reservoir = 1000 


# NOISE LEVELS
# for group_idx in range(5):
#     noise_level = noise_level_arr[group_idx]
#     for seed in range(5):
#         # Format the command with the current seed
#         if seed == 0:
#             command = base_command_1.format(seed=seed, group_idx=base_idx + group_idx, noise_level=noise_level)
#         else:
#             command = base_command.format(seed=seed, group_idx=base_idx + group_idx, noise_level=noise_level)
#         # Print the command for visibility
#         print(f"Running: {command}")
        
#         # Execute the command
#         os.system(command)

# WAIT STEPS
# for group_idx in range(1):
#     wait_steps = wait_steps_arr[group_idx]
#     for seed in range(1):
#         # Format the command with the current seed
#         if seed == 0:
#             command = base_command_1.format(seed=seed, group_idx=base_idx + group_idx, wait_steps=wait_steps)
#         else:
#             command = base_command.format(seed=seed, group_idx=base_idx + group_idx, wait_steps=wait_steps)
#         # Print the command for visibility
#         print(f"Running: {command}")
        
#         # Execute the command
#         os.system(command)

# # # BASIC RUN
# for seed in range(5):
#     # Format the command with the current seed
#     if seed == 0:
#         command = base_command_1.format(seed=seed, group_idx=base_idx)
#     else:
#         command = base_command.format(seed=seed, group_idx=base_idx)
#     # Print the command for visibility
#     print(f"Running: {command}")
    
#     # Execute the command
#     os.system(command)

# Reservoir size
# for group_idx in range(len(N_reservoir_list)):
#     N_reservoir = N_reservoir_list[group_idx]
#     for seed in range(5):
#         hyp_file_name = f"local_results/KS/{run_name_list[group_idx][seed]}/esn_hyperparameters.h5"
#         # Format the command with the current seed
#         if seed == 0:
#             command = base_command_1.format(seed=seed, group_idx=base_idx + group_idx, N_reservoir=N_reservoir, hyp_file_name=hyp_file_name)
#         else:
#             command = base_command.format(seed=seed, group_idx=base_idx + group_idx, N_reservoir=N_reservoir, hyp_file_name=hyp_file_name)
#         # Print the command for visibility
#         print(f"Running: {command}")
        
#         # Execute the command
#         os.system(command)

# for r_idx in range(len(N_reservoir_list)):
#     N_reservoir = N_reservoir_list[r_idx]
#     for n_sensor_idx in range(len(n_sensor_list)):
#         n_sensor = n_sensor_list[n_sensor_idx]
#         group_idx = r_idx * len(n_sensor_list) + n_sensor_idx
#         for seed in range(5):
#             # Format the command with the current seed
#             if seed == 0:
#                 command = base_command_1.format(seed=seed, group_idx=base_idx + group_idx, N_reservoir=N_reservoir, n_sensor=n_sensor)
#             else:
#                 command = base_command.format(seed=seed, group_idx=base_idx + group_idx, N_reservoir=N_reservoir, n_sensor=n_sensor)
#             # Print the command for visibility
#             print(f"Running: {command}")
            
#             # Execute the command
#             os.system(command)

for r_idx in range(len(N_reservoir_list)):
    N_reservoir = N_reservoir_list[r_idx]
    for n_sensor_idx in range(len(n_sensor_list)):
        n_sensor = n_sensor_list[n_sensor_idx]
        group_idx = r_idx * len(n_sensor_list) + n_sensor_idx
        for seed in range(5):
            hyp_file_name = f"local_results/KS/{run_name_list[r_idx][seed]}/esn_hyperparameters.h5"
            # Format the command with the current seed
            if seed == 0:
                command = base_command_1.format(seed=seed, group_idx=base_idx + group_idx, N_reservoir=N_reservoir, hyp_file_name=hyp_file_name, n_sensor=n_sensor)
            else:
                command = base_command.format(seed=seed, group_idx=base_idx + group_idx, N_reservoir=N_reservoir, hyp_file_name=hyp_file_name, n_sensor=n_sensor)
            # Print the command for visibility
            print(f"Running: {command}")
            
            # Execute the command
            os.system(command)

# # BASIC RUN with hyp file
# for seed in range(5):
#     # Format the command with the current seed
#     hyp_file_name = f"local_results/KS/{run_name_list[seed]}/esn_hyperparameters.h5"
#     if seed == 0:
#         command = base_command_1.format(seed=seed, group_idx=base_idx, hyp_file_name=hyp_file_name)
#     else:
#         command = base_command.format(seed=seed, group_idx=base_idx, hyp_file_name=hyp_file_name)
#     # Print the command for visibility
#     print(f"Running: {command}")
    
#     # Execute the command
#     os.system(command)

# Low order N
# for group_idx in range(5):
#     low_order_N = low_order_N_arr[group_idx]
#     for seed in range(5):
#         # Format the command with the current seed
#         if seed == 0:
#             command = base_command_1.format(seed=seed, group_idx=base_idx + group_idx, low_order_N=low_order_N)
#         else:
#             command = base_command.format(seed=seed, group_idx=base_idx + group_idx, low_order_N=low_order_N)
#         # Print the command for visibility
#         print(f"Running: {command}")
        
#         # Execute the command
#         os.system(command)


# Low order N
# for order_idx in range(len(low_order_N_arr)):
#     low_order_N = low_order_N_arr[order_idx]
#     # wait 
#     for wait_idx in range(len(wait_steps_arr)):
#         wait_steps = wait_steps_arr[wait_idx]
#         group_idx = order_idx * len(wait_steps_arr) + wait_idx
#         for seed in range(5):
#             # Format the command with the current seed
#             if seed == 0:
#                 command = base_command_1.format(seed=seed, group_idx=base_idx + group_idx, low_order_N=low_order_N, wait_steps=wait_steps)
#             else:
#                 command = base_command.format(seed=seed, group_idx=base_idx + group_idx, low_order_N=low_order_N, wait_steps=wait_steps)
#             # Print the command for visibility
#             print(f"Running: {command}")
            
#             # Execute the command
#             os.system(command)