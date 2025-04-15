from pathlib import Path
import utils.training_visualizations as vis
import utils.file_processing as fp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

figure_data_folder = Path("local_figure_data")
figure_data_folder.mkdir(exist_ok=True)
figure_folder = Path("local_figures")
figure_folder.mkdir(exist_ok=True)

locs_ga = ((2 * np.pi) / 8) * np.arange(8)
locs_gb = (2 * np.pi) / 16 + ((2 * np.pi) / 8) * np.arange(8)
# MODEL FREE 
# Load the presaved data, else download it from wandb
figure_data_file = Path(figure_data_folder / "fig_mf_2.pickle")
if figure_data_file.exists() == True:
    figure_data = fp.unpickle_file(figure_data_file)[0]
    group_averages_MF = figure_data["group_averages_MF"]
    group_stds_MF = figure_data["group_stds_MF"]
    labels_MF = figure_data["labels_MF"]
    sensor_labels_MF = figure_data["sensor_labels_MF"]
    actuator_labels_MF = figure_data["actuator_labels_MF"]
    group_idx_arr_MF = figure_data["group_idx_arr_MF"]
else:
    group_histories_MF = {}
    labels_MF = []
    sensor_labels_MF = []
    actuator_labels_MF = []
    # [[sensors (a), actuators (a)],
    #  [sensors (b), actuators (a)],
    #  [sensors (a), actuators (b)],
    #  [sensors (b), actuators (b)]]
    group_idx_arr_MF = [3, 30, 31, 32]
    # group_idx_arr_MF = [0]
    project_name = "DA-RL-MF"
    for group_idx in group_idx_arr_MF:
        group_data, label, config_dict = vis.retrieve_group_data(project_name, group_idx)
        group_histories_MF[group_idx] = group_data
        labels_MF.append(label)
        sensor_locs = config_dict["env"]["sensor_locs"]
        actuator_locs = config_dict["env"]["actuator_locs"]
        if all(sensor_locs == locs_ga):
            sensor_labels_MF.append('ga')
        elif all(sensor_locs == locs_gb):
            sensor_labels_MF.append('gb')
        if all(actuator_locs == locs_ga):
            actuator_labels_MF.append('ga')
        elif all(actuator_locs == locs_gb):
            actuator_labels_MF.append('gb')
    group_averages_MF, group_stds_MF = vis.compute_group_averages(group_histories_MF)
    # Save the plotting data
    fp.pickle_file(figure_data_file, 
                   {"group_averages_MF": group_averages_MF, 
                    "group_stds_MF": group_stds_MF,
                    "labels_MF": labels_MF,
                    "sensor_labels_MF": sensor_labels_MF,
                    "actuator_labels_MF": actuator_labels_MF,
                    "group_idx_arr_MF": group_idx_arr_MF,
                    })

# colors  
legend_labels = []
for sensor_label, actuator_label in zip(sensor_labels_MF, actuator_labels_MF):
    legend_label = f"$\mathbf{{x}}_a = \mathbf{{x}}_{{{actuator_label}}}, \; \mathbf{{x}}_o = \mathbf{{x}}_{{{sensor_label}}}$"
    legend_labels.append(legend_label)
colors = sns.color_palette("husl", len(legend_labels)).as_hex()    
colors_dict = {label: color for label, color in zip(legend_labels, colors)}

# ylimits
ylims = [[-2000,-100],
         [0,0.5],
         [0,100]]

plt.style.use("stylesheet.mplstyle")
fig, axs = vis.plot_training_curves(group_averages=group_averages_MF, 
                        group_stds=group_stds_MF, 
                        labels=legend_labels, 
                        group_idx_arr=group_idx_arr_MF, 
                        hist_name="train.episode_return",
                        label_type="grid_config",
                        num_sensors_to_plot=None,
                        colors_dict=colors_dict,
                        ylims=ylims)
axs[0].annotate('(a)', xy=(0.03, 0.7), xycoords='axes fraction')
axs[1].annotate('(b)', xy=(0.03, 0.7), xycoords='axes fraction')
axs[2].annotate('(c)', xy=(0.03, 0.7), xycoords='axes fraction')
handles, _ = axs[0].get_legend_handles_labels()
plt.figlegend(handles, 
              legend_labels, 
              ncols=len(legend_labels)//2, 
              loc ='upper right', 
              bbox_to_anchor=(1.0, 1.3), 
              frameon=False)
# plt.legend()
fig.savefig(figure_folder/"fig_mf_2.png", bbox_inches='tight')
fig.savefig(figure_folder/"fig_mf_2.pdf", bbox_inches='tight')