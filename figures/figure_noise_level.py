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

# MODEL FREE 
# Load the presaved data, else download it from wandb
cov_type = 'max'
if cov_type == 'max':
    figure_data_file = Path(figure_data_folder / "fig_mf_3.pickle")
elif cov_type == 'const':
    figure_data_file = Path(figure_data_folder / "fig_mf_5.pickle")
if figure_data_file.exists() == True:
    figure_data = fp.unpickle_file(figure_data_file)[0]
    group_averages_MF = figure_data["group_averages_MF"]
    group_stds_MF = figure_data["group_stds_MF"]
    labels_MF = figure_data["labels_MF"]
    noise_labels_MF = figure_data["noise_labels_MF"]
    group_idx_arr_MF = figure_data["group_idx_arr_MF"]
else:
    group_histories_MF = {}
    labels_MF = []
    noise_labels_MF = []
    # noise_levels = [0, 0.01, 0.025, 0.05, 0.1, 0.2]
    if cov_type == 'max':
        group_idx_arr_MF = [0, 11, 12, 13, 14, 15]
    elif cov_type == 'const':
        group_idx_arr_MF = [0, 22, 23, 24, 25, 26]
    # group_idx_arr_MF = [0]
    project_name = "DA-RL-MF"
    for group_idx in group_idx_arr_MF:
        group_data, label, config_dict = vis.retrieve_group_data(project_name, group_idx)
        group_histories_MF[group_idx] = group_data
        labels_MF.append(label)
        noise_labels_MF.append(config_dict["enKF"]["std_obs"])
    group_averages_MF, group_stds_MF = vis.compute_group_averages(group_histories_MF)
    # Save the plotting data
    fp.pickle_file(figure_data_file, 
                   {"group_averages_MF": group_averages_MF, 
                    "group_stds_MF": group_stds_MF,
                    "labels_MF": labels_MF,
                    "noise_labels_MF": noise_labels_MF,
                    "group_idx_arr_MF": group_idx_arr_MF,
                    })

# colors  
legend_labels = []
for noise_label in noise_labels_MF:
    legend_label = f"${100 * noise_label}\%$"
    legend_labels.append(legend_label)
colors = sns.color_palette("husl", len(legend_labels)).as_hex()    
colors_dict = {label: color for label, color in zip(legend_labels, colors)}

# ylimits
ylims = [[-2000,-100],
         [0,4],
         [0,100]]

plt.style.use("stylesheet.mplstyle")
fig, axs = vis.plot_training_curves(group_averages=group_averages_MF, 
                        group_stds=group_stds_MF, 
                        labels=legend_labels, 
                        group_idx_arr=group_idx_arr_MF, 
                        hist_name="train.episode_return",
                        label_type="noise_level",
                        num_sensors_to_plot=None,
                        colors_dict=colors_dict,
                        ylims=ylims)
# axs[0].annotate('(a)', xy=(0.03, 0.7), xycoords='axes fraction')
# axs[1].annotate('(b)', xy=(0.03, 0.7), xycoords='axes fraction')
# axs[2].annotate('(c)', xy=(0.03, 0.7), xycoords='axes fraction')
handles, _ = axs[0].get_legend_handles_labels()
plt.figlegend(handles, 
              legend_labels, 
            #   ncols=len(legend_labels), 
              loc ='right', 
            #   bbox_to_anchor=(1.0, 1.15), 
              bbox_to_anchor=(1.2, 0.5), 
              frameon=False)
# plt.legend()
if cov_type == 'max':
    fig.savefig(figure_folder/"ppt_fig_mf_3.png", bbox_inches='tight')
    # fig.savefig(figure_folder/"fig_mf_3.pdf", bbox_inches='tight')
elif cov_type == 'const':
    fig.savefig(figure_folder/"ppt_fig_mf_5.png", bbox_inches='tight')
    # fig.savefig(figure_folder/"fig_mf_5.pdf", bbox_inches='tight')