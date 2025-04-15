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
figure_data_file = Path(figure_data_folder / "fig_mf_1.pickle")
if figure_data_file.exists() == True:
    figure_data = fp.unpickle_file(figure_data_file)[0]
    group_averages_MF = figure_data["group_averages_MF"]
    group_stds_MF = figure_data["group_stds_MF"]
    labels_MF = figure_data["labels_MF"]
    group_idx_arr_MF = figure_data["group_idx_arr_MF"]
else:
    group_histories_MF = {}
    labels_MF = []
    # num_sensors = [64, 12, 8, 6, 4, 7]
    group_idx_arr_MF = [0, 2, 3, 4, 5, 6]
    # group_idx_arr_MF = [0]
    project_name = "DA-RL-MF"
    for group_idx in group_idx_arr_MF:
        group_data, label, _ = vis.retrieve_group_data(project_name, group_idx)
        group_histories_MF[group_idx] = group_data
        labels_MF.append(label)
    group_averages_MF, group_stds_MF = vis.compute_group_averages(group_histories_MF)
    # Save the plotting data
    fp.pickle_file(figure_data_file, 
                   {"group_averages_MF": group_averages_MF, 
                    "group_stds_MF": group_stds_MF,
                    "labels_MF": labels_MF,
                    "group_idx_arr_MF": group_idx_arr_MF})
    
# MODEL BASED FOURIER
# Load the presaved data, else download it from wandb
figure_data_file = Path(figure_data_folder / "fig_mb_fo_1.pickle")
if figure_data_file.exists() == True:
    figure_data = fp.unpickle_file(figure_data_file)[0]
    group_averages_MB_Fo = figure_data["group_averages_MB_Fo"]
    group_stds_MB_Fo = figure_data["group_stds_MB_Fo"]
    labels_MB_Fo = figure_data["labels_MB_Fo"]
    group_idx_arr_MB_Fo = figure_data["group_idx_arr_MB_Fo"]
else:
    group_histories_MB_Fo = {}
    labels_MB_Fo = []
    # num_sensors = [4, 3, 2, 5, 6, 7]
    group_idx_arr_MB_Fo = [2, 7, 12, 17, 22, 27] 
    project_name = "DA-RL-MB-Fo"
    for group_idx in group_idx_arr_MB_Fo:
        group_data, label, _ = vis.retrieve_group_data(project_name, group_idx)
        group_histories_MB_Fo[group_idx] = group_data
        labels_MB_Fo.append(label)

    group_averages_MB_Fo, group_stds_MB_Fo = vis.compute_group_averages(group_histories_MB_Fo)
    # Save the plotting data
    fp.pickle_file(figure_data_file, 
                   {"group_averages_MB_Fo": group_averages_MB_Fo, 
                    "group_stds_MB_Fo": group_stds_MB_Fo,
                    "labels_MB_Fo": labels_MB_Fo,
                    "group_idx_arr_MB_Fo": group_idx_arr_MB_Fo})

# MODEL BASED ESN
# Load the presaved data, else download it from wandb
figure_data_file = Path(figure_data_folder / "fig_mb_esn_2.pickle")
if figure_data_file.exists() == True:
    figure_data = fp.unpickle_file(figure_data_file)[0]
    group_averages_MB_ESN = figure_data["group_averages_MB_ESN"]
    group_stds_MB_ESN = figure_data["group_stds_MB_ESN"]
    labels_MB_ESN = figure_data["labels_MB_ESN"]
    group_idx_arr_MB_ESN = figure_data["group_idx_arr_MB_ESN"]
else:
    group_histories_MB_ESN = {}
    labels_MB_ESN = []
    # group_idx_arr_MB_ESN = [0, 1, 2, 3, 4, 5] # r2_mode
    group_idx_arr_MB_ESN = [7,9,10,11,12,13] # input bias
    project_name = "DA-RL-MB-ESN"
    for group_idx in group_idx_arr_MB_ESN:
        group_data, label, _ = vis.retrieve_group_data(project_name, group_idx)
        group_histories_MB_ESN[group_idx] = group_data
        labels_MB_ESN.append(label)

    group_averages_MB_ESN, group_stds_MB_ESN = vis.compute_group_averages(group_histories_MB_ESN)
    # Save the plotting data
    fp.pickle_file(figure_data_file, 
                   {"group_averages_MB_ESN": group_averages_MB_ESN, 
                    "group_stds_MB_ESN": group_stds_MB_ESN,
                    "labels_MB_ESN": labels_MB_ESN,
                    "group_idx_arr_MB_ESN": group_idx_arr_MB_ESN})

# colors  
all_labels = labels_MF + labels_MB_Fo + labels_MB_ESN
all_labels = list(set(all_labels))
colors = sns.color_palette("husl", len(all_labels)).as_hex()    
colors_dict = {label: color for label, color in zip(np.sort(all_labels), colors)}

# ylimits
ylims = [[-2000,-100],
         [0,0.5],
         [0,100]]

plt.style.use("stylesheet.mplstyle")
fig, axs = vis.plot_training_curves(group_averages=group_averages_MF, 
                        group_stds=group_stds_MF, 
                        labels=labels_MF, 
                        group_idx_arr=group_idx_arr_MF, 
                        hist_name="train.episode_return",
                        num_sensors_to_plot=None,
                        colors_dict=colors_dict,
                        ylims=ylims)
# axs[0].annotate('(a)', xy=(0.03, 0.7), xycoords='axes fraction')
# axs[1].annotate('(b)', xy=(0.03, 0.7), xycoords='axes fraction')
# axs[2].annotate('(c)', xy=(0.03, 0.7), xycoords='axes fraction')
handles, legend_labels = axs[0].get_legend_handles_labels()
plt.figlegend(handles, 
              legend_labels, 
            #   ncols=len(legend_labels), 
              loc ='right', 
            #   bbox_to_anchor=(1.0, 1.15), 
              bbox_to_anchor=(1.2, 0.5), 
              frameon=False)
fig.savefig(figure_folder/"ppt_fig_mf_1.png", bbox_inches='tight')
# fig.savefig(figure_folder/"fig_mf_1.pdf", bbox_inches='tight')

fig, axs = vis.plot_training_curves(group_averages=group_averages_MB_Fo, 
                         group_stds=group_stds_MB_Fo, 
                         labels=labels_MB_Fo, 
                         group_idx_arr=group_idx_arr_MB_Fo, 
                         hist_name="train.episode_return_env",
                         num_sensors_to_plot=None,
                         colors_dict=colors_dict,
                         ylims=ylims)
num_sensors_64_idx = labels_MF.index(64)
group_idx_64 = group_idx_arr_MF[num_sensors_64_idx]
axs[0].hlines(group_averages_MF[group_idx_64]["train.episode_return"].max(), 
          0, len(group_averages_MF[group_idx_64]["_step"])-1, 
          colors='k', 
          linestyles='dashed',
          label='Ref.')
# axs[0].annotate('(a)', xy=(0.03, 0.7), xycoords='axes fraction')
# axs[1].annotate('(b)', xy=(0.03, 0.7), xycoords='axes fraction')
# axs[2].annotate('(c)', xy=(0.03, 0.7), xycoords='axes fraction')
handles, legend_labels = axs[0].get_legend_handles_labels()
plt.figlegend(handles, 
              legend_labels, 
            #   ncols=len(legend_labels), 
              loc ='right', 
            #   bbox_to_anchor=(1.0, 1.15), 
              bbox_to_anchor=(1.2, 0.5), 
              frameon=False)
fig.savefig(figure_folder/"ppt_fig_mb_fo_1.png", bbox_inches='tight')
# fig.savefig(figure_folder/"fig_mb_fo_1.pdf", bbox_inches='tight')

fig, axs = vis.plot_training_curves(group_averages=group_averages_MB_ESN, 
                         group_stds=group_stds_MB_ESN, 
                         labels=labels_MB_ESN, 
                         group_idx_arr=group_idx_arr_MB_ESN, 
                         hist_name="train.episode_return_env",
                         num_sensors_to_plot=None,
                         colors_dict=colors_dict,
                         ylims=ylims)
axs[0].hlines(group_averages_MF[group_idx_64]["train.episode_return"].max(), 
          0, len(group_averages_MF[group_idx_64]["_step"])-1, 
          colors='k', 
          linestyles='dashed',
          label='Ref.')
# axs[0].annotate('(a)', xy=(0.03, 0.7), xycoords='axes fraction')
# axs[1].annotate('(b)', xy=(0.03, 0.7), xycoords='axes fraction')
# axs[2].annotate('(c)', xy=(0.03, 0.7), xycoords='axes fraction')
handles, legend_labels = axs[0].get_legend_handles_labels()
plt.figlegend(handles, 
              legend_labels, 
            #   ncols=len(legend_labels), 
              loc ='right', 
            #   bbox_to_anchor=(1.0, 1.15),
              bbox_to_anchor=(1.2, 0.5),  
              frameon=False)
fig.savefig(figure_folder/"ppt_fig_mb_esn_2.png", bbox_inches='tight')
# fig.savefig(figure_folder/"fig_mb_esn_2.pdf", bbox_inches='tight')

