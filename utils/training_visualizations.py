import wandb
from pathlib import Path
import utils.file_processing as fp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

api = wandb.Api()
results_folder = Path("local_results/KS")

def retrieve_group_data(project_name, group_idx):
    runs = api.runs(
        f"defneozan/{project_name}",
        {"$and": [{"group": f"{group_idx}"}]},
        )
    group_data = []  # Store all runs in this group
    for run_idx, run in enumerate(runs):
        results_path = Path(results_folder / run.config["local_path"])
        # cfg = fp.load_config(results_path)

        total_learning_steps = run.config["total_steps"] - run.config["learning_starts"]
        scan_interval = 10000  # How many steps to fetch per page

        if run_idx == 0:
            label = len(run.config["env"]["sensor_locs"])
            config_dict = run.config

        # Fetch full history with proper pagination
        hist = []
        min_step = 0
        while min_step < total_learning_steps:
            partial_history = list(run.scan_history(
                keys=["_step", "q_loss", "policy_loss"], 
                min_step=min_step, 
                max_step=min_step + scan_interval,
                page_size=scan_interval
            ))

            if not partial_history:
                break  # Stop when no more data is returned

            hist.extend(partial_history)
            min_step += scan_interval  # Move to next batch

        # Convert to DataFrame and store
        df1 = pd.DataFrame(hist)

        # Get the history of train.episode_return
        if project_name == "DA-RL-MF":
            hist_names = ["train.episode_return"]
        elif project_name == "DA-RL-MB-Fo" or project_name == "DA-RL-MB-ESN":
            hist_names = ["train.episode_return_env", "train.episode_return_model"]

        df2 = run.history(keys=hist_names, pandas=True) # we can do this because number of episodes is low

        # Merge both DataFrames on `_step`
        df = pd.merge(df1, df2, on="_step", how="outer").sort_values("_step")

        # Forward fill NaNs in train.episode_return before averaging
        for hist_name in hist_names:
            df[hist_name] = df[hist_name].ffill()

        group_data.append(df)

    return group_data, label, config_dict

def compute_group_averages(group_histories):
    group_averages = {}
    group_stds = {}
    for group_idx, runs in group_histories.items():
        if runs:
            group_df = pd.concat(runs, ignore_index=True)  # Combine all runs in the group
            avg_df = group_df.groupby("_step").mean().reset_index()  # Compute average per step
            std_df = group_df.groupby("_step").std().reset_index()  # Compute standard deviation per step
            group_averages[group_idx] = avg_df
            group_stds[group_idx] = std_df
    return group_averages, group_stds

def plot_training_curves(group_averages, 
                         group_stds, 
                         labels, 
                         group_idx_arr, 
                         hist_name="train.episode_return",
                         label_type="num_sensors",
                         num_sensors_to_plot=None,
                         fig=None,
                         axs=None,
                         colors_dict=None,
                         plt_step=1000,
                         linewidth=0.8,
                         ylims=[None,None,None]):
    if colors_dict is None:
        colors = sns.color_palette("husl", len(labels)).as_hex()    
        colors_dict = {label: color for label, color in zip(labels, colors)}

    if label_type == "num_sensors":
        if num_sensors_to_plot is None:
            num_sensors_to_plot = labels
            labels_idxs = np.argsort(num_sensors_to_plot)
        else:
            # Find the index of the specified number of sensors in the labels list
            labels_idxs = [labels.index(num_sensors) for num_sensors in np.sort(num_sensors_to_plot)]
    else:
        labels_idxs = np.arange(len(labels))

    if axs is None:
        # fig, axs = plt.subplots(1,3,figsize=(10,3), constrained_layout=True)
        fig, axs = plt.subplots(3,1,figsize=(8,6), constrained_layout=True)
    for k, label_idx in enumerate(labels_idxs):
        group_idx = group_idx_arr[label_idx]
        avg_df = group_averages[group_idx]
        std_df = group_stds[group_idx]
        
        # plot training return
        axs[0].plot(avg_df["_step"].loc[::plt_step], avg_df[hist_name].loc[::plt_step], 
                label = labels[label_idx],
                color = colors_dict[labels[label_idx]],
                linewidth = linewidth)
        axs[0].fill_between(avg_df["_step"].loc[::plt_step], 
                        avg_df[hist_name].loc[::plt_step]-std_df[hist_name].loc[::plt_step],
                        avg_df[hist_name].loc[::plt_step]+std_df[hist_name].loc[::plt_step],
                        color = colors_dict[labels[label_idx]],
                        alpha=0.1)

        # plot q loss
        axs[1].plot(avg_df["_step"].loc[::plt_step], avg_df["q_loss"].loc[::plt_step], 
                label = labels[label_idx],
                color = colors_dict[labels[label_idx]],
                linewidth = linewidth
                )
        axs[1].fill_between(avg_df["_step"].loc[::plt_step], 
                        avg_df["q_loss"].loc[::plt_step]-std_df["q_loss"].loc[::plt_step],
                        avg_df["q_loss"].loc[::plt_step]+std_df["q_loss"].loc[::plt_step],
                        color = colors_dict[labels[label_idx]],
                        alpha=0.1)

        # plot policy loss
        axs[2].plot(avg_df["_step"].loc[::plt_step], avg_df["policy_loss"].loc[::plt_step], 
                label = labels[label_idx],
                color = colors_dict[labels[label_idx]],
                linewidth = linewidth
                )
        axs[2].fill_between(avg_df["_step"].loc[::plt_step], 
                        avg_df["policy_loss"].loc[::plt_step]-std_df["policy_loss"].loc[::plt_step],
                        avg_df["policy_loss"].loc[::plt_step]+std_df["policy_loss"].loc[::plt_step],
                        color = colors_dict[labels[label_idx]],
                        alpha=0.1)
        
    # axs[0].set_xlabel("Training Steps")
    axs[0].set_xticklabels([])
    axs[0].set_ylabel("Episode return")
    if ylims[0] is not None:
        axs[0].set_ylim(ylims[0])
    # axs[1].set_xlabel("Training Steps")
    axs[1].set_xticklabels([])
    axs[1].set_ylabel("Q loss")
    if ylims[1] is not None:
        axs[1].set_ylim(ylims[1])
    axs[2].set_xlabel("Training Steps")
    axs[2].set_ylabel("Policy loss")
    if ylims[2] is not None:
        axs[2].set_ylim(ylims[2])
    return fig, axs

def plot_return(group_averages, 
                group_stds, 
                labels, 
                group_idx_arr, 
                hist_name="train.episode_return",
                num_sensors_to_plot=None,
                fig=None,
                ax=None,
                colors_dict=None,
                plt_step=1000,
                linewidth=0.8,
                ylims=None):

    if colors_dict is None:
        colors = sns.color_palette("husl", len(labels)).as_hex()    
        colors_dict = {label: color for label, color in zip(labels, colors)}

    if num_sensors_to_plot is None:
        num_sensors_to_plot = labels
        labels_idxs = np.argsort(num_sensors_to_plot)
    else:
        # Find the index of the specified number of sensors in the labels list
        labels_idxs = [labels.index(num_sensors) for num_sensors in np.sort(num_sensors_to_plot)]

    colors = sns.color_palette("husl", len(labels_idxs)).as_hex() 

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(5,4), constrained_layout=True)

    for k, label_idx in enumerate(labels_idxs):
        group_idx = group_idx_arr[label_idx]
        avg_df = group_averages[group_idx]
        std_df = group_stds[group_idx]
        
        # plot training return
        ax.plot(avg_df["_step"].loc[::plt_step], avg_df[hist_name].loc[::plt_step], 
                label = labels[label_idx],
                color = colors_dict[labels[label_idx]],
                linewidth = linewidth)
        ax.fill_between(avg_df["_step"].loc[::plt_step], 
                        avg_df[hist_name].loc[::plt_step]-std_df[hist_name].loc[::plt_step],
                        avg_df[hist_name].loc[::plt_step]+std_df[hist_name].loc[::plt_step],
                        color = colors_dict[labels[label_idx]],
                        alpha=0.1)
        
    ax.set_xlabel("Training Steps $[t]$")
    ax.set_ylabel("Episode return")
    if ylims is not None:
        ax.set_ylim(ylims)

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles, 
              legend_labels, 
              ncols=int(len(labels)/2), 
              loc ='upper right', 
              bbox_to_anchor=(1.0, 1.2), 
              frameon=False)
    return fig, ax