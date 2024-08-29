import matplotlib.pyplot as plt
import numpy as np


def plot_episode(
    x,
    x_obs,
    target,
    full_state_arr,
    full_state_ens_arr,
    full_state_mean_arr,
    mag_state_arr,
    mag_state_ens_arr,
    mag_state_mean_arr,
    true_obs_arr,
    obs_arr,
    obs_ens_arr,
    obs_mean_arr,
):
    # set the colours
    true_color = "black"
    noisy_color = "silver"
    model_color = "red"
    target_color = "limegreen"

    # initialize figure
    fig = plt.figure(layout="constrained", figsize=(10, 10))
    subfigs = fig.subfigures(2, 1, height_ratios=[1.0, 0.8])
    axs = subfigs[0].subplots(
        2,
        3,
        width_ratios=[1, 0.55, 1],
    )
    axs2 = subfigs[1].subplots(3, 2)

    # plot the full state and reward
    im = axs[0, 0].imshow(
        full_state_arr.T,
        extent=[0, len(full_state_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
    )
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_ylabel("x")
    cbar = fig.colorbar(im, ax=[axs[0, 0]], location="left")
    cbar.ax.set_title("u")
    # axs[0,1].set_title(
    #     f"Return={info['episode']['r'][0]:.2f}, Ave. Reward={info['episode']['r'][0]/info['episode']['l'][0]:.2f}"
    # )
    axs[0, 0].set_title("True")

    # plot the full state reconstruction from low order model
    im = axs[1, 0].imshow(
        full_state_mean_arr.T,
        extent=[0, len(full_state_mean_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
    )
    axs[1, 0].set_xlabel("t")
    axs[1, 0].set_yticks(axs[0, 0].get_yticks())
    axs[1, 0].set_ylim(axs[0, 0].get_ylim())
    axs[1, 0].set_ylabel("x")
    cbar = fig.colorbar(im, ax=[axs[1, 0]], location="left")
    cbar.ax.set_title("u")
    axs[1, 0].set_title("Model reconstruction")

    # plot the last state and target
    axs[0, 1].plot(full_state_arr[-1, :], x, color=true_color, label="Final")
    axs[0, 1].plot(target, x, "-.", color=target_color, label="Target")
    axs[0, 1].set_yticks(axs[0, 0].get_yticks())
    axs[0, 1].set_yticklabels([])
    axs[0, 1].set_ylim(axs[0, 0].get_ylim())
    # axs[0,2].set_title(f"Last Reward={reward:.2f}")
    axs[0, 1].grid()
    axs[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=2)

    # plot initial condition
    axs[1, 1].plot(full_state_ens_arr[0, :, :], x, ":", linewidth=0.5)
    axs[1, 1].plot(full_state_arr[0, :], x, color=true_color, label="Initial")
    axs[1, 1].plot(full_state_mean_arr[0, :], x, "--", color=model_color, label="Model")
    axs[1, 1].grid()
    axs[1, 1].set_xlabel("u")
    axs[1, 1].set_yticks(axs[0, 0].get_yticks())
    axs[1, 1].set_yticklabels([])
    axs[1, 1].set_ylim(axs[0, 0].get_ylim())
    axs[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=2)

    # plot the error from target
    err = np.abs(target[:, None] - full_state_arr.T)
    im = axs[0, 2].imshow(
        err,
        extent=[0, len(full_state_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
        cmap="Greens",
    )
    axs[0, 2].set_xticklabels([])
    axs[0, 2].set_yticks(axs[0, 0].get_yticks())
    axs[0, 2].set_yticklabels([])
    axs[0, 2].set_ylim(axs[0, 0].get_ylim())
    cbar = fig.colorbar(im, ax=[axs[0, 2]], location="right")
    axs[0, 2].set_title("|Target - u|")

    # plot the error between true and reconstruction
    err2 = np.abs(full_state_mean_arr.T - full_state_arr.T)
    im = axs[1, 2].imshow(
        err2,
        extent=[0, len(full_state_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
        cmap="Reds",
    )
    axs[1, 2].set_xlabel("t")
    axs[1, 2].set_yticks(axs[0, 0].get_yticks())
    axs[1, 2].set_yticklabels([])
    axs[1, 2].set_ylim(axs[0, 0].get_ylim())
    cbar = fig.colorbar(im, ax=[axs[1, 2]], location="right")
    axs[1, 2].set_title("|u_recon - u|")

    # plot the tracked Fourier modes and measurements

    for j in range(3):
        axs2[j, 0].plot(mag_state_ens_arr[:, j + 1, :], ":", linewidth=0.5)
        axs2[j, 0].plot(
            mag_state_arr[:, j + 1],
            color=true_color,
            label=f"True N={2*(mag_state_arr.shape[1]-1)}",
        )
        axs2[j, 0].plot(
            mag_state_mean_arr[:, j + 1],
            "--",
            color=model_color,
            label=f"Model N={2*(mag_state_mean_arr.shape[1]-1)}",
        )
        axs2[j, 0].set_ylabel(f"F(u)[{j+1}]")
        if j < 2:
            axs2[j, 0].set_xticklabels([])
    axs2[2, 0].set_xlabel("t")
    axs2[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncols=2)

    for j in range(3):
        axs2[j, 1].plot(obs_arr[:, j], color=noisy_color, label="Noisy")
        axs2[j, 1].plot(obs_ens_arr[:, j, :], ":", linewidth=0.5)
        axs2[j, 1].plot(true_obs_arr[:, j], color=true_color, label="True")
        axs2[j, 1].plot(obs_mean_arr[:, j], "--", color=model_color, label="Model")
        axs2[j, 1].set_ylabel(f"u(x={x_obs[j]:.2f})")
        if j < 2:
            axs2[j, 1].set_xticklabels([])
    axs2[2, 1].set_xlabel("t")
    axs2[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncols=3)

    return fig
