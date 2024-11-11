import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_episode_wo_KF(
    x,
    x_obs,
    x_act,
    target,
    full_state_arr,
    mag_state_arr,
    true_obs_arr,
    obs_arr,
    action_arr,
    reward_arr,
    observation_starts,
    wait_steps,
):
    non_nan_idxs = jnp.where(~jnp.isnan(obs_arr))

    # set the colours
    true_color = "black"
    noisy_color = "silver"
    model_color = "red"
    target_color = "limegreen"

    # initialize figure
    fig = plt.figure(layout="constrained", figsize=(16, 10))
    subfigs = fig.subfigures(2, 1, height_ratios=[1.0, 0.8])
    subfigs0 = subfigs[0].subfigures(1, 2, width_ratios=[1.0, 0.4])
    axs0 = subfigs0[0].subplots(
        2,
        3,
        width_ratios=[1, 0.55, 1],
    )
    axs1 = subfigs0[1].subplots(3, 1)
    axs2 = subfigs[1].subplots(3, 3)

    # plot the full state
    im = axs0[0, 0].imshow(
        full_state_arr.T,
        extent=[0, len(full_state_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
    )
    axs0[0, 0].set_xticklabels([])
    axs0[0, 0].set_ylabel("x")
    cbar = fig.colorbar(im, ax=[axs0[0, 0]], location="left")
    cbar.ax.set_title("u")
    axs0[0, 0].set_title("True")

    # plot the state that actor-critic receives
    non_nan_obs = obs_arr[non_nan_idxs]
    non_nan_obs = non_nan_obs.reshape(
        non_nan_obs.shape[0] // obs_arr.shape[1], obs_arr.shape[1]
    )
    # concatenate with nans for the unobserved part
    nan_arr = jnp.nan * jnp.ones((observation_starts // wait_steps, obs_arr.shape[1]))
    stacked_obs = jnp.vstack((nan_arr, non_nan_obs))

    im = axs0[1, 0].imshow(
        stacked_obs.T,
        extent=[0, len(full_state_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
    )
    axs0[1, 0].set_xlabel("t")
    axs0[1, 0].set_yticks(axs0[0, 0].get_yticks())
    axs0[1, 0].set_ylim(axs0[0, 0].get_ylim())
    axs0[1, 0].set_ylabel("x")
    cbar = fig.colorbar(im, ax=[axs0[1, 0]], location="left")
    cbar.ax.set_title("u")
    axs0[1, 0].set_title("RL state")

    # plot the last state and target
    axs0[0, 1].plot(full_state_arr[-1, :], x, color=true_color, label="Final")
    axs0[0, 1].plot(target, x, "-.", color=target_color, label="Target")
    axs0[0, 1].set_yticks(axs0[0, 0].get_yticks())
    axs0[0, 1].set_yticklabels([])
    axs0[0, 1].set_ylim(axs0[0, 0].get_ylim())
    axs0[0, 1].grid()
    axs0[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=2)

    # plot initial condition
    axs0[1, 1].plot(
        full_state_arr[non_nan_idxs[0][0], :], x, color=true_color, label="Initial"
    )
    axs0[1, 1].plot(
        non_nan_obs[0, :],
        x_obs,
        marker="o",
        linestyle="",
        markersize=4,
        color=model_color,
        label="Obs",
    )
    axs0[1, 1].grid()
    axs0[1, 1].set_xlabel("u")
    axs0[1, 1].set_yticks(axs0[0, 0].get_yticks())
    axs0[1, 1].set_yticklabels([])
    axs0[1, 1].set_ylim(axs0[0, 0].get_ylim())
    axs0[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=2)

    # plot the error from target
    err = jnp.abs(target[:, None] - full_state_arr.T)
    im = axs0[0, 2].imshow(
        err,
        extent=[0, len(full_state_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
        cmap="Greens",
    )
    axs0[0, 2].set_xticklabels([])
    axs0[0, 2].set_yticks(axs0[0, 0].get_yticks())
    axs0[0, 2].set_yticklabels([])
    axs0[0, 2].set_ylim(axs0[0, 0].get_ylim())
    cbar = fig.colorbar(im, ax=[axs0[0, 2]], location="right")
    axs0[0, 2].set_title("|Target - u|")

    # plot the error between true obs and obs
    non_nan_true_obs = true_obs_arr[non_nan_idxs]
    non_nan_true_obs = non_nan_true_obs.reshape(
        non_nan_true_obs.shape[0] // obs_arr.shape[1], obs_arr.shape[1]
    )
    err2 = jnp.abs(non_nan_true_obs - non_nan_obs)
    # concatenate with nans for the unobserved part
    stacked_err2 = jnp.vstack((nan_arr, err2))
    im = axs0[1, 2].imshow(
        stacked_err2.T,
        extent=[0, len(full_state_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
        cmap="Reds",
    )
    axs0[1, 2].set_xlabel("t")
    axs0[1, 2].set_yticks(axs0[0, 0].get_yticks())
    axs0[1, 2].set_yticklabels([])
    axs0[1, 2].set_ylim(axs0[0, 0].get_ylim())
    cbar = fig.colorbar(im, ax=[axs0[1, 2]], location="right")
    axs0[1, 2].set_title("|True Obs - Obs|")

    # plot the energy, actuation energy and reward
    state_norm = jnp.linalg.norm(full_state_arr, axis=1)
    axs1[0].plot(state_norm, color=true_color)
    axs1[0].set_xticklabels([])
    axs1[0].set_ylabel("||u||")
    axs1[0].set_title(f"Final ||u|| = {state_norm[-1]:.2f}")
    axs1[0].grid()

    action_norm = jnp.linalg.norm(action_arr, axis=1)
    axs1[1].plot(action_norm)
    axs1[1].set_xticklabels([])
    axs1[1].set_ylabel("||a||")
    axs1[1].grid()

    axs1[2].plot(reward_arr, color=true_color)
    axs1[2].set_ylabel("r(u,a)")
    axs1[2].grid()
    axs1[2].set_xlabel("t")

    # plot the tracked Fourier modes
    for j in range(3):
        axs2[j, 0].plot(
            mag_state_arr[:, j + 1],
            color=true_color,
            label=f"True N={2*(mag_state_arr.shape[1]-1)}",
        )
        axs2[j, 0].set_ylabel(f"F(u)[{j+1}]")
        if j < 2:
            axs2[j, 0].set_xticklabels([])
    axs2[2, 0].set_xlabel("t")
    axs2[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncols=2)

    # plot the observations
    for j in range(3):
        axs2[j, 1].plot(
            obs_arr[:, j],
            marker="o",
            linestyle="",
            markersize=3,
            color=model_color,
            label="Noisy",
        )
        axs2[j, 1].plot(true_obs_arr[:, j], color=true_color, label="True")
        axs2[j, 1].set_ylabel(f"u(x={x_obs[j]:.2f})")
        if j < 2:
            axs2[j, 1].set_xticklabels([])
    axs2[2, 1].set_xlabel("t")
    axs2[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncols=3)

    # plot the actuation
    for j in range(3):
        axs2[j, 2].plot(action_arr[:, j])
        axs2[j, 2].set_ylabel(f"a(x={x_act[j]:.2f})")
        if j < 2:
            axs2[j, 2].set_xticklabels([])
    axs2[2, 2].set_xlabel("t")

    return fig


def plot_episode(
    x,
    x_obs,
    x_act,
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
    action_arr,
    reward_env_arr,
    reward_model_arr,
):
    # set the colours
    true_color = "black"
    noisy_color = "silver"
    model_color = "red"
    target_color = "limegreen"

    # initialize figure
    fig = plt.figure(layout="constrained", figsize=(16, 10))
    subfigs = fig.subfigures(2, 1, height_ratios=[1.0, 0.8])
    subfigs0 = subfigs[0].subfigures(1, 2, width_ratios=[1.0, 0.4])
    axs0 = subfigs0[0].subplots(
        2,
        3,
        width_ratios=[1, 0.55, 1],
    )
    axs1 = subfigs0[1].subplots(3, 1)
    axs2 = subfigs[1].subplots(3, 3)

    # plot the full state and reward
    im = axs0[0, 0].imshow(
        full_state_arr.T,
        extent=[0, len(full_state_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
    )
    axs0[0, 0].set_xticklabels([])
    axs0[0, 0].set_ylabel("x")
    cbar = fig.colorbar(im, ax=[axs0[0, 0]], location="left")
    cbar.ax.set_title("u")
    axs0[0, 0].set_title("True")

    # plot the full state reconstruction from low order model
    im = axs0[1, 0].imshow(
        full_state_mean_arr.T,
        extent=[0, len(full_state_mean_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
    )
    axs0[1, 0].set_xlabel("t")
    axs0[1, 0].set_yticks(axs0[0, 0].get_yticks())
    axs0[1, 0].set_ylim(axs0[0, 0].get_ylim())
    axs0[1, 0].set_ylabel("x")
    cbar = fig.colorbar(im, ax=[axs0[1, 0]], location="left")
    cbar.ax.set_title("u")
    axs0[1, 0].set_title("Model reconstruction")

    # plot the last state and target
    axs0[0, 1].plot(full_state_arr[-1, :], x, color=true_color, label="Final")
    axs0[0, 1].plot(target, x, "-.", color=target_color, label="Target")
    axs0[0, 1].set_yticks(axs0[0, 0].get_yticks())
    axs0[0, 1].set_yticklabels([])
    axs0[0, 1].set_ylim(axs0[0, 0].get_ylim())
    axs0[0, 1].grid()
    axs0[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=2)

    # plot initial condition
    axs0[1, 1].plot(full_state_ens_arr[0, :, :], x, ":", linewidth=0.5)
    axs0[1, 1].plot(full_state_arr[0, :], x, color=true_color, label="Initial")
    axs0[1, 1].plot(
        full_state_mean_arr[0, :], x, "--", color=model_color, label="Model"
    )
    axs0[1, 1].grid()
    axs0[1, 1].set_xlabel("u")
    axs0[1, 1].set_yticks(axs0[0, 0].get_yticks())
    axs0[1, 1].set_yticklabels([])
    axs0[1, 1].set_ylim(axs0[0, 0].get_ylim())
    axs0[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncols=2)

    # plot the error from target
    err = jnp.abs(target[:, None] - full_state_arr.T)
    im = axs0[0, 2].imshow(
        err,
        extent=[0, len(full_state_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
        cmap="Greens",
    )
    axs0[0, 2].set_xticklabels([])
    axs0[0, 2].set_yticks(axs0[0, 0].get_yticks())
    axs0[0, 2].set_yticklabels([])
    axs0[0, 2].set_ylim(axs0[0, 0].get_ylim())
    cbar = fig.colorbar(im, ax=[axs0[0, 2]], location="right")
    axs0[0, 2].set_title("|Target - u|")

    # plot the error between true and reconstruction
    err2 = jnp.abs(full_state_mean_arr.T - full_state_arr.T)
    im = axs0[1, 2].imshow(
        err2,
        extent=[0, len(full_state_arr), x[0], x[-1]],
        origin="lower",
        aspect="auto",
        cmap="Reds",
    )
    axs0[1, 2].set_xlabel("t")
    axs0[1, 2].set_yticks(axs0[0, 0].get_yticks())
    axs0[1, 2].set_yticklabels([])
    axs0[1, 2].set_ylim(axs0[0, 0].get_ylim())
    cbar = fig.colorbar(im, ax=[axs0[1, 2]], location="right")
    axs0[1, 2].set_title("|u_recon - u|")

    # plot the energy, actuation energy and reward
    state_norm = jnp.linalg.norm(full_state_arr, axis=1)
    axs1[0].plot(state_norm, color=true_color, label="True")
    axs1[0].set_xticklabels([])
    axs1[0].set_ylabel("||u||")
    axs1[0].set_title(f"Final ||u|| = {state_norm[-1]:.2f}")
    axs1[0].legend()
    axs1[0].grid()

    action_norm = jnp.linalg.norm(action_arr, axis=1)
    axs1[1].plot(action_norm)
    axs1[1].set_xticklabels([])
    axs1[1].set_ylabel("||a||")
    axs1[1].grid()

    axs1[2].plot(reward_env_arr, color=true_color, label="True")
    axs1[2].plot(reward_model_arr, "--", color=model_color, label="Model")
    axs1[2].set_ylabel("r(u,a)")
    axs1[2].set_xlabel("t")
    axs1[2].grid()
    axs1[2].legend()

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
        axs2[j, 1].plot(obs_ens_arr[:, j, :], ":", linewidth=0.5)
        axs2[j, 1].plot(true_obs_arr[:, j], color=true_color, label="True")
        axs2[j, 1].plot(obs_mean_arr[:, j], "--", color=model_color, label="Model")
        axs2[j, 1].plot(
            obs_arr[:, j],
            marker="o",
            linestyle="",
            markersize=3,
            color=noisy_color,
            label="Noisy",
        )
        axs2[j, 1].set_ylabel(f"u(x={x_obs[j]:.2f})")
        if j < 2:
            axs2[j, 1].set_xticklabels([])
    axs2[2, 1].set_xlabel("t")
    axs2[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncols=3)

    # plot the actuation
    for j in range(3):
        axs2[j, 2].plot(action_arr[:, j])
        axs2[j, 2].set_ylabel(f"a(x={x_act[j]:.2f})")
        if j < 2:
            axs2[j, 2].set_xticklabels([])
    axs2[2, 2].set_xlabel("t")

    return fig
