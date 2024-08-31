import os
import sys
from pathlib import Path

import orbax.checkpoint
from absl import app, flags
from ml_collections import config_flags

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from datetime import datetime

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax
from flax.training import orbax_utils
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from scipy import linalg

import utils.file_processing as fp
import utils.flags as myflags
import wandb
from ddpg import DDPG
from envs.KS_environment import KSenv
from envs.KS_solver import KS
from replay_buffer import ReplayBuffer
from utils import visualizations as vis

# system config
FLAGS = flags.FLAGS
myflags.DEFINE_path("experiment_path", None, "Directory to store experiment results.")
flags.DEFINE_bool("log_wandb", False, "Use --log_wandb to log the experiment to wandb.")
flags.DEFINE_bool(
    "log_offline", False, "Use --log_offline to log the experiment to local."
)
flags.DEFINE_bool(
    "save_checkpoints",
    False,
    "Use --save_checkpoints to save intermediate model weights.",
)
_CONFIG = config_flags.DEFINE_config_file(
    "config", "configs/enKF_config.py", "Contains configs to run the experiment"
)
_WANDB = config_flags.DEFINE_config_file(
    "wandb_config", "configs/wandb_config.py", "Contains configs to log to wandb."
)
_ENV = config_flags.DEFINE_config_file(
    "env_config", "configs/KS_config.py", "Contains configs for the environment."
)

# flags.mark_flags_as_required(['config'])


def save_config():
    with open(FLAGS.experiment_path / "config.yml", "w") as f:
        FLAGS.config.to_yaml(stream=f)


def log_metrics_wandb(wandb_run, metrics, step):
    wandb_run.log(data=metrics, step=step)


def log_metrics_offline(logs, metrics):
    for metric_name, metric_value in metrics.items():
        if metric_name in logs.keys():
            logs[metric_name].append(metric_value)
        else:
            logs[metric_name] = [metric_value]


def add_gaussian_noise(key, x, stddev):
    noise = stddev * jax.random.normal(key, shape=x.shape)
    return x + noise


def evaluate_episode(config, env, actor, params, model, obs_mat):
    # reset the environment
    true_obs, _ = env.reset()
    full_state = env.unwrapped.u

    # initialize enKF
    state_ens = initialize_ensemble(
        env, model, env.unwrapped.u, config.enKF.std_init, config.enKF.m
    )

    # Preallocate arrays
    max_steps = config.episode_steps
    # Calculate the total number of steps
    total_steps = max_steps + config.enKF.observation_starts

    # Preallocate arrays using the types of the existing variables
    true_obs_arr = np.empty((total_steps, *true_obs.shape), dtype=true_obs.dtype)
    obs_arr = np.empty((total_steps, *true_obs.shape), dtype=true_obs.dtype)
    full_state_arr = np.empty((total_steps, *full_state.shape), dtype=full_state.dtype)
    state_ens_arr = np.empty((total_steps, *state_ens.shape), dtype=state_ens.dtype)

    # create an action of zeros to pass
    null_action = np.zeros(env.action_space.shape[0])

    # forecast until first observation
    for i in range(config.enKF.observation_starts):
        # Fill in the preallocated arrays
        true_obs_arr[i] = true_obs
        obs_arr[i] = true_obs
        full_state_arr[i] = full_state
        state_ens_arr[i] = state_ens

        # advance true environment
        true_obs, _, _, _, _ = env.step(action=null_action)
        full_state = env.unwrapped.u

        # advance model
        state_ens = forecast(
            model, state_ens, action=null_action, frame_skip=env.unwrapped.frame_skip
        )

    episode_step = 0
    state = ensemble_to_state(state_ens)

    terminated = False
    truncated = False

    while not terminated and not truncated and episode_step < max_steps:
        index = episode_step + config.enKF.observation_starts
        true_obs_arr[index] = true_obs
        full_state_arr[index] = full_state

        if episode_step % config.enKF.wait_steps == 0:
            # define observation covariance matrix
            obs_cov = (
                np.diag((config.enKF.std_obs * np.ones(len(true_obs))))
                * np.max(abs(true_obs), axis=0) ** 2
            )

            # add noise on the observation
            obs = np.random.multivariate_normal(true_obs, obs_cov)

            # apply enkf to correct the state estimation
            state_ens = apply_enKF(model, state_ens, obs, obs_cov, obs_mat)
            state = ensemble_to_state(state_ens)

            obs_arr[index] = obs
        else:
            obs_arr[index] = true_obs

        # save the analysis state
        state_ens_arr[index] = state_ens

        # get action from the target actor network
        action = actor.apply(params, state)
        true_obs, reward, terminated, truncated, _ = env.step(action)
        full_state = env.unwrapped.u

        # forecast
        state_ens = forecast(
            model, state_ens, action, frame_skip=env.unwrapped.frame_skip
        )
        episode_step += 1

    # Return only the filled parts of the arrays
    return (
        true_obs,
        full_state,
        state_ens,
        true_obs_arr[: index + 1],
        obs_arr[: index + 1],
        full_state_arr[: index + 1],
        state_ens_arr[: index + 1],
        reward,
    )


def evaluate(config, env, actor, params, model, obs_mat):
    # if env in jax, vmap over eval_episodes and jax.lax.scan over episode_steps
    last_reward_arr = []

    for _ in range(config.eval_episodes):
        _, _, _, _, _, _, _, reward = evaluate_episode(
            config, env, actor, params, model, obs_mat
        )
        last_reward_arr.append(reward)

    return_queue = np.asarray(env.return_queue).flatten()[-config.eval_episodes :]
    length_queue = np.asarray(env.length_queue).flatten()[-config.eval_episodes :]
    average_return = return_queue.mean()
    average_reward = (return_queue / length_queue).mean()
    average_last_reward = np.asarray(last_reward_arr).mean()
    return average_return, average_reward, average_last_reward


def evaluate_KS_for_plotting(env, actor, params, model, config, eval_episodes=1):
    x = env.unwrapped.KS.x
    x_obs = env.unwrapped.observation_locs
    target = env.unwrapped.target

    full_obs_mat = get_observation_matrix(model, x)
    obs_mat = get_observation_matrix(model, x_obs)
    figs = []
    for i in range(eval_episodes):
        (
            true_obs,
            full_state,
            state_ens,
            true_obs_arr,
            obs_arr,
            full_state_arr,
            state_ens_arr,
            _,
        ) = evaluate_episode(config, env, actor, params, model, obs_mat)
        true_obs_arr = np.vstack((true_obs_arr, true_obs))
        obs_arr = np.vstack((obs_arr, true_obs))
        full_state_arr = np.vstack((full_state_arr, full_state))
        state_ens_arr = np.concatenate((state_ens_arr, np.array([state_ens])), axis=0)

        # get full state from low order model
        state_ens_arr_ = np.hstack(
            (state_ens_arr, np.conjugate(np.flip(state_ens_arr[:, 1:-1], axis=1)))
        )
        full_state_ens_arr = np.real(
            np.einsum("kjm,ij->kim", state_ens_arr_, full_obs_mat)
        )
        # get the mean
        full_state_mean_arr = np.mean(full_state_ens_arr, axis=-1)

        # get fourier coefficients
        full_state_arr_f = np.fft.rfft(full_state_arr, axis=1)
        mag_state_arr = 2 / env.unwrapped.N * np.abs(full_state_arr_f)
        mag_state_ens_arr = 2 / model.n * np.abs(state_ens_arr)
        mag_state_mean_arr = np.mean(mag_state_ens_arr, axis=-1)

        # get observations from low order model
        obs_ens_arr = np.real(np.einsum("kjm,ij->kim", state_ens_arr_, obs_mat))
        # get the mean
        obs_mean_arr = np.mean(obs_ens_arr, axis=-1)

        fig = vis.plot_episode(
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
        )
        figs.append(fig)
    return figs


def initialize_ensemble(env, model, u0, std_init, m):
    # fourier transform the initial condition
    u0_f = np.fft.rfft(u0, axis=-1)
    # get lower order
    # make sure the magnitude of fourier modes match
    u0_f_low = model.n / env.unwrapped.N * u0_f[: len(model.k)]
    # create an ensemble by perturbing the real and imaginary parts
    # with the given uncertainty
    Af_0_real = np.random.multivariate_normal(
        u0_f_low.real, np.diag((u0_f_low.real * std_init) ** 2), m
    ).T
    Af_0_complex = np.random.multivariate_normal(
        u0_f_low.imag, np.diag((u0_f_low.imag * std_init) ** 2), m
    ).T
    Af_0 = Af_0_real + Af_0_complex * 1j
    return Af_0


def get_observation_matrix(model, x):
    # get the matrix to do inverse fft on observation points
    k = model.n * np.fft.fftfreq(model.n) * 2 * np.pi / model.L
    k_x = np.einsum("i,j->ij", x, k) * 1j
    exp_k_x = np.exp(k_x)
    M = 1 / model.n * exp_k_x
    return M


def ensemble_to_state(state_ens):
    state = np.mean(state_ens, axis=-1)
    # inverse rfft before passing to the neural network
    state = np.fft.irfft(state)
    return state


def forecast(model, state_ens, action, frame_skip):
    for _ in range(frame_skip):
        # advance forecast with low order model
        for m_idx in range(state_ens.shape[-1]):
            state_ens[:, m_idx] = model.advance_f(state_ens[:, m_idx], action)
    return state_ens


def EnKF(Af, d, Cdd, M):
    """Taken from real-time-bias-aware-DA by Novoa.
    Ensemble Kalman Filter as derived in Evensen (2009) eq. 9.27.
    Inputs:
        Af: forecast ensemble at time t
        d: observation at time t
        Cdd: observation error covariance matrix
        M: matrix mapping from state to observation space
    Returns:
        Aa: analysis ensemble
    """
    m = np.size(Af, 1)

    psi_f_m = np.mean(Af, 1, keepdims=True)
    Psi_f = Af - psi_f_m

    # Create an ensemble of observations
    if d.ndim == 2 and d.shape[-1] == m:
        D = d
    else:
        D = np.random.multivariate_normal(d, Cdd, m).transpose()
    # Mapped forecast matrix M(Af) and mapped deviations M(Af')
    Y = np.real(np.dot(M, Af))
    S = np.real(np.dot(M, Psi_f))
    # because we are multiplying with M first, we get real values
    # so we never actually compute the covariance of the complex-valued state
    # if i have to do that, then make sure to do it properly with the complex conjugate!!
    # Matrix to invert
    C = (m - 1) * Cdd + np.dot(S, S.T)
    Cinv = linalg.inv(C)

    X = np.dot(S.T, np.dot(Cinv, (D - Y)))

    Aa = Af + np.dot(Af, X)
    return Aa


def apply_enKF(model, Af, d, Cdd, M):
    Af_full = np.vstack((Af, np.conjugate(np.flip(Af[1:-1, :], axis=0))))
    Aa_full = EnKF(Af_full, d, Cdd, M)
    Aa = Aa_full[: len(model.k), :]
    return Aa


def run_experiment(
    config, env, eval_env, agent, model, wandb_run=None, logs=None, checkpoint_dir=None
):
    # random seed for initialization
    key = jax.random.PRNGKey(config.seed)

    # initialize networks
    # sample state and action to get the correct shape
    state_0 = jnp.array([jnp.zeros(model.n)])
    action_0 = jnp.array([env.action_space.sample()])
    actor_state, critic_state = agent.initial_network_state(key, state_0, action_0)

    # checkpoint the initial weights
    if checkpoint_dir is not None:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(
            save_interval_steps=config.total_steps // 5
        )
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
            checkpoint_dir, orbax_checkpointer, options
        )
        checkpoint = {"actor": actor_state, "critic": critic_state}
        save_args = orbax_utils.save_args_from_target(checkpoint)
        checkpoint_manager.save(0, checkpoint, save_kwargs={"save_args": save_args})

    # initialize buffer
    replay_buffer = ReplayBuffer(config.replay_buffer.capacity)

    # get the observation matrix that maps state to observations
    obs_mat = get_observation_matrix(model, env.unwrapped.observation_locs)

    # standard deviation of the exploration scales with the range of actions in the environment
    exploration_stddev = (
        env.action_space.high - env.action_space.low
    ) * config.train.exploration_stddev

    # create a action of zeros to pass
    null_action = np.zeros(env.action_space.shape[0])

    global_step = 0
    n_episode = 0
    truncated = False
    terminated = True
    while global_step < config.total_steps:
        # initialize episode
        if truncated or terminated:
            # reset the environment
            true_obs, info = env.reset()

            # initialize enKF
            state_ens = initialize_ensemble(
                env, model, env.unwrapped.u, config.enKF.std_init, config.enKF.m
            )

            # forecast until first observation
            for _ in range(config.enKF.observation_starts):
                # advance true environment
                true_obs, _, _, _, _ = env.step(action=null_action)

                # advance model
                state_ens = forecast(
                    model,
                    state_ens,
                    action=null_action,
                    frame_skip=env.unwrapped.frame_skip,
                )

                global_step += 1
            # reset
            action = None
            reward = None
            terminated = None
            episode_step = 0

        state = ensemble_to_state(state_ens)

        # push to buffer
        replay_buffer.push(state, action, reward, terminated)

        if episode_step % config.enKF.wait_steps == 0:
            # we got an observation
            # define observation covariance matrix
            obs_cov = (
                np.diag((config.enKF.std_obs * np.ones(len(true_obs))))
                * np.max(abs(true_obs), axis=0) ** 2
            )

            # add noise on the observation
            obs = np.random.multivariate_normal(true_obs, obs_cov)

            # apply enkf to correct the state estimation
            state_ens = apply_enKF(model, state_ens, obs, obs_cov, obs_mat)
            state = ensemble_to_state(state_ens)

            # push to buffer, because my starting state has been updated
            replay_buffer.push(state, action=None, reward=None, terminated=None)

        key, _ = jax.random.split(key)

        if global_step < config.learning_starts:
            # collect samples before starting to train
            action = env.action_space.sample()
        else:
            # get action from the learning actor network
            action = agent.actor.apply(actor_state.params, state)

            # add exploration noise on the action
            # original paper adds Ornstein-Uhlenbeck process noise
            # but in other papers this is deemed unnecessary
            action = add_gaussian_noise(key, action, stddev=exploration_stddev)
            # clip the action so that it obeys the limits set by the environment
            action = jnp.clip(
                action, min=env.action_space.low, max=env.action_space.high
            )

        # get the next observation and reward with this action
        true_obs, reward, terminated, truncated, info = env.step(action)

        # forecast
        state_ens = forecast(
            model, state_ens, action, frame_skip=env.unwrapped.frame_skip
        )

        episode_step += 1
        global_step += 1
        metrics = {"train": {}, "eval": {}}
        if "episode" in info:  # RecordEpisodeStatistics wrapper from gymnasium
            n_episode += 1
            print(
                f'Episode {n_episode}, Step={global_step}/{config.total_steps}, Return={info["episode"]["r"][0]}',
                flush=True,
            )
            # episode statistics to log
            metrics["train"]["episode_return"] = info["episode"]["r"][0]
            metrics["train"]["episode_length"] = info["episode"]["l"][0]
            metrics["train"]["episode_average_reward"] = (
                info["episode"]["r"][0] / info["episode"]["l"][0]
            )
            metrics["train"]["episode_last_reward"] = reward

        # if there is enough data in the buffer optimize
        if global_step >= config.learning_starts and replay_buffer.is_ready(
            config.train.batch_size
        ):
            (
                sampled_state,
                sampled_action,
                sampled_next_state,
                sampled_reward,
                sampled_terminated,
            ) = replay_buffer.sample(config.train.batch_size)
            critic_state, q_loss = agent.update_critic(
                actor_state,
                critic_state,
                sampled_state,
                sampled_action,
                sampled_next_state,
                sampled_reward,
                sampled_terminated,
            )

            actor_state, policy_loss = agent.update_actor(
                actor_state, critic_state, sampled_state
            )

            actor_state, critic_state = agent.update_target_networks(
                actor_state, critic_state
            )

            # losses to log
            metrics["train"]["q_loss"] = q_loss
            metrics["train"]["policy_loss"] = policy_loss

            if global_step % config.eval_freq == 0:
                # evaluate the performance of the target parameters ?
                eval_ave_return, eval_ave_reward, eval_ave_last_reward = evaluate(
                    config,
                    eval_env,
                    agent.actor,
                    actor_state.target_params,
                    model,
                    obs_mat,
                )
                print(
                    f"\n Evaluation, Step={global_step}/{config.total_steps}, Average Return={eval_ave_return} \n",
                    flush=True,
                )
                metrics["eval"]["average_return"] = eval_ave_return
                metrics["eval"]["average_reward"] = eval_ave_reward
                metrics["eval"]["average_last_reward"] = eval_ave_last_reward

        if wandb_run is not None:
            log_metrics_wandb(wandb_run, metrics, step=global_step)

        if logs is not None:
            log_metrics_offline(logs, metrics)

        # checkpoint the model
        if checkpoint_dir is not None:
            checkpoint = {"actor": actor_state, "critic": critic_state}
            save_args = orbax_utils.save_args_from_target(checkpoint)
            checkpoint_manager.save(
                global_step, checkpoint, save_kwargs={"save_args": save_args}
            )

    return actor_state, critic_state, logs


def main(_):
    config = FLAGS.config

    if not FLAGS.experiment_path:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        FLAGS.experiment_path = (
            Path.cwd() / "local_results" / config.env_name / f"run_{dt_string}"
        )

    # setup the experiment path
    FLAGS.experiment_path.mkdir(parents=True, exist_ok=True)

    # redirect print to text file
    orig_stdout = sys.stdout
    f = open(FLAGS.experiment_path / "out.txt", "w")
    sys.stdout = f
    print(f"Experiment will be saved to {FLAGS.experiment_path}", flush=True)

    # add environment config to the config
    config.env = FLAGS.env_config

    save_config()

    # initialize wandb logging
    if FLAGS.log_wandb:
        wandb_run = wandb.init(config=config.to_dict(), **FLAGS.wandb_config)
    else:
        wandb_run = None

    # initialize offline logging
    if FLAGS.log_offline:
        logs = {}
    else:
        logs = None

    # create model directory if checkpoint saving
    if FLAGS.save_checkpoints:
        checkpoint_dir = FLAGS.experiment_path / "models"
    else:
        checkpoint_dir = None

    # create environment
    if config.env_name == "KS":
        env = KSenv(**config.env)
        env = TimeLimit(env, max_episode_steps=config.episode_steps)
        env = RecordEpisodeStatistics(env)

        eval_env = KSenv(**config.env)
        eval_env = TimeLimit(eval_env, max_episode_steps=config.episode_steps)
        eval_env = RecordEpisodeStatistics(eval_env)
    else:
        env = gym.make(config.env_name)
        env = RecordEpisodeStatistics(env)

        eval_env = gym.make(config.env_name)
        eval_env = RecordEpisodeStatistics(eval_env)

    # create agent and run
    agent = DDPG(config, env)
    print("Starting experiment.", flush=True)

    # create low order model
    model = KS(
        nu=config.env.nu,
        N=config.enKF.low_order_N,
        dt=env.unwrapped.dt,
        actuator_locs=config.env.actuator_locs,
        actuator_scale=config.env.actuator_scale,
    )

    actor_state, critic_state, logs = run_experiment(
        config, env, eval_env, agent, model, wandb_run, logs, checkpoint_dir
    )

    # save the final model weights
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    final_checkpoint = {"actor": actor_state, "critic": critic_state}
    save_args = orbax_utils.save_args_from_target(final_checkpoint)
    orbax_checkpointer.save(
        FLAGS.experiment_path / "final_model", final_checkpoint, save_args=save_args
    )

    # finish logging
    if FLAGS.log_wandb:
        wandb_run.finish()

    if FLAGS.log_offline:
        print(f"Saving logs to {FLAGS.experiment_path}.")
        fp.pickle_file(FLAGS.experiment_path / "logs.pickle", logs)

    # evaluate with the final weights and plot episode
    if config.env_name == "KS":
        figs = evaluate_KS_for_plotting(
            eval_env,
            agent.actor,
            actor_state.target_params,
            model,
            config,
            eval_episodes=2,
        )
        for fig_idx, fig in enumerate(figs):
            fig.savefig(FLAGS.experiment_path / f"final_evaluation_{fig_idx}.png")
        plt.show()

    # close text
    sys.stdout = orig_stdout
    f.close()
    return


if __name__ == "__main__":
    app.run(main)
