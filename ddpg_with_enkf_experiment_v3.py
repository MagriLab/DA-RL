import os
import sys

# add the root directory to the path before importing from the library
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)

from datetime import datetime
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax
import orbax.checkpoint
from absl import app, flags
from flax.training import orbax_utils
from ml_collections import config_flags
from scipy import linalg

import utils.file_processing as fp
import utils.flags as myflags
import wandb
from ddpg import DDPG
from envs.KS_environment_jax import KSenv
from envs.KS_solver_jax import KS
from replay_buffer_jax import add_experience, init_replay_buffer, sample_experiences
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


# def evaluate_KS_for_plotting(env, actor, params, model, config, eval_episodes=1):
#     x = env.unwrapped.KS.x
#     x_obs = env.unwrapped.observation_locs
#     target = env.unwrapped.target

#     full_obs_mat = get_observation_matrix(model, x)
#     obs_mat = get_observation_matrix(model, x_obs)
#     figs = []
#     for i in range(eval_episodes):
#         (
#             true_obs,
#             full_state,
#             state_ens,
#             true_obs_arr,
#             obs_arr,
#             full_state_arr,
#             state_ens_arr,
#             _,
#         ) = evaluate_episode(config, env, actor, params, model, obs_mat)
#         true_obs_arr = np.vstack((true_obs_arr, true_obs))
#         obs_arr = np.vstack((obs_arr, true_obs))
#         full_state_arr = np.vstack((full_state_arr, full_state))
#         state_ens_arr = np.concatenate((state_ens_arr, np.array([state_ens])), axis=0)

#         # get full state from low order model
#         state_ens_arr_ = np.hstack(
#             (state_ens_arr, np.conjugate(np.flip(state_ens_arr[:, 1:-1], axis=1)))
#         )
#         full_state_ens_arr = np.real(
#             np.einsum("kjm,ij->kim", state_ens_arr_, full_obs_mat)
#         )
#         # get the mean
#         full_state_mean_arr = np.mean(full_state_ens_arr, axis=-1)

#         # get fourier coefficients
#         full_state_arr_f = np.fft.rfft(full_state_arr, axis=1)
#         mag_state_arr = 2 / env.N * np.abs(full_state_arr_f)
#         mag_state_ens_arr = 2 / model.N * np.abs(state_ens_arr)
#         mag_state_mean_arr = np.mean(mag_state_ens_arr, axis=-1)

#         # get observations from low order model
#         obs_ens_arr = np.real(np.einsum("kjm,ij->kim", state_ens_arr_, obs_mat))
#         # get the mean
#         obs_mean_arr = np.mean(obs_ens_arr, axis=-1)

#         fig = vis.plot_episode(
#             x,
#             x_obs,
#             target,
#             full_state_arr,
#             full_state_ens_arr,
#             full_state_mean_arr,
#             mag_state_arr,
#             mag_state_ens_arr,
#             mag_state_mean_arr,
#             true_obs_arr,
#             obs_arr,
#             obs_ens_arr,
#             obs_mean_arr,
#         )
#         figs.append(fig)
#     return figs


def initialize_ensemble(env_N, model_N, model_k, u0, std_init, m, key):
    # fourier transform the initial condition
    u0_f = jnp.fft.rfft(u0, axis=-1)
    # get lower order
    # make sure the magnitude of fourier modes match
    u0_f_low = model_N / env_N * u0_f[: len(model_k)]
    # create an ensemble by perturbing the real and imaginary parts
    # with the given uncertainty
    key, subkey = jax.random.split(key)
    Af_0_real = jax.random.multivariate_normal(
        key,
        u0_f_low.real,
        jnp.diag((u0_f_low.real * std_init) ** 2),
        (m,),
        method="svd",
    ).T
    # covariance matrix is rank deficient because zeroth
    Af_0_complex = jax.random.multivariate_normal(
        subkey,
        u0_f_low.imag,
        jnp.diag((u0_f_low.imag * std_init) ** 2),
        (m,),
        method="svd",
    ).T
    Af_0 = Af_0_real + Af_0_complex * 1j
    return Af_0


def get_observation_matrix(model_N, model_L, x):
    # get the matrix to do inverse fft on observation points
    k = model_N * jnp.fft.fftfreq(model_N) * 2 * jnp.pi / model_L
    k_x = jnp.einsum("i,j->ij", x, k) * 1j
    exp_k_x = jnp.exp(k_x)
    M = 1 / model_N * exp_k_x
    return M


def ensemble_to_state(state_ens):
    state = jnp.mean(state_ens, axis=-1)
    # inverse rfft before passing to the neural network
    state = jnp.fft.irfft(state)
    return state


def forecast(state_ens, action, frame_skip, B, lin, ik, dt):
    """
    Forecast the state ensemble over a number of steps.

    Args:
        state_ens: Ensemble of states. Shape [n_ensemble, n_state].
        action: Action applied to the system.
        frame_skip: Number of steps to advance.
        B, lin, ik, dt: KS model parameters.

    Returns:
        Updated state ensemble.
    """

    def step_fn(state, _):
        return (
            jax.vmap(
                KS.advance_f, in_axes=(-1, None, None, None, None, None), out_axes=-1
            )(state, action, B, lin, ik, dt),
            None,
        )

    # Use lax.scan to iterate over frame_skip and advance the state
    state_ens, _ = jax.lax.scan(step_fn, state_ens, jnp.arange(frame_skip))

    return state_ens


def EnKF(m, Af, d, Cdd, M, key):
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
    psi_f_m = jnp.mean(Af, 1, keepdims=True)
    Psi_f = Af - psi_f_m

    # Create an ensemble of observations
    D = jax.random.multivariate_normal(key, d, Cdd, (m,), method="svd").T
    # Mapped forecast matrix M(Af) and mapped deviations M(Af')
    Y = jnp.real(jnp.dot(M, Af))
    S = jnp.real(jnp.dot(M, Psi_f))
    # because we are multiplying with M first, we get real values
    # so we never actually compute the covariance of the complex-valued state
    # if i have to do that, then make sure to do it properly with the complex conjugate!!
    # Matrix to invert
    C = (m - 1) * Cdd + jnp.dot(S, S.T)
    # Cinv = jnp.linalg.inv(C)

    # X = jnp.dot(S.T, jnp.dot(Cinv, (D - Y)))
    X = jnp.dot(S.T, jnp.linalg.solve(C, D - Y))

    Aa = Af + jnp.dot(Af, X)
    return Aa


def apply_enKF(m, k, Af, d, Cdd, M, key):
    Af_full = jnp.vstack((Af, jnp.conjugate(jnp.flip(Af[1:-1, :], axis=0))))
    Aa_full = EnKF(m, Af_full, d, Cdd, M, key)
    Aa = Aa_full[:k, :]
    return Aa


def run_experiment(
    config, env, agent, model, wandb_run=None, logs=None, checkpoint_dir=None
):
    # random seed for initialization
    key = jax.random.PRNGKey(config.seed)
    key, key_network, key_buffer, key_env, key_obs, key_action = jax.random.split(
        key, 6
    )

    # initialize networks
    # sample state and action to get the correct shape
    state_0 = jnp.array([jnp.zeros(model.N)])
    action_0 = jnp.array([jnp.zeros(env.action_size)])
    actor_state, critic_state = agent.initial_network_state(
        key_network, state_0, action_0
    )

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
    replay_buffer = init_replay_buffer(
        capacity=config.replay_buffer.capacity,
        state_dim=(model.N,),
        action_dim=(env.action_size,),
        rng_key=key_buffer,
    )

    # get the observation matrix that maps state to observations
    obs_mat = get_observation_matrix(model.N, model.L, env.observation_locs)

    # standard deviation of the exploration scales with the range of actions in the environment
    exploration_stddev = (
        env.action_high - env.action_low
    ) * config.train.exploration_stddev

    # create a action of zeros to pass
    null_action = jnp.zeros(env.action_size)

    # jit the necessary environment functions
    initialize_ensemble = partial(
        initialize_ensemble,
        env_N=env.N,
        model_N=model.N,
        model_k=model.k,
        std_init=config.enKF.std_init,
        m=config.enKF.m,
    )
    initialize_ensemble = jax.jit(initialize_ensemble)

    env_reset = partial(
        KSenv.reset,
        N=env.N,
        B=env.ks_solver.B,
        lin=env.ks_solver.lin,
        ik=env.ks_solver.ik,
        dt=env.ks_solver.dt,
        initial_amplitude=env.initial_amplitude,
        action_size=env.action_size,
        burn_in=env.burn_in,
        observation_inds=env.observation_inds,
    )
    env_reset = jax.jit(env_reset)
    env_step = partial(
        KSenv.step,
        frame_skip=env.frame_skip,
        B=env.ks_solver.B,
        lin=env.ks_solver.lin,
        ik=env.ks_solver.ik,
        dt=env.ks_solver.dt,
        target=env.target,
        actuator_loss_weight=env.actuator_loss_weight,
        termination_threshold=env.termination_threshold,
        observation_inds=env.observation_inds,
    )
    env_step = jax.jit(env_step)
    env_sample_action = partial(
        KSenv.sample_continuous_space,
        low=env.action_low,
        high=env.action_high,
        shape=(env.action_size,),
    )
    env_sample_action = jax.jit(env_sample_action)

    model_forecast = partial(
        forecast,
        frame_skip=env.frame_skip,
        B=model.B,
        lin=model.lin,
        ik=model.ik,
        dt=model.dt,
    )
    model_forecast = jax.jit(model_forecast)

    model_apply_enKF = partial(
        apply_enKF,
        m=config.enKF.m,
        k=len(model.k),
        M=obs_mat,
    )
    model_apply_enKF = jax.jit(model_apply_enKF)

    global_step = 0
    n_episode = 0
    truncated = False
    terminated = True

    def until_first_observation(true_state, true_obs, state_ens, observation_starts):
        def body_fun(carry, _):
            true_state, true_obs, state_ens = carry
            # advance true environment
            true_state, true_obs, _, _, _, _ = env_step(
                state=true_state, action=null_action
            )
            # advance model
            state_ens = model_forecast(state_ens=state_ens, action=null_action)
            return (true_state, true_obs, state_ens), (true_state, true_obs, state_ens)

        (true_state, true_obs, state_ens), (
            true_state_arr,
            true_obs_arr,
            state_ens_arr,
        ) = jax.lax.scan(
            body_fun, (true_state, true_obs, state_ens), jnp.arange(observation_starts)
        )
        return (
            true_state,
            true_obs,
            state_ens,
            true_state_arr,
            true_obs_arr,
            state_ens_arr,
        )

    def act_observe_and_forecast(
        true_state, true_obs, state_ens, params, wait_steps, episode_steps, key_obs
    ):
        def forecast_fun(carry, _):
            true_state, true_obs, state_ens = carry
            state = ensemble_to_state(state_ens)

            # get action
            action = agent.actor.apply(params, state)

            # get the next observation and reward with this action
            true_state, true_obs, reward, _, _, _ = env_step(
                state=true_state, action=action
            )

            # forecast
            state_ens = model_forecast(state_ens=state_ens, action=action)
            return (true_state, true_obs, state_ens), (
                true_state,
                true_obs,
                state_ens,
                reward,
            )

        def body_fun(carry, _):
            # observe
            # we got an observation
            # define observation covariance matrix
            true_state, true_obs, state_ens, key_obs = carry
            obs_cov = (
                jnp.diag((config.enKF.std_obs * jnp.ones(len(true_obs))))
                * jnp.max(abs(true_obs), axis=0) ** 2
            )

            # add noise on the observation
            key_obs, key_enKF = jax.random.split(key_obs)
            obs = jax.random.multivariate_normal(
                key_obs, true_obs, obs_cov, method="svd"
            )

            # apply enkf to correct the state estimation
            state_ens = model_apply_enKF(Af=state_ens, d=obs, Cdd=obs_cov, key=key_enKF)
            (true_state, true_obs, state_ens), (
                true_state_arr,
                true_obs_arr,
                state_ens_arr,
                reward_arr,
            ) = jax.lax.scan(
                forecast_fun, (true_state, true_obs, state_ens), jnp.arange(wait_steps)
            )
            return (true_state, true_obs, state_ens, key_obs), (
                true_state_arr,
                true_obs_arr,
                obs,
                state_ens_arr,
                reward_arr,
            )

        n_loops = episode_steps // wait_steps
        (true_state, true_obs, state_ens, _), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            reward_arr,
        ) = jax.lax.scan(
            body_fun, (true_state, true_obs, state_ens, key_obs), jnp.arange(n_loops)
        )
        return (
            true_state,
            true_obs,
            state_ens,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            reward_arr,
        )

    def random_observe_and_forecast(
        true_state,
        true_obs,
        state_ens,
        wait_steps,
        episode_steps,
        key_obs,
        key_action,
        replay_buffer,
    ):
        def forecast_fun(carry, _):
            true_state, true_obs, state_ens, state, key_action, replay_buffer = carry

            # get action
            key_action, _ = jax.random.split(key_action)
            action = env_sample_action(key=key_action)

            # get the next observation and reward with this action
            next_true_state, next_true_obs, reward, terminated, _, _ = env_step(
                state=true_state, action=action
            )

            # forecast
            next_state_ens = model_forecast(state_ens=state_ens, action=action)
            next_state = ensemble_to_state(next_state_ens)

            replay_buffer = add_experience(
                replay_buffer, state, action, reward, next_state, terminated
            )
            return (
                next_true_state,
                next_true_obs,
                next_state_ens,
                next_state,
                key_action,
                replay_buffer,
            ), (true_state, true_obs, state_ens, reward)

        def body_fun(carry, _):
            # observe
            # we got an observation
            # define observation covariance matrix
            true_state, true_obs, state_ens, key_obs, key_action, replay_buffer = carry
            obs_cov = (
                jnp.diag((config.enKF.std_obs * jnp.ones(len(true_obs))))
                * jnp.max(abs(true_obs), axis=0) ** 2
            )

            # add noise on the observation
            key_obs, key_enKF = jax.random.split(key_obs)
            obs = jax.random.multivariate_normal(
                key_obs, true_obs, obs_cov, method="svd"
            )

            # apply enkf to correct the state estimation
            state_ens = model_apply_enKF(Af=state_ens, d=obs, Cdd=obs_cov, key=key_enKF)
            (true_state, true_obs, state_ens, state, key_action, replay_buffer), (
                true_state_arr,
                true_obs_arr,
                state_ens_arr,
                reward_arr,
            ) = jax.lax.scan(
                forecast_fun,
                (true_state, true_obs, state, state_ens, key_action, replay_buffer),
                jnp.arange(wait_steps),
            )
            return (true_state, true_obs, state_ens, key_obs, key_action), (
                true_state_arr,
                true_obs_arr,
                obs,
                state_ens_arr,
                reward_arr,
            )

        n_loops = episode_steps // wait_steps
        (true_state, true_obs, state_ens, _), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            reward_arr,
        ) = jax.lax.scan(
            body_fun,
            (
                true_state,
                true_obs,
                state_ens,
                state,
                key_obs,
                key_action,
                replay_buffer,
            ),
            jnp.arange(n_loops),
        )
        return (
            true_state,
            true_obs,
            state_ens,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            reward_arr,
        )

    def learn_observe_and_forecast(
        true_state,
        true_obs,
        state_ens,
        params,
        wait_steps,
        episode_steps,
        key_obs,
        key_action,
        replay_buffer,
        actor_state,
        critic_state,
    ):
        def forecast_fun(carry, _):
            (
                true_state,
                true_obs,
                state_ens,
                state,
                key_action,
                replay_buffer,
                actor_state,
                critic_state,
            ) = carry

            # get action from the learning actor network
            action = agent.actor.apply(params, state)

            # add exploration noise on the action
            # original paper adds Ornstein-Uhlenbeck process noise
            # but in other papers this is deemed unnecessary
            key_action, _ = jax.random.split(key_action)
            action = add_gaussian_noise(key_action, action, stddev=exploration_stddev)
            # clip the action so that it obeys the limits set by the environment
            action = jnp.clip(action, min=env.action_low, max=env.action_high)

            # get the next observation and reward with this action
            next_true_state, next_true_obs, reward, terminated, _, _ = env_step(
                state=true_state, action=action
            )

            # forecast
            next_state_ens = model_forecast(state_ens=state_ens, action=action)
            next_state = ensemble_to_state(next_state_ens)

            replay_buffer = add_experience(
                replay_buffer, state, action, reward, next_state, terminated
            )

            sampled, replay_buffer = sample_experiences(
                replay_buffer, config.train.batch_size
            )
            (
                sampled_state,
                sampled_action,
                sampled_next_state,
                sampled_reward,
                sampled_terminated,
            ) = sampled
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
            return (
                next_true_state,
                next_true_obs,
                next_state_ens,
                next_state,
                key_action,
                replay_buffer,
                actor_state,
                critic_state,
            ), (true_state, true_obs, state_ens, reward, q_loss, policy_loss)

        def body_fun(carry, _):
            # observe
            # we got an observation
            # define observation covariance matrix
            (
                true_state,
                true_obs,
                state_ens,
                key_obs,
                key_action,
                replay_buffer,
                actor_state,
                critic_state,
            ) = carry
            obs_cov = (
                jnp.diag((config.enKF.std_obs * jnp.ones(len(true_obs))))
                * jnp.max(abs(true_obs), axis=0) ** 2
            )

            # add noise on the observation
            key_obs, key_enKF = jax.random.split(key_obs)
            obs = jax.random.multivariate_normal(
                key_obs, true_obs, obs_cov, method="svd"
            )

            # apply enkf to correct the state estimation
            state_ens = model_apply_enKF(Af=state_ens, d=obs, Cdd=obs_cov, key=key_enKF)
            (
                true_state,
                true_obs,
                state_ens,
                state,
                key_action,
                replay_buffer,
                actor_state,
                critic_state,
            ), (
                true_state_arr,
                true_obs_arr,
                state_ens_arr,
                reward_arr,
                q_loss_arr,
                policy_loss_arr,
            ) = jax.lax.scan(
                forecast_fun,
                (
                    true_state,
                    true_obs,
                    state_ens,
                    state,
                    key_action,
                    replay_buffer,
                    actor_state,
                    critic_state,
                ),
                jnp.arange(wait_steps),
            )
            return (
                true_state,
                true_obs,
                state_ens,
                state,
                key_obs,
                key_action,
                replay_buffer,
                actor_state,
                critic_state,
            ), (true_state_arr, true_obs_arr, obs, state_ens_arr, reward_arr)

        n_loops = episode_steps // wait_steps
        (true_state, true_obs, state_ens, _), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            reward_arr,
            q_loss_arr,
            policy_loss_arr,
        ) = jax.lax.scan(
            body_fun,
            (
                true_state,
                true_obs,
                state_ens,
                key_obs,
                key_action,
                replay_buffer,
                actor_state,
                critic_state,
            ),
            jnp.arange(n_loops),
        )
        return (
            true_state,
            true_obs,
            state_ens,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            reward_arr,
            q_loss_arr,
            policy_loss_arr,
        )

    until_first_observation = partial(
        until_first_observation, observation_starts=config.enKF.observation_starts
    )
    until_first_observation = jax.jit(until_first_observation)

    act_observe_and_forecast = partial(
        act_observe_and_forecast,
        wait_steps=config.enKF.wait_steps,
        episode_steps=config.episode_steps - config.enKF.observation_starts,
    )
    act_observe_and_forecast = jax.jit(act_observe_and_forecast)

    random_observe_and_forecast = partial(
        random_observe_and_forecast,
        wait_steps=config.enKF.wait_steps,
        episode_steps=config.episode_steps - config.enKF.observation_starts,
    )
    random_observe_and_forecast = jax.jit(random_observe_and_forecast)

    learn_observe_and_forecast = partial(
        learn_observe_and_forecast,
        wait_steps=config.enKF.wait_steps,
        episode_steps=config.episode_steps - config.enKF.observation_starts,
    )
    learn_observe_and_forecast = jax.jit(learn_observe_and_forecast)

    def act_evaluate_episode(key_env, key_obs, params):
        # reset the environment
        key_env, key_ens = jax.random.split(key_env)
        init_true_state, init_true_obs, _ = env_reset(key=key_env)

        # initialize enKF
        init_state_ens = initialize_ensemble(u0=init_true_state, key=key_ens)

        # forecast until first observation
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr0,
            true_obs_arr0,
            state_ens_arr0,
        ) = until_first_observation(
            true_state=init_true_state, true_obs=init_true_obs, state_ens=init_state_ens
        )
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            reward_arr,
        ) = act_observe_and_forecast(
            true_state=true_state,
            true_obs=true_obs,
            state_ens=state_ens,
            params=params,
            key_obs=key_obs,
        )

        true_state_arr = jnp.reshape(
            true_state_arr,
            (
                true_state_arr.shape[0] * true_state_arr.shape[1],
                true_state_arr.shape[2],
            ),
        )
        state_ens_arr = jnp.reshape(
            state_ens_arr,
            (
                state_ens_arr.shape[0] * state_ens_arr.shape[1],
                state_ens_arr.shape[2],
                state_ens_arr.shape[3],
            ),
        )
        true_obs_arr = jnp.reshape(
            true_obs_arr,
            (true_obs_arr.shape[0] * true_obs_arr.shape[1], true_obs_arr.shape[2]),
        )

        stack = lambda a, b, c: jnp.vstack((jnp.expand_dims(a, axis=0), b, c))

        return (
            stack(init_true_state, true_state_arr0, true_state_arr),
            stack(init_true_obs, true_obs_arr0, true_obs_arr),
            obs_arr,
            stack(init_state_ens, state_ens_arr0, state_ens_arr),
            reward_arr,
        )

    def random_evaluate_episode(key_env, key_obs, key_action, replay_buffer):
        # do i need to pass actor state?
        # reset the environment
        key_env, key_ens = jax.random.split(key_env)
        init_true_state, init_true_obs, _ = env_reset(key=key_env)

        # initialize enKF
        init_state_ens = initialize_ensemble(u0=init_true_state, key=key_ens)

        # forecast until first observation
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr0,
            true_obs_arr0,
            state_ens_arr0,
        ) = until_first_observation(
            true_state=init_true_state, true_obs=init_true_obs, state_ens=init_state_ens
        )
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            reward_arr,
        ) = random_observe_and_forecast(
            true_state=true_state,
            true_obs=true_obs,
            state_ens=state_ens,
            key_obs=key_obs,
            key_action=key_action,
            replay_buffer=replay_buffer,
        )

        true_state_arr = jnp.reshape(
            true_state_arr,
            (
                true_state_arr.shape[0] * true_state_arr.shape[1],
                true_state_arr.shape[2],
            ),
        )
        state_ens_arr = jnp.reshape(
            state_ens_arr,
            (
                state_ens_arr.shape[0] * state_ens_arr.shape[1],
                state_ens_arr.shape[2],
                state_ens_arr.shape[3],
            ),
        )
        true_obs_arr = jnp.reshape(
            true_obs_arr,
            (true_obs_arr.shape[0] * true_obs_arr.shape[1], true_obs_arr.shape[2]),
        )

        stack = lambda a, b, c: jnp.vstack((jnp.expand_dims(a, axis=0), b, c))

        return (
            stack(init_true_state, true_state_arr0, true_state_arr),
            stack(init_true_obs, true_obs_arr0, true_obs_arr),
            obs_arr,
            stack(init_state_ens, state_ens_arr0, state_ens_arr),
            reward_arr,
        )

    def learn_evaluate_episode(
        key_env, key_obs, key_action, replay_buffer, actor_state, critic_state
    ):
        # do i need to pass actor state?
        # reset the environment
        key_env, key_ens = jax.random.split(key_env)
        init_true_state, init_true_obs, _ = env_reset(key=key_env)

        # initialize enKF
        init_state_ens = initialize_ensemble(u0=init_true_state, key=key_ens)

        # forecast until first observation
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr0,
            true_obs_arr0,
            state_ens_arr0,
        ) = until_first_observation(
            true_state=init_true_state, true_obs=init_true_obs, state_ens=init_state_ens
        )
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            reward_arr,
        ) = learn_observe_and_forecast(
            true_state=true_state,
            true_obs=true_obs,
            state_ens=state_ens,
            key_obs=key_obs,
            key_action=key_action,
            replay_buffer=replay_buffer,
            actor_state=actor_state,
            critic_state=critic_state,
        )

        true_state_arr = jnp.reshape(
            true_state_arr,
            (
                true_state_arr.shape[0] * true_state_arr.shape[1],
                true_state_arr.shape[2],
            ),
        )
        state_ens_arr = jnp.reshape(
            state_ens_arr,
            (
                state_ens_arr.shape[0] * state_ens_arr.shape[1],
                state_ens_arr.shape[2],
                state_ens_arr.shape[3],
            ),
        )
        true_obs_arr = jnp.reshape(
            true_obs_arr,
            (true_obs_arr.shape[0] * true_obs_arr.shape[1], true_obs_arr.shape[2]),
        )

        stack = lambda a, b, c: jnp.vstack((jnp.expand_dims(a, axis=0), b, c))

        return (
            stack(init_true_state, true_state_arr0, true_state_arr),
            stack(init_true_obs, true_obs_arr0, true_obs_arr),
            obs_arr,
            stack(init_state_ens, state_ens_arr0, state_ens_arr),
            reward_arr,
        )

    while global_step < config.total_steps:
        # initialize episode
        if truncated or terminated:
            # reset the environment
            key_env, key_ens = jax.random.split(key_env)
            true_state, true_obs, info = env_reset(key=key_env)

            # initialize enKF
            state_ens = initialize_ensemble(
                env.N,
                model.N,
                model.k,
                true_state,
                config.enKF.std_init,
                config.enKF.m,
                key_ens,
            )

            # forecast until first observation
            (
                next_true_state,
                next_true_obs,
                next_state_ens,
                _,
                _,
                _,
            ) = until_first_observation(
                true_state=true_state, true_obs=true_obs, state_ens=state_ens
            )
            next_state = ensemble_to_state(next_state_ens)

            # reset
            episode_step = 0
            episode_return = 0
            global_step = global_step + config.enKF.observation_starts

        true_state = next_true_state
        true_obs = next_true_obs
        state_ens = next_state_ens
        state = next_state

        if episode_step % config.enKF.wait_steps == 0:
            # we got an observation
            # define observation covariance matrix
            obs_cov = (
                jnp.diag((config.enKF.std_obs * jnp.ones(len(true_obs))))
                * jnp.max(abs(true_obs), axis=0) ** 2
            )

            # add noise on the observation
            key_obs, key_enKF = jax.random.split(key_obs)
            obs = jax.random.multivariate_normal(
                key_obs, true_obs, obs_cov, method="svd"
            )

            # apply enkf to correct the state estimation
            state_ens = model_apply_enKF(Af=state_ens, d=obs, Cdd=obs_cov, key=key_enKF)
            state = ensemble_to_state(state_ens)

        if global_step < config.learning_starts:
            # collect samples before starting to train
            key_action, _ = jax.random.split(key_action)
            action = env_sample_action(key=key_action)
        else:
            # get action from the learning actor network
            action = agent.actor.apply(actor_state.params, state)

            # add exploration noise on the action
            # original paper adds Ornstein-Uhlenbeck process noise
            # but in other papers this is deemed unnecessary
            key_action, _ = jax.random.split(key_action)
            action = add_gaussian_noise(key_action, action, stddev=exploration_stddev)
            # clip the action so that it obeys the limits set by the environment
            action = jnp.clip(action, min=env.action_low, max=env.action_high)

        # get the next observation and reward with this action
        next_true_state, next_true_obs, reward, terminated, truncated, info = env_step(
            state=true_state, action=action
        )

        # forecast
        next_state_ens = model_forecast(state_ens=state_ens, action=action)
        next_state = ensemble_to_state(next_state_ens)

        replay_buffer = add_experience(
            replay_buffer, state, action, reward, next_state, terminated
        )
        episode_step += 1
        global_step += 1
        episode_return += reward
        metrics = {"train": {}, "eval": {}}

        if (
            episode_step + config.enKF.observation_starts
        ) == config.episode_steps or terminated:
            truncated = True
            n_episode += 1
            print(
                f"Episode {n_episode}, Step={global_step}/{config.total_steps}, Return={episode_return}",
                flush=True,
            )
            # episode statistics to log
            metrics["train"]["episode_return"] = episode_return
            metrics["train"]["episode_length"] = episode_step
            metrics["train"]["episode_average_reward"] = episode_return / episode_step
            metrics["train"]["episode_last_reward"] = reward

        # if there is enough data in the buffer optimize
        if (
            global_step >= config.learning_starts
            and replay_buffer["size"] >= config.train.batch_size
        ):
            sampled, replay_buffer = sample_experiences(
                replay_buffer, config.train.batch_size
            )
            (
                sampled_state,
                sampled_action,
                sampled_next_state,
                sampled_reward,
                sampled_terminated,
            ) = sampled
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
                (_, _, _, _, reward_arr) = act_evaluate_episode(
                    key_env=key_env, key_obs=key_obs, params=actor_state.target_params
                )
                eval_ave_return = jnp.sum(reward_arr)
                eval_ave_reward = jnp.mean(reward_arr)
                eval_ave_last_reward = reward_arr[-1]
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
        eval_env = KSenv(**config.env)

    # create agent and run
    agent = DDPG(config, env)
    print("Starting experiment.", flush=True)

    # create low order model
    model = KS(
        nu=config.env.nu,
        N=config.enKF.low_order_N,
        dt=env.dt,
        actuator_locs=config.env.actuator_locs,
        actuator_scale=config.env.actuator_scale,
    )

    actor_state, critic_state, logs = run_experiment(
        config, env, agent, model, wandb_run, logs, checkpoint_dir
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

    # # evaluate with the final weights and plot episode
    # if config.env_name == "KS":
    #     figs = evaluate_KS_for_plotting(
    #         eval_env,
    #         agent.actor,
    #         actor_state.target_params,
    #         model,
    #         config,
    #         eval_episodes=2,
    #     )
    #     for fig_idx, fig in enumerate(figs):
    #         fig.savefig(FLAGS.experiment_path / f"final_evaluation_{fig_idx}.png")
    #     plt.show()

    # close text
    sys.stdout = orig_stdout
    f.close()
    return


if __name__ == "__main__":
    app.run(main)
