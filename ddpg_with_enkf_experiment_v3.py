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
import orbax
import orbax.checkpoint
from absl import app, flags
from flax.training import orbax_utils
from ml_collections import config_flags

import utils.file_processing as fp
import utils.flags as myflags
import wandb
from ddpg import DDPG
from envs.KS_environment_jax import KSenv
from envs.KS_solver_jax import KS
from replay_buffer_jax import add_experience, init_replay_buffer, sample_experiences
from utils import visualizations as vis
from utils import covariance_matrix as cov
from utils.system import set_gpu

# system config
FLAGS = flags.FLAGS
myflags.DEFINE_path("experiment_path", None, "Directory to store experiment results.")
flags.DEFINE_integer("gpu_id", None, "Which gpu to use.")
flags.DEFINE_float("gpu_mem", 0.9, "Fraction of gpu memory to use.")
flags.DEFINE_bool("log_wandb", False, "Use --log_wandb to log the experiment to wandb.")
flags.DEFINE_bool(
    "log_offline", False, "Use --log_offline to log the experiment to local."
)
flags.DEFINE_bool(
    "save_checkpoints",
    False,
    "Use --save_checkpoints to save intermediate model weights.",
)
flags.DEFINE_bool(
    "save_episode_data",
    False,
    "Use --save_episode_data to save the data from episodes.",
)
flags.DEFINE_bool(
    "make_plots",
    False,
    "Use --make_plots to plot an episode and save it.",
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


def log_metrics_wandb(wandb_run, metrics, step=None):
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


def plot_KS_episode(
    env,
    model,
    true_state_arr,
    true_obs_arr,
    unfilled_obs_arr,
    state_ens_arr,
    action_arr,
    reward_arr,
    wait_steps,
    observation_starts,
):
    x = env.ks_solver.x
    x_obs = env.observation_locs
    x_act = env.actuator_locs * (env.ks_solver.L / (2 * jnp.pi))
    target = env.target

    full_obs_mat = get_observation_matrix(model.N, model.L, x)
    obs_mat = get_observation_matrix(model.N, model.L, x_obs)
    # fill the observations
    obs_arr = jnp.nan * jnp.ones_like(true_obs_arr)
    for i in range(len(obs_arr)):
        obs_arr = obs_arr.at[wait_steps * i + observation_starts].set(
            unfilled_obs_arr[i]
        )

    # get full state from low order model
    state_ens_arr_ = jnp.hstack(
        (state_ens_arr, jnp.conjugate(jnp.flip(state_ens_arr[:, 1:-1], axis=1)))
    )
    full_state_ens_arr = jnp.real(
        jnp.einsum("kjm,ij->kim", state_ens_arr_, full_obs_mat)
    )
    # get the mean
    full_state_mean_arr = jnp.mean(full_state_ens_arr, axis=-1)

    # get fourier coefficients
    true_state_arr_f = jnp.fft.rfft(true_state_arr, axis=1)
    mag_state_arr = 2 / env.N * jnp.abs(true_state_arr_f)
    mag_state_ens_arr = 2 / model.N * jnp.abs(state_ens_arr)
    mag_state_mean_arr = jnp.mean(mag_state_ens_arr, axis=-1)

    # get observations from low order model
    obs_ens_arr = jnp.real(jnp.einsum("kjm,ij->kim", state_ens_arr_, obs_mat))
    # get the mean
    obs_mean_arr = jnp.mean(obs_ens_arr, axis=-1)

    # fill the observations
    fig = vis.plot_episode(
        x,
        x_obs,
        x_act,
        target,
        true_state_arr,
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
        reward_arr,
    )
    return fig


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


def draw_initial_condition(u0, std_init, key):
    # fourier transform the initial condition
    u0_f = jnp.fft.rfft(u0, axis=-1)
    key, subkey = jax.random.split(key)
    u0_real = jax.random.multivariate_normal(
        key,
        u0_f.real,
        jnp.diag((u0_f.real * std_init) ** 2),
        (1,),
        method="svd",
    )
    # covariance matrix is rank deficient because zeroth
    u0_complex = jax.random.multivariate_normal(
        subkey,
        u0_f.imag,
        jnp.diag((u0_f.imag * std_init) ** 2),
        (1,),
        method="svd",
    )
    u0_f = u0_real + u0_complex * 1j
    u0 = jnp.fft.irfft(u0_f.squeeze())
    return u0


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
    Ensemble Kalman Filter as derived in Evensen book (2009) eq. 9.27.
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
    # Aa = Af + jnp.dot(Psi_f, X) # gives same result as above
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
    model_initialize_ensemble = partial(
        initialize_ensemble,
        env_N=env.N,
        model_N=model.N,
        model_k=model.k,
        std_init=config.enKF.std_init,
        m=config.enKF.m,
    )
    model_initialize_ensemble = jax.jit(model_initialize_ensemble)
    env_draw_initial_condition = partial(
        draw_initial_condition,
        std_init=config.enKF.std_init,
    )
    env_draw_initial_condition = jax.jit(env_draw_initial_condition)

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

    def until_first_observation(true_state, true_obs, state_ens, observation_starts):
        def body_fun(carry, _):
            true_state, true_obs, state_ens = carry
            # advance true environment
            action = null_action
            true_state, true_obs, _, _, _, _ = env_step(state=true_state, action=action)
            # advance model
            state_ens = model_forecast(state_ens=state_ens, action=action)
            return (true_state, true_obs, state_ens), (
                true_state,
                true_obs,
                state_ens,
                action,
            )

        (true_state, true_obs, state_ens), (
            true_state_arr,
            true_obs_arr,
            state_ens_arr,
            action_arr,
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
            action_arr,
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
                action,
                reward,
            )

        def body_fun(carry, _):
            # observe
            # we got an observation
            # define observation covariance matrix
            true_state, true_obs, state_ens, key_obs = carry
            obs_cov = cov.get_max(std=config.enKF.std_obs, y=true_obs)

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
                action_arr,
                reward_arr,
            ) = jax.lax.scan(
                forecast_fun, (true_state, true_obs, state_ens), jnp.arange(wait_steps)
            )
            return (true_state, true_obs, state_ens, key_obs), (
                true_state_arr,
                true_obs_arr,
                obs,
                state_ens_arr,
                action_arr,
                reward_arr,
            )

        n_loops = episode_steps // wait_steps
        (true_state, true_obs, state_ens, key_obs), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_arr,
        ) = jax.lax.scan(
            body_fun, (true_state, true_obs, state_ens, key_obs), jnp.arange(n_loops)
        )
        return (
            true_state,
            true_obs,
            state_ens,
            key_obs,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
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
            true_state, true_obs, state_ens, key_action, replay_buffer = carry
            state = ensemble_to_state(state_ens)

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
                key_action,
                replay_buffer,
            ), (true_state, true_obs, state_ens, action, reward)

        def body_fun(carry, _):
            # observe
            # we got an observation
            # define observation covariance matrix
            true_state, true_obs, state_ens, key_obs, key_action, replay_buffer = carry
            obs_cov = cov.get_max(std=config.enKF.std_obs, y=true_obs)

            # add noise on the observation
            key_obs, key_enKF = jax.random.split(key_obs)
            obs = jax.random.multivariate_normal(
                key_obs, true_obs, obs_cov, method="svd"
            )

            # apply enkf to correct the state estimation
            state_ens = model_apply_enKF(Af=state_ens, d=obs, Cdd=obs_cov, key=key_enKF)
            (true_state, true_obs, state_ens, key_action, replay_buffer), (
                true_state_arr,
                true_obs_arr,
                state_ens_arr,
                action_arr,
                reward_arr,
            ) = jax.lax.scan(
                forecast_fun,
                (true_state, true_obs, state_ens, key_action, replay_buffer),
                jnp.arange(wait_steps),
            )
            return (
                true_state,
                true_obs,
                state_ens,
                key_obs,
                key_action,
                replay_buffer,
            ), (
                true_state_arr,
                true_obs_arr,
                obs,
                state_ens_arr,
                action_arr,
                reward_arr,
            )

        n_loops = episode_steps // wait_steps
        (true_state, true_obs, state_ens, key_obs, key_action, replay_buffer), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_arr,
        ) = jax.lax.scan(
            body_fun,
            (true_state, true_obs, state_ens, key_obs, key_action, replay_buffer),
            jnp.arange(n_loops),
        )
        return (
            true_state,
            true_obs,
            state_ens,
            key_obs,
            key_action,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_arr,
            replay_buffer,
        )

    def learn_observe_and_forecast(
        true_state,
        true_obs,
        state_ens,
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
                key_action,
                replay_buffer,
                actor_state,
                critic_state,
            ) = carry

            state = ensemble_to_state(state_ens)

            # get action from the learning actor network
            action = agent.actor.apply(actor_state.params, state)

            # add exploration noise on the action
            # original paper adds Ornstein-Uhlenbeck process noise
            # but in other papers this is deemed unnecessary
            key_action, _ = jax.random.split(key_action)
            action = add_gaussian_noise(key_action, action, stddev=exploration_stddev)
            # clip the action so that it obeys the limits set by the environment
            action = jnp.clip(action, a_min=env.action_low, a_max=env.action_high)

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
                key_action,
                replay_buffer,
                actor_state,
                critic_state,
            ), (true_state, true_obs, state_ens, action, reward, q_loss, policy_loss)

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
            obs_cov = cov.get_max(std=config.enKF.std_obs, y=true_obs)

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
                key_action,
                replay_buffer,
                actor_state,
                critic_state,
            ), (
                true_state_arr,
                true_obs_arr,
                state_ens_arr,
                action_arr,
                reward_arr,
                q_loss_arr,
                policy_loss_arr,
            ) = jax.lax.scan(
                forecast_fun,
                (
                    true_state,
                    true_obs,
                    state_ens,
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
                key_obs,
                key_action,
                replay_buffer,
                actor_state,
                critic_state,
            ), (
                true_state_arr,
                true_obs_arr,
                obs,
                state_ens_arr,
                action_arr,
                reward_arr,
                q_loss_arr,
                policy_loss_arr,
            )

        n_loops = episode_steps // wait_steps
        (
            true_state,
            true_obs,
            state_ens,
            key_obs,
            key_action,
            replay_buffer,
            actor_state,
            critic_state,
        ), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
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
            key_obs,
            key_action,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_arr,
            q_loss_arr,
            policy_loss_arr,
            replay_buffer,
            actor_state,
            critic_state,
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

    def act_episode(key_env, key_obs, params):
        # reset the environment
        key_env, key_ens, key_init = jax.random.split(key_env, 3)
        init_true_state_mean, _, _ = env_reset(key=key_env)
        init_true_state = env_draw_initial_condition(
            u0=init_true_state_mean, key=key_init
        )
        init_true_obs = init_true_state[env.observation_inds]

        # initialize enKF
        init_state_ens = model_initialize_ensemble(u0=init_true_state_mean, key=key_ens)

        # forecast until first observation
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr0,
            true_obs_arr0,
            state_ens_arr0,
            action_arr0,
        ) = until_first_observation(
            true_state=init_true_state, true_obs=init_true_obs, state_ens=init_state_ens
        )
        (
            true_state,
            true_obs,
            state_ens,
            key_obs,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
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
        true_obs_arr = jnp.reshape(
            true_obs_arr,
            (true_obs_arr.shape[0] * true_obs_arr.shape[1], true_obs_arr.shape[2]),
        )
        state_ens_arr = jnp.reshape(
            state_ens_arr,
            (
                state_ens_arr.shape[0] * state_ens_arr.shape[1],
                state_ens_arr.shape[2],
                state_ens_arr.shape[3],
            ),
        )
        action_arr = jnp.reshape(
            action_arr,
            (action_arr.shape[0] * action_arr.shape[1], action_arr.shape[2]),
        )
        reward_arr = jnp.reshape(
            reward_arr,
            (reward_arr.shape[0] * reward_arr.shape[1],),
        )
        stack = lambda a, b, c: jnp.vstack((jnp.expand_dims(a, axis=0), b, c))

        return (
            stack(init_true_state, true_state_arr0, true_state_arr),
            stack(init_true_obs, true_obs_arr0, true_obs_arr),
            obs_arr,
            stack(init_state_ens, state_ens_arr0, state_ens_arr),
            stack(null_action, action_arr0, action_arr),
            reward_arr,
            key_env,
            key_obs,
        )

    def random_episode(key_env, key_obs, key_action, replay_buffer):
        # reset the environment
        key_env, key_ens, key_init = jax.random.split(key_env, 3)
        init_true_state_mean, _, _ = env_reset(key=key_env)
        init_true_state = env_draw_initial_condition(
            u0=init_true_state_mean, key=key_init
        )
        init_true_obs = init_true_state[env.observation_inds]

        # initialize enKF
        init_state_ens = model_initialize_ensemble(u0=init_true_state_mean, key=key_ens)

        # forecast until first observation
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr0,
            true_obs_arr0,
            state_ens_arr0,
            action_arr0,
        ) = until_first_observation(
            true_state=init_true_state, true_obs=init_true_obs, state_ens=init_state_ens
        )
        (
            true_state,
            true_obs,
            state_ens,
            key_obs,
            key_action,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_arr,
            replay_buffer,
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
        true_obs_arr = jnp.reshape(
            true_obs_arr,
            (true_obs_arr.shape[0] * true_obs_arr.shape[1], true_obs_arr.shape[2]),
        )
        state_ens_arr = jnp.reshape(
            state_ens_arr,
            (
                state_ens_arr.shape[0] * state_ens_arr.shape[1],
                state_ens_arr.shape[2],
                state_ens_arr.shape[3],
            ),
        )
        action_arr = jnp.reshape(
            action_arr,
            (action_arr.shape[0] * action_arr.shape[1], action_arr.shape[2]),
        )
        reward_arr = jnp.reshape(
            reward_arr,
            (reward_arr.shape[0] * reward_arr.shape[1],),
        )

        stack = lambda a, b, c: jnp.vstack((jnp.expand_dims(a, axis=0), b, c))

        return (
            stack(init_true_state, true_state_arr0, true_state_arr),
            stack(init_true_obs, true_obs_arr0, true_obs_arr),
            obs_arr,
            stack(init_state_ens, state_ens_arr0, state_ens_arr),
            stack(null_action, action_arr0, action_arr),
            reward_arr,
            replay_buffer,
            key_env,
            key_obs,
            key_action,
        )

    def learn_episode(
        key_env, key_obs, key_action, replay_buffer, actor_state, critic_state
    ):
        # reset the environment
        key_env, key_ens, key_init = jax.random.split(key_env, 3)
        init_true_state_mean, _, _ = env_reset(key=key_env)
        init_true_state = env_draw_initial_condition(
            u0=init_true_state_mean, key=key_init
        )
        init_true_obs = init_true_state[env.observation_inds]

        # initialize enKF
        init_state_ens = model_initialize_ensemble(u0=init_true_state_mean, key=key_ens)

        # forecast until first observation
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr0,
            true_obs_arr0,
            state_ens_arr0,
            action_arr0,
        ) = until_first_observation(
            true_state=init_true_state, true_obs=init_true_obs, state_ens=init_state_ens
        )
        (
            true_state,
            true_obs,
            state_ens,
            key_obs,
            key_action,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_arr,
            q_loss_arr,
            policy_loss_arr,
            replay_buffer,
            actor_state,
            critic_state,
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

        true_obs_arr = jnp.reshape(
            true_obs_arr,
            (true_obs_arr.shape[0] * true_obs_arr.shape[1], true_obs_arr.shape[2]),
        )

        state_ens_arr = jnp.reshape(
            state_ens_arr,
            (
                state_ens_arr.shape[0] * state_ens_arr.shape[1],
                state_ens_arr.shape[2],
                state_ens_arr.shape[3],
            ),
        )
        action_arr = jnp.reshape(
            action_arr,
            (action_arr.shape[0] * action_arr.shape[1], action_arr.shape[2]),
        )
        reward_arr = jnp.reshape(
            reward_arr,
            (reward_arr.shape[0] * reward_arr.shape[1],),
        )

        q_loss_arr = jnp.reshape(
            q_loss_arr,
            (q_loss_arr.shape[0] * q_loss_arr.shape[1],),
        )

        policy_loss_arr = jnp.reshape(
            policy_loss_arr,
            (policy_loss_arr.shape[0] * policy_loss_arr.shape[1],),
        )

        stack = lambda a, b, c: jnp.vstack((jnp.expand_dims(a, axis=0), b, c))

        return (
            stack(init_true_state, true_state_arr0, true_state_arr),
            stack(init_true_obs, true_obs_arr0, true_obs_arr),
            obs_arr,
            stack(init_state_ens, state_ens_arr0, state_ens_arr),
            stack(null_action, action_arr0, action_arr),
            reward_arr,
            q_loss_arr,
            policy_loss_arr,
            replay_buffer,
            actor_state,
            critic_state,
            key_env,
            key_obs,
            key_action,
        )

    # fill replay buffer with episodes with random inputs
    random_episodes = config.learning_starts // config.episode_steps
    n_plot = config.plot_freq // config.episode_steps
    for i in range(random_episodes):
        (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_arr,
            replay_buffer,
            key_env,
            key_obs,
            key_action,
        ) = random_episode(key_env, key_obs, key_action, replay_buffer)
        random_return = jnp.sum(reward_arr)
        random_ave_reward = jnp.mean(reward_arr)
        random_last_reward = reward_arr[-1]
        print(
            f"Random input, Episode={i+1}/{random_episodes}, Return={random_return},  Average Reward={random_ave_reward}, Last Reward ={random_last_reward}",
            flush=True,
        )
        if i == 0 or (i + 1) % n_plot == 0:
            if FLAGS.make_plots == True:
                fig = plot_KS_episode(
                    env,
                    model,
                    true_state_arr,
                    true_obs_arr,
                    obs_arr,
                    state_ens_arr,
                    action_arr,
                    reward_arr,
                    config.enKF.wait_steps,
                    config.enKF.observation_starts,
                )
                fig.savefig(
                    FLAGS.experiment_path / "plots" / f"random_episode_{i+1}.png"
                )
            if FLAGS.save_episode_data == True:
                episode_dict = {
                    "true_state": true_state_arr,
                    "true_obs": true_obs_arr,
                    "obs": obs_arr,
                    "state_ens": state_ens_arr,
                    "action": action_arr,
                    "reward": reward_arr,
                }
                fp.write_h5(
                    FLAGS.experiment_path / "episode_data" / f"random_episode_{i+1}.h5",
                    episode_dict,
                )

    print("\n")
    # learn and evaluate
    learn_episodes = (
        config.total_steps - config.learning_starts
    ) // config.episode_steps
    n_eval = config.eval_freq // config.episode_steps
    learn_steps = config.episode_steps - config.enKF.observation_starts
    metrics = {"train": {}, "eval": {}}
    for i in range(learn_episodes):
        (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_arr,
            q_loss_arr,
            policy_loss_arr,
            replay_buffer,
            actor_state,
            critic_state,
            key_env,
            key_obs,
            key_action,
        ) = learn_episode(
            key_env, key_obs, key_action, replay_buffer, actor_state, critic_state
        )
        train_return = jnp.sum(reward_arr)
        train_ave_reward = jnp.mean(reward_arr)
        train_last_reward = reward_arr[-1]
        print(
            f"Training, Episode={i+1}/{learn_episodes}, Return={train_return},  Average Reward={train_ave_reward}, Last Reward ={train_last_reward}",
            flush=True,
        )

        if i == 0 or (i + 1) % n_plot == 0:
            if FLAGS.make_plots == True:
                fig = plot_KS_episode(
                    env,
                    model,
                    true_state_arr,
                    true_obs_arr,
                    obs_arr,
                    state_ens_arr,
                    action_arr,
                    reward_arr,
                    config.enKF.wait_steps,
                    config.enKF.observation_starts,
                )
                fig.savefig(
                    FLAGS.experiment_path / "plots" / f"learn_episode_{i+1}.png"
                )
            if FLAGS.save_episode_data == True:
                episode_dict = {
                    "true_state": true_state_arr,
                    "true_obs": true_obs_arr,
                    "obs": obs_arr,
                    "state_ens": state_ens_arr,
                    "action": action_arr,
                    "reward": reward_arr,
                }
                fp.write_h5(
                    FLAGS.experiment_path / "episode_data" / f"learn_episode_{i+1}.h5",
                    episode_dict,
                )

        for j, (q_loss, policy_loss) in enumerate(zip(q_loss_arr, policy_loss_arr)):
            metrics["q_loss"] = q_loss
            metrics["policy_loss"] = policy_loss
            if wandb_run is not None:
                log_metrics_wandb(
                    wandb_run,
                    metrics={"q_loss": q_loss, "policy_loss": policy_loss},
                    step=i * learn_steps + j,
                )

        metrics["train"]["episode_return"] = train_return
        metrics["train"]["episode_average_reward"] = train_ave_reward
        metrics["train"]["episode_last_reward"] = train_last_reward
        if (i + 1) % n_eval == 0:
            eval_ave_return = 0
            eval_ave_reward = 0
            eval_ave_last_reward = 0
            for j in range(config.eval_episodes):
                (
                    true_state_arr,
                    true_obs_arr,
                    obs_arr,
                    state_ens_arr,
                    action_arr,
                    reward_arr,
                    key_env,
                    key_obs,
                ) = act_episode(key_env, key_obs, actor_state.params)
                eval_ave_return += jnp.sum(reward_arr)
                eval_ave_reward += jnp.mean(reward_arr)
                eval_ave_last_reward += reward_arr[-1]
                if (i + 1) % n_plot == 0 and j == 0:
                    if FLAGS.make_plots == True:
                        fig = plot_KS_episode(
                            env,
                            model,
                            true_state_arr,
                            true_obs_arr,
                            obs_arr,
                            state_ens_arr,
                            action_arr,
                            reward_arr,
                            config.enKF.wait_steps,
                            config.enKF.observation_starts,
                        )
                        fig.savefig(
                            FLAGS.experiment_path / "plots" / f"eval_episode_{i+1}.png"
                        )
                    if FLAGS.save_episode_data == True:
                        episode_dict = {
                            "true_state": true_state_arr,
                            "true_obs": true_obs_arr,
                            "obs": obs_arr,
                            "state_ens": state_ens_arr,
                            "action": action_arr,
                            "reward": reward_arr,
                        }
                        fp.write_h5(
                            FLAGS.experiment_path
                            / "episode_data"
                            / f"eval_episode_{i+1}.h5",
                            episode_dict,
                        )
            eval_ave_return = eval_ave_return / config.eval_episodes
            eval_ave_reward = eval_ave_reward / config.eval_episodes
            eval_ave_last_reward = eval_ave_last_reward / config.eval_episodes
            print(
                f"\n Evaluation, Episode={i+1}/{learn_episodes}, Average Return={eval_ave_return}, Average Reward={eval_ave_reward}, Average Last Reward ={eval_ave_last_reward} \n ",
                flush=True,
            )
            metrics["eval"]["average_return"] = eval_ave_return
            metrics["eval"]["average_reward"] = eval_ave_reward
            metrics["eval"]["average_last_reward"] = eval_ave_last_reward

        if (i + 1) == learn_episodes:
            (
                true_state_arr,
                true_obs_arr,
                obs_arr,
                state_ens_arr,
                action_arr,
                reward_arr,
                key_env,
                key_obs,
            ) = act_episode(key_env, key_obs, actor_state.params)
            final_eval_return = jnp.sum(reward_arr)
            final_eval_ave_reward = jnp.mean(reward_arr)
            final_eval_last_reward = reward_arr[-1]
            fig = plot_KS_episode(
                env,
                model,
                true_state_arr,
                true_obs_arr,
                obs_arr,
                state_ens_arr,
                action_arr,
                reward_arr,
                config.enKF.wait_steps,
                config.enKF.observation_starts,
            )
            fig.savefig(FLAGS.experiment_path / "plots" / f"final_eval_episode.png")
            print(
                f"\n Final evaluation, Episode={i+1}/{learn_episodes}, Return={final_eval_return}, Average Reward={final_eval_ave_reward}, Last Reward ={final_eval_last_reward} \n ",
                flush=True,
            )
            episode_dict = {
                "true_state": true_state_arr,
                "true_obs": true_obs_arr,
                "obs": obs_arr,
                "state_ens": state_ens_arr,
                "action": action_arr,
                "reward": reward_arr,
            }
            fp.write_h5(
                FLAGS.experiment_path / "episode_data" / f"final_eval_episode.h5",
                episode_dict,
            )

        if wandb_run is not None:
            log_metrics_wandb(wandb_run, metrics, step=(i + 1) * learn_steps - 1)
        # checkpoint the model
        if checkpoint_dir is not None:
            checkpoint = {"actor": actor_state, "critic": critic_state}
            save_args = orbax_utils.save_args_from_target(checkpoint)
            checkpoint_manager.save(
                (i + 1) * learn_steps, checkpoint, save_kwargs={"save_args": save_args}
            )
    return (
        actor_state,
        critic_state,
    )


def main(_):
    config = FLAGS.config

    # set up system
    if FLAGS.gpu_id:
        set_gpu(FLAGS.gpu_id, FLAGS.gpu_mem)

    if not FLAGS.experiment_path:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        FLAGS.experiment_path = (
            Path.cwd() / "local_results" / config.env_name / f"run_{dt_string}"
        )

    # setup the experiment path
    FLAGS.experiment_path.mkdir(parents=True, exist_ok=True)

    if FLAGS.make_plots == True:
        plots_path = FLAGS.experiment_path / "plots"
        plots_path.mkdir(parents=True, exist_ok=True)

    if FLAGS.save_episode_data == True:
        episode_data_path = FLAGS.experiment_path / "episode_data"
        episode_data_path.mkdir(parents=True, exist_ok=True)

    # redirect print to text file
    orig_stdout = sys.stdout
    f = open(FLAGS.experiment_path / "out.txt", "w")
    sys.stdout = f
    print(f"Experiment will be saved to {FLAGS.experiment_path}", flush=True)

    # add environment config to the config
    config.env = FLAGS.env_config

    fp.save_config(FLAGS.experiment_path, config)

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

    actor_state, critic_state = run_experiment(
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
        fp.write_h5(FLAGS.experiment_path / "logs.h5", logs)

    # close text
    sys.stdout = orig_stdout
    f.close()
    return


if __name__ == "__main__":
    app.run(main)
