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
from utils import preprocessing as pp

import numpy as np
from esn.esn import ESN
from esn.validation import validate, set_ESN
from esn.utils import errors, scalers
import jax_esn.esn as jesn
from jax_esn.esn import ESN as JESN
import matplotlib.pyplot as plt

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
    True,
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
_ESN = config_flags.DEFINE_config_file(
    "esn_config", "configs/ESN_config.py", "Contains configs for the ESN."
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
    state_ens_arr,  # has shape (time_steps, state_dimension, ensemble_size)
    before_readout,
    action_arr,
    reward_env_arr,
    reward_model_arr,
    wait_steps,
    observation_starts,
):
    x = env.ks_solver.x
    x_obs = env.observation_locs
    x_act = env.actuator_locs * (env.ks_solver.L / (2 * jnp.pi))
    target = env.target

    # fill the observations
    obs_arr = jnp.nan * jnp.ones_like(true_obs_arr)
    for i in range(len(obs_arr)):
        obs_arr = obs_arr.at[wait_steps * i + observation_starts].set(
            unfilled_obs_arr[i]
        )

    # get full state from low order model
    single_readout_fn = (
        lambda x: before_readout(x, model.output_bias) @ model.output_weights
    )
    time_readout_fn = jax.vmap(single_readout_fn, in_axes=0, out_axes=0)
    ensemble_readout_fn = jax.vmap(time_readout_fn, in_axes=2, out_axes=2)

    # Apply the function
    full_state_ens_arr = ensemble_readout_fn(state_ens_arr)

    # get the mean
    full_state_mean_arr = jnp.mean(full_state_ens_arr, axis=-1)

    # get observations from low order model
    obs_ens_arr = full_state_ens_arr[:, env.observation_inds, :]

    # get the mean
    obs_mean_arr = jnp.mean(obs_ens_arr, axis=-1)

    # get fourier coefficients
    true_state_arr_f = jnp.fft.rfft(true_state_arr, axis=1)

    # get full state from low order model
    state_ens_arr_f = jax.vmap(
        lambda x: jnp.fft.rfft(x, axis=1), in_axes=-1, out_axes=-1
    )(full_state_ens_arr)

    mag_state_arr = 2 / env.N * jnp.abs(true_state_arr_f)
    mag_state_ens_arr = 2 / model.N_dim * jnp.abs(state_ens_arr_f)
    mag_state_mean_arr = jnp.mean(mag_state_ens_arr, axis=-1)

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
        reward_env_arr,
        reward_model_arr,
    )
    return fig


def initialize_ensemble_with_washout(my_ESN, U_washout, P_washout, std_init, m, key):
    # initialize an ensemble of ESNs
    r0 = jesn.run_washout(my_ESN, U_washout, P_washout)

    # create an ensemble by perturbing the reservoir state
    key, subkey = jax.random.split(key)

    # this is slow when r big
    r0_ens = jax.random.multivariate_normal(
        key,
        r0,
        jnp.diag((r0 * std_init) ** 2),
        (m,),
        method="svd",
    ).T

    return r0_ens


def initialize_ensemble_with_auto_washout(my_ESN, N_washout, u0, p0, std_init, m, key):
    key, subkey = jax.random.split(key)

    # Sample ensemble of u0
    u0_sampled = jax.random.multivariate_normal(
        subkey,
        u0,
        jnp.diag((u0 * std_init) ** 2),
        shape=(m,),
        method="svd",
    )

    # Prepare p_washout for all members (shape: [N_washout, param_dim])
    p_washout = jnp.tile(p0[None, :], (N_washout, 1))  # Repeat p0 over washout steps

    # Create repeated inputs for washout
    def create_washout_inputs(u0_member):
        u_washout = jnp.tile(
            u0_member[None, :], (N_washout, 1)
        )  # Repeat u0_member over washout steps
        return u_washout

    # Define a single instance of washout
    def single_washout(u0_member):
        u_washout = create_washout_inputs(u0_member)
        return jesn.run_washout(my_ESN, u_washout, p_washout)

    # Vectorize the washout process over the ensemble
    r0_ens = jax.vmap(single_washout)(u0_sampled)

    # transpose so we have column vectors for the reservoir state
    return r0_ens.T


def draw_initial_condition(u0, std_init, key):
    key, subkey = jax.random.split(key)

    u0_sampled = jax.random.multivariate_normal(
        key,
        u0,
        jnp.diag((u0 * std_init) ** 2),
        (1,),
        method="svd",
    )
    return u0_sampled.squeeze()


def get_observation_matrix(my_ESN, observation_indices):
    """
    Create an observation matrix for the given ESN based on observation indices.
    """
    # Initialize a zero matrix with appropriate dimensions
    M = jnp.zeros((len(observation_indices), my_ESN.output_weights.T.shape[0]))

    # Set the appropriate entries to 1 based on observation_indices
    M = M.at[jnp.arange(len(observation_indices)), observation_indices].set(1)

    # Multiply with the transposed output weight matrix and return
    return M @ my_ESN.output_weights[: -len(my_ESN.output_bias)].T


def ensemble_to_state(state_ens, my_ESN, before_readout):
    state = jnp.mean(state_ens, axis=-1)
    # inverse rfft before passing to the neural network
    # pass the full observed state
    # another idea is to pass the reservoir state
    state = before_readout(state, my_ESN.output_bias) @ my_ESN.output_weights
    return state


def forecast(state_ens, action, frame_skip, my_ESN, before_readout):
    """
    Forecast the state ensemble over a number of steps.

    Args:
        state_ens: Ensemble of states. Shape [n_ensemble, n_state].
        action: Action applied to the system.
        frame_skip: Number of steps to advance.

    Returns:
        Updated state ensemble.
    """

    def closed_loop_step_fn(x):
        # Closed-loop step of ESN
        x_augmented = before_readout(x, my_ESN.output_bias)
        y = jnp.dot(x_augmented, my_ESN.W_out)
        x_next = jesn.step(my_ESN, x, y, action)
        return x_next

    def step_fn(x, _):
        return (jax.vmap(closed_loop_step_fn, in_axes=-1, out_axes=-1)(x), None)

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

    # Matrix to invert
    C = (m - 1) * Cdd + jnp.dot(S, S.T)
    # Cinv = jnp.linalg.inv(C)

    # X = jnp.dot(S.T, jnp.dot(Cinv, (D - Y)))
    X = jnp.dot(S.T, jnp.linalg.solve(C, D - Y))

    Aa = Af + jnp.dot(Af, X)
    # Aa = Af + jnp.dot(Psi_f, X) # gives same result as above
    return Aa


def inflate_ensemble(A, rho):
    A_m = jnp.mean(A, -1, keepdims=True)
    return A_m + rho * (A - A_m)


def before_EnKF_r1(x):
    return x


def before_EnKF_r2(x):
    # go to r2 mode
    x2 = x.at[1::2].set(x[1::2] ** 2)
    return x2


def after_EnKF_r1(x):
    return x


def after_EnKF_r2(x, y):
    # to go back to r from r2 mode
    x2 = x.at[1::2].set(jnp.sign(y[1::2]) * jnp.sqrt(jnp.maximum(x[1::2], 0)))
    return x2


def apply_enKF(m, Af, d, Cdd, M, my_ESN, before_EnKF, after_EnKF, key, rho=1.0):
    # get the reservoir state before readout
    # before the readout because we need our observation matrix to be linear
    # if we're using r2 mode then the EnKF is applied on the r2 state
    # without the output bias because we apply EnKF on the r/r2 state and bias is constant
    Af_full = before_EnKF(Af)

    # remove the bias from the data
    Aa_full = EnKF(m, Af_full, d - my_ESN.observation_bias, Cdd, M, key)

    Aa = after_EnKF(Aa_full, Af)

    # inflate analysed state ensemble
    # helps with the collapse of variance when using small ensemble
    Aa = inflate_ensemble(Aa, rho)
    return Aa


def generate_training_episode(config, env, episode_type="null_action"):
    # create a action of zeros to pass
    null_action = jnp.zeros(env.action_size)

    # jit the necessary environment functions
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

    def null_action_observe(
        true_state,
        true_obs,
        wait_steps,
        episode_steps,
        key_obs,
    ):
        def step_fun(carry, _):
            true_state, true_obs = carry

            # get action
            action = null_action

            # get the next observation and reward with this action
            true_state, true_obs, reward, _, _, _ = env_step(
                state=true_state, action=action
            )

            return (true_state, true_obs), (
                true_state,
                true_obs,
                action,
                reward,
            )

        def body_fun(carry, _):
            # observe
            # we got an observation
            # define observation covariance matrix
            true_state, true_obs, key_obs = carry

            obs_cov = cov.get_max(std=config.enKF.std_obs, y=true_obs)

            # add noise on the observation
            key_obs, _ = jax.random.split(key_obs)
            obs = jax.random.multivariate_normal(
                key_obs, true_obs, obs_cov, method="svd"
            )

            (true_state, true_obs), (
                true_state_arr,
                true_obs_arr,
                action_arr,
                reward_arr,
            ) = jax.lax.scan(step_fun, (true_state, true_obs), jnp.arange(wait_steps))
            return (true_state, true_obs, key_obs), (
                true_state_arr,
                true_obs_arr,
                obs,
                action_arr,
                reward_arr,
            )

        n_loops = episode_steps // wait_steps
        (true_state, true_obs, key_obs), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            action_arr,
            reward_arr,
        ) = jax.lax.scan(body_fun, (true_state, true_obs, key_obs), jnp.arange(n_loops))
        return (
            true_state,
            true_obs,
            key_obs,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            action_arr,
            reward_arr,
        )

    def random_action_observe(
        true_state,
        true_obs,
        wait_steps,
        episode_steps,
        key_obs,
        key_action,
    ):
        def step_fun(carry, _):
            true_state, true_obs, action = carry

            # get the next observation and reward with this action
            true_state, true_obs, reward, _, _, _ = env_step(
                state=true_state, action=action
            )

            return (true_state, true_obs, action), (
                true_state,
                true_obs,
                action,
                reward,
            )

        def body_fun(carry, _):
            # observe
            # we got an observation
            true_state, true_obs, key_obs, key_action = carry

            # get action
            key_action, _ = jax.random.split(key_action)
            action = env_sample_action(key=key_action)

            obs_cov = cov.get_max(std=config.enKF.std_obs, y=true_obs)

            # add noise on the observation
            key_obs, _ = jax.random.split(key_obs)
            obs = jax.random.multivariate_normal(
                key_obs, true_obs, obs_cov, method="svd"
            )

            # propagate environment with the given action
            (true_state, true_obs, action), (
                true_state_arr,
                true_obs_arr,
                action_arr,
                reward_arr,
            ) = jax.lax.scan(
                step_fun,
                (true_state, true_obs, action),
                jnp.arange(wait_steps),
            )

            return (
                true_state,
                true_obs,
                key_obs,
                key_action,
            ), (true_state_arr, true_obs_arr, obs, action_arr, reward_arr)

        n_loops = episode_steps // wait_steps
        (true_state, true_obs, key_obs, key_action), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            action_arr,
            reward_arr,
        ) = jax.lax.scan(
            body_fun,
            (true_state, true_obs, key_obs, key_action),
            jnp.arange(n_loops),
        )
        return (
            true_state,
            true_obs,
            key_obs,
            key_action,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            action_arr,
            reward_arr,
        )

    null_action_observe = partial(
        null_action_observe,
        wait_steps=1,  # WE GENERATE TRAINING DATA BY SAMPLING AT EVERY TIME STEP
        episode_steps=config.episode_steps,
    )
    null_action_observe = jax.jit(null_action_observe)

    random_action_observe = partial(
        random_action_observe,
        wait_steps=1,  # WE GENERATE TRAINING DATA BY SAMPLING AT EVERY TIME STEP
        episode_steps=config.episode_steps,
    )
    random_action_observe = jax.jit(random_action_observe)

    def episode(key_env, key_obs, key_action):
        # reset the environment
        key_env, key_init = jax.random.split(key_env, 2)
        init_true_state_mean, _, _ = env_reset(key=key_env)
        init_true_state = env_draw_initial_condition(
            u0=init_true_state_mean, key=key_init
        )
        init_true_obs = init_true_state[env.observation_inds]

        init_reward = jnp.nan

        if episode_type == "null_action":
            (
                true_state,
                true_obs,
                key_obs,
                true_state_arr,
                true_obs_arr,
                obs_arr,
                action_arr,
                reward_arr,
            ) = null_action_observe(
                true_state=init_true_state,
                true_obs=init_true_obs,
                key_obs=key_obs,
            )
        elif episode_type == "random_action":
            # add noise on the observation
            (
                true_state,
                true_obs,
                key_obs,
                key_action,
                true_state_arr,
                true_obs_arr,
                obs_arr,
                action_arr,
                reward_arr,
            ) = random_action_observe(
                true_state=init_true_state,
                true_obs=init_true_obs,
                key_obs=key_obs,
                key_action=key_action,
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
        action_arr = jnp.reshape(
            action_arr,
            (action_arr.shape[0] * action_arr.shape[1], action_arr.shape[2]),
        )

        reward_arr = jnp.reshape(
            reward_arr,
            (reward_arr.shape[0] * reward_arr.shape[1],),
        )
        stack = lambda a, b: jnp.vstack((jnp.expand_dims(a, axis=0), b))
        hstack = lambda a, b: jnp.hstack((jnp.expand_dims(a, axis=0), b))

        return (
            stack(init_true_state, true_state_arr),
            stack(init_true_obs, true_obs_arr),
            obs_arr,
            stack(null_action, action_arr),
            hstack(init_reward, reward_arr),
            key_env,
            key_obs,
            key_action,
        )

    return episode


def train_ESN(config, env, key):
    key, key_env, key_obs, key_action = jax.random.split(key, 4)
    esn_config = config.esn
    episode = generate_training_episode(
        config, env, episode_type=esn_config.episode_type
    )

    # Define batched random keys for parallel processing
    key_env, subkey_env = jax.random.split(key_env)
    key_obs, subkey_obs = jax.random.split(key_obs)
    key_action, subkey_action = jax.random.split(key_action)

    # Create batched keys for all episodes
    total_episodes = (
        esn_config.train_episodes + esn_config.val_episodes + esn_config.test_episodes
    )
    batch_keys_env = jax.random.split(subkey_env, total_episodes)
    batch_keys_obs = jax.random.split(subkey_obs, total_episodes)
    batch_keys_action = jax.random.split(subkey_action, total_episodes)

    # Use vmap to process all episodes in parallel
    batched_results = jax.vmap(episode)(
        batch_keys_env, batch_keys_obs, batch_keys_action
    )

    print("Creating training dataset.", flush=True)
    # Unpack results
    (
        true_state_arrs,
        true_obs_arrs,
        obs_arrs,
        action_arrs,
        _,  # ignore rewards
        _,
        _,
        _,  # Keys can be ignored if not needed further
    ) = batched_results
    RAW_DATA = {
        "true_state": [],
        "true_observation": [],
        "observation": [],
        "action": [],
        "forcing": [],
    }
    # can include POD time coefficients

    RAW_DATA["true_state"] = true_state_arrs
    RAW_DATA["true_observation"] = true_obs_arrs
    RAW_DATA["observation"] = obs_arrs
    RAW_DATA["action"] = action_arrs
    RAW_DATA["forcing"] = jax.vmap(lambda x: (env.ks_solver.B @ x.T).T)(
        RAW_DATA["action"]
    )

    train_idxs = np.arange(esn_config.train_episodes)
    val_idxs = np.arange(
        esn_config.train_episodes, esn_config.train_episodes + esn_config.val_episodes
    )
    test_idxs = np.arange(
        esn_config.train_episodes + esn_config.val_episodes,
        esn_config.train_episodes + esn_config.val_episodes + esn_config.test_episodes,
    )
    idxs_list = np.concatenate((train_idxs, val_idxs, test_idxs), axis=None)

    total_time = env.dt * config.episode_steps
    train_time = total_time - esn_config.model.washout_time
    network_dt = esn_config.model.network_dt
    t = env.dt * jnp.arange(config.episode_steps + 1)

    loop_times = [train_time]
    DATA = {
        "u_washout": [],
        "p_washout": [],
        "u": [],
        "p": [],
        "y": [],
        "full_state": [],
        "t": [],
    }

    for i in idxs_list:
        y = RAW_DATA[esn_config.model.which_state][i]
        a = RAW_DATA[esn_config.model.which_control][i][1:]

        full_state = RAW_DATA["true_state"][i]

        episode_data = pp.create_dataset(
            full_state,
            y,
            t,
            a,
            network_dt,
            transient_time=0,
            washout_time=esn_config.model.washout_time,
            loop_times=loop_times,
        )
        [
            DATA[var].append(np.asarray(episode_data["loop_0"][var]))
            for var in DATA.keys()
        ]
        # convert to numpy here because validation of ESN is in numpy

    # dimension of the inputs
    dim = DATA["u"][0].shape[1]
    action_dim = DATA["p"][0].shape[1]

    print("Dimension", dim)
    print("Creating hyperparameter search range.", flush=True)

    hyp_param_names = [name for name in esn_config.val.hyperparameters.keys()]

    # scale for the hyperparameter range
    hyp_param_scales = [
        esn_config.val.hyperparameters[name].scale for name in hyp_param_names
    ]

    # range for hyperparameters
    grid_range = [
        [
            esn_config.val.hyperparameters[name].min,
            esn_config.val.hyperparameters[name].max,
        ]
        for name in hyp_param_names
    ]

    # scale the ranges
    for i in range(len(grid_range)):
        for j in range(2):
            scaler = getattr(scalers, hyp_param_scales[i])
            grid_range[i][j] = scaler(grid_range[i][j])

    # create base ESN
    ESN_dict = {
        "dimension": dim,
        "reservoir_size": esn_config.model.reservoir_size,
        "parameter_dimension": action_dim,
        "reservoir_connectivity": esn_config.model.connectivity,
        "r2_mode": esn_config.model.r2_mode,
        "input_weights_mode": esn_config.model.input_weights_mode,
        "reservoir_weights_mode": esn_config.model.reservoir_weights_mode,
        "tikhonov": esn_config.tikhonov,
    }
    if esn_config.model.normalize_input:
        data_mean = np.mean(np.vstack(DATA["u"]), axis=0)
        data_std = np.std(np.vstack(DATA["u"]), axis=0)
        ESN_dict["input_normalization"] = [data_mean, data_std]
        ESN_dict["output_bias"] = np.array(
            [1.0]
        )  # if subtracting the mean, need the output bias

    N_washout = pp.get_steps(esn_config.model.washout_time, esn_config.model.network_dt)
    N_val = pp.get_steps(esn_config.val.fold_time, esn_config.model.network_dt)
    N_transient = 0
    error_measure = getattr(errors, esn_config.val.error_measure)
    print("Starting validation.", flush=True)
    min_dict = validate(
        grid_range,
        hyp_param_names,
        hyp_param_scales,
        n_calls=esn_config.val.n_calls,
        n_initial_points=esn_config.val.n_initial_points,
        ESN_dict=ESN_dict,
        U_washout_train=DATA["u_washout"],
        U_train=DATA["u"],
        U_val=DATA["u"],
        Y_train=DATA["y"],
        Y_val=DATA["y"],
        P_washout_train=DATA["p_washout"],
        P_train=DATA["p"],
        P_val=DATA["p"],
        n_folds=esn_config.val.n_folds,
        n_realisations=esn_config.val.n_realisations,
        N_washout_steps=N_washout,
        N_val_steps=N_val,
        N_transient_steps=N_transient,
        train_idx_list=train_idxs,
        val_idx_list=val_idxs,
        random_seed=esn_config.seed,
        error_measure=error_measure,
        network_dt=esn_config.model.network_dt,
    )

    print("Train JAX ESN with the same hyperparameters.", flush=True)
    hyp_params = []
    for name in hyp_param_names:
        hyp_params.append(min_dict[name][0])

    # hyp_param_scales_ = ["uniform"] * len(hyp_param_names)
    # after validation, hyperparameters are saved with uniform scaling

    # fix the seeds
    # my_ESN = ESN(
    #     input_seeds=[config.val.seed + 1, config.val.seed + 2, config.val.seed + 3],
    #     reservoir_seeds=[config.val.seed + 4, config.val.seed + 5],
    #     **ESN_dict,
    # )
    # set_ESN(my_ESN, hyp_param_names, hyp_param_scales_, hyp_params)
    # my_ESN.train(
    #     DATA["u_washout"],
    #     DATA["u"],
    #     DATA["y"],
    #     P_washout=DATA["p_washout"],
    #     P_train=DATA["p"],
    #     train_idx_list=train_idxs,
    # )
    # print("Testing on the test set.", flush=True)
    # for episode_idx in test_idxs:
    #     _, y_pred = my_ESN.closed_loop_with_washout(
    #         U_washout=DATA["u_washout"][episode_idx],
    #         P_washout=DATA["p_washout"][episode_idx],
    #         P=DATA["p"][episode_idx],
    #         N_t=len(DATA["u_washout"][episode_idx]),
    #     )
    #     y_pred = y_pred[1:]
    #     episode_error = error_measure(DATA["y"][episode_idx], y_pred)
    #     print(f"Error test episode {episode_idx}: {episode_error:.4f}")
    #     # plot prediction on test set and save it in the folder

    my_ESN = JESN(
        input_seed=esn_config.seed + 1,
        reservoir_seed=esn_config.seed + 2,
        verbose=False,
        **ESN_dict,
    )
    for hyp_param_name, hyp_param in zip(hyp_param_names, hyp_params):
        setattr(my_ESN, hyp_param_name, hyp_param)

    before_readout = (
        jesn.before_readout_r2 if my_ESN.r2_mode == True else jesn.before_readout_r1
    )
    # convert back to jax arrray?
    W_out = jesn.train(
        my_ESN,
        DATA["u_washout"],
        DATA["u"],
        DATA["y"],
        DATA["p_washout"],
        DATA["p"],
        train_idx_list=train_idxs,
        before_readout=before_readout,
    )
    my_ESN.output_weights = W_out

    print("Testing on the test set.", flush=True)
    for episode_idx in test_idxs:
        _, y_pred = jesn.closed_loop_with_washout(
            my_ESN,
            U_washout=DATA["u_washout"][episode_idx],
            P_washout=DATA["p_washout"][episode_idx],
            P=DATA["p"][episode_idx],
            N_t=len(DATA["u"][episode_idx]),
            before_readout=before_readout,
        )
        y_pred = y_pred[1:]

        # plt_idxs = [int(my_idx) for my_idx in np.linspace(0,my_ESN.N_dim-1,5)]
        # plt.figure(figsize=(5*len(plt_idxs),5))
        # for k, plt_idx in enumerate(plt_idxs):
        #     plt.subplot(1,len(plt_idxs),k+1)
        #     plt.plot(DATA['t'][episode_idx],DATA['y'][episode_idx][:,plt_idx])
        #     plt.plot(DATA['t'][episode_idx],y_pred[:,plt_idx],'--')
        #     # plt.xlim([0,500])
        # plt.figure(figsize=(20,5))
        # plt.subplot(2,1,1)
        # plt.imshow(DATA['y'][episode_idx].T, aspect='auto')
        # plt.subplot(2,1,2)
        # plt.imshow(y_pred.T, aspect='auto')
        # plt.show()
    return my_ESN


def run_experiment(
    config, env, agent, model, key, wandb_run=None, logs=None, checkpoint_dir=None
):
    # random seed for initialization
    key, key_network, key_buffer, key_env, key_obs, key_action = jax.random.split(
        key, 6
    )

    # initialize networks
    # sample state and action to get the correct shape
    state_0 = jnp.array([jnp.zeros(model.N_dim)])
    action_0 = jnp.array([jnp.zeros(env.action_size)])
    actor_state, critic_state = agent.initial_network_state(
        key_network, state_0, action_0
    )

    # set up checkpointers
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=config.total_steps // 5
    )
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, options=options
    )

    best_model_dir = checkpoint_dir / "best_model"
    final_model_dir = checkpoint_dir / "final_model"
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    # checkpoint the initial weights
    if FLAGS.save_checkpoints == True:
        checkpoint = {"actor": actor_state, "critic": critic_state}
        save_args = orbax_utils.save_args_from_target(checkpoint)
        checkpoint_manager.save(0, checkpoint, save_kwargs={"save_args": save_args})

    # initialize buffer
    replay_buffer = init_replay_buffer(
        capacity=config.replay_buffer.capacity,
        state_dim=(model.N_dim,),
        action_dim=(env.action_size,),
        rng_key=key_buffer,
    )

    # get the observation matrix that maps state to observations
    obs_mat = get_observation_matrix(model, env.observation_inds)

    # standard deviation of the exploration scales with the range of actions in the environment
    exploration_stddev = (
        env.action_high - env.action_low
    ) * config.train.exploration_stddev

    # create a action of zeros to pass
    null_action = jnp.zeros(env.action_size)

    # jit the necessary environment functions
    N_washout = pp.get_steps(config.esn.model.washout_time, config.esn.model.network_dt)
    model_initialize_ensemble = partial(
        initialize_ensemble_with_auto_washout,
        my_ESN=model,
        N_washout=N_washout,
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

    before_readout = (
        jesn.before_readout_r2 if model.r2_mode == True else jesn.before_readout_r1
    )
    model_forecast = partial(
        forecast, frame_skip=env.frame_skip, my_ESN=model, before_readout=before_readout
    )
    model_forecast = jax.jit(model_forecast)

    before_EnKF = before_EnKF_r2 if model.r2_mode == True else jesn.before_EnKF_r1
    after_EnKF = after_EnKF_r2 if model.r2_mode == True else jesn.after_EnKF_r1
    model_apply_enKF = partial(
        apply_enKF,
        m=config.enKF.m,
        M=obs_mat,
        my_ESN=model,
        before_EnKF=before_EnKF,
        after_EnKF=after_EnKF,
        rho=config.enKF.inflation_factor,
    )
    model_apply_enKF = jax.jit(model_apply_enKF)

    model_target = env.target  # ONLY WORKS IF MODEL OUTPUT IS SAME AS ENV DIMENSION
    get_model_reward = partial(
        KSenv.get_reward,
        target=model_target,
        actuator_loss_weight=env.actuator_loss_weight,
    )
    get_model_reward = jax.jit(get_model_reward)

    model_ensemble_to_state = partial(
        ensemble_to_state, my_ESN=model, before_readout=before_readout
    )
    model_ensemble_to_state = jax.jit(model_ensemble_to_state)

    def until_first_observation(true_state, true_obs, state_ens, observation_starts):
        def body_fun(carry, _):
            true_state, true_obs, state_ens = carry
            # advance true environment
            action = null_action
            true_state, true_obs, reward_env, _, _, _ = env_step(
                state=true_state, action=action
            )
            # advance model
            state_ens = model_forecast(state_ens=state_ens, action=action)
            state = model_ensemble_to_state(state_ens)
            reward_model = get_model_reward(next_state=state, action=action)

            return (true_state, true_obs, state_ens), (
                true_state,
                true_obs,
                state_ens,
                action,
                reward_env,
                reward_model,
            )

        (true_state, true_obs, state_ens), (
            true_state_arr,
            true_obs_arr,
            state_ens_arr,
            action_arr,
            reward_env_arr,
            reward_model_arr,
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
            reward_env_arr,
            reward_model_arr,
        )

    def act_observe_and_forecast(
        true_state,
        true_obs,
        state_ens,
        params,
        wait_steps,
        episode_steps,
        key_obs,
    ):
        def forecast_fun(carry, _):
            true_state, true_obs, state_ens = carry
            state = model_ensemble_to_state(state_ens)

            # get action
            action = agent.actor.apply(params, state)

            # get the next observation and reward with this action
            true_state, true_obs, reward_env, _, _, _ = env_step(
                state=true_state, action=action
            )

            # forecast
            state_ens = model_forecast(state_ens=state_ens, action=action)
            state = model_ensemble_to_state(state_ens)
            reward_model = get_model_reward(next_state=state, action=action)

            return (true_state, true_obs, state_ens), (
                true_state,
                true_obs,
                state_ens,
                action,
                reward_env,
                reward_model,
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
                reward_env_arr,
                reward_model_arr,
            ) = jax.lax.scan(
                forecast_fun, (true_state, true_obs, state_ens), jnp.arange(wait_steps)
            )
            return (true_state, true_obs, state_ens, key_obs), (
                true_state_arr,
                true_obs_arr,
                obs,
                state_ens_arr,
                action_arr,
                reward_env_arr,
                reward_model_arr,
            )

        n_loops = episode_steps // wait_steps
        (true_state, true_obs, state_ens, key_obs), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_env_arr,
            reward_model_arr,
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
            reward_env_arr,
            reward_model_arr,
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
        use_reward="env",
    ):
        def forecast_fun(carry, _):
            true_state, true_obs, state_ens, key_action, replay_buffer = carry
            state = model_ensemble_to_state(state_ens)

            # get action
            key_action, _ = jax.random.split(key_action)
            action = env_sample_action(key=key_action)

            # get the next observation and reward with this action
            next_true_state, next_true_obs, reward_env, terminated, _, _ = env_step(
                state=true_state, action=action
            )

            # forecast
            next_state_ens = model_forecast(state_ens=state_ens, action=action)
            next_state = model_ensemble_to_state(next_state_ens)
            reward_model = get_model_reward(next_state=next_state, action=action)

            # choose which reward
            if use_reward == "env":
                reward = reward_env
            elif use_reward == "model":
                reward = reward_model

            replay_buffer = add_experience(
                replay_buffer, state, action, reward, next_state, terminated
            )
            return (
                next_true_state,
                next_true_obs,
                next_state_ens,
                key_action,
                replay_buffer,
            ), (true_state, true_obs, state_ens, action, reward_env, reward_model)

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
                reward_env_arr,
                reward_model_arr,
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
                reward_env_arr,
                reward_model_arr,
            )

        n_loops = episode_steps // wait_steps
        (true_state, true_obs, state_ens, key_obs, key_action, replay_buffer), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_env_arr,
            reward_model_arr,
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
            reward_env_arr,
            reward_model_arr,
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
        use_reward="env",
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

            state = model_ensemble_to_state(state_ens)

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
            next_true_state, next_true_obs, reward_env, terminated, _, _ = env_step(
                state=true_state, action=action
            )

            # forecast
            next_state_ens = model_forecast(state_ens=state_ens, action=action)
            next_state = model_ensemble_to_state(next_state_ens)
            reward_model = get_model_reward(next_state=state, action=action)

            # choose which reward
            if use_reward == "env":
                reward = reward_env
            elif use_reward == "model":
                reward = reward_model

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
            ), (
                true_state,
                true_obs,
                state_ens,
                action,
                reward_env,
                reward_model,
                q_loss,
                policy_loss,
            )

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
                reward_env_arr,
                reward_model_arr,
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
                reward_env_arr,
                reward_model_arr,
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
            reward_env_arr,
            reward_model_arr,
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
            reward_env_arr,
            reward_model_arr,
            q_loss_arr,
            policy_loss_arr,
            replay_buffer,
            actor_state,
            critic_state,
        )

    until_first_observation = partial(
        until_first_observation,
        observation_starts=config.enKF.observation_starts,
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
        use_reward=config.enKF.use_reward,
    )
    random_observe_and_forecast = jax.jit(random_observe_and_forecast)

    learn_observe_and_forecast = partial(
        learn_observe_and_forecast,
        wait_steps=config.enKF.wait_steps,
        episode_steps=config.episode_steps - config.enKF.observation_starts,
        use_reward=config.enKF.use_reward,
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
        init_reward = jnp.nan

        # initialize enKF
        init_state_ens = model_initialize_ensemble(
            u0=init_true_state_mean, p0=null_action, key=key_ens
        )

        # forecast until first observation
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr0,
            true_obs_arr0,
            state_ens_arr0,
            action_arr0,
            reward_env_arr0,
            reward_model_arr0,
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
            reward_env_arr,
            reward_model_arr,
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
        reward_env_arr = jnp.reshape(
            reward_env_arr,
            (reward_env_arr.shape[0] * reward_env_arr.shape[1],),
        )
        reward_model_arr = jnp.reshape(
            reward_model_arr,
            (reward_model_arr.shape[0] * reward_model_arr.shape[1],),
        )
        stack = lambda a, b, c: jnp.vstack((jnp.expand_dims(a, axis=0), b, c))
        hstack = lambda a, b, c: jnp.hstack((jnp.expand_dims(a, axis=0), b, c))

        return (
            stack(init_true_state, true_state_arr0, true_state_arr),
            stack(init_true_obs, true_obs_arr0, true_obs_arr),
            obs_arr,
            stack(init_state_ens, state_ens_arr0, state_ens_arr),
            stack(null_action, action_arr0, action_arr),
            hstack(init_reward, reward_env_arr0, reward_env_arr),
            hstack(init_reward, reward_model_arr0, reward_model_arr),
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

        init_reward = jnp.nan

        # initialize enKF
        init_state_ens = model_initialize_ensemble(
            u0=init_true_state_mean, p0=null_action, key=key_ens
        )

        # forecast until first observation
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr0,
            true_obs_arr0,
            state_ens_arr0,
            action_arr0,
            reward_env_arr0,
            reward_model_arr0,
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
            reward_env_arr,
            reward_model_arr,
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
        reward_env_arr = jnp.reshape(
            reward_env_arr,
            (reward_env_arr.shape[0] * reward_env_arr.shape[1],),
        )

        reward_model_arr = jnp.reshape(
            reward_model_arr,
            (reward_model_arr.shape[0] * reward_model_arr.shape[1],),
        )
        stack = lambda a, b, c: jnp.vstack((jnp.expand_dims(a, axis=0), b, c))
        hstack = lambda a, b, c: jnp.hstack((jnp.expand_dims(a, axis=0), b, c))

        return (
            stack(init_true_state, true_state_arr0, true_state_arr),
            stack(init_true_obs, true_obs_arr0, true_obs_arr),
            obs_arr,
            stack(init_state_ens, state_ens_arr0, state_ens_arr),
            stack(null_action, action_arr0, action_arr),
            hstack(init_reward, reward_env_arr0, reward_env_arr),
            hstack(init_reward, reward_model_arr0, reward_model_arr),
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

        init_reward = jnp.nan

        # initialize enKF
        init_state_ens = model_initialize_ensemble(
            u0=init_true_state_mean, p0=null_action, key=key_ens
        )

        # forecast until first observation
        (
            true_state,
            true_obs,
            state_ens,
            true_state_arr0,
            true_obs_arr0,
            state_ens_arr0,
            action_arr0,
            reward_env_arr0,
            reward_model_arr0,
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
            reward_env_arr,
            reward_model_arr,
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
        reward_env_arr = jnp.reshape(
            reward_env_arr,
            (reward_env_arr.shape[0] * reward_env_arr.shape[1],),
        )
        reward_model_arr = jnp.reshape(
            reward_model_arr,
            (reward_model_arr.shape[0] * reward_model_arr.shape[1],),
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
        hstack = lambda a, b, c: jnp.hstack((jnp.expand_dims(a, axis=0), b, c))

        return (
            stack(init_true_state, true_state_arr0, true_state_arr),
            stack(init_true_obs, true_obs_arr0, true_obs_arr),
            obs_arr,
            stack(init_state_ens, state_ens_arr0, state_ens_arr),
            stack(null_action, action_arr0, action_arr),
            hstack(init_reward, reward_env_arr0, reward_env_arr),
            hstack(init_reward, reward_model_arr0, reward_model_arr),
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
            reward_env_arr,
            reward_model_arr,
            replay_buffer,
            key_env,
            key_obs,
            key_action,
        ) = random_episode(key_env, key_obs, key_action, replay_buffer)
        random_return_env = jnp.sum(
            reward_env_arr[config.enKF.observation_starts + 1 :]
        )
        random_last_reward_env = reward_env_arr[-1]

        random_return_model = jnp.sum(
            reward_model_arr[config.enKF.observation_starts + 1 :]
        )
        random_last_reward_model = reward_model_arr[-1]

        print(
            f"Random input, Episode={i+1}/{random_episodes}, (ENV) Return = {random_return_env}, (MODEL) Return = {random_return_model},(ENV) Last Reward ={random_last_reward_env}, (MODEL) Last Reward ={random_last_reward_model}",
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
                    before_readout,
                    action_arr,
                    reward_env_arr,
                    reward_model_arr,
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
                    "reward_env": reward_env_arr,
                    "reward_model": reward_model_arr,
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
    max_eval_return = -jnp.inf
    for i in range(learn_episodes):
        (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_env_arr,
            reward_model_arr,
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
        train_return_env = jnp.sum(reward_env_arr[config.enKF.observation_starts + 1 :])
        train_ave_reward_env = jnp.mean(
            reward_env_arr[config.enKF.observation_starts :]
        )
        train_last_reward_env = reward_env_arr[-1]

        train_return_model = jnp.sum(
            reward_model_arr[config.enKF.observation_starts + 1 :]
        )
        train_ave_reward_model = jnp.mean(
            reward_model_arr[config.enKF.observation_starts :]
        )
        train_last_reward_model = reward_model_arr[-1]

        print(
            f"Training, Episode={i+1}/{learn_episodes}, (ENV) Return = {train_return_env}, (MODEL) Return = {train_return_model}, (ENV) Last Reward ={train_last_reward_env}, (MODEL) Last Reward ={train_last_reward_model}",
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
                    before_readout,
                    action_arr,
                    reward_env_arr,
                    reward_model_arr,
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
                    "reward_env": reward_env_arr,
                    "reward_model": reward_model_arr,
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

        metrics["train"]["episode_return_env"] = train_return_env
        metrics["train"]["episode_average_reward_env"] = train_ave_reward_env
        metrics["train"]["episode_last_reward_env"] = train_last_reward_env

        metrics["train"]["episode_return_model"] = train_return_model
        metrics["train"]["episode_average_reward_model"] = train_ave_reward_model
        metrics["train"]["episode_last_reward_model"] = train_last_reward_model

        if (i + 1) % n_eval == 0:
            eval_ave_return_env = 0
            eval_ave_reward_env = 0
            eval_ave_last_reward_env = 0

            eval_ave_return_model = 0
            eval_ave_reward_model = 0
            eval_ave_last_reward_model = 0

            for j in range(config.eval_episodes):
                (
                    true_state_arr,
                    true_obs_arr,
                    obs_arr,
                    state_ens_arr,
                    action_arr,
                    reward_env_arr,
                    reward_model_arr,
                    key_env,
                    key_obs,
                ) = act_episode(key_env, key_obs, actor_state.params)

                eval_ave_return_env += jnp.sum(
                    reward_env_arr[config.enKF.observation_starts + 1 :]
                )
                eval_ave_reward_env += jnp.mean(
                    reward_env_arr[config.enKF.observation_starts + 1 :]
                )
                eval_ave_last_reward_env += reward_env_arr[-1]

                eval_ave_return_model += jnp.sum(
                    reward_model_arr[config.enKF.observation_starts + 1 :]
                )
                eval_ave_reward_model += jnp.mean(
                    reward_model_arr[config.enKF.observation_starts + 1 :]
                )
                eval_ave_last_reward_model += reward_model_arr[-1]

                if (i + 1) % n_plot == 0 and j == 0:
                    if FLAGS.make_plots == True:
                        fig = plot_KS_episode(
                            env,
                            model,
                            true_state_arr,
                            true_obs_arr,
                            obs_arr,
                            state_ens_arr,
                            before_readout,
                            action_arr,
                            reward_env_arr,
                            reward_model_arr,
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
                            "reward_env": reward_env_arr,
                            "reward_model": reward_model_arr,
                        }
                        fp.write_h5(
                            FLAGS.experiment_path
                            / "episode_data"
                            / f"eval_episode_{i+1}.h5",
                            episode_dict,
                        )
            eval_ave_return_env = eval_ave_return_env / config.eval_episodes
            eval_ave_reward_env = eval_ave_reward_env / config.eval_episodes
            eval_ave_last_reward_env = eval_ave_last_reward_env / config.eval_episodes

            eval_ave_return_model = eval_ave_return_model / config.eval_episodes
            eval_ave_reward_model = eval_ave_reward_model / config.eval_episodes
            eval_ave_last_reward_model = (
                eval_ave_last_reward_model / config.eval_episodes
            )

            print(
                f"\n Evaluation, Episode={i+1}/{learn_episodes}, (ENV) Return = {eval_ave_return_env}, (MODEL) Return = {eval_ave_return_model}, (ENV) Last Reward ={eval_ave_last_reward_env}, (MODEL) Last Reward ={eval_ave_last_reward_model}  \n ",
                flush=True,
            )

            prev_max_eval_return = max_eval_return
            max_eval_return = max(max_eval_return, eval_ave_return_model)
            if max_eval_return > prev_max_eval_return:
                best_checkpoint = {"actor": actor_state, "critic": critic_state}
                save_args = orbax_utils.save_args_from_target(best_checkpoint)
                checkpointer.save(
                    best_model_dir, best_checkpoint, save_args=save_args, force=True
                )

            metrics["eval"]["average_return_env"] = eval_ave_return_env
            metrics["eval"]["average_reward_env"] = eval_ave_reward_env
            metrics["eval"]["average_last_reward_env"] = eval_ave_last_reward_env

            metrics["eval"]["average_return_model"] = eval_ave_return_model
            metrics["eval"]["average_reward_model"] = eval_ave_reward_model
            metrics["eval"]["average_last_reward_model"] = eval_ave_last_reward_model

        if (i + 1) == learn_episodes:
            (
                true_state_arr,
                true_obs_arr,
                obs_arr,
                state_ens_arr,
                action_arr,
                reward_env_arr,
                reward_model_arr,
                key_env,
                key_obs,
            ) = act_episode(key_env, key_obs, actor_state.params)
            final_eval_return_env = jnp.sum(
                reward_env_arr[config.enKF.observation_starts + 1 :]
            )
            final_eval_ave_reward_env = jnp.mean(
                reward_env_arr[config.enKF.observation_starts + 1 :]
            )
            final_eval_last_reward_env = reward_env_arr[-1]

            final_eval_return_model = jnp.sum(
                reward_model_arr[config.enKF.observation_starts + 1 :]
            )
            final_eval_ave_reward_model = jnp.mean(
                reward_model_arr[config.enKF.observation_starts + 1 :]
            )
            final_eval_last_reward_model = reward_model_arr[-1]
            if FLAGS.make_plots == True:
                fig = plot_KS_episode(
                    env,
                    model,
                    true_state_arr,
                    true_obs_arr,
                    obs_arr,
                    state_ens_arr,
                    before_readout,
                    action_arr,
                    reward_env_arr,
                    reward_model_arr,
                    config.enKF.wait_steps,
                    config.enKF.observation_starts,
                )
                fig.savefig(FLAGS.experiment_path / "plots" / f"final_eval_episode.png")
            if FLAGS.save_episode_data == True:
                episode_dict = {
                    "true_state": true_state_arr,
                    "true_obs": true_obs_arr,
                    "obs": obs_arr,
                    "state_ens": state_ens_arr,
                    "action": action_arr,
                    "reward_env": reward_env_arr,
                    "reward_model": reward_model_arr,
                }
                fp.write_h5(
                    FLAGS.experiment_path / "episode_data" / f"final_eval_episode.h5",
                    episode_dict,
                )
            print(
                f"\n Final evaluation, Episode={i+1}/{learn_episodes}, (ENV) Return = {final_eval_return_env}, (MODEL) Return {final_eval_return_model}, (ENV) Last Reward = {final_eval_last_reward_env}, (MODEL) Last Reward = {final_eval_last_reward_model}",
                flush=True,
            )

        if wandb_run is not None:
            log_metrics_wandb(wandb_run, metrics, step=(i + 1) * learn_steps - 1)
        # checkpoint the model
        if FLAGS.save_checkpoints == True:
            checkpoint = {"actor": actor_state, "critic": critic_state}
            save_args = orbax_utils.save_args_from_target(checkpoint)
            checkpoint_manager.save(
                (i + 1) * learn_steps, checkpoint, save_kwargs={"save_args": save_args}
            )
    final_checkpoint = {"actor": actor_state, "critic": critic_state}
    save_args = orbax_utils.save_args_from_target(final_checkpoint)
    checkpointer.save(final_model_dir, final_checkpoint, save_args=save_args)
    return (
        actor_state,
        critic_state,
    )


def main(_):
    config = FLAGS.config  # Access experiment config
    wandb_config = FLAGS.wandb_config  # Access wandb config
    env_config = FLAGS.env_config  # Access environment config

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

    # add esn config to the config
    config.esn = FLAGS.esn_config

    fp.save_config(FLAGS.experiment_path, config)

    # initialize wandb logging
    config.experiment = "ddpg_with_enkf_esn"
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
    checkpoint_dir = FLAGS.experiment_path / "models"

    # create environment
    if config.env_name == "KS":
        env = KSenv(**config.env)

    # create agent and run
    agent = DDPG(config, env)
    print("Starting experiment.", flush=True)

    # generate keys for running experiments
    key = jax.random.PRNGKey(config.seed)
    key_ESN, key_experiment = jax.random.split(key)

    # validate and train an ESN
    print("Training Echo State Network.", flush=True)
    model = train_ESN(config, env, key_ESN)

    # determine the bias term of the ESN for observations
    model.observation_bias = jnp.multiply(
        model.output_bias, model.W_out[-len(model.output_bias)]
    )[env.observation_inds]

    # running experiment with the trained model
    actor_state, critic_state = run_experiment(
        config, env, agent, model, key_experiment, wandb_run, logs, checkpoint_dir
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
