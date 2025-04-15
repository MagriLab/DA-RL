from functools import partial

import jax
import jax.numpy as jnp

import utils.file_processing as fp
from envs.KS_environment_jax import KSenv
from utils import covariance_matrix as cov
from utils import preprocessing as pp

import numpy as np
from esn.utils import scalers
import jax_esn.esn as jesn
from jax_esn.esn import ESN as JESN

def initialize_ensemble_with_auto_washout(my_ESN, N_washout, u0, p0, std_init, m, key):
    key, subkey = jax.random.split(key)

    # Sample ensemble of u0
    # u0_sampled = jax.random.multivariate_normal(
    #     subkey,
    #     u0,
    #     jnp.diag((u0 * std_init) ** 2),
    #     shape=(m,),
    #     method="svd",
    # )

    u0_f = jnp.fft.rfft(u0, axis=-1)
    key, subkey = jax.random.split(key)
    u0_real = jax.random.multivariate_normal(
        key,
        u0_f.real,
        jnp.diag((u0_f.real * std_init) ** 2),
        (m,),
        method="svd",
    )
    # covariance matrix is rank deficient because zeroth
    u0_complex = jax.random.multivariate_normal(
        subkey,
        u0_f.imag,
        jnp.diag((u0_f.imag * std_init) ** 2),
        (m,),
        method="svd",
    )
    u0_f_sampled = u0_real + u0_complex * 1j
    u0_sampled = jax.vmap(lambda x: jnp.fft.irfft(x))(u0_f_sampled)

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


def get_observation_matrix(my_ESN, observation_indices):
    """
    Create an observation matrix for the given ESN based on observation indices.
    """
    # Initialize a zero matrix with appropriate dimensions
    M = jnp.zeros((len(observation_indices), my_ESN.output_weights.T.shape[0]))

    # Set the appropriate entries to 1 based on observation_indices
    M = M.at[jnp.arange(len(observation_indices)), observation_indices].set(1)

    # Multiply with the transposed output weight matrix and return
    # return M @ my_ESN.output_weights[: -len(my_ESN.output_bias)].T
    return M @ my_ESN.output_weights.T


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

def apply_enKF(m, Af, d, Cdd, M, my_ESN, before_readout, key, rho=1.0):
    # NONLINEAR METHOD
    # get the reservoir state before readout
    # before the readout because we need our observation matrix to be linear
    # if we're using r2 mode then the EnKF is applied on the r2 state
    # without the output bias because we apply EnKF on the r/r2 state and bias is constant
    # Af_full = before_EnKF(Af)

    # # remove the bias from the data, if using this M should exclude the output bias!!
    # Aa_full = EnKF(m, Af_full, d - my_ESN.observation_bias, Cdd, M, key)

    # Aa = after_EnKF(Aa_full, Af)

    # Af2 = before_EnKF(Af)

    # STATE AUGMENTATION METHOD
    # Vectorize the before_readout function for the ensemble
    before_readout_ensemble = jax.vmap(before_readout, in_axes=(1, None), out_axes=1)

    # Apply the vectorized before_readout function to the ensemble Af
    Af2 = before_readout_ensemble(Af, my_ESN.output_bias)
    M_Af2 = M @ Af2

    # concatenate with observations to avoid nonlinear observation operator
    # the end result is equivalent to 
    # psi_a = psi_f + [C_psi_m(psi); C_m(psi)_m(psi)] @ (C_dd +  C_m(psi)_m(psi))^-1
    Af_full = jnp.vstack([Af, M_Af2])
    M_new = jnp.hstack([jnp.zeros((M.shape[0],Af.shape[0])), jnp.eye(M.shape[0])])
    # don't need to remove the bias from the data because it's part of the mean
    # it doesn't affect the covariance of M(Af)
    Aa_full = EnKF(m, Af_full, d, Cdd, M_new, key)
    Aa = Aa_full[: my_ESN.N_reservoir, :]

    # INFLATION
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

    if config.enKF.cov_type == "const":
        get_cov = partial(
            cov.get_const, std=NOISE_DICT[f"{env.nu}"] * config.enKF.std_obs
        )
    elif config.enKF.cov_type == "max":
        get_cov = partial(cov.get_max, std=config.enKF.std_obs)
    elif config.enKF.cov_type == "prop":
        get_cov = partial(cov.get_prop, std=config.enKF.std_obs)

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

            obs_cov = get_cov(y=true_obs)

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

            obs_cov = get_cov(y=true_obs)

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

def train_ESN(config, env, key, esn_hyp_file_name):
    key, key_env, key_obs, key_action = jax.random.split(key, 4)
    episode = generate_training_episode(
        config, env, episode_type=config.esn.episode_type
    )

    # Define batched random keys for parallel processing
    key_env, subkey_env = jax.random.split(key_env)
    key_obs, subkey_obs = jax.random.split(key_obs)
    key_action, subkey_action = jax.random.split(key_action)

    # Create batched keys for all episodes
    total_episodes = (
        config.esn.train_episodes + config.esn.val_episodes + config.esn.test_episodes
    )
    batch_keys_env = jax.random.split(subkey_env, total_episodes)
    batch_keys_obs = jax.random.split(subkey_obs, total_episodes)
    batch_keys_action = jax.random.split(subkey_action, total_episodes)

    # Use vmap to process all episodes in parallel
    batched_results = jax.vmap(episode)(
        batch_keys_env, batch_keys_obs, batch_keys_action
    )

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

    train_idxs = jnp.arange(config.esn.train_episodes)
    val_idxs = jnp.arange(
        config.esn.train_episodes, config.esn.train_episodes + config.esn.val_episodes
    )
    test_idxs = jnp.arange(
        config.esn.train_episodes + config.esn.val_episodes,
        config.esn.train_episodes + config.esn.val_episodes + config.esn.test_episodes,
    )
    idxs_list = jnp.concatenate((train_idxs, val_idxs, test_idxs), axis=None)

    total_time = env.dt * config.episode_steps
    train_time = total_time - config.esn.model.washout_time
    network_dt = config.esn.model.network_dt
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
        y = RAW_DATA[config.esn.model.which_state][i]
        a = RAW_DATA[config.esn.model.which_control][i][1:]

        full_state = RAW_DATA["true_state"][i]

        episode_data = pp.create_dataset(
            full_state,
            y,
            t,
            a,
            network_dt,
            transient_time=0,
            washout_time=config.esn.model.washout_time,
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

    hyp_param_names = [name for name in config.esn.val.hyperparameters.keys()]

    # scale for the hyperparameter range
    hyp_param_scales = [
        config.esn.val.hyperparameters[name].scale for name in hyp_param_names
    ]

    # range for hyperparameters
    grid_range = [
        [
            config.esn.val.hyperparameters[name].min,
            config.esn.val.hyperparameters[name].max,
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
        "reservoir_size": config.esn.model.reservoir_size,
        "parameter_dimension": action_dim,
        "reservoir_connectivity": config.esn.model.connectivity,
        "r2_mode": config.esn.model.r2_mode,
        "input_weights_mode": config.esn.model.input_weights_mode,
        "reservoir_weights_mode": config.esn.model.reservoir_weights_mode,
        "tikhonov": config.esn.tikhonov,
    }
    if config.esn.model.normalize_input:
        data_mean = jnp.mean(np.vstack(DATA["u"]), axis=0)
        data_std = jnp.std(np.vstack(DATA["u"]), axis=0)
        ESN_dict["input_normalization"] = [data_mean, data_std]
        ESN_dict["output_bias"] = np.array(
            [1.0]
        )  # if subtracting the mean, need the output bias
    if config.esn.model.input_bias:
        ESN_dict["input_bias"] = jnp.array([1.0])
        
    min_dict = fp.read_h5(esn_hyp_file_name)

    hyp_params = []
    for name in hyp_param_names:
        hyp_params.append(min_dict[name][0])

    my_ESN = JESN(
        input_seed=config.esn.seed + 1,
        reservoir_seed=config.esn.seed + 2,
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
    return my_ESN

NOISE_DICT = {"0.08": 1.15, "0.05": 1.29, "0.03": 1.33}

def generate_DA_RL_episode(config, env, model, agent, key_experiment):
    # random seed for initialization
    key, key_network, key_buffer, key_env, key_obs, key_action = jax.random.split(
        key_experiment, 6
    )

    # random seed for ESN evaluation
    key, key_ESN = jax.random.split(key)

    # initialize networks
    # sample state and action to get the correct shape
    state_0 = jnp.array([jnp.zeros(model.N_dim)])
    action_0 = jnp.array([jnp.zeros(env.action_size)])
    actor_state, critic_state = agent.initial_network_state(
        key_network, state_0, action_0
    )

    # get the observation matrix that maps state to observations
    obs_mat = get_observation_matrix(model, env.observation_inds)

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

    # before_EnKF = before_EnKF_r2 if model.r2_mode == True else jesn.before_EnKF_r1
    # after_EnKF = after_EnKF_r2 if model.r2_mode == True else jesn.after_EnKF_r1
    model_apply_enKF = partial(
        apply_enKF,
        m=config.enKF.m,
        M=obs_mat,
        my_ESN=model,
        before_readout=before_readout,
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

    if config.enKF.cov_type == "const":
        get_cov = partial(
            cov.get_const, std=NOISE_DICT[f"{env.nu}"] * config.enKF.std_obs
        )
    elif config.enKF.cov_type == "max":
        get_cov = partial(cov.get_max, std=config.enKF.std_obs)
    elif config.enKF.cov_type == "prop":
        get_cov = partial(cov.get_prop, std=config.enKF.std_obs)

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
            obs_cov = get_cov(y=true_obs)

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
    ):
        def forecast_fun(carry, _):
            true_state, true_obs, state_ens, key_action = carry
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

            return (
                next_true_state,
                next_true_obs,
                next_state_ens,
                key_action,
            ), (true_state, true_obs, state_ens, action, reward_env, reward_model)

        def body_fun(carry, _):
            # observe
            # we got an observation
            # define observation covariance matrix
            true_state, true_obs, state_ens, key_obs, key_action = carry
            obs_cov = get_cov(y=true_obs)

            # add noise on the observation
            key_obs, key_enKF = jax.random.split(key_obs)
            obs = jax.random.multivariate_normal(
                key_obs, true_obs, obs_cov, method="svd"
            )

            # apply enkf to correct the state estimation
            state_ens = model_apply_enKF(Af=state_ens, d=obs, Cdd=obs_cov, key=key_enKF)
            (true_state, true_obs, state_ens, key_action), (
                true_state_arr,
                true_obs_arr,
                state_ens_arr,
                action_arr,
                reward_env_arr,
                reward_model_arr,
            ) = jax.lax.scan(
                forecast_fun,
                (true_state, true_obs, state_ens, key_action),
                jnp.arange(wait_steps),
            )
            return (
                true_state,
                true_obs,
                state_ens,
                key_obs,
                key_action,
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
        (true_state, true_obs, state_ens, key_obs, key_action), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            state_ens_arr,
            action_arr,
            reward_env_arr,
            reward_model_arr,
        ) = jax.lax.scan(
            body_fun,
            (true_state, true_obs, state_ens, key_obs, key_action),
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
    )
    random_observe_and_forecast = jax.jit(random_observe_and_forecast)

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

    def random_episode(key_env, key_obs, key_action):
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
        ) = random_observe_and_forecast(
            true_state=true_state,
            true_obs=true_obs,
            state_ens=state_ens,
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
            key_action,
        )
    return random_episode, act_episode

def predictability_horizon(y, y_pred, threshold=0.2):
    # Predictability horizon defined as in Racca & Magri, Neural Networks, 2021.
    nom = jnp.linalg.norm(y - y_pred, axis=1)
    denom = jnp.sqrt(
        jnp.cumsum(jnp.linalg.norm(y, axis=1) ** 2) / jnp.arange(1, len(y) + 1)
    )
    eps = nom / denom
    PH_list = jnp.where(eps > threshold)[0]
    if len(PH_list) == 0:
        print("Predictability horizon longer than given time series.")
        PH = len(y)
    else:
        PH = PH_list[0]
    return PH

def test_prediction(
    my_ESN,
    before_readout,
    U_washout,
    P_washout,
    U_test,
    P_test,
    Y_test,
    N_washout,
    N_val,
    n_folds,
    error_measure,
    key,
):
    fold_error = jnp.zeros(n_folds)
    for fold in range(n_folds):
        # select washout and validation
        # start_step = fold * (N_val-N_washout)
        key, _ = jax.random.split(key)
        start_step = jax.random.randint(
            key=key, shape=(1,), minval=0, maxval=len(U_test) - (N_washout + N_val)
        )[0]
        U_washout_fold = U_test[start_step : start_step + N_washout]
        Y_test_fold = Y_test[start_step + N_washout : start_step + N_washout + N_val]
        P_washout_fold = P_test[start_step : start_step + N_washout]
        P_test_fold = P_test[start_step + N_washout : start_step + N_washout + N_val]

        # predict output validation in closed-loop
        _, Y_test_pred = jesn.closed_loop_with_washout(
            my_ESN,
            U_washout=U_washout_fold,
            N_t=N_val,
            P_washout=P_washout_fold,
            P=P_test_fold,
            before_readout=before_readout,
        )
        Y_test_pred = Y_test_pred[1:, :]
        fold_error = fold_error.at[fold].set(error_measure(Y_test_fold, Y_test_pred))

    episode_test_error = jnp.mean(fold_error)

    # test on the whole episode
    _, y_pred = jesn.closed_loop_with_washout(
        my_ESN,
        U_washout=U_washout,
        P_washout=P_washout,
        P=P_test,
        N_t=len(U_test),
        before_readout=before_readout,
    )
    y_pred = y_pred[1:]
    # Determine the predictability horizon
    episode_PH = predictability_horizon(Y_test, y_pred)

    # Determine the error of entire episode
    episode_error = error_measure(Y_test, y_pred)
    return episode_test_error, episode_error, episode_PH, y_pred