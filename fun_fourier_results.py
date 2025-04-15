from functools import partial

import jax
import jax.numpy as jnp

from envs.KS_environment_jax import KSenv
from envs.KS_solver_jax import KS
from utils import covariance_matrix as cov

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


def inflate_ensemble(A, rho):
    A_m = jnp.mean(A, -1, keepdims=True)
    return A_m + rho * (A - A_m)


def apply_enKF(m, k, Af, d, Cdd, M, key, rho=1.0):
    Af_full = jnp.vstack((Af, jnp.conjugate(jnp.flip(Af[1:-1, :], axis=0))))
    Aa_full = EnKF(m, Af_full, d, Cdd, M, key)
    Aa = Aa_full[:k, :]

    # inflate analysed state ensemble
    # helps with the collapse of variance when using small ensemble
    Aa = inflate_ensemble(Aa, rho)
    return Aa

def generate_DA_RL_episode(config, env, model, agent, key_experiment):
       # random seed for initialization
    key, key_network, key_buffer, key_env, key_obs, key_action = jax.random.split(
        key_experiment, 6
    )

    # initialize networks
    # sample state and action to get the correct shape
    state_0 = jnp.array([jnp.zeros(model.N)])
    action_0 = jnp.array([jnp.zeros(env.action_size)])
    actor_state, critic_state = agent.initial_network_state(
        key_network, state_0, action_0
    )

    # get the observation matrix that maps state to observations
    obs_mat = get_observation_matrix(model.N, model.L, env.observation_locs)

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
        rho=config.enKF.inflation_factor,
    )
    model_apply_enKF = jax.jit(model_apply_enKF)

    model_target = KSenv.determine_target(
        target=config.env.target,
        N=model.N,
        action_size=env.action_size,
        B=model.B,
        lin=model.lin,
        ik=model.ik,
        dt=model.dt,
    )
    get_model_reward = partial(
        KSenv.get_reward,
        target=model_target,
        actuator_loss_weight=env.actuator_loss_weight,
    )
    get_model_reward = jax.jit(get_model_reward)

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
            state = ensemble_to_state(state_ens)
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
            state = ensemble_to_state(state_ens)

            # get action
            action = agent.actor.apply(params, state)

            # get the next observation and reward with this action
            true_state, true_obs, reward_env, _, _, _ = env_step(
                state=true_state, action=action
            )

            # forecast
            state_ens = model_forecast(state_ens=state_ens, action=action)
            state = ensemble_to_state(state_ens)
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
            state = ensemble_to_state(state_ens)

            # get action
            key_action, _ = jax.random.split(key_action)
            action = env_sample_action(key=key_action)

            # get the next observation and reward with this action
            next_true_state, next_true_obs, reward_env, terminated, _, _ = env_step(
                state=true_state, action=action
            )

            # forecast
            next_state_ens = model_forecast(state_ens=state_ens, action=action)
            next_state = ensemble_to_state(next_state_ens)
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
            key_obs
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
            key_action
        )

    return random_episode, act_episode