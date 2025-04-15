from functools import partial

import jax
import jax.numpy as jnp

from envs.KS_environment_jax import KSenv
from envs.KS_solver_jax import KS
from utils import covariance_matrix as cov

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

def generate_DA_RL_episode(config, env, agent, key_experiment):
    key, key_network, key_buffer, key_env, key_obs, key_action = jax.random.split(
        key_experiment, 6
    )
    # initialize networks
    # sample state and action to get the correct shape
    state_0 = jnp.array([jnp.zeros(env.num_observations)])
    action_0 = jnp.array([jnp.zeros(env.action_size)])
    actor_state, critic_state = agent.initial_network_state(
        key_network, state_0, action_0
    )

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

    if hasattr(config.enKF,'cov_type'):
        if config.enKF.cov_type == "const":
            get_cov = partial(
                cov.get_const, std=NOISE_DICT[f"{env.nu}"] * config.enKF.std_obs
            )
        elif config.enKF.cov_type == "max":
            get_cov = partial(cov.get_max, std=config.enKF.std_obs)
        elif config.enKF.cov_type == "prop":
            get_cov = partial(cov.get_prop, std=config.enKF.std_obs)
    else:
        get_cov = partial(cov.get_max, std=config.enKF.std_obs)

    def until_first_observation(true_state, true_obs, observation_starts):
        def body_fun(carry, _):
            true_state, true_obs = carry
            # advance true environment
            action = null_action
            true_state, true_obs, reward, _, _, _ = env_step(
                state=true_state, action=action
            )
            return (true_state, true_obs), (true_state, true_obs, action, reward)

        (true_state, true_obs), (
            true_state_arr,
            true_obs_arr,
            action_arr,
            reward_arr,
        ) = jax.lax.scan(
            body_fun, (true_state, true_obs), jnp.arange(observation_starts)
        )
        return (
            true_state,
            true_obs,
            true_state_arr,
            true_obs_arr,
            action_arr,
            reward_arr,
        )

    def act_observe_and_forecast(
        true_state, true_obs, params, wait_steps, episode_steps, key_obs
    ):
        def forecast_fun(carry, _):
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
            # define observation covariance matrix
            true_state, true_obs, key_obs = carry
            obs_cov = get_cov(y=true_obs)

            # add noise on the observation
            key_obs, _ = jax.random.split(key_obs)
            obs = jax.random.multivariate_normal(
                key_obs, true_obs, obs_cov, method="svd"
            )

            # get action
            action = agent.actor.apply(params, obs)

            # propagate environment with the given action
            (true_state, true_obs, action), (
                true_state_arr,
                true_obs_arr,
                action_arr,
                reward_arr,
            ) = jax.lax.scan(
                forecast_fun, (true_state, true_obs, action), jnp.arange(wait_steps)
            )
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

    def random_observe_and_forecast(
        true_state,
        true_obs,
        obs,
        wait_steps,
        episode_steps,
        key_obs,
        key_action,
    ):
        def forecast_fun(carry, _):
            true_state, true_obs, action = carry

            # get the next observation and reward with this action
            true_state, true_obs, reward, terminated, _, _ = env_step(
                state=true_state, action=action
            )

            return (true_state, true_obs, action), (
                true_state,
                true_obs,
                action,
                reward,
                terminated,
            )

        def body_fun(carry, _):
            # observe
            # we got an observation
            true_state, true_obs, obs, key_obs, key_action = carry

            # get action
            key_action, _ = jax.random.split(key_action)
            action = env_sample_action(key=key_action)

            # propagate environment with the given action
            (next_true_state, next_true_obs, action), (
                true_state_arr,
                true_obs_arr,
                action_arr,
                reward_arr,
                terminated_arr,
            ) = jax.lax.scan(
                forecast_fun,
                (true_state, true_obs, action),
                jnp.arange(wait_steps),
            )

            obs_cov = get_cov(y=true_obs)

            # add noise on the observation
            key_obs, _ = jax.random.split(key_obs)
            next_obs = jax.random.multivariate_normal(
                key_obs, next_true_obs, obs_cov, method="svd"
            )

            return (
                next_true_state,
                next_true_obs,
                next_obs,
                key_obs,
                key_action,
            ), (true_state_arr, true_obs_arr, next_obs, action_arr, reward_arr)

        n_loops = episode_steps // wait_steps
        (true_state, true_obs, obs, key_obs, key_action), (
            true_state_arr,
            true_obs_arr,
            obs_arr,
            action_arr,
            reward_arr,
        ) = jax.lax.scan(
            body_fun,
            (true_state, true_obs, obs, key_obs, key_action),
            jnp.arange(n_loops),
        )
        return (
            true_state,
            true_obs,
            obs,
            key_obs,
            key_action,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            action_arr,
            reward_arr,
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

    def act_episode(key_env, key_obs, params):
        # reset the environment
        key_env, _, key_init = jax.random.split(key_env, 3)
        init_true_state_mean, _, _ = env_reset(key=key_env)
        init_true_state = env_draw_initial_condition(
            u0=init_true_state_mean, key=key_init
        )
        init_true_obs = init_true_state[env.observation_inds]

        init_reward = jnp.nan

        # forecast until first observation
        (
            true_state,
            true_obs,
            true_state_arr0,
            true_obs_arr0,
            action_arr0,
            reward_arr0,
        ) = until_first_observation(true_state=init_true_state, true_obs=init_true_obs)
        (
            true_state,
            true_obs,
            key_obs,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            action_arr,
            reward_arr,
        ) = act_observe_and_forecast(
            true_state=true_state,
            true_obs=true_obs,
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
        action_arr = jnp.reshape(
            action_arr,
            (action_arr.shape[0] * action_arr.shape[1], action_arr.shape[2]),
        )
        reward_arr = jnp.reshape(
            reward_arr,
            (reward_arr.shape[0] * reward_arr.shape[1],),
        )
        stack = lambda a, b, c: jnp.vstack((jnp.expand_dims(a, axis=0), b, c))
        hstack = lambda a, b, c: jnp.hstack((jnp.expand_dims(a, axis=0), b, c))

        return (
            stack(init_true_state, true_state_arr0, true_state_arr),
            stack(init_true_obs, true_obs_arr0, true_obs_arr),
            obs_arr,
            stack(null_action, action_arr0, action_arr),
            hstack(init_reward, reward_arr0, reward_arr),
            key_env,
            key_obs,
        )

    def random_episode(key_env, key_obs, key_action):
        # reset the environment
        key_env, _, key_init = jax.random.split(key_env, 3)
        init_true_state_mean, _, _ = env_reset(key=key_env)
        init_true_state = env_draw_initial_condition(
            u0=init_true_state_mean, key=key_init
        )
        init_true_obs = init_true_state[env.observation_inds]

        init_reward = jnp.nan

        # forecast until first observation
        (
            true_state,
            true_obs,
            true_state_arr0,
            true_obs_arr0,
            action_arr0,
            reward_arr0,
        ) = until_first_observation(true_state=init_true_state, true_obs=init_true_obs)
        obs_cov = get_cov(y=true_obs)

        # add noise on the observation
        key_obs, _ = jax.random.split(key_obs)
        first_obs = jax.random.multivariate_normal(
            key_obs, true_obs, obs_cov, method="svd"
        )

        (
            true_state,
            true_obs,
            obs,
            key_obs,
            key_action,
            true_state_arr,
            true_obs_arr,
            obs_arr,
            action_arr,
            reward_arr,
        ) = random_observe_and_forecast(
            true_state=true_state,
            true_obs=true_obs,
            obs=first_obs,
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

        stack = lambda a, b, c: jnp.vstack((jnp.expand_dims(a, axis=0), b, c))
        stack2 = lambda a, b: jnp.vstack((jnp.expand_dims(a, axis=0), b))
        hstack = lambda a, b, c: jnp.hstack((jnp.expand_dims(a, axis=0), b, c))

        return (
            stack(init_true_state, true_state_arr0, true_state_arr),
            stack(init_true_obs, true_obs_arr0, true_obs_arr),
            stack2(first_obs, obs_arr),
            stack(null_action, action_arr0, action_arr),
            hstack(init_reward, reward_arr0, reward_arr),
            key_env,
            key_obs,
            key_action,
        )
    return random_episode, act_episode