import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
from jax import jit, lax
from scipy.optimize import minimize 
# jax scipy.optimize.minimize doesn't converge!! 
# alternative is to do gradient descent with optax

from envs.KS_solver_jax import KS  # Import the KS solver
from functools import partial

class KSenv:
    def __init__(
        self,
        nu,
        actuator_locs,
        sensor_locs,
        burn_in=0,
        target="e0",
        frame_skip=1,
        actuator_loss_weight=0.0,
        initial_amplitude=1e-2,
        actuator_scale=0.1,
        N=64,
    ):
        """
        Initialize the KS environment.

        Args:
            nu: Viscosity parameter.
            actuator_locs: Locations of the actuators.
            sensor_locs: Locations of the sensors.
            burn_in: Number of initial steps to burn in.
            target: Target state (e.g., "e0", "e1", etc.).
            frame_skip: Number of steps per action.
            actuator_loss_weight: Weight for actuator penalty in reward.
            initial_amplitude: Initial amplitude for the state initialization.
            actuator_scale: Scale of the actuator influence.
            N: Number of grid points.
        """
        # Simulation parameters
        self.nu = nu
        self.N = N
        self.dt = 0.05

        # Action configuration
        if isinstance(actuator_locs, int):
            actuator_locs = ((2 * jnp.pi) / actuator_locs) * jnp.arange(actuator_locs) 

        self.action_size = actuator_locs.shape[-1]
        self.actuator_locs = actuator_locs
        self.actuator_scale = actuator_scale
        self.action_low = -1.0
        self.action_high = 1.0
        self.actuator_loss_weight = actuator_loss_weight

        # Initialization
        self.burn_in = burn_in
        self.initial_amplitude = initial_amplitude

        # Observations
        if isinstance(sensor_locs, int):
            sensor_locs = ((2 * jnp.pi) / sensor_locs) * jnp.arange(sensor_locs) 

        self.observation_inds = [
            int(jnp.round(x)) for x in (self.N / (2 * jnp.pi)) * sensor_locs
        ]
        self.num_observations = len(self.observation_inds)
        assert len(self.observation_inds) == len(set(self.observation_inds))
        self.observation_inds = jnp.array(self.observation_inds)

        self.termination_threshold = 20.0  # Termination condition based on max(u)

        self.frame_skip = frame_skip

        # Initialize the KS solver
        self.ks_solver = KS(
            actuator_locs=actuator_locs,
            actuator_scale=actuator_scale,
            nu=self.nu,
            N=self.N,
            dt=self.dt,
        )
        self.observation_locs = self.ks_solver.x[self.observation_inds]

        # Determine target solution
        self.target = KSenv.determine_target(target, self.N, self.action_size, self.ks_solver.B, self.ks_solver.lin, self.ks_solver.ik, self.ks_solver.dt)

        # # Jit compile the step function
        # self.jit_step = jit(self._step)

    @staticmethod
    def fixed_point(u, action_size, B, lin, ik, dt):
        """
        Compute the fixed point for the given initial condition.

        Args:
            u: Initial condition for the state.

        Returns:
            The norm of the difference between the new and old state.
        """
        u_new = KS.advance(
            u,
            jnp.zeros(action_size),
            B,
            lin,
            ik,
            dt
        )
        return jnp.linalg.norm(u_new - u)
    
    @staticmethod
    def determine_target(target, N, action_size, B, lin, ik, dt):
        x = ((2 * jnp.pi) / N) * jnp.arange(N)
        if target == "e0":
            return jnp.zeros(N)
        elif target == "e1":
            u0 = -jnp.cos(x)
        elif target == "e2":
            u0 = -jnp.cos(2 * x)
        elif target == "e3":
            u0 = -3 * jnp.cos(3 * x)
        else:
            raise ValueError("Target not recognized.")

        partial_fixed_point = partial(KSenv.fixed_point, 
                                      action_size = action_size, 
                                      B = B,
                                      lin = lin, 
                                      ik = ik, 
                                      dt = dt)
        result = minimize(partial_fixed_point, u0, method="BFGS")
        return result.x

    @staticmethod
    def sample_continuous_space(key, low, high, shape):
        """
        Sample from a continuous action space defined by low and high bounds.

        Args:
            key: JAX PRNG key.
            low: The lower bound of the action space (scalar or array).
            high: The upper bound of the action space (scalar or array).
            shape: The shape of the action space.

        Returns:
            Sampled action with the given shape.
        """
        return jax.random.uniform(key, shape=shape, minval=low, maxval=high)

    @staticmethod
    def get_reward_state(state, target):
        return jnp.sqrt(1/len(state)) * jnp.linalg.norm(state - target)

    @staticmethod
    def get_reward_action(action):
        return jnp.linalg.norm(action)

    @staticmethod
    def get_reward(next_state, target, action, actuator_loss_weight):
        return -KSenv.get_reward_state(
            next_state, target
        ) - actuator_loss_weight * KSenv.get_reward_action(action)

    @staticmethod
    def step(
        state,
        action,
        B,
        lin,
        ik,
        dt,
        target,
        frame_skip,
        actuator_loss_weight,
        termination_threshold,
        observation_inds,
    ):
        """
        Perform one step in the environment.

        Args:
            u: Current state of the system.
            action: Actuation input.

        Returns:
            next_u: State after the step.
            observation: Observation after the step.
            reward: Reward received after the step.
            terminated: Whether the simulation is terminated.
            truncated: Whether the simulation is truncated.
        """

        def single_step(state, _):
            # Take a step using the KS solver
            next_state = KS.advance(state, action, B, lin, ik, dt)
            # Compute reward
            reward = KSenv.get_reward(next_state, target, action, actuator_loss_weight)
            return next_state, reward

        next_state, total_reward = lax.scan(single_step, state, jnp.arange(frame_skip))
        reward = jnp.mean(total_reward)

        # Observe the state at the sensor locations
        observation = next_state[observation_inds]

        terminated = jnp.max(jnp.abs(next_state)) > termination_threshold
        truncated = False
        return next_state, observation, reward, terminated, truncated, {}

    @staticmethod
    def reset(
        N,
        B,
        lin,
        ik,
        dt,
        initial_amplitude,
        action_size,
        burn_in,
        observation_inds,
        key,
    ):
        """
        Reset the environment to the initial state.

        Args:
            seed: Random seed for initialization.

        Returns:
            u: Initial state.
            observation: Initial observation.
            info: Additional information.
        """
        state = initial_amplitude * random.normal(key, (N,))
        state = state - state.mean()

        def burn_in_step(u, _):
            return KS.advance(u, jnp.zeros(action_size), B, lin, ik, dt), None

        state, _ = lax.scan(burn_in_step, state, None, length=burn_in)

        observation = state[observation_inds]
        return state, observation, {}


if __name__ == "__main__":
    # Example usage
    env = KSenv(
        nu=0.08,
        actuator_locs=jnp.array([0.0, jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4]),
        sensor_locs=jnp.array([0.0, jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4]),
        burn_in=0,
    )
    state, observation, _ = KSenv.reset(
        env.N,
        env.ks_solver.B,
        env.ks_solver.lin,
        env.ks_solver.ik,
        env.ks_solver.dt,
        env.initial_amplitude,
        env.action_size,
        env.burn_in,
        env.observation_inds,
    )
    action = jnp.zeros(env.ks_solver.B.shape[1])
    state, reward, terminated, truncated, _ = KSenv.step(
        state,
        action,
        env.ks_solver.B,
        env.ks_solver.lin,
        env.ks_solver.ik,
        env.ks_solver.dt,
        env.target,
        env.frame_skip,
        env.actuator_loss_weight,
        env.termination_threshold,
        env.observation_inds,
    )
