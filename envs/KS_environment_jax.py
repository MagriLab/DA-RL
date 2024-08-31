import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
from jax import jit, lax
from jax.scipy.optimize import minimize

from envs.KS_solver_jax import KS  # Import the KS solver


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
        self.observation_inds = [int(x) for x in (self.N / (2 * jnp.pi)) * sensor_locs]
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
        self.target = self.determine_target(target)

        # # Jit compile the step function
        # self.jit_step = jit(self._step)

    def fixed_point(self, u):
        """
        Compute the fixed point for the given initial condition.

        Args:
            u: Initial condition for the state.

        Returns:
            The norm of the difference between the new and old state.
        """
        u_new = KS.advance(
            u,
            jnp.zeros(self.action_size),
            self.ks_solver.B,
            self.ks_solver.lin,
            self.ks_solver.ik,
            self.ks_solver.dt,
        )
        return jnp.linalg.norm(u_new - u)

    def determine_target(self, target):
        x = ((2 * jnp.pi) / self.N) * jnp.arange(self.N)
        if target == "e0":
            return jnp.zeros(self.N)
        elif target == "e1":
            u0 = -jnp.cos(x)
        elif target == "e2":
            u0 = -jnp.cos(2 * x)
        elif target == "e3":
            u0 = -3 * jnp.cos(3 * x)
        else:
            raise ValueError("Target not recognized.")

        result = minimize(self.fixed_point(u0))
        return result.x

    @staticmethod
    def sample_continuous_space(rng_key, low, high, shape):
        """
        Sample from a continuous action space defined by low and high bounds.

        Args:
            rng_key: JAX PRNG key.
            low: The lower bound of the action space (scalar or array).
            high: The upper bound of the action space (scalar or array).
            shape: The shape of the action space.

        Returns:
            Sampled action with the given shape.
        """
        return jax.random.uniform(rng_key, shape=shape, minval=low, maxval=high)

    @staticmethod
    def step(
        u,
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

        def single_step(u, _):
            # Take a step using the KS solver
            next_u = KS.advance(u, action, B, lin, ik, dt)
            # Compute reward
            reward = -jnp.linalg.norm(
                next_u - target
            ) - actuator_loss_weight * jnp.linalg.norm(action)
            return next_u, reward

        next_u, total_reward = lax.scan(single_step, u, jnp.arange(frame_skip))
        reward = jnp.mean(total_reward)

        # Observe the state at the sensor locations
        observation = u[observation_inds]

        terminated = jnp.max(jnp.abs(next_u)) > termination_threshold
        truncated = False
        return next_u, observation, reward, terminated, truncated, {}

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
        u = initial_amplitude * random.normal(key, (N,))
        u = u - u.mean()

        def burn_in_step(u, _):
            return KS.advance(u, jnp.zeros(action_size), B, lin, ik, dt), None

        u, _ = lax.scan(burn_in_step, u, None, length=burn_in)

        observation = u[observation_inds]
        return u, observation, {}


if __name__ == "__main__":
    # Example usage
    env = KSenv(
        nu=0.08,
        actuator_locs=jnp.array([0.0, jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4]),
        sensor_locs=jnp.array([0.0, 1.0, 2.0]),
        burn_in=0,
    )
    u, observation, _ = KSenv.reset(
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
    u, reward, terminated, truncated, _ = KSenv.step(
        u,
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
