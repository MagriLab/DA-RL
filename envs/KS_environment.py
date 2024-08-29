# Here we wrap the numerical KS solver into a Gym environment
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.optimize import minimize

from envs.KS_solver import KS


class KSenv(gym.Env):
    metadata = {}

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
        noise_stddev=0.0,
        seed=None,
        device="cpu",
        N=64,
    ):
        # Specify simulation parameters
        self.nu = nu
        self.N = N
        self.dt = 0.05
        self.action_size = actuator_locs.shape[-1]
        self.actuator_locs = actuator_locs
        self.actuator_scale = actuator_scale
        self.burn_in = burn_in
        self.initial_amplitude = initial_amplitude
        self.observation_inds = [int(x) for x in (self.N / (2 * np.pi)) * sensor_locs]
        self.num_observations = len(self.observation_inds)
        assert len(self.observation_inds) == len(set(self.observation_inds))
        self.termination_threshold = (
            20.0  # Terminate the simulation if max(u) exceeds this threshold
        )
        self.action_low = -1.0  # Minimum allowed actuation (per actuator)
        self.action_high = 1.0  # Maximum allowed actuation (per actuator)
        self.actuator_loss_weight = actuator_loss_weight
        self.frame_skip = frame_skip
        self.device = device
        self.noise_stddev = noise_stddev
        super().__init__()

        self.KS = KS(
            nu=self.nu,
            N=self.N,
            dt=self.dt,
            actuator_locs=self.actuator_locs,
            actuator_scale=self.actuator_scale,
            device=self.device,
        )
        self.solver_step = self.KS.advance
        self.observation_locs = self.KS.x[self.observation_inds]

        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, shape=(self.action_size,)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,)
        )

        # determine target solution
        if target == "e0":  # steer towards zero solution
            self.target = np.zeros(self.N)
        elif target == "e1":
            x = ((2 * np.pi) / self.N) * np.arange(self.N)
            u0 = -np.cos(x)
            self.target = minimize(self.fixed_point, x0=u0).x
        elif target == "e2":
            x = ((2 * np.pi) / self.N) * np.arange(self.N)
            u0 = -np.cos(2 * x)
            self.target = minimize(self.fixed_point, x0=u0).x
        elif target == "e3":
            x = ((2 * np.pi) / self.N) * np.arange(self.N)
            u0 = -3 * np.cos(3 * x)
            self.target = minimize(self.fixed_point, x0=u0).x
        else:
            raise ValueError("Target not recognized.")

    def fixed_point(self, u):
        u_new = self.solver_step(u0=u, action=np.zeros(self.action_size))
        return np.linalg.norm(u_new - u)

    def step(self, action):
        u = self.u  # Solution at previous timestep
        reward_sum = np.zeros([])
        for i in range(self.frame_skip):  # Take frame_skip many steps
            u = self.solver_step(u, action)  # Take a step using the PDE solver
            # reward = - (L2 norm of solution + hyperparameter * L2 norm of action)
            reward = -np.linalg.norm(
                u - self.target, axis=-1
            ) - self.actuator_loss_weight * np.linalg.norm(action, axis=-1)
            reward_sum += reward
        reward = (
            reward_sum / self.frame_skip
        )  # Compute the average reward over frame_skip steps
        # reward_mean = reward_mean.view(*tensordict.shape, 1)
        self.u = u
        observation = u[self.observation_inds]  # Evaluate at desired indices
        # To allow for batched computations, use this instead:
        # ... however the KS solver needs to be compatible with np.vmap!
        # u = np.vmap(self.solver_step)(u, action)
        terminated = np.max(np.abs(u)) > self.termination_threshold

        # add measurement noise
        observation += self.noise_stddev * np.random.randn(self.num_observations)

        # done = done.view(*tensordict.shape, 1)
        truncated = False
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        # Initial data drawn from IID normal distributions
        zrs = np.zeros([self.N])
        ons = np.ones([self.N])
        u = np.random.normal(zrs, ons)
        u = self.initial_amplitude * u
        u = u - u.mean(axis=-1)
        # Burn in
        for _ in range(self.burn_in):
            u = self.solver_step(u, np.zeros(self.action_size))
        # Store the solution in class variable
        self.u = u
        # Compute observation
        observation = u[self.observation_inds]
        observation += self.noise_stddev * np.random.randn(self.num_observations)
        return observation, {}


if __name__ == "__main__":
    # Defining env
    env = KSenv(
        nu=0.08,
        actuator_locs=np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
        sensor_locs=np.array([0.0, 1.0, 2.0]),
        burn_in=0,
    )
    env.reset()
    env.step(action=np.zeros(4))
