from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


class Actor(nn.Module):
    """Actor network that learns a stochastic policy."""

    n_hidden_units: Sequence[int]
    n_action: int
    log_std_min: jnp.ndarray
    log_std_max: jnp.ndarray
    activation: nn.activation = nn.relu

    @nn.compact
    def __call__(self, state):
        """
        Args:
            state: (batch_size x n_state)
        Returns
            mean: (batch_size x n_action)
            log_std: (batch_size x n_action)
        """
        x = self.activation(nn.Dense(self.n_hidden_units[0])(state))
        for n in self.n_hidden_units[1:]:
            x = self.activation(nn.Dense(n)(x))

        # output layer
        x = nn.Dense(2 * self.n_action)(x)
        mean, log_std = jnp.split(x, 2, axis=1)

        log_std = nn.tanh(
            x
        )  # last activation is a tanh to squash the std outputs to between -1 and 1
        # scale the std outputs to the limits given by the environment
        # such that -1 corresponds to the minimum std and 1 corresponds to maximum std
        log_std = 0.5 * (
            log_std * (self.log_std_max - self.log_std_min)
            + (self.log_std_max + self.log_std_min)
        )
        return mean, log_std


class Critic(nn.Module):
    """Critic network that learns the action-value function."""

    n_hidden_units: Sequence[int]
    activation: nn.activation = nn.relu

    @nn.compact
    def __call__(self, state, action):
        """
        Args:
            state: (batch_size x n_state)
            action: (batch_size x n_action)
        Returns
            action-value function q(s, a): (batch_size x 1)
        """
        x = jnp.concatenate([state, action], axis=-1)
        for n in self.n_hidden_units:
            x = self.activation(nn.Dense(n)(x))

        # output layer
        q = nn.Dense(1)(x)
        return q


class TrainState(TrainState):
    target_params: flax.core.FrozenDict  # add the parameters of the target networks


class SAC:
    """Soft Actor Critic."""

    def __init__(self, config, env):
        # training parameters
        self.actor_lr = config.train.actor_learning_rate
        self.critic_lr = config.train.critic_learning_rate

        # create actor network
        # TO BE MODIFIED
        log_std_min = config.train.exploration_stddev * jnp.log(
            env.action_high - env.action_low
        )
        log_std_max = jnp.log(env.action_high - env.action_low)

        self.actor = Actor(
            config.network.actor_hidden_units,
            n_action=env.action_size,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            activation=getattr(nn, config.network.activation_function),
        )

        # create critic network
        self.critic = [
            Critic(
                config.network.critic_hidden_units,
                activation=getattr(nn, config.network.activation_function),
            )
            for _ in range(config.network.n_critics)
        ]

        self.actor.apply = jax.jit(self.actor.apply)

        # Vectorize the application of critics using vmap
        self.vmap_critic_apply = jax.vmap(
            lambda critic, critic_state, state, action: critic.apply(
                critic_state.params, state, action
            )
        )

        # Apply jitting to the vectorized critic apply function
        self.vmap_critic_apply = jax.jit(self.vmap_critic_apply)

        # self.update_actor = jax.jit(self.update_actor)
        # self.update_critic = jax.jit(self.update_critic)
        # self.update_target_networks = jax.jit(self.update_target_networks)

    def initial_network_state(self, key, state, action):
        key, actor_key, critic_key = jax.random.split(key, 3)

        # initialize the parameters of the actor network
        actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, state),
            target_params=self.actor.init(actor_key, state),
            tx=optax.adam(learning_rate=self.actor_lr),
        )

        # Initialize the parameters of the critic network using vmap
        # Generate a random key for each critic
        critic_keys = jax.random.split(critic_key, len(self.critic))

        # Use vmap to initialize all critics with their respective keys
        critic_state = [
            TrainState.create(
                apply_fn=critic.apply,
                params=critic.init(key, state, action),
                target_params=critic.init(critic_key, state, action),
                tx=optax.adam(learning_rate=self.critic_lr),
            )
            for (critic, critic_key) in zip(self.critic, critic_keys)
        ]

        return actor_state, critic_state

    # def update_critic(
    #     self,
    #     actor_state: TrainState,
    #     critic_state: TrainState,
    #     state: jnp.ndarray,
    #     action: jnp.ndarray,
    #     next_state: jnp.ndarray,
    #     reward: jnp.float64,
    #     terminated: jnp.ndarray,
    # ):
