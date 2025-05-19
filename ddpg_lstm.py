from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

# using old flax!!


class Actor(nn.Module):
    """Actor network that learns a deterministic policy."""

    n_hidden_state: int
    n_hidden_units: Sequence[int]
    n_action: int
    action_min: jnp.ndarray  # (1 x n_action)
    action_max: jnp.ndarray  # (1 x n_action)
    activation: nn.activation = nn.relu

    @nn.compact
    def __call__(self, obs_seq, carry):
        """
        Args:
            obs_seq: (batch_size x seq_len x n_obs)
            carry: (hidden, cell) state of the LSTM that is carried
        Returns
            action: (batch_size x seq_len x n_action)
            new_carry: updated LSTM carry
        """
        lstm = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,  # scan over the sequence
            out_axes=1,  # scan over the sequence
        )(name="lstm", features=self.n_hidden_state)

        # apply LSTM across the sequence
        new_carry, outputs = lstm(carry, obs_seq)

        # pass the outputs to MLP
        x = outputs
        for n in self.n_hidden_units:
            x = self.activation(nn.Dense(n)(x))

        # output layer
        x = nn.Dense(self.n_action)(x)
        action = nn.tanh(
            x
        )  # last activation is a tanh to squash the action outputs to between -1 and 1

        # scale the action outputs to the limits given by the environment
        # such that -1 corresponds to the minimum action and 1 corresponds to maximum action
        action = 0.5 * (
            action * (self.action_max - self.action_min)
            + (self.action_max + self.action_min)
        )
        return action, new_carry


class Critic(nn.Module):
    """Critic network that learns the action-value function."""

    n_hidden_state: int
    n_hidden_units: Sequence[int]
    activation: nn.activation = nn.relu

    @nn.compact
    def __call__(self, obs_seq, action_seq, carry):
        """
        Args:
            obs_seq: (batch_size x seq_len x n_obs)
            action_seq: (batch_size x seq_len x n_action)
            carry: (hidden, cell) state of the LSTM that is carried
        Returns
            action-value function q(s, a): (batch_size x seq_len x 1)
            new_carry: updated LSTM carry
        """
        lstm = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,  # scan over the sequence
            out_axes=1,  # scan over the sequence
        )(name="lstm", features=self.n_hidden_state)

        # apply LSTM across the sequence
        new_carry, outputs = lstm(
            carry, jnp.concatenate([obs_seq, action_seq], axis=-1)
        )

        x = outputs
        for n in self.n_hidden_units:
            x = self.activation(nn.Dense(n)(x))

        # output layer
        q = nn.Dense(1)(x)
        return q, new_carry


class TrainState(TrainState):
    target_params: flax.core.FrozenDict  # add the parameters of the target networks


class DDPG:
    """Deep Deterministic Policy Gradient."""

    def __init__(self, config, env):
        # training parameters
        self.actor_lr = config.train.actor_learning_rate
        self.critic_lr = config.train.critic_learning_rate
        self.gamma = config.train.discount_factor
        self.tau = config.train.soft_update_rate

        # create actor network
        self.actor = Actor(
            n_hidden_state=config.network.actor_n_hidden_state,
            n_hidden_units=config.network.actor_hidden_units,
            n_action=env.action_size,
            action_min=env.action_low,
            action_max=env.action_high,
            activation=getattr(nn, config.network.activation_function),
        )

        # create critic network
        self.critic = Critic(
            n_hidden_state=config.network.actor_n_hidden_state,
            n_hidden_units=config.network.critic_hidden_units,
            activation=getattr(nn, config.network.activation_function),
        )

        self.actor.apply = jax.jit(self.actor.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.update_actor = jax.jit(self.update_actor)
        self.update_critic = jax.jit(self.update_critic)
        self.update_target_networks = jax.jit(self.update_target_networks)

    def initial_network_state(self, key, obs_seq, action_seq):
        key, actor_key, critic_key = jax.random.split(key, 3)

        batch_size = obs_seq.shape[0]
        # initialize carry
        carry_actor = nn.LSTMCell(self.actor.n_hidden_state).initialize_carry(
            jax.random.PRNGKey(0), input_shape=(batch_size, self.actor.n_hidden_state)
        )
        carry_actor = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64), carry_actor
        )

        carry_critic = nn.LSTMCell(self.critic.n_hidden_state).initialize_carry(
            jax.random.PRNGKey(0), input_shape=(batch_size, self.critic.n_hidden_state)
        )
        carry_critic = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64), carry_critic
        )

        # initialize the parameters of the actor network
        actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs_seq, carry_actor),
            target_params=self.actor.init(actor_key, obs_seq, carry_actor),
            tx=optax.adam(learning_rate=self.actor_lr),
        )

        # initialize the parameters of the critic network
        critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, obs_seq, action_seq, carry_critic),
            target_params=self.critic.init(
                critic_key, obs_seq, action_seq, carry_critic
            ),
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        return actor_state, critic_state

    def initialise_actor_carry(self, batch_size):
        carry_actor = nn.LSTMCell(self.actor.n_hidden_state).initialize_carry(
            jax.random.PRNGKey(0), input_shape=(batch_size, self.actor.n_hidden_state)
        )
        carry_actor = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64), carry_actor
        )
        return carry_actor

    def initialise_critic_carry(self, batch_size):
        carry_critic = nn.LSTMCell(self.critic.n_hidden_state).initialize_carry(
            jax.random.PRNGKey(0), input_shape=(batch_size, self.critic.n_hidden_state)
        )
        carry_critic = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64), carry_critic
        )
        return carry_critic

    # We are using LSTMs for both actor and critic,
    # so we are training on observation-action sequences, not just single steps!
    # (crucial for learning temporal dependencies)
    # We obtained the observation sequence by using that action sequence
    # So when we are doing off-policy learning we can't just use the action sequence predicted by the actor
    # That's why we append the last action predicted by the network with our previous action sequence

    # If you want to model long-term dependencies (like POMDPs),
    # consider sampling overlapping sequences with a burn-in period
    # where you don’t update the loss but still evolve the LSTM state.

    def update_critic(
        self,
        actor_state: TrainState,
        critic_state: TrainState,
        obs_seq: jnp.ndarray,  # (batch_size x seq_len x n_obs)
        action_seq: jnp.ndarray,  # (batch_size x seq_len x n_action)
        next_obs_seq: jnp.ndarray,  # (batch_size x seq_len x n_obs)
        reward_seq: jnp.ndarray,  # (batch_size x seq_len x 1)
        terminated_seq: jnp.ndarray,  # (batch_size x seq_len x 1)
    ):
        batch_size = obs_seq.shape[0]

        # initialise carry for the actor
        carry_actor = self.initialise_actor_carry(batch_size)

        # get next action using the target policy on the sequence of next observations
        next_action_seq, _ = self.actor.apply(
            actor_state.target_params, next_obs_seq, carry_actor
        )
        next_action = next_action_seq[:, -1]  # get the last action
        next_action = next_action[:, None, :]  # for concatenation

        # create full next action sequence by shifting + appending predicted next_action
        # we have taken the same actions up until the last one where we take the action given by actor
        new_action_seq = jnp.concatenate([action_seq[:, 1:], next_action], axis=1)

        # initialise carry for the critic
        carry_critic = self.initialise_critic_carry(batch_size)

        # get the q value at the next state (at time t+1) using the target policy and target q network
        # Given the past observations and actions, what is the value of taking the next action at the next observation?
        next_q_target_seq, _ = self.critic.apply(
            critic_state.target_params, next_obs_seq, new_action_seq, carry_critic
        )
        next_q_target = jax.lax.stop_gradient(next_q_target_seq[:, -1])

        # compute the target value for the current state (this is denoted by y in the paper)
        # the Bellman update target estimated using the reward r at time t and q value at time t+1
        # i.e., one-step temporal difference

        # when an episode ends due to termination we don’t bootstrap,
        # when it ends due to truncation, we bootstrap.
        target = (
            reward_seq[:, -1] + self.gamma * (1 - terminated_seq[:, -1]) * next_q_target
        )

        # define the q loss
        # we want the gradient of the MSE loss from learning q network with respect to its parameters
        def compute_q_loss(params):
            carry = self.initialise_critic_carry(batch_size)

            # get the q value of the learning q network
            q_seq, _ = self.critic.apply(params, obs_seq, action_seq, carry)
            q = q_seq[:, -1]
            # compute mean squared error
            mse = jnp.mean((q - target) ** 2)
            return mse

        # compute gradient
        q_loss, q_grad = jax.value_and_grad(compute_q_loss)(critic_state.params)

        # update critic
        critic_state = critic_state.apply_gradients(grads=q_grad)

        return critic_state, q_loss

    def update_actor(
        self,
        actor_state: TrainState,
        critic_state: TrainState,
        obs_seq: jnp.ndarray,  # (batch_size x seq_len x n_obs)
        action_seq: jnp.ndarray,  # (batch_size x seq_len x n_obs)
    ):
        # define the policy loss that will give us the policy gradient,
        # once differentiated with respect to the parameters of the learning actor network
        # this is given by the the chain rule
        # dJ/dtheta^mu = mean (dq/da * dmu/dtheta^mu))
        # we choose this loss as the negative average q values
        # because we want a policy that maximizes q
        batch_size = obs_seq.shape[0]

        def compute_policy_loss(params):
            # initialise carry
            carry_actor = self.initialise_actor_carry(batch_size)
            carry_critic = self.initialise_critic_carry(batch_size)

            # get new action sequence from actor
            new_action_seq, _ = self.actor.apply(params, obs_seq, carry_actor)
            new_action = new_action_seq[:, -1]
            new_action = new_action[:, None, :]

            # create full next action sequence by shifting + appending predicted next_action
            train_action_seq = jnp.concatenate([action_seq[:, :-1], new_action], axis=1)

            q_seq, _ = self.critic.apply(
                critic_state.params, obs_seq, train_action_seq, carry_critic
            )
            q_last = q_seq[:, -1]
            J = -jnp.mean(q_last)
            return J

        # compute gradient
        policy_loss, policy_grad = jax.value_and_grad(compute_policy_loss)(
            actor_state.params
        )

        # update actor
        actor_state = actor_state.apply_gradients(grads=policy_grad)

        return actor_state, policy_loss

    def update_target_networks(self, actor_state: TrainState, critic_state: TrainState):
        # update the target networks' parameters such that they are updated as a moving average
        # theta = tau * theta_new + (1-tau) * theta
        # this enables stability during learning
        critic_state = critic_state.replace(
            target_params=optax.incremental_update(
                critic_state.params, critic_state.target_params, self.tau
            )
        )
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(
                actor_state.params, actor_state.target_params, self.tau
            )
        )
        return actor_state, critic_state
