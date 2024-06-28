import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from typing import Sequence
import optax

class Actor(nn.Module):
    """Actor network that learns a deterministic policy."""
    n_hidden_units: Sequence[int]
    n_action: int
    action_min: jnp.ndarray # (1 x n_action)
    action_max: jnp.ndarray # (1 x n_action)
    activation: nn.activation = nn.relu
    @nn.compact
    def __call__(self, state):
        """
        Args:
            state: (batch_size x n_state)
        Returns 
            action: (batch_size x n_action)
        """
        x = self.activation(nn.Dense(self.n_hidden_units[0])(state))
        for n in self.n_hidden_units[1:]:
            x = self.activation(nn.Dense(n)(x))

        # output layer
        x = nn.Dense(self.n_action)(x)
        action = nn.tanh(x) # last activation is a tanh to squash the action outputs to between -1 and 1

        # scale the action outputs to the limits given by the environment
        # such that -1 corresponds to the minimum action and 1 corresponds to maximum action
        action = 0.5 * (action * (self.action_max-self.action_min) + (self.action_max+self.action_min))
        return action

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
        x = jnp.concatenate([state, action], axis = -1)
        for n in self.n_hidden_units:
            x = self.activation(nn.Dense(n)(x))
        
        # output layer
        q = nn.Dense(1)(x)
        return q

class TrainState(TrainState):
    target_params: flax.core.FrozenDict # add the parameters of the target networks 

class DDPG:
    """Deep Deterministic Policy Gradient."""
    def __init__(self, config, env):
        # training parameters
        self.actor_lr = config.train.actor_learning_rate 
        self.critic_lr = config.train.critic_learning_rate 
        self.gamma = config.train.discount_factor
        self.tau = config.train.soft_update_rate

        # create actor network
        self.actor = Actor(config.network.actor_hidden_units, 
                      n_action=env.action_space.shape[0], 
                      action_min=env.action_space.low, action_max=env.action_space.high,
                      activation=getattr(nn,config.network.activation_function))
        
        # create critic network
        self.critic = Critic(config.network.critic_hidden_units,
                             activation=getattr(nn,config.network.activation_function))

        self.actor.apply = jax.jit(self.actor.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.update_actor = jax.jit(self.update_actor)
        self.update_critic = jax.jit(self.update_critic)
        self.update_target_networks = jax.jit(self.update_target_networks)

    def initial_network_state(self, env, key):
        key, actor_key, critic_key = jax.random.split(key, 3)

        # sample state and action to get the correct shape
        state = jnp.array([env.observation_space.sample()])
        action = jnp.array([env.action_space.sample()])

        # initialize the parameters of the actor network
        actor_state = TrainState.create(apply_fn=self.actor.apply,
                                        params=self.actor.init(actor_key, state),
                                        target_params=self.actor.init(actor_key, state),
                                        tx=optax.adam(learning_rate=self.actor_lr))

        # initialize the parameters of the critic network
        critic_state = TrainState.create(apply_fn=self.critic.apply,
                                         params=self.critic.init(critic_key, state, action),
                                         target_params=self.critic.init(critic_key, state, action),
                                         tx=optax.adam(learning_rate=self.critic_lr))
        
        return actor_state, critic_state
    
    def update_critic(self,
                      actor_state: TrainState, 
                      critic_state: TrainState,
                      state: jnp.ndarray,
                      action: jnp.ndarray,
                      next_state: jnp.ndarray,
                      reward: jnp.float64,
                      terminated: jnp.ndarray):
        # get next action using the target policy on the next state
        next_action = self.actor.apply(actor_state.target_params, next_state)

        # get the q value at the next state (at time t+1) using the target policy and target q network
        next_q_target = self.critic.apply(critic_state.target_params, next_state, next_action).reshape(-1)
        # we need to reshape because next_q_target is batch_size x 1, whereas reward and terminated are batch_size,

        # compute the target value for the current state (this is denoted by y in the paper)
        # the Bellman update target estimated using the reward r at time t and q value at time t+1
        # i.e., one-step temporal difference

        # when an episode ends due to termination we don’t bootstrap, 
        # when it ends due to truncation, we bootstrap.
        
        target = reward + self.gamma * (1-terminated) * next_q_target

        # define the q loss
        # we want the gradient of the MSE loss from learning q network with respect to its parameters
        def compute_q_loss(params):
            # get the q value of the learning q network 
            q = self.critic.apply(params, state, action).reshape(-1)
            # compute mean squared error
            mse = jnp.mean((q - target) ** 2)
            return mse

        # compute gradient
        q_loss, q_grad = jax.value_and_grad(compute_q_loss)(critic_state.params)

        # update critic
        critic_state = critic_state.apply_gradients(grads=q_grad)

        return critic_state, q_loss
    
    def update_actor(self,
                     actor_state: TrainState, 
                     critic_state: TrainState,
                     state: jnp.ndarray):
        
        # define the policy loss that will give us the policy gradient,
        # once differentiated with respect to the parameters of the learning actor network
        # this is given by the the chain rule
        # dJ/dtheta^mu = mean (dq/da * dmu/dtheta^mu))
        # we choose this loss as the negative average q values
        # because we want a policy that maximizes q
        def compute_policy_loss(params):
            J = -jnp.mean(self.critic.apply(critic_state.params, state, self.actor.apply(params, state)))
            return J 

        # compute gradient
        policy_loss, policy_grad = jax.value_and_grad(compute_policy_loss)(actor_state.params)

        # update actor
        actor_state = actor_state.apply_gradients(grads=policy_grad)

        return actor_state, policy_loss
    
    def update_target_networks(self,
                               actor_state: TrainState, 
                               critic_state: TrainState):
        # update the target networks' parameters such that they are updated as a moving average
        # theta = tau * theta_new + (1-tau) * theta
        # this enables stability during learning
        critic_state = critic_state.replace(target_params = optax.incremental_update(critic_state.params, critic_state.target_params, self.tau))
        actor_state = actor_state.replace(target_params = optax.incremental_update(actor_state.params, actor_state.target_params, self.tau))
        return actor_state, critic_state
