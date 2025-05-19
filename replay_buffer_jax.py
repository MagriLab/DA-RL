from functools import partial

import jax
import jax.numpy as jnp
from jax import jit


def init_replay_buffer(capacity, state_dim, action_dim, rng_key):
    """
    Initialize the replay buffer.

    Args:
        capacity: Integer, size of the replay buffer.
        state_dim: Tuple, shape of the state space.
        action_dim: Tuple, shape of the action space.
        rng_key: JAX PRNG key.

    Returns:
        A dictionary containing the initialized replay buffer.
    """
    return {
        "state_buffer": jnp.zeros((capacity, *state_dim)),
        "action_buffer": jnp.zeros((capacity, *action_dim)),
        "next_state_buffer": jnp.zeros((capacity, *state_dim)),
        "reward_buffer": jnp.zeros(capacity),
        "terminated_buffer": jnp.zeros(capacity),
        "position": 0,
        "size": 0,
        "capacity": capacity,
        "rng_key": rng_key,
    }


@jit
def add_experience(buffer, state, action, reward, next_state, terminated):
    """
    Add a new experience to the replay buffer.

    Args:
        buffer: The replay buffer dictionary.
        state: Current state.
        action: Action taken.
        next_state: Next state.
        reward: Reward received.
        terminated: Boolean, whether the episode is terminated.

    Returns:
        Updated replay buffer dictionary.
    """
    pos = buffer["position"] % buffer["capacity"]

    state_buffer = buffer["state_buffer"].at[pos].set(state)
    action_buffer = buffer["action_buffer"].at[pos].set(action)
    next_state_buffer = buffer["next_state_buffer"].at[pos].set(next_state)
    reward_buffer = buffer["reward_buffer"].at[pos].set(reward)
    terminated_buffer = buffer["terminated_buffer"].at[pos].set(terminated)

    position = buffer["position"] + 1
    size = jnp.minimum(buffer["size"] + 1, buffer["capacity"])

    return {
        "state_buffer": state_buffer,
        "action_buffer": action_buffer,
        "next_state_buffer": next_state_buffer,
        "reward_buffer": reward_buffer,
        "terminated_buffer": terminated_buffer,
        "position": position,
        "size": size,
        "capacity": buffer["capacity"],
        "rng_key": buffer["rng_key"],
    }


@partial(jit, static_argnums=(1,))
def sample_experiences(buffer, batch_size):
    """
    Sample a batch of experiences from the replay buffer.

    Args:
        buffer: The replay buffer dictionary.
        batch_size: Integer, number of samples to draw.

    Returns:
        A tuple of (states, actions, rewards, next_states, terminateds).
    """
    rng_key, subkey = jax.random.split(buffer["rng_key"])
    indices = jax.random.randint(subkey, (batch_size,), minval=0, maxval=buffer["size"])

    batch_states = buffer["state_buffer"][indices]
    batch_actions = buffer["action_buffer"][indices]
    batch_rewards = buffer["reward_buffer"][indices]
    batch_terminateds = buffer["terminated_buffer"][indices]
    batch_next_states = buffer["next_state_buffer"][indices]

    updated_buffer = {
        "state_buffer": buffer["state_buffer"],
        "action_buffer": buffer["action_buffer"],
        "reward_buffer": buffer["reward_buffer"],
        "next_state_buffer": buffer["next_state_buffer"],
        "terminated_buffer": buffer["terminated_buffer"],
        "position": buffer["position"],
        "size": buffer["size"],
        "capacity": buffer["capacity"],
        "rng_key": rng_key,  # the rng key changes, that's why we return the updated buffer
    }

    return (
        batch_states,
        batch_actions,
        batch_next_states,
        batch_rewards,
        batch_terminateds,
    ), updated_buffer


@partial(jit, static_argnums=(1, 2, 3))
def sample_sequences(buffer, batch_size, seq_len, episode_len):
    """
    Sample a batch of experiences from the replay buffer.

    Args:
        buffer: The replay buffer dictionary.
        batch_size: Integer, number of samples to draw.
        seq_len: Integer, length of sequence
        episode_len: Integer, length of episode, we assume fixed episode length

        BE CAREFUL, THIS IMPLEMENTATION DOESN'T TAKE INTO ACCOUNT CIRCULAR BUFFER!

    Returns:
        Tuple of sequences of (states, actions, rewards, next_states, terminateds).
    """
    total_steps = buffer["size"]
    num_episodes = total_steps // episode_len

    # Sample which episodes to sample sequences from
    rng_key, subkey = jax.random.split(buffer["rng_key"])
    episode_indices = jax.random.randint(subkey, (batch_size,), 0, num_episodes)

    # Choose start positions inside episodes (optional if seq_len == episode_length)
    max_start = episode_len - seq_len
    rng_key, subkey = jax.random.split(rng_key)
    start_in_episode = jax.random.randint(subkey, (batch_size,), 0, max_start + 1)

    # Convert episode index and start to flat indices
    flat_start_indices = episode_indices * episode_len + start_in_episode
    seq_offsets = jnp.arange(seq_len)
    seq_indices = flat_start_indices[:, None] + seq_offsets[None, :]

    # Gather sequences
    batch_states = buffer["state_buffer"][seq_indices]
    batch_actions = buffer["action_buffer"][seq_indices]
    batch_rewards = buffer["reward_buffer"][seq_indices]
    batch_terminateds = buffer["terminated_buffer"][seq_indices]
    batch_next_states = buffer["next_state_buffer"][seq_indices]

    updated_buffer = {
        "state_buffer": buffer["state_buffer"],
        "action_buffer": buffer["action_buffer"],
        "reward_buffer": buffer["reward_buffer"],
        "next_state_buffer": buffer["next_state_buffer"],
        "terminated_buffer": buffer["terminated_buffer"],
        "position": buffer["position"],
        "size": buffer["size"],
        "capacity": buffer["capacity"],
        "rng_key": rng_key,  # the rng key changes, that's why we return the updated buffer
    }

    return (
        batch_states,
        batch_actions,
        batch_next_states,
        batch_rewards,
        batch_terminateds,
    ), updated_buffer
