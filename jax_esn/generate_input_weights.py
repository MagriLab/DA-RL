# Input weights generation methods
import jax
import jax.numpy as jnp


def random_sparse(W_in_shape, W_in_seed):
    """
    Create the input weights matrix.
      Inputs are not connected and the weights are randomly placed one per row.

    Args:
        W_in_shape : (N_reservoir, N_inputs + N_input_bias + N_param_dim)
        W_in_seed: Seed for the random generator

    Returns:
        W_in: Sparse matrix containing the input weights
    """

    W_in = jnp.zeros(W_in_shape)
    key = jax.random.PRNGKey(W_in_seed)
    rnd0, rnd1 = jax.random.split(key, 2)

    # only one element different from zero sample from the uniform distribution
    rnd_idx = jax.random.randint(
        rnd0, shape=(W_in.shape[0],), minval=0, maxval=W_in.shape[1]
    )
    row_idx = jnp.arange(0, W_in.shape[0])
    uniform_values = jax.random.uniform(
        rnd1, shape=(W_in.shape[0],), minval=-1.0, maxval=1.0
    )

    # Set values in the matrix at the specified indices
    W_in = W_in.at[row_idx, rnd_idx].set(uniform_values)
    return W_in


def random_sparse_input_sparse_param(W_in_shape, N_param_dim, W_in_seed):
    """
    Create the input weights matrix using JAX.
    Inputs are not connected, but the parameters/forcing are sparsely connected.
    The weights are randomly placed one per row.

    Args:
        W_in_shape: (N_reservoir, N_inputs + N_input_bias + N_param_dim).
        N_param_dim: Number of parameter dimensions.
        W_in_seed: Seed for the random generator

    Returns:
        W_in: Sparse matrix containing the input weights.
    """
    N_reservoir, N_cols = W_in_shape

    # Initialize random key and split for reproducibility
    key = jax.random.PRNGKey(W_in_seed)
    key_idx1, key_vals1, key_idx2, key_vals2 = jax.random.split(key, 4)

    # Initialize weight matrix
    W_in = jnp.zeros(W_in_shape)

    # Random indices for nonzero elements in each row
    rnd_idx1 = jax.random.randint(key_idx1, (N_reservoir,), 0, N_cols - N_param_dim)
    rnd_vals1 = jax.random.uniform(key_vals1, (N_reservoir,), minval=-1, maxval=1)

    # Place the random values in the matrix
    W_in = W_in.at[jnp.arange(N_reservoir), rnd_idx1].set(rnd_vals1)

    if N_param_dim > 0:
        rnd_idx2 = jax.random.randint(key_idx2, (N_reservoir,), 1, N_param_dim + 1)
        rnd_vals2 = jax.random.uniform(key_vals2, (N_reservoir,), minval=-1, maxval=1)

        # Place parameter-related random values in the matrix
        W_in = W_in.at[jnp.arange(N_reservoir), N_cols - rnd_idx2].set(rnd_vals2)

    return W_in


def random_sparse_input_dense_param(W_in_shape, N_param_dim, W_in_seed):
    """
    Create the input weights matrix using JAX.
    Inputs are not connected, but the parameters/forcing are fully connected.
    The weights are randomly placed one per row.

    Args:
        W_in_shape: Tuple (N_reservoir, N_inputs + N_input_bias + N_param_dim).
        N_param_dim: Number of parameter dimensions.
        W_in_seed: Single seed for the random generators.
    Returns:
        W_in: Dense matrix containing the input weights.
    """
    N_reservoir, N_cols = W_in_shape

    # Initialize random key and split for reproducibility
    key = jax.random.PRNGKey(W_in_seed)
    key_idx, key_vals1, key_vals2 = jax.random.split(key, 3)

    # Initialize dense weight matrix
    W_in = jnp.zeros(W_in_shape)

    # Random indices for nonzero elements in each row
    rnd_idx = jax.random.randint(key_idx, (N_reservoir,), 0, N_cols - N_param_dim)
    rnd_vals = jax.random.uniform(key_vals1, (N_reservoir,), minval=-1, maxval=1)

    # Place the random values in the dense matrix
    W_in = W_in.at[jnp.arange(N_reservoir), rnd_idx].set(rnd_vals)

    # Fully connect the parameter-related columns
    if N_param_dim > 0:
        param_vals = jax.random.uniform(
            key_vals2, (N_reservoir, N_param_dim), minval=-1, maxval=1
        )
        W_in = W_in.at[:, -N_param_dim:].set(param_vals)

    return W_in


def grouped_sparse(W_in_shape, W_in_seed):
    """
    Create a grouped input weights matrix.
    The inputs are not connected, but they are grouped within the matrix.

    Args:
        W_in_shape: (N_reservoir, N_inputs + N_input_bias + N_param_dim)
        W_in_seed: Seed for the random generator
    """

    W_in = jnp.zeros(W_in_shape)
    key = jax.random.PRNGKey(W_in_seed)
    rnd0 = jax.random.split(key, 1)

    # Generate row and column indices
    row_idx = jnp.arange(0, W_in.shape[0])
    column_idx = jnp.floor(row_idx * (W_in_shape[1]) / W_in_shape[0]).astype(int)
    uniform_values = jax.random.uniform(
        rnd0, shape=(W_in.shape[0],), minval=-1.0, maxval=1.0
    )

    # Set values in the matrix at the specified indices
    W_in = W_in.at[row_idx, column_idx].set(uniform_values)
    return W_in


def grouped_sparse_input_dense_param(W_in_shape, N_param_dim, W_in_seed):
    """
    Create the input weights matrix using JAX.
    Inputs are grouped within the matrix, and parameters/forcing are fully connected.

    Args:
        W_in_shape: Tuple (N_reservoir, N_inputs + N_input_bias + N_param_dim).
        N_param_dim: Number of parameter dimensions.
        W_in_seed: Single seed for the random generators.
    Returns:
        W_in: Dense matrix containing the input weights.
    """
    N_reservoir, N_cols = W_in_shape

    # Initialize random key and split for reproducibility
    key = jax.random.PRNGKey(W_in_seed)
    key_uniform, key_params = jax.random.split(key, 2)

    # Initialize dense weight matrix
    W_in = jnp.zeros(W_in_shape)

    # Assign grouped indices for nonzero elements in each row
    group_indices = jnp.floor(
        jnp.arange(N_reservoir) * (N_cols - N_param_dim) / N_reservoir
    ).astype(int)
    group_vals = jax.random.uniform(key_uniform, (N_reservoir,), minval=-1, maxval=1)

    # Place grouped random values in the dense matrix
    W_in = W_in.at[jnp.arange(N_reservoir), group_indices].set(group_vals)

    # Fully connect the parameter-related columns
    if N_param_dim > 0:
        param_vals = jax.random.uniform(
            key_params, (N_reservoir, N_param_dim), minval=-1, maxval=1
        )
        W_in = W_in.at[:, -N_param_dim:].set(param_vals)

    return W_in


def dense(W_in_shape: tuple, W_in_seeds: list):
    """
    Create a dense input weights matrix. All inputs are connected.

    Args:
        W_in_shape: (N_reservoir, N_inputs + N_input_bias + N_param_dim)
        W_in_seed: Seed for the random generator

    Returns:
        jnp.ndarray: Dense matrix containing the input weights
    """

    # Generate random matrix with uniform values
    rnd_key = jax.random.split(jax.random.PRNGKey(W_in_seeds[0]), 1)
    W_in = jax.random.uniform(rnd_key, shape=W_in_shape, minval=-1.0, maxval=1.0)

    return W_in
