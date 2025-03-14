# Input weights generation methods
import numpy as np
from scipy.sparse import lil_matrix

# Input is [u_in, p, b_in] (input state, parameters, bias)


def random_sparse(W_in_shape, W_in_seeds, input_bias_len):
    """Create the input weights matrix
    Inputs are not connected and the weights are randomly placed one per row

    Args:
        W_in_shape: N_reservoir x (N_inputs + N_param_dim + N_input_bias)
        W_in_seeds: a list of seeds for the random generators;
            one for the column index, one for the uniform sampling
    Returns:
        W_in: sparse matrix containing the input weights
    """
    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    # set the seeds
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])

    # make W_in
    n_cols = W_in_shape[1] - input_bias_len
    for j in range(W_in_shape[0]):
        rnd_idx = rnd0.randint(0, n_cols)  # low inclusive, high exclusive
        # only one element different from zero
        # sample from the uniform distribution
        W_in[j, rnd_idx] = rnd1.uniform(-1, 1)

    # input bias is fully connected to the reservoir states
    if input_bias_len > 0:
        W_in[:, W_in_shape[1] - input_bias_len] = rnd1.uniform(
            -1, 1, (W_in_shape[0], input_bias_len)
        )

    W_in = W_in.tocsr()

    return W_in


def random_sparse_input_sparse_param(
    W_in_shape, N_param_dim, W_in_seeds, input_bias_len
):
    """Create the input weights matrix
    Inputs are not connected but the parameters/forcing are sparsely connected
    The weights are randomly placed one per row

    Args:
        W_in_shape: N_reservoir x (N_inputs + N_param_dim + N_input_bias)
        W_in_seeds: a list of seeds for the random generators;
            one for the column index, one for the uniform sampling
    Returns:
        W_in: sparse matrix containing the input weights
    """
    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    # set the seeds
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])
    rnd2 = np.random.RandomState(W_in_seeds[2])

    # make W_in
    n_cols = W_in_shape[1] - N_param_dim - input_bias_len
    for j in range(W_in_shape[0]):
        rnd_idx1 = rnd0.randint(0, n_cols)

        # only one element different from zero
        # sample from the uniform distribution
        W_in[j, rnd_idx1] = rnd1.uniform(-1, 1)

        if N_param_dim > 0:
            # input associated with parameters/forcing are
            # sparsely connected to the reservoir states
            rnd_idx2 = rnd0.randint(1, N_param_dim + 1)  # low inclusive, high exclusive
            W_in[j, W_in_shape[1] - input_bias_len - rnd_idx2] = rnd2.uniform(-1, 1)

    # input bias is fully connected to the reservoir states
    if input_bias_len > 0:
        W_in[:, W_in_shape[1] - input_bias_len] = rnd1.uniform(
            -1, 1, (W_in_shape[0], input_bias_len)
        )

    W_in = W_in.tocsr()

    return W_in


def random_sparse_input_dense_param(
    W_in_shape, N_param_dim, W_in_seeds, input_bias_len
):
    """Create the input weights matrix
    Inputs are not connected but the parameters/forcing are fully connected
    The weights are randomly placed one per row

    Args:
        W_in_shape: N_reservoir x (N_inputs + N_param_dim + N_input_bias)
        W_in_seeds: a list of seeds for the random generators;
            one for the column index, one for the uniform sampling
    Returns:
        W_in: sparse matrix containing the input weights
    """
    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    # set the seeds
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])
    rnd2 = np.random.RandomState(W_in_seeds[2])

    # make W_in
    n_cols = W_in_shape[1] - N_param_dim - input_bias_len
    for j in range(W_in_shape[0]):
        rnd_idx = rnd0.randint(0, n_cols)
        # only one element different from zero
        # sample from the uniform distribution
        W_in[j, rnd_idx] = rnd1.uniform(-1, 1)

    # input associated with parameters/forcing are
    # fully connected to the reservoir states
    if N_param_dim > 0:
        W_in[
            :,
            W_in_shape[1]
            - input_bias_len
            - N_param_dim : W_in_shape[1]
            - input_bias_len,
        ] = rnd2.uniform(-1, 1, (W_in_shape[0], N_param_dim))

    # input bias is fully connected to the reservoir states
    if input_bias_len > 0:
        W_in[:, W_in_shape[1] - input_bias_len] = rnd1.uniform(
            -1, 1, (W_in_shape[0], input_bias_len)
        )

    W_in = W_in.tocsr()

    return W_in


def grouped_sparse(W_in_shape, W_in_seeds, input_bias_len):
    # Sparse input matrix that is not connected to the parameter

    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    rnd0 = np.random.RandomState(W_in_seeds[0])

    n_cols = W_in_shape[1] - input_bias_len
    for i in range(W_in_shape[0]):
        W_in[
            i,
            int(np.floor(i * n_cols / W_in_shape[0])),
        ] = rnd0.uniform(-1, 1)

    # input bias is fully connected to the reservoir states
    if input_bias_len > 0:
        W_in[:, W_in_shape[1] - input_bias_len] = rnd0.uniform(
            -1, 1, (W_in_shape[0], input_bias_len)
        )
    W_in = W_in.tocsr()
    return W_in


def grouped_sparse_input_dense_param(
    W_in_shape, N_param_dim, W_in_seeds, input_bias_len
):
    # The inputs are not connected but they are grouped within the matrix

    # initialize W_in with zeros
    W_in = lil_matrix(W_in_shape)
    rnd0 = np.random.RandomState(W_in_seeds[0])
    rnd1 = np.random.RandomState(W_in_seeds[1])

    n_cols = W_in_shape[1] - N_param_dim - input_bias_len
    for i in range(W_in_shape[0]):
        W_in[
            i,
            int(np.floor(i * (n_cols) / W_in_shape[0])),
        ] = rnd0.uniform(-1, 1)

    # input associated with parameters/forcing are
    # fully connected to the reservoir states
    if N_param_dim > 0:
        W_in[
            :,
            W_in_shape[1]
            - input_bias_len
            - N_param_dim : W_in_shape[1]
            - input_bias_len,
        ] = rnd1.uniform(-1, 1, (W_in_shape[0], N_param_dim))

    # input bias is fully connected to the reservoir states
    if input_bias_len > 0:
        W_in[:, W_in_shape[1] - input_bias_len] = rnd1.uniform(
            -1, 1, (W_in_shape[0], input_bias_len)
        )

    W_in = W_in.tocsr()
    return W_in


def dense(W_in_shape, W_in_seeds):
    # The inputs are all connected

    rnd0 = np.random.RandomState(W_in_seeds[0])
    W_in = rnd0.uniform(-1, 1, W_in_shape)
    return W_in
