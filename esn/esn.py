from functools import partial

import numpy as np
from scipy.sparse import csr_matrix

import esn.generate_input_weights as generate_input_weights
import esn.generate_reservoir_weights as generate_reservoir_weights
from esn.utils.discretizations import finite_differences


class ESN:
    def __init__(
        self,
        reservoir_size,
        dimension,
        reservoir_connectivity=0,
        parameter_dimension=0,
        input_normalization=None,
        parameter_normalization=None,
        input_scaling=1.0,
        spectral_radius=1.0,
        leak_factor=1.0,
        input_bias=np.array([]),
        output_bias=np.array([]),
        input_seeds=[None, None, None],
        reservoir_seeds=[None, None],
        tikhonov=None,
        verbose=True,
        r2_mode=False,
        input_only_mode=False,
        input_weights_mode="sparse_grouped",
        reservoir_weights_mode="erdos_renyi2",
    ):
        """Creates an Echo State Network with the given parameters
        Args:
            reservoir_size: number of neurons in the reservoir
            dimension: dimension of the state space of the input and output
                they must have the same size in order for the closed-loop to work
            parameter_dimension: dimension of the system's bifurcation parameters
            reservoir_connectivity: connectivity of the reservoir weights,
                how many connections does each neuron have (on average)
            input_normalization: normalization applied to the input before activation
                tuple with (mean, norm) such that u is updated as (u-mean)/norm
            input_scaling: scaling applied to the input weights matrix
            spectral_radius: spectral radius (maximum absolute eigenvalue)
                of the reservoir weights matrix
            leak_factor: factor for the leaky integrator
                if set to 1 (default), then no leak is applied
            input_bias: bias that is augmented to the input vector
            input_seeds: seeds to generate input weights matrix
            reservoir_seeds: seeds to generate reservoir weights matrix
        Returns:
            ESN object

        """
        self.verbose = verbose
        self.r2_mode = r2_mode
        self.input_only_mode = input_only_mode

        ## Hyperparameters
        # these should be fixed during initialization and not changed since they affect
        # the matrix dimensions, and the matrices can become incompatible
        self.N_reservoir = reservoir_size
        self.N_dim = dimension
        self.N_param_dim = parameter_dimension

        self.leak_factor = leak_factor

        ## Biases
        self.input_bias = input_bias
        self.output_bias = output_bias

        ## Input normalization
        if not input_normalization:
            input_normalization = [None] * 2
            input_normalization[0] = np.zeros(self.N_dim)
            input_normalization[1] = np.ones(self.N_dim)

        self.input_normalization = input_normalization

        if not parameter_normalization:
            parameter_normalization = [None] * 2
            parameter_normalization[0] = np.zeros(self.N_param_dim)
            parameter_normalization[1] = np.ones(self.N_param_dim)

        self.parameter_normalization = parameter_normalization

        ## Weights
        # the object should also store the seeds for reproduction
        # initialise input weights
        self.W_in_seeds = input_seeds
        self.W_in_shape = (
            self.N_reservoir,
            self.N_dim + len(self.input_bias) + self.N_param_dim,
        )
        # N_dim+length of input bias because we augment the inputs with a bias
        # if no bias, then this will be + 0
        self.input_weights_mode = input_weights_mode
        self.input_weights = self.generate_input_weights()
        self.input_scaling = input_scaling
        # input weights are automatically scaled if input scaling is updated

        # initialise reservoir weights
        if not self.input_only_mode:
            self.reservoir_connectivity = reservoir_connectivity
            self.W_seeds = reservoir_seeds
            self.W_shape = (self.N_reservoir, self.N_reservoir)
            self.reservoir_weights_mode = reservoir_weights_mode
            valid_W = False
            while not valid_W:
                try:
                    self.reservoir_weights = self.generate_reservoir_weights()
                    valid_W = True
                except:
                    # perturb the seeds
                    for i in range(2):
                        if self.W_seeds[i]:
                            self.W_seeds[i] = self.W_seeds[i] + 1
                    valid_W = False
                    print("Not valid reservoir encountered, changing seed.")
            self.spectral_radius = spectral_radius
            # reservoir weights are automatically scaled if spectral radius is updated

        # tikhonov coefficient
        if tikhonov:
            self.tikhonov = tikhonov

        # initialise output weights
        self.W_out_shape = (self.N_reservoir + len(self.output_bias), self.N_dim)
        # N_reservoir+length of output bias because we augment the outputs with a bias
        # if no bias, then this will be + 0
        self.output_weights = np.zeros(self.W_out_shape)

    @property
    def reservoir_connectivity(self):
        return self.connectivity

    @reservoir_connectivity.setter
    def reservoir_connectivity(self, new_reservoir_connectivity):
        # set connectivity
        if new_reservoir_connectivity <= 0:
            raise ValueError("Connectivity must be greater than 0.")
        self.connectivity = new_reservoir_connectivity
        # regenerate reservoir with the new connectivity
        if hasattr(self, "W"):
            if self.verbose:
                print("Reservoir weights are regenerated for the new connectivity.")
            self.reservoir_weights = self.generate_reservoir_weights()
        return

    @property
    def leak_factor(self):
        return self.alpha

    @leak_factor.setter
    def leak_factor(self, new_leak_factor):
        # set leak factor
        if new_leak_factor < 0 or new_leak_factor > 1:
            raise ValueError("Leak factor must be between 0 and 1 (including).")
        self.alpha = new_leak_factor
        return

    @property
    def tikhonov(self):
        return self.tikh

    @tikhonov.setter
    def tikhonov(self, new_tikhonov):
        # set tikhonov coefficient
        if new_tikhonov <= 0:
            raise ValueError("Tikhonov coefficient must be greater than 0.")
        self.tikh = new_tikhonov
        return

    @property
    def input_normalization(self):
        return self.norm_in

    @input_normalization.setter
    def input_normalization(self, new_input_normalization):
        self.norm_in = new_input_normalization
        if self.verbose:
            print("Input normalization is changed, training must be done again.")

    @property
    def parameter_normalization_mean(self):
        return self.norm_p[0]

    @parameter_normalization_mean.setter
    def parameter_normalization_mean(self, new_parameter_normalization_mean):
        self.norm_p[0] = new_parameter_normalization_mean
        if self.verbose:
            print("Parameter normalization is changed, training must be done again.")

    @property
    def parameter_normalization_var(self):
        return self.norm_p[1]

    @parameter_normalization_var.setter
    def parameter_normalization_var(self, new_parameter_normalization_var):
        self.norm_p[1] = new_parameter_normalization_var
        if self.verbose:
            print("Parameter normalization is changed, training must be done again.")

    @property
    def parameter_normalization(self):
        return self.norm_p

    @parameter_normalization.setter
    def parameter_normalization(self, new_parameter_normalization):
        self.norm_p = new_parameter_normalization
        if self.verbose:
            print("Parameter normalization is changed, training must be done again.")

    @property
    def input_scaling(self):
        return self.sigma_in

    @input_scaling.setter
    def input_scaling(self, new_input_scaling):
        """Setter for the input scaling, if new input scaling is given,
        then the input weight matrix is also updated
        """
        if hasattr(self, "sigma_in"):
            # rescale the input matrix
            if self.N_param_dim > 0:
                self.W_in[:, : -self.N_param_dim] = (1 / self.sigma_in) * self.W_in[
                    :, : -self.N_param_dim
                ]
            else:
                self.W_in = (1 / self.sigma_in) * self.W_in

        # set input scaling
        self.sigma_in = new_input_scaling
        if self.verbose:
            print("Input weights are rescaled with the new input scaling.")
        if self.N_param_dim > 0:
            self.W_in[:, : -self.N_param_dim] = (
                self.sigma_in * self.W_in[:, : -self.N_param_dim]
            )
        else:
            self.W_in = self.sigma_in * self.W_in
        return

    @property
    def spectral_radius(self):
        return self.rho

    @spectral_radius.setter
    def spectral_radius(self, new_spectral_radius):
        """Setter for the spectral_radius, if new spectral_radius is given,
        then the reservoir weight matrix is also updated
        """
        if hasattr(self, "rho"):
            # rescale the reservoir matrix
            self.W = (1 / self.rho) * self.W
        # set spectral radius
        self.rho = new_spectral_radius
        if self.verbose:
            print("Reservoir weights are rescaled with the new spectral radius.")
        self.W = self.rho * self.W
        return

    @property
    def input_weights(self):
        return self.W_in

    @input_weights.setter
    def input_weights(self, new_input_weights):
        # first check the dimensions
        if new_input_weights.shape != self.W_in_shape:
            raise ValueError(
                f"The shape of the provided input weights does not match with the network, {new_input_weights.shape} != {self.W_in_shape}"
            )

        # set the new input weights
        self.W_in = new_input_weights

        # set the input scaling to 1.0
        if self.verbose:
            print("Input scaling is set to 1, set it separately if necessary.")
        self.sigma_in = 1.0
        return

    @property
    def reservoir_weights(self):
        return self.W

    @reservoir_weights.setter
    def reservoir_weights(self, new_reservoir_weights):
        # first check the dimensions
        if new_reservoir_weights.shape != self.W_shape:
            raise ValueError(
                f"The shape of the provided reservoir weights does not match with the network,"
                f"{new_reservoir_weights.shape} != {self.W_shape}"
            )

        # set the new reservoir weights
        self.W = new_reservoir_weights

        # set the spectral radius to 1.0
        if self.verbose:
            print("Spectral radius is set to 1, set it separately if necessary.")
        self.rho = 1.0
        return

    @property
    def output_weights(self):
        return self.W_out

    @output_weights.setter
    def output_weights(self, new_output_weights):
        # first check the dimensions
        if new_output_weights.shape != self.W_out_shape:
            raise ValueError(
                f"The shape of the provided output weights does not match with the network,"
                f"{new_output_weights.shape} != {self.W_out_shape}"
            )
        # set the new reservoir weights
        self.W_out = new_output_weights
        return

    @property
    def input_bias(self):
        return self.b_in

    @input_bias.setter
    def input_bias(self, new_input_bias):
        self.b_in = new_input_bias
        return

    @property
    def output_bias(self):
        return self.b_out

    @output_bias.setter
    def output_bias(self, new_output_bias):
        self.b_out = new_output_bias
        return

    @property
    def sparseness(self):
        """Define sparseness from connectivity"""
        # probability of non-connections = 1 - probability of connection
        # probability of connection = (number of connections)/(total number of neurons - 1)
        # -1 to exclude the neuron itself
        return 1 - (self.connectivity / (self.N_reservoir - 1))

    def generate_input_weights(self):
        if self.input_weights_mode == "random_sparse":
            return generate_input_weights.random_sparse(
                self.W_in_shape, self.W_in_seeds
            )
        elif self.input_weights_mode == "random_sparse_input_sparse_param":
            return generate_input_weights.random_sparse_input_sparse_param(
                self.W_in_shape, self.N_param_dim, self.W_in_seeds
            )
        elif self.input_weights_mode == "random_sparse_input_dense_param":
            return generate_input_weights.random_sparse_input_dense_param(
                self.W_in_shape, self.N_param_dim, self.W_in_seeds
            )
        elif self.input_weights_mode == "grouped_sparse":
            return generate_input_weights.grouped_sparse(
                self.W_in_shape, self.W_in_seeds
            )
        elif self.input_weights_mode == "grouped_sparse_input_dense_param":
            return generate_input_weights.grouped_sparse_input_dense_param(
                self.W_in_shape, self.N_param_dim, self.W_in_seeds
            )
        elif self.input_weights_mode == "dense":
            return generate_input_weights.dense(self.W_in_shape, self.W_in_seeds)
        else:
            raise ValueError("Not valid input weights generator.")

    def generate_reservoir_weights(self):
        if self.reservoir_weights_mode == "erdos_renyi1":
            return generate_reservoir_weights.erdos_renyi1(
                self.W_shape, self.sparseness, self.W_seeds
            )
        if self.reservoir_weights_mode == "erdos_renyi2":
            return generate_reservoir_weights.erdos_renyi2(
                self.W_shape, self.sparseness, self.W_seeds
            )
        else:
            raise ValueError("Not valid reservoir weights generator.")

    def step(self, x_prev, u, p=None):
        """Advances ESN time step.
        Args:
            x_prev: reservoir state in the previous time step (n-1)
            u: input in this time step (n)
            p: systems bifucation parameters vector
        Returns:
            x_next: reservoir state in this time step (n)
        """
        # normalise the input
        u_norm = (u - self.norm_in[0]) / self.norm_in[1]
        # we normalize here, so that the input is normalised
        # in closed-loop run too?

        # augment the input with the input bias
        u_augmented = np.hstack((u_norm, self.b_in))

        # augment the input with the parameters
        if self.N_param_dim > 0:
            u_augmented = np.hstack(
                (u_augmented, (p - self.norm_p[0]) * self.norm_p[1])
            )

        # update the reservoir
        if self.input_only_mode:
            x_tilde = np.tanh(self.W_in.dot(u_augmented))
        else:
            x_tilde = np.tanh(self.W_in.dot(u_augmented) + self.W.dot(x_prev))

        # apply the leaky integrator
        x = (1 - self.alpha) * x_prev + self.alpha * x_tilde
        return x

    def open_loop(self, x0, U, P=None):
        """Advances ESN in open-loop.
        Args:
            x0: initial reservoir state
            U: input time series in matrix form (N_t x N_dim)
            P: parameter time series (N_t x N_param_dim)
        Returns:
            X: time series of the reservoir states (N_t x N_reservoir)
        """
        N_t = U.shape[0]  # number of time steps

        # create an empty matrix to hold the reservoir states in time
        X = np.empty((N_t + 1, self.N_reservoir))
        # N_t+1 because at t = 0, we don't have input

        # initialise with the given initial reservoir states
        X[0, :] = x0
        # X = [x0]
        # step in time
        for n in np.arange(1, N_t + 1):
            # update the reservoir
            if self.N_param_dim > 0:
                X[n] = self.step(X[n - 1, :], U[n - 1, :], P[n - 1, :])
                # X.append(self.step(X[n - 1], U[n - 1], P[n - 1]))
            else:
                X[n] = self.step(X[n - 1, :], U[n - 1, :])
                # X.append(self.step(X[n - 1], U[n - 1]))
        # X = np.array(X)
        return X

    def before_readout_r1(self, x):
        # augment with bias before readout
        return np.hstack((x, self.b_out))

    def before_readout_r2(self, x):
        # replaces r with r^2 if even, r otherwise
        x2 = x.copy()
        x2[1::2] = x2[1::2] ** 2
        return np.hstack((x2, self.b_out))

    @property
    def before_readout(self):
        if not hasattr(self, "_before_readout"):
            if self.r2_mode:
                self._before_readout = self.before_readout_r2
            else:
                self._before_readout = self.before_readout_r1
        return self._before_readout

    def closed_loop(self, x0, N_t, P=None):
        # @todo: make it an option to hold X or just x in memory
        """Advances ESN in closed-loop.
        Args:
            N_t: number of time steps
            x0: initial reservoir state
            P: parameter time series (N_t x N_param_dim)
        Returns:
            X: time series of the reservoir states (N_t x N_reservoir)
            Y: time series of the output (N_t x N_dim)
        """
        # create an empty matrix to hold the reservoir states in time
        X = np.empty((N_t + 1, self.N_reservoir))
        # create an empty matrix to hold the output states in time
        Y = np.empty((N_t + 1, self.N_dim))

        # initialize with the given initial reservoir states
        X[0, :] = x0

        # augment the reservoir states with the bias
        x0_augmented = self.before_readout(x0)

        # initialise with the calculated output states
        Y[0, :] = np.dot(x0_augmented, self.W_out)

        # step in time
        for n in range(1, N_t + 1):
            # update the reservoir with the feedback from the output
            if self.N_param_dim > 0:
                X[n, :] = self.step(X[n - 1, :], Y[n - 1, :], P[n - 1, :])
            else:
                X[n, :] = self.step(X[n - 1, :], Y[n - 1, :])

            # augment the reservoir states with bias
            x_augmented = self.before_readout(X[n, :])

            # update the output with the reservoir states
            Y[n, :] = np.dot(x_augmented, self.W_out)
        return X, Y

    def run_washout(self, U_washout, P_washout=None):
        # Wash-out phase to get rid of the effects of reservoir states initialised as zero
        # initialise the reservoir states before washout
        x0_washout = np.zeros(self.N_reservoir)

        # let the ESN run in open-loop for the wash-out
        # get the initial reservoir to start the actual open/closed-loop,
        # which is the last reservoir state
        x0 = self.open_loop(x0=x0_washout, U=U_washout, P=P_washout)[-1, :]
        return x0

    def open_loop_with_washout(self, U_washout, U, P_washout=None, P=None):
        x0 = self.run_washout(U_washout, P_washout)
        X = self.open_loop(x0=x0, U=U, P=P)
        return X

    def closed_loop_with_washout(self, U_washout, N_t, P_washout=None, P=None):
        x0 = self.run_washout(U_washout, P_washout)
        X, Y = self.closed_loop(x0=x0, N_t=N_t, P=P)
        return X, Y

    def solve_ridge(X, Y, tikh):
        """Solves the ridge regression problem
        Args:
            X: input data (N_t x (N_reservoir + N_bias))
            Y: output data (N_t x N_dim)
            tikh: weighing tikhonov coefficient that regularises L2 norm
        Output: W_out of size ((N_reservoir+N_bias) x N_dim)
        """

        A = X.T @ X + tikh * np.eye(X.shape[1])
        b = X.T @ Y
        return np.linalg.solve(A, b)

    def reservoir_for_train(self, U_washout, U_train, P_washout=None, P_train=None):
        X_train = self.open_loop_with_washout(U_washout, U_train, P_washout, P_train)

        # X_train is one step longer than U_train and Y_train, we discard the initial state
        X_train = X_train[1:, :]

        # augment with the bias
        N_t = X_train.shape[0]  # number of time steps

        if self.r2_mode:
            X_train2 = X_train.copy()
            X_train2[:, 1::2] = X_train2[:, 1::2] ** 2
            X_train_augmented = np.hstack((X_train2, self.b_out * np.ones((N_t, 1))))
        else:
            X_train_augmented = np.hstack((X_train, self.b_out * np.ones((N_t, 1))))

        return X_train_augmented

    def train(
        self,
        U_washout,
        U_train,
        Y_train,
        P_washout=None,
        P_train=None,
        tikhonov=1e-12,
        train_idx_list=None,
    ):
        """Trains ESN and sets the output weights.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            Y_train: training output time series
            (list of time series if more than one trajectories)
            P_washout: parameters in washout
            P_train: parameters in training
            tikhonov: regularization coefficient
            train_idx_list: if list of time series, then which ones to use in training
                if not specified, all are used
        """
        # get the training input
        # this is the reservoir states augmented with the bias after a washout phase
        if isinstance(U_train, list):
            X_train_augmented = np.empty((0, self.W_out_shape[0]))
            if train_idx_list is None:
                train_idx_list = range(len(U_train))
            for train_idx in train_idx_list:
                X_train_augmented_ = self.reservoir_for_train(
                    U_washout[train_idx],
                    U_train[train_idx],
                    P_washout[train_idx],
                    P_train[train_idx],
                )
                X_train_augmented = np.vstack((X_train_augmented, X_train_augmented_))

            Y_train = [Y_train[train_idx] for train_idx in train_idx_list]
            Y_train = np.vstack(Y_train)
        else:
            X_train_augmented = self.reservoir_for_train(
                U_washout, U_train, P_washout, P_train
            )

        # solve for W_out using ridge regression
        if not self.tikhonov:
            self.tikhonov = tikhonov  # set the tikhonov during training

        self.output_weights = ESN.solve_ridge(X_train_augmented, Y_train, self.tikhonov)
        return

    @property
    def dfdu_const(self):
        # constant part of gradient of x(i+1) with respect to u_in(i)
        # sparse matrix
        if not hasattr(self, "_dfdu_const"):
            try:
                self._dfdu_const = self.alpha * self.W_in[:, : self.N_dim].multiply(
                    1.0 / self.norm_in[1][: self.N_dim]
                )
            except:
                self._dfdu_const = self.alpha * np.multiply(
                    self.W_in[:, : self.N_dim], 1.0 / self.norm_in[1][: self.N_dim]
                )
        return self._dfdu_const

    @property
    def dudx_const(self):
        # gradient of u_in(i) with respect to x(i)
        # not sparse matrix
        if not hasattr(self, "_dudx_const"):
            self._dudx_const = self.W_out[: self.N_reservoir, :].T
        return self._dudx_const

    @property
    def dfdu_dudx_const(self):
        # constant part of gradient of x(i+1) with respect to x(i) due to u_in(i)
        # not sparse matrix
        if not hasattr(self, "_dfdu_dudx_const"):
            self._dfdu_dudx_const = self.dfdu_const.dot(self.dudx_const)
        return self._dfdu_dudx_const

    @property
    def dfdx_x_const(self):
        # constant part of gradient of x(i+1) with respect to x(i) due to x(i)
        # sparse matrix
        if not hasattr(self, "_dfdx_x_const"):
            self._dfdx_x_const = csr_matrix((1 - self.alpha) * np.eye(self.N_reservoir))
        return self._dfdx_x_const

    def dtanh(self, x, x_prev):
        """Derivative of the tanh part
        This derivative appears in different gradient calculations
        So, it makes sense to calculate it only once and call as required

        Args:
        x: reservoir states at time i+1, x(i+1)
        x_prev: reservoir states at time i, x(i)
        """
        # first we find tanh(...)
        x_tilde = (x - (1 - self.alpha) * x_prev) / self.alpha

        # derivative of tanh(...) is 1-tanh**2(...)
        dtanh = 1.0 - x_tilde**2

        return dtanh

    def dfdx_u_r1(self, dtanh, x_prev=None):
        return np.multiply(self.dfdu_dudx_const, dtanh)

    def dfdx_u_r2(self, dtanh, x_prev=None):
        # derivative of x**2 terms
        dx_prev = np.ones(self.N_reservoir)
        dx_prev[1::2] = 2 * x_prev[1::2]

        dudx = np.multiply(self.dudx_const, dx_prev)
        dfdu_dudx = self.dfdu_const.dot(dudx)
        return np.multiply(dfdu_dudx, dtanh)

    @property
    def dfdx_u(self):
        if not hasattr(self, "_dfdx_u"):
            if self.r2_mode:
                self._dfdx_u = self.dfdx_u_r2
            else:
                self._dfdx_u = self.dfdx_u_r1
        return self._dfdx_u

    def jac(self, dtanh, x_prev=None):
        """Jacobian of the reservoir states, ESN in closed loop
        taken from
        Georgios Margazoglou, Luca Magri:
        Stability analysis of chaotic systems from data, arXiv preprint arXiv:2210.06167

        x(i+1) = f(x(i),u(i),p)
        df(x(i),u(i))/dx(i) = \partial f/\partial x(i) + \partial f/\partial u(i)*\partial u(i)/\partial x(i)

        x(i+1) = (1-alpha)*x(i)+alpha*tanh(W_in*[u(i);p]+W*x(i))

        Args:
        dtanh: derivative of tanh at x(i+1), x(i)

        Returns:
        dfdx: jacobian of the reservoir states
        """
        # gradient of x(i+1) with x(i) due to u(i) (in closed-loop)
        dfdx_u = self.dfdx_u(dtanh, x_prev)

        # gradient of x(i+1) with x(i) due to x(i) that appears explicitly
        # no reservoir connections
        dfdx_x = self.dfdx_x_const
        if not self.input_only_mode:
            dfdx_x += self.alpha * self.W.multiply(dtanh)

        # total derivative
        dfdx = dfdx_x.toarray() + dfdx_u
        return dfdx

    @property
    def dfdx_const(self):
        if not hasattr(self, "_dfdx_const"):
            self._dfdx_const = self.dfdu_dudx_const + self.alpha * self.W.toarray()
        return self._dfdx_const

    def fast_jac(self, dtanh):
        """Faster implementation of the Jacobian for the basic ESN
        Only valid when input_only_mode = False and r2_mode = False
        """
        dfdx = np.multiply(self.dfdx_const, dtanh)
        dfdx.ravel()[:: dfdx.shape[1] + 1] += 1 - self.alpha
        return dfdx

    @property
    def dfdp_const(self):
        # constant part of gradient of x(i+1) with respect to p
        if not hasattr(self, "_dfdp_const"):
            try:
                self._dfdp_const = self.alpha * self.W_in[
                    :, -self.N_param_dim :
                ].multiply(1.0 * self.norm_p[1])
                self._dfdp_const = self._dfdp_const.toarray()
            except:
                self._dfdp_const = self.alpha * np.multiply(
                    self.W_in[:, -self.N_param_dim :], 1.0 * self.norm_p[1]
                )
        return self._dfdp_const

    def dfdp(self, dtanh):
        """Jacobian of the reservoir states with respect to the parameters
        \partial x(i+1) / \partial p

        x(i+1) = f(x(i),u(i),p)
        x(i+1) = (1-alpha)*x(i)+alpha*tanh(W_in*[u(i);p]+W*x(i))

        Args:
        dtanh: derivative of tanh at x(i+1), x(i)

        Returns:
        dfdp: csr_matrix
        """
        # gradient of x(i+1) with respect to p
        dfdp = np.multiply(self.dfdp_const, dtanh)
        return dfdp

    def dydf_r1(self, x=None):
        return self.W_out[: self.N_reservoir, :].T

    def dydf_r2(self, x):
        # derivative of x**2 terms
        df = np.ones(self.N_reservoir)
        df[1::2] = 2 * x[1::2]
        return np.multiply(self.W_out[: self.N_reservoir, :].T, df)

    @property
    def dydf(self):
        # gradient of output galerkin amplitudes with respect to
        # reservoir states
        if not hasattr(self, "_dydf"):
            if self.r2_mode:
                self._dydf = self.dydf_r2
            else:
                self._dydf = self.dydf_r1
        return self._dydf

    def reset_grad_attrs(self):
        # reset the attributes
        attr_list = [
            "_dfdu_const",
            "_dudx_const",
            "_dfdu_dudx_const",
            "_dfdx_x_const",
            "_dfdx_u",
            "_dfdp_const",
            "_dydf",
            "_dfdx_const",
        ]
        for attr in attr_list:
            if hasattr(self, attr):
                delattr(self, attr)

    def direct_sensitivity(self, X, Y, N, dJdy_fun=None, fast_jac=False):
        """Sensitivity of the ESN with respect to the parameters
        Calculated using DIRECT method
        Objective is some time-averaged function of the output

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            P: parameter
            N: number of steps

        Returns:
            dJdp: adjoint sensitivity to parameters
        """
        # reset grad attributes
        self.reset_grad_attrs()

        # choose fast jacobian
        if fast_jac == True:
            jac_fun = lambda dtanh, x_prev: self.fast_jac(dtanh)
        else:
            jac_fun = lambda dtanh, x_prev: self.jac(dtanh, x_prev)

        # initialize direct variables, dx(i+1)/dp
        # dJ_dp doesn't depend on the initial reservoir state, i.e. q[0] = 0
        q = np.zeros((N + 1, self.N_reservoir, self.N_param_dim))

        # initialize sensitivity,
        dJdp = np.zeros(self.N_param_dim)

        for i in np.arange(1, N + 1):
            dtanh = self.dtanh(X[i, :], X[i - 1, :])[:, None]

            # partial derivative with respect to parameters
            dfdp = self.dfdp(dtanh)

            # jacobian of the reservoir dynamics
            jac = jac_fun(dtanh, X[i - 1, :])

            # integrate direct variables forwards in time
            q[i] = dfdp + np.dot(jac, q[i - 1])

            # get objective with respect to output states
            dJdy = dJdy_fun(Y[i])

            # gradient of objective with respect to reservoir states
            dydf = self.dydf(X[i, :])
            dJdf = (1 / N) * np.dot(dJdy, dydf)

            # sensitivity to parameters
            dJdp += np.dot(dJdf, q[i])

        return dJdp

    def direct_sensitivity_x0(self, X, Y, N, dJdy_fun=None, fast_jac=False):
        """Sensitivity of the ESN with respect to the initial reservoir state x0
        Calculated using DIRECT method
        Objective is some function of the terminal output

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            Y:
            N: number of steps
            dJdy_fun: derivative of objective function

        Returns:
            dJdx0: direct sensitivity to initial reservoir state
        """
        # reset grad attributes
        self.reset_grad_attrs()

        # choose fast jacobian
        if fast_jac == True:
            jac_fun = lambda dtanh, x_prev: self.fast_jac(dtanh)
        else:
            jac_fun = lambda dtanh, x_prev: self.jac(dtanh, x_prev)

        # initialize direct variables, dx(i+1)/dx(0)
        q = np.zeros((N + 1, self.N_reservoir, self.N_reservoir))
        q[0] = np.eye(self.N_reservoir)

        for i in np.arange(1, N + 1):
            dtanh = self.dtanh(X[i, :], X[i - 1, :])[:, None]

            # jacobian of the reservoir dynamics
            jac = jac_fun(dtanh, X[i - 1, :])

            # integrate direct variables forwards in time
            q[i] = np.dot(jac, q[i - 1])

        # get objective with respect to terminal output states
        dJdy = dJdy_fun(Y[-1])

        # gradient of objective with respect to reservoir states
        dydf = self.dydf(X[-1, :])
        dJdf = np.dot(dJdy, dydf)

        # sensitivity to initial reservoir state
        dJdx0 = np.dot(dJdf, q[-1])

        return dJdx0

    def direct_sensitivity_u0(self, X, Y, N, dJdy_fun=None, fast_jac=False):
        """Sensitivity of the ESN with respect to the initial input u0
        Calculated using DIRECT method
        Objective is some function of the terminal output

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            Y:
            N: number of steps
            dJdy_fun: derivative of objective function

        Returns:
            dJdu0: direct sensitivity to the initial input u0
        """
        # reset grad attributes
        self.reset_grad_attrs()

        # choose fast jacobian
        if fast_jac == True:
            jac_fun = lambda dtanh, x_prev: self.fast_jac(dtanh)
        else:
            jac_fun = lambda dtanh, x_prev: self.jac(dtanh, x_prev)

        # initialize direct variables, dx(i+1)/du(0)
        # dJ_dp doesn't depend on the initial reservoir state, i.e. q[0] = 0
        dtanh = self.dtanh(X[1, :], X[0, :])[:, None]
        dfdu0 = np.multiply(self.dfdu_const.toarray(), dtanh)
        q = np.zeros((N + 1, self.N_reservoir, self.N_dim))
        q[1] = dfdu0

        for i in np.arange(2, N + 1):
            dtanh = self.dtanh(X[i, :], X[i - 1, :])[:, None]

            # jacobian of the reservoir dynamics
            jac = jac_fun(dtanh, X[i - 1, :])

            # integrate direct variables forwards in time
            q[i] = np.dot(jac, q[i - 1])

        # get objective with respect to output states
        dJdy = dJdy_fun(Y[-1])

        # gradient of objective with respect to reservoir states
        dydf = self.dydf(X[-1, :])
        dJdf = np.dot(dJdy, dydf)

        # sensitivity to parameters
        dJdu0 = np.dot(dJdf, q[-1])

        return dJdu0

    def adjoint_sensitivity(self, X, Y, N, dJdy_fun=None, fast_jac=False):
        """Sensitivity of the ESN with respect to the parameters
        Calculated using ADJOINT method
        Objective is some time-averaged function of the output

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            P: parameter
            N: number of steps

        Returns:
            dJdp: adjoint sensitivity to parameters
        """
        # reset grad attributes
        self.reset_grad_attrs()

        # choose fast jacobian
        if fast_jac == True:
            jac_fun = lambda dtanh, x_prev: self.fast_jac(dtanh)
        else:
            jac_fun = lambda dtanh, x_prev: self.jac(dtanh, x_prev)

        # initialize adjoint variables
        v = np.zeros((N + 1, self.N_reservoir))

        # initialize sensitivity
        dJdp = np.zeros(self.N_param_dim)

        # integrate backwards
        # predict galerkin amplitudes
        dJdy = dJdy_fun(Y[N])

        # terminal condition,
        # i.e. gradient of the objective at the terminal state
        dydf = self.dydf(X[N, :])
        v[N] = (1 / N) * np.dot(dJdy, dydf).T

        for i in np.arange(N, 0, -1):
            dtanh = self.dtanh(X[i, :], X[i - 1, :])[:, None]

            # partial derivative with respect to parameters
            dfdp = self.dfdp(dtanh)

            # sensitivity to parameters
            dJdp += np.dot(v[i], dfdp)

            # get the derivative of the objective with respect to the outputs
            dJdy = dJdy_fun(Y[i - 1])

            # gradient of objective with respect to reservoir states
            dydf = self.dydf(X[i - 1, :])
            dJdf = (1 / N) * np.dot(dJdy, dydf).T

            # jacobian of the reservoir dynamics
            jac = jac_fun(dtanh, X[i - 1, :])

            # integrate adjoint variables backwards in time
            v[i - 1] = np.dot(jac.T, v[i]) + dJdf

        return dJdp

    def adjoint_sensitivity_x0(self, X, Y, N, dJdy_fun=None, fast_jac=False):
        """Sensitivity of the ESN with respect to the initial reservoir state x0
        Calculated using ADJOINT method
        Objective is some function of the terminal output

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            Y:
            N: number of steps
            dJdy_fun: derivative of objective function

        Returns:
            dJdx0: adjoint sensitivity to initial reservoir state
        """
        # reset grad attributes
        self.reset_grad_attrs()

        # choose fast jacobian
        if fast_jac == True:
            jac_fun = lambda dtanh, x_prev: self.fast_jac(dtanh)
        else:
            jac_fun = lambda dtanh, x_prev: self.jac(dtanh, x_prev)

        # initialize adjoint variables
        v = np.zeros((N + 1, self.N_reservoir))

        # integrate backwards
        dJdy = dJdy_fun(Y[N])

        # terminal condition,
        # i.e. gradient of the objective at the terminal state
        dydf = self.dydf(X[N, :])
        v[N] = np.dot(dJdy, dydf).T

        for i in np.arange(N, 0, -1):
            dtanh = self.dtanh(X[i, :], X[i - 1, :])[:, None]

            # jacobian of the reservoir dynamics
            jac = jac_fun(dtanh, X[i - 1, :])

            # integrate adjoint variables backwards in time
            v[i - 1] = np.dot(jac.T, v[i])

        dJdx0 = v[0]
        return dJdx0

    def adjoint_sensitivity_u0(self, X, Y, N, dJdy_fun=None, fast_jac=False):
        """Sensitivity of the ESN with respect to the initial input u0
        Calculated using ADJOINT method
        Objective is some function of the terminal output

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            Y:
            N: number of steps
            dJdy_fun: derivative of objective function

        Returns:
            dJdu0: adjoint sensitivity to the initial input u0
        """
        # reset grad attributes
        self.reset_grad_attrs()

        # choose fast jacobian
        if fast_jac == True:
            jac_fun = lambda dtanh, x_prev: self.fast_jac(dtanh)
        else:
            jac_fun = lambda dtanh, x_prev: self.jac(dtanh, x_prev)

        # initialize adjoint variables
        v = np.zeros((N + 1, self.N_reservoir))

        # integrate backwards
        dJdy = dJdy_fun(Y[N])

        # terminal condition,
        # i.e. gradient of the objective at the terminal state
        dydf = self.dydf(X[N, :])
        v[N] = np.dot(dJdy, dydf).T

        for i in np.arange(N, 0, -1):
            dtanh = self.dtanh(X[i, :], X[i - 1, :])[:, None]

            # jacobian of the reservoir dynamics
            jac = jac_fun(dtanh, X[i - 1, :])

            # integrate adjoint variables backwards in time
            v[i - 1] = np.dot(jac.T, v[i])

        dfdu0 = np.multiply(self.dfdu_const.toarray(), dtanh)
        dJdu0 = np.dot(v[1], dfdu0)
        return dJdu0

    def finite_difference_sensitivity(
        self,
        X,
        Y,
        P,
        N,
        h=1e-5,
        method="central",
        J_fun=None,
    ):
        """Sensitivity of the ESN with respect to the parameters
        Calculated using CENTRAL FINITE DIFFERENCES
        Objective is some time-averaged function of the output

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            P: parameter
            N: number of steps
            h: perturbation

        Returns:
            dJdp: numerical sensitivity to parameters
        """
        # initialize sensitivity
        dJdp = np.zeros((self.N_param_dim))

        # compute the energy of the base
        J = J_fun(Y[1:])

        # define which finite difference method to use
        finite_difference = partial(finite_differences, method=method)

        # perturbed by h
        for i in range(self.N_param_dim):
            P_left = P.copy()
            P_left[:, i] -= h
            P_right = P.copy()
            P_right[:, i] += h
            _, Y_left = self.closed_loop(X[0, :], N, P_left)
            _, Y_right = self.closed_loop(X[0, :], N, P_right)
            J_left = J_fun(Y_left[1:])
            J_right = J_fun(Y_right[1:])

            dJdp[i] = finite_difference(J, J_right, J_left, h)

        return dJdp

    def finite_difference_sensitivity_x0(
        self, X, Y, P, N, h=1e-5, method="central", J_fun=None
    ):
        """Sensitivity of the ESN with respect to the initial reservoir state x0
        Calculated using CENTRAL FINITE DIFFERENCES
        Objective is some function of the terminal output

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            Y:
            P: parameter
            N: number of steps
            h: perturbation

        Returns:
            dJdx0: numerical sensitivity to initial reservoir state
        """
        # initialize sensitivity
        dJdx0 = np.zeros((self.N_reservoir))

        # compute the energy of the base
        J = J_fun(Y[-1, :])

        # define which finite difference method to use
        finite_difference = partial(finite_differences, method=method)

        # perturbed by h
        for i in range(self.N_reservoir):
            x0_left = X[0, :].copy()
            x0_left[i] -= h
            x0_right = X[0, :].copy()
            x0_right[i] += h
            _, Y_left = self.closed_loop(x0_left, N, P)
            _, Y_right = self.closed_loop(x0_right, N, P)
            J_left = J_fun(Y_left[-1, :])
            J_right = J_fun(Y_right[-1, :])

            dJdx0[i] = finite_difference(J, J_right, J_left, h)

        return dJdx0

    def finite_difference_sensitivity_u0(
        self, X, Y, P, N, h=1e-5, method="central", J_fun=None
    ):
        """Sensitivity of the ESN with respect to the initial input u0
        Calculated using CENTRAL FINITE DIFFERENCES
        Objective is some function of the terminal output

        Args:
            X: trajectory of reservoir states around which we find the sensitivity
            P: parameter
            N: number of steps
            h: perturbation

        Returns:
            dJdu0: numerical sensitivity to the initial input u0
        """
        # initialize sensitivity
        dJdu0 = np.zeros((self.N_dim))

        # compute the energy of the base
        J = J_fun(Y[-1, :])

        # define which finite difference method to use
        finite_difference = partial(finite_differences, method=method)
        # perturbed by h
        for i in range(self.N_dim):
            y0_left = Y[0, :].copy()
            y0_left[i] -= h
            y0_right = Y[0, :].copy()
            y0_right[i] += h

            x1_left = self.step(X[0, :], y0_left, P[0, :])
            x1_right = self.step(X[0, :], y0_right, P[0, :])
            _, Y_left = self.closed_loop(x1_left, N - 1, P)
            _, Y_right = self.closed_loop(x1_right, N - 1, P)

            J_left = J_fun(Y_left[-1, :])
            J_right = J_fun(Y_right[-1, :])

            dJdu0[i] = finite_difference(J, J_right, J_left, h)

        return dJdu0
