# author: Yaxin Mo, https://github.com/MagriLab/MD-CNN-AE/blob/main/MD_AE_tools/mode_decomposition.py
# contains POD and DMD
from time import time
import numpy as np
import sys
import typing

StrOrArray = typing.Union[str, np.array]


# Performs DMD
# reference: Brunton, S. L. & Kutz, J. N. (2019) Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control . 1st edition. Cambridge, UK, Cambridge Univeristy Press. Chapter 7.
class DMD:
    # Inputs:
    # X: [x0, x1 ... xn-1], X can have any shape, but the last dimension must be time
    # Xprime: [x1, x2, ... xn], has the same number of snapshots as X
    # r: how many modes to keep. r <= rank(Sigma), r<=n
    # keep_shape: if true, the returned matrix 'self.modes' will have the same spatial dimension as the data matrix X
    #       example X has shape (2,3,2,10), then self.modes will have shape (2,3,2,modes)

    # Attributes
    # self.r: how many modes to keep
    # self.keep_shape: keep_shape
    # self.grid_shape: the shape of the grid
    # self.modes: DMD modes, the last dimension is time
    # self.lam: eigenvalues
    # self.b: amplitude

    # Methods
    # get_modes(): returns modes, lambda and amplitude
    # get_frequency(dt=1): dt is the interval between snapshots. Returns the rate of growth/decay and the corresponding physical frequency.
    # get_amplitude(): returns |b|, which is the contribution of each mode
    def __init__(self, X: np.array, Xprime: np.array, r: int, keep_shape=False):
        self.r = r
        self.keep_shape = keep_shape

        # prepare data
        nx, nt, self.grid_shape, Q, Qprime = self.prepare_data(X, Xprime)
        print("Calculating DMD...")
        self.modes, self.lam, self.b = self.dmd(Q, Qprime, self.r)
        print("DMD done.")

        if keep_shape:
            self.restore_shape()

    def prepare_data(self, X, Xprime):
        Q = np.copy(X)
        Qprime = np.copy(Xprime)
        grid_shape = list(Q.shape[:-1])
        nt = Q.shape[-1]
        Q = np.reshape(Q, (-1, nt))
        Qprime = np.reshape(Qprime, (-1, nt))
        nx = Q.shape[0]
        return nx, nt, grid_shape, Q, Qprime

    def dmd(self, Q, Qprime, r):
        U, Sigma, VT = np.linalg.svd(Q, full_matrices=0)  # Step 1
        Ur = U[:, :r]
        Sigmar = np.diag(Sigma[:r])
        VTr = VT[:r, :]
        Atilde = np.linalg.solve(Sigmar.T, (Ur.T @ Qprime @ VTr.T).T).T  # Step 2
        Lambda, W = np.linalg.eig(Atilde)  # Step 3
        Lambda = np.diag(Lambda)

        Phi = Qprime @ np.linalg.solve(Sigmar.T, VTr).T @ W  # Step 4
        alpha1 = Sigmar @ VTr[:, 0]
        b = np.linalg.solve(W @ Lambda, alpha1)

        # sort modes based on their contribution ('amplitude')
        idx = np.argsort(np.abs(b))
        idx = np.flip(idx)
        b = b[idx]
        Lambda = np.diag(np.diag(Lambda)[idx])
        Phi = Phi[:, idx]

        return Phi, Lambda, b

    def restore_shape(self):
        original_shape = self.grid_shape.copy()
        original_shape.extend([-1])
        self.modes = np.reshape(self.modes, original_shape)

    def get_modes(self):
        return self.modes, self.lam, self.b

    def get_frequency(self, dt=1.0):
        lam = np.diag(self.lam)
        lam_r = np.abs(lam)  # rate of decay/growth
        lam_f = np.angle(lam) / (2 * np.pi) / dt  # physical frequency
        return lam_r, lam_f

    def get_amplitude(self):
        return np.abs(self.b)


# calculate POD modes, contains a reconstruction function
# Originally a matlab script
class POD:
    """Performs POD for a data matrix.

    Initiation this class will calculate POD for the data matrix.

    Methods:
        reconstruct: reconstruct the data with the specified number of POD modes.
        get_modes: returns modes and eigenvalues.
        get_time_coefficient: returns the time coefficients
    """

    # Attributes:
    # self.nt: number of snapshots
    # self.Q: reshaped X', has shape [nx,nt]
    # self.Q_mean: time averaged X, has shape [nx,nt]
    # self.typePOD: which method to use
    # self.Phi: eigenvectors Phi
    # self.modes: matrix of POD modes
    # self.lam: eigenvalues
    # self.nt: number of snapshots
    # self.nx: total number of grid points
    # self.w: weights
    # self.grid_shape: the shape of the grid

    def __init__(
        self,
        X: np.array,
        method: str = "auto",
        weight: StrOrArray = "ones",
        keep_shape: bool = False,
    ):
        """
        Arguments:

        X: data tensor, the last dimension must be time.
        method: default 'auto' will select the method base on the size of the data matrix.
                'classic' is recommended for when Nx < Nt.
                'snapshot' is recommended for when Nx > Nt.
        weight: select weights that are applied to each snapshot, or pass an numpy array as weights.
        keep_shape: if True, the POD modes will be reshaped to the spatial dimension of the input data X. For example X has shape (2,3,2,10), then self.modes will have shape (2,3,2,modes)
        """
        self.method = method
        self.weight = weight
        self.keep_shape = keep_shape

        # Prepare data matrix Q
        self.nx, self.nt, self.grid_shape, self.Q, self.Q_mean = self.prepare_data(X)
        self.w = self.set_weight()

        # choose POD method
        print("Calculating POD ...")
        if self.method == "auto":
            if self.nt >= self.nx:
                self.typePOD = "classic"
                print("Use classic POD, Nx <= Nt.")
                self.modes, self.lam, self.Phi = self.classic_POD(self.Q)
            else:
                self.typePOD = "snapshot"
                print("Use snapshot POD, Nx > Nt.")
                self.modes, self.lam, self.Phi = self.snapshot_POD(self.Q)
        else:
            self.typePOD = self.method
            print("User has selected " + self.typePOD + " POD")
            if self.typePOD == "classic":
                self.modes, self.lam, self.Phi = self.classic_POD(self.Q)
            elif self.typePOD == "snapshot":
                self.modes, self.lam, self.Phi = self.snapshot_POD(self.Q)
            else:
                sys.exit(
                    "Method does not exist. Please choose between 'auto', 'classic' or 'snapshot'."
                )
        print("POD done.")

        if self.keep_shape:
            self.restore_shape()

    @staticmethod
    def prepare_data(X: np.array):
        """Prepare the data for decomposition.

        Data is reshaped into [nx, nt], mean is removed.

        Returns:
        nx: number of data points
        nt: number of snapshots
        grid_shape: the shape of the input (original data)
        Q: fluctuating velocity with shape [nx, nt]
        Q_mean: mean velocity with length [nx]
        """
        Q = np.copy(X)
        grid_shape = list(Q.shape[:-1])
        nt = Q.shape[-1]
        Q = np.reshape(Q, (-1, nt))
        nx = Q.shape[0]

        # remove mean
        Q_mean = np.mean(Q, axis=1)
        for ti in range(0, nt):
            Q[:, ti] = Q[:, ti] - Q_mean

        return nx, nt, grid_shape, Q, Q_mean

    def set_weight(self):
        if self.weight == "ones":
            weights = np.ones((self.nx, 1))
        elif isinstance(self.weight, np.ndarray):
            weights = self.weight
        else:
            raise ValueError(
                "Choose weights from available options or provide a numpy array."
            )
        return weights

    def classic_POD(self, Q: np.array):
        """Calculate POD using the classic method.

        Suitable for when number of snapshots is larger than the number of data points.

        Arguments:
            Q: np.array of fluctuating velocity, with shape [nx,nt], where nx is the number of data points and nt is the number of snapshots.
        """
        C = (
            (Q * self.w) @ ((Q.T) * (self.w.T)) / (self.nt - 1)
        )  # 2-point spatial correlation tesnsor: Q*Q'
        # print('C is Hermitian?',np.allclose(C,np.conj(C.T)))
        lam, Phi = np.linalg.eigh(C)  # right eigenvectors and eigenvalues
        idx = np.argsort(lam)  # sort
        idx = np.flip(idx)
        Q_POD = Phi[:, idx]
        lam = lam[idx]
        Phi = np.copy(Q_POD)
        # normalise energy in the weighted inner product
        normQ = (Q_POD.T @ Q_POD * self.w).real ** 0.5
        Q_POD = Q_POD @ np.diag(1 / np.diag(normQ))
        return Q_POD, lam, Phi

    def snapshot_POD(self, Q: np.array):
        """Calculate POD using the snapshot method.

        Suitable for when number of snapshots is smaller than the number of data points.

        Arguments:
            Q: np.array of fluctuating velocity, with shape [nx,nt], where nx is the number of data points and nt is the number of snapshots.
        """
        C = (
            (Q.T * self.w.T) @ (Q * self.w) / (self.nt - 1)
        )  # 2-point temporal correlation tesnsor: Q'*Q
        lam, Phi = np.linalg.eigh(C)
        idx = np.argsort(np.abs(lam))  # sort
        idx = np.flip(idx)
        Phi = Phi[:, idx]
        lam = lam[idx]
        # get spatial POD modes: PSI = Q*Phi, must be normalised
        Q_POD = (Q @ Phi) * (1 / (lam**0.5).T)
        norms = np.einsum("m n -> n", Q_POD**2) ** 0.5
        Q_POD = Q_POD / norms.reshape((1, -1))
        return Q_POD, lam, Phi

    def restore_shape(self):
        """Reshape self.modes into the shape of the input data."""
        original_shape = self.grid_shape.copy()
        original_shape.extend([-1])
        self.modes = np.reshape(self.modes, original_shape)

    def reconstruct(
        self,
        which_modes: typing.Union[int, list],
        A: StrOrArray = "self",
        Q_mean: StrOrArray = "self",
        Q: StrOrArray = "self",
        Phi: StrOrArray = "self",
        shape="self",
    ) -> np.array:
        """Reconstruct data with the selected number of POD modes.

        Arguments:
            which_modes: which modes to use. \n
                If given integer, mode up to the value given will be used. \n
                If given list of 2 values [val1,val2], mode [val1:val2] will be used. \n
                If given of more than 2 values, mode specified in the list is used.\n
            Q_mean, Q, Phi: If any of 'Q_mean', 'Q', or 'Phi' is not given, the method uses the values stored in self.
            shape: reshape the reconstructed vector into this shape, if left as default the result is reshaped to self.grid_shape.

        Returns:
            Reconstructed flow field with the shape specified by 'shape'
        """
        if Q_mean == "self" or Q == "self" or Phi == "self":
            Q_mean = self.Q_mean
            Q = self.Q
            Phi = self.Phi
        if isinstance(A, str):
            if A == "self":
                A = self.get_time_coefficient
        if isinstance(which_modes, int):
            idx = np.s_[:, 0:which_modes]
        elif isinstance(which_modes, list):
            if len(which_modes) == 2:
                idx = np.s_[:, which_modes[0] : which_modes[1]]
                print(f"Using mode [{which_modes[0]}:{which_modes[1]}]")
            else:
                idx = np.s_[:, which_modes]
                print(f"Using modes named by user in argument 'which_modes'.")
        else:
            raise ValueError("Invalid argument: which_modes.")
        if self.typePOD == "classic":
            Q_add = Phi[idx] @ A[idx].T
        elif self.typePOD == "snapshot":
            Q_add = A[idx] @ Phi[idx].T
        rebuildv = np.reshape(Q_mean, (-1, 1)) + Q_add
        if shape == "self":
            new_shape = np.copy(self.grid_shape).tolist()
            new_shape.append(-1)
            rebuildv = np.reshape(rebuildv, tuple(new_shape))
        else:
            rebuildv = np.reshape(rebuildv, shape)
        return rebuildv

    @property
    def get_modes(self):
        return self.modes, self.lam

    @property
    def get_time_coefficient(self) -> np.array:
        """Return the temporal or spatial coefficients depending on the method

        The shape of the returned array depends on the type of POD.
        """
        if self.typePOD == "classic":
            # temporal coefficient A
            print("Returning temporal coefficients for classic POD.")
            self._A = self.Q.T @ self.Phi  # (nt,nx)
        elif self.typePOD == "snapshot":
            # spatial coefficient A
            print(" Returning spatial coefficients for snapshot POD.")
            self._A = self.Q @ self.Phi  # (nx,nt)
        return self._A
