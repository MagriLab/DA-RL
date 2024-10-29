# -------------------------------------------------------------------------------------
# Implementation based on Bucci-Semeraro-Allauzen-Wisniewski-Cordier-Mathelin (2019)
# https://doi.org/10.1098/rspa.2019.0351
# -------------------------------------------------------------------------------------
# The code has been refactored into PyTorch and is compatible with the torchrl package.
# The actuation is implemented in a different fashion, to allow more control over
# actuator locations and spread.


import numpy as np
from scipy.stats import norm


def normal_pdf(x, loc, scale):
    return norm.pdf(x, loc, scale)


class KS:
    def __init__(
        self, actuator_locs, actuator_scale=0.1, nu=1.0, N=256, dt=0.5, device="cpu"
    ):
        """
        :param nu: 'Viscosity' parameter of the KS equation.
        :param N: Number of collocation points
        :param dt: Time step
        :param actuator_locs: np array. Specifies the locations of the actuators in the interval [0, 2*pi].
                              Cannot be empty or unspecified. Must be of shape [n] for some n > 0.
        """
        # np.set_default_dtype(np.float64)
        self.device = device

        # Convert the 'viscosity' parameter to a length parameter - this is numerically more stable
        self.L = 2 * np.pi / np.sqrt(np.array(nu))
        self.n = int(N)  # Ensure that N is integer, N must also be even?
        self.dt = np.array(dt)
        self.x = np.arange(self.n) * self.L / self.n
        self.k = (
            self.n * np.fft.fftfreq(self.n)[0 : self.n // 2 + 1] * 2 * np.pi / self.L
        )  # goes until -pi
        self.ik = 1j * self.k  # spectral derivative operator
        self.lin = self.k**2 - self.k**4  # Fourier multipliers for linear term

        # Actuation set-up
        self.num_actuators = actuator_locs.shape[-1]
        self.scale = (
            self.L / (2 * np.pi) * actuator_scale
        )  # Rescale so that we represent the same actuator shape in [0, 2*pi]
        # This should really be doable with vmap...
        B_list = []
        for loc in actuator_locs:
            B_list.append(self.normal_pdf_periodic(self.L / (2 * np.pi) * loc))
        self.B = np.stack(B_list, axis=1)
        self.B = self.B.astype(dtype=np.float64)

    def nlterm(self, u, f):
        # compute tendency from nonlinear term. advection + forcing
        # rfft outputs the positive frequencies n/2+1 if even, (n+1)/2 if odd
        ur = np.fft.irfft(u, axis=-1)
        return -0.5 * self.ik * np.fft.rfft(ur**2, axis=-1) + f

    def advance_f(self, u, action):
        # just advances the fourier coefficients
        f0 = self.B @ action
        f_fine = np.fft.rfft(f0, axis=-1)
        f = self.n / self.n_fine * f_fine[: len(self.k)]
        u_save = np.copy(u)
        for n in range(3):
            dt = self.dt / (3 - n)
            # explicit RK3 step for nonlinear term
            u = u_save + dt * self.nlterm(u, f)
            # implicit trapezoidal adjustment for linear term
            u = (u + 0.5 * self.lin * dt * u_save) / (1.0 - 0.5 * self.lin * dt)
        return u

    def advance(self, u0, action):
        """
        :param u0: np.array or np.array. Solution at previous timestep.
        :param action: np.array or np.array of shape [len(sensor_locations)].
        :return: Same type and shape as u0. The solution at the next timestep.
        """
        # print(self.B.shape)
        # print(action.shape)
        """
        if action.dtype != np.float32 or self.B.dtype != np.float32:
            print(f'Action dtype {action.dtype} || B dtype {self.B.dtype}')
        """
        f0 = self.B @ action
        f_fine = np.fft.rfft(f0, axis=-1)
        f = self.n / self.n_fine * f_fine[: len(self.k)]
        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        u = np.fft.rfft(u0, axis=-1)
        u_save = np.copy(u)
        for n in range(3):
            dt = self.dt / (3 - n)
            # explicit RK3 step for nonlinear term
            u = u_save + dt * self.nlterm(u, f)
            # implicit trapezoidal adjustment for linear term
            u = (u + 0.5 * self.lin * dt * u_save) / (1.0 - 0.5 * self.lin * dt)
        u = np.fft.irfft(u, axis=-1)
        return u

    def normal_pdf_periodic(self, loc):
        """
        Return the pdf of the normal distribution centred at loc with variance self.scale,
        wrapped around the circle of length self.L
        :param loc: Float
        :return: np.array of shape self.x.shape
        """
        self.n_fine = max(64, self.n)
        x_fine = np.arange(self.n_fine) * self.L / self.n_fine
        y = np.zeros(x_fine.shape)
        y_max = 0
        for shift in range(-3, 3):
            y += normal_pdf(x_fine + shift * self.L, loc, self.scale)
            # compute y_max by looking at the peak
            # this is to ensure that the max is the same even when the grid points
            # don't align with the actuator locations
            y_max += normal_pdf(loc + shift * self.L, loc, self.scale)
        y = y / y_max  # Normalize the distribution
        return y


if __name__ == "__main__":
    ks = KS(actuator_locs=np.array([0.0, np.pi]), device="cuda")
    u = np.zeros(ks.n)
    ks.advance(u, np.zeros(2))

    print("here")
