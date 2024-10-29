import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.scipy.stats import norm


# Function to calculate the probability density function (PDF) of a normal distribution
def normal_pdf(x, loc, scale):
    return norm.pdf(x, loc, scale)


# KS (Kuramoto-Sivashinsky) class definition
class KS:
    def __init__(self, actuator_locs, actuator_scale=0.1, nu=1.0, N=256, dt=0.5):
        """
        Initialize the KS model.

        Args:
            actuator_locs: jnp.array. Specifies the locations of the actuators in the interval [0, 2*pi].
                        Cannot be empty or unspecified. Must be of shape [n] for some n > 0.
            actuator_scale: Standard deviation of the actuators.
            nu: Viscosity parameter of the KS equation.
            N: Number of collocation points.
            dt: Time step for the simulation.
        """
        # Convert the 'viscosity' parameter to a length parameter - this is numerically more stable
        self.L = 2 * jnp.pi / jnp.sqrt(jnp.array(nu))
        self.N = int(N)  # Ensure that N is an integer
        self.dt = jnp.array(dt)
        self.x = jnp.arange(self.N) * self.L / self.N  # Spatial grid points

        # Define wavenumbers for Fourier transform, considering only the positive frequencies
        self.k = (
            self.N * jnp.fft.fftfreq(self.N)[0 : self.N // 2 + 1] * 2 * jnp.pi / self.L
        )

        # Spectral derivative operator
        self.ik = 1j * self.k

        # Fourier multipliers for the linear term
        self.lin = self.k**2 - self.k**4

        # Actuation setup
        self.num_actuators = actuator_locs.shape[-1]
        self.scale = (
            self.L / (2 * jnp.pi) * actuator_scale
        )  # Rescale actuator influence

        # Create the actuator influence matrix B using vectorization
        self.B = jnp.stack(
            vmap(self.normal_pdf_periodic)(self.L / (2 * jnp.pi) * actuator_locs),
            axis=1,
        )
        self.B = self.B.astype(dtype=jnp.float64)

    @staticmethod
    @jit
    def nlterm(u, f, ik):
        """
        Compute the nonlinear term of the KS equation.

        Args:
            u: Fourier coefficients of the solution.
            f: Forcing term in Fourier space.

        Returns:
            Nonlinear term in Fourier space.
        """
        ur = jnp.fft.irfft(u, axis=-1)
        # Compute the advection term + forcing
        return -0.5 * ik * jnp.fft.rfft(ur * ur, axis=-1) + f

    @staticmethod
    @jit
    def advance_f(u, action, B, lin, ik, dt):
        """
        Advance the Fourier coefficients.

        Args:
            u: Fourier coefficients of the solution at the current time step.
            action: Actuation input vector.

        Returns:
            Updated Fourier coefficients.
        """
        # Calculate the forcing term in real space
        f0 = B @ action
        # Transform the forcing term to Fourier space
        f_fine = jnp.fft.rfft(
            f0, axis=-1
        )  # we do this to eliminate aliasing from the excitation
        N = 2 * (u.shape[0] - 1)  # because u in fourier
        N_fine = B.shape[0]
        f = N / N_fine * f_fine[: ik.shape[0]]

        # Save the current Fourier coefficients
        u_save = jnp.copy(u)

        # Define a single RK3 step function
        def rk3_step(u, n):
            dt_n = dt / (3 - n)  # Adjust the time step for the RK3 method
            # Perform explicit RK3 step for the nonlinear term
            u = u_save + dt_n * KS.nlterm(u, f, ik)
            # Implicit trapezoidal rule for the linear term
            u = (u + 0.5 * lin * dt_n * u_save) / (1.0 - 0.5 * lin * dt_n)
            return u, None

        # Use lax.scan to efficiently apply the RK3 steps
        u, _ = lax.scan(rk3_step, u, jnp.arange(3))
        return u

    @staticmethod
    @jit  # Just-in-time compile for performance optimization
    def advance(u0, action, B, lin, ik, dt):
        """
        Advance the solution in time.

        Args:
            u0: Velocity in real space at grid points.
            action: Actuation input vector.

        Returns:
            Updated velocity in real space.
        """
        # Calculate the forcing term in real space
        f0 = B @ action
        # Transform the solution and forcing term to Fourier space
        f_fine = jnp.fft.rfft(
            f0, axis=-1
        )  # we do this to eliminate aliasing from the excitation
        N = u0.shape[0]  # because u in velocity
        N_fine = B.shape[0]
        f = N / N_fine * f_fine[: ik.shape[0]]

        u = jnp.fft.rfft(u0, axis=-1)
        # Save the current Fourier coefficients
        u_save = jnp.copy(u)

        # Define a single RK3 step function
        def rk3_step(u, n):
            dt_n = dt / (3 - n)  # Adjust the time step for the RK3 method
            # Perform explicit RK3 step for the nonlinear term
            u = u_save + dt_n * KS.nlterm(u, f, ik)
            # Implicit trapezoidal rule for the linear term
            u = (u + 0.5 * lin * dt_n * u_save) / (1.0 - 0.5 * lin * dt_n)
            return u, None

        # Use lax.scan to efficiently apply the RK3 steps
        u, _ = lax.scan(rk3_step, u, jnp.arange(3))
        # Transform back to real space from Fourier space
        u = jnp.fft.irfft(u, axis=-1)
        return u

    def normal_pdf_periodic(self, loc):
        """
        Compute a periodic normal distribution centered at `loc`.

        Args:
            loc: Center of the normal distribution.

        Returns:
            Periodic normal distribution values over the grid `self.x`.
        """
        self.N_fine = max(64, self.N)
        x_fine = jnp.arange(self.N_fine) * self.L / self.N_fine
        shifts = jnp.arange(
            -3, 3
        )  # Consider periodic copies within the range [-3L, 3L]
        # Sum the contributions from all periodic copies of the normal distribution
        y = jnp.sum(
            vmap(lambda shift: normal_pdf(x_fine + shift * self.L, loc, self.scale))(
                shifts
            ),
            axis=0,
        )
        # compute y_max by looking at the peak
        # this is to ensure that the max is the same even when the grid points
        # don't align with the actuator locations
        y_max = jnp.sum(
            vmap(lambda shift: normal_pdf(loc + shift * self.L, loc, self.scale))(
                shifts
            ),
            axis=0,
        )
        y = y / y_max  # Normalize the distribution
        # can also check this against the max you get from the fine grid
        return y


# Example usage
if __name__ == "__main__":
    # Create an instance of the KS solver
    ks_solver = KS(
        actuator_locs=jnp.array([0.0, jnp.pi / 4, jnp.pi / 2, 3 * jnp.pi / 4])
    )

    # To advance the solution, call the advance method with the necessary data.
    u0 = jnp.zeros(ks_solver.N)
    action = jnp.zeros(ks_solver.B.shape[1])
    u_next = KS.advance(
        u0, action, ks_solver.B, ks_solver.lin, ks_solver.ik, ks_solver.dt
    )

    print("here")
