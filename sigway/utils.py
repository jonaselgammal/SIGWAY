# Local
import jax
from jax import numpy as jnp
from jax import jit

jax.config.update("jax_enable_x64", True)

# Conversion factors
c = 299792458.0  # Speed of light in m/s
M_p = 2.176e-8  # Planck mass in kg (in natural units this would be 1)
Mpc_to_m = 3.086e22  # 1 Mpc in meters
CMB_scale = 0.05  # Mpc^-1
CMB_scale_k = CMB_scale / Mpc_to_m * c  # Mpc^-1 to s^-1

# Some cosmological parameters
Omega_radiation_h2_today = 4.2e-5  # Omega_r h^2 today
SM_CG_factor = 0.39  # CG factor for the SM


# Utils to convert between e-folds, wavenumber, and Hubble parameter
@jit
def wavenumber_from_efolds_si_units(N, H, N_CMB, H_CMB):
    """
    Calculate the wavenumber k as a function of the number of e-folds N and
    the Hubble parameter H, normalized such that k=0.05 Mpc^-1 corresponds
    to N_CMB_to_end e-folds before the end of inflation.

    Parameters
    ----------
    N : array-like
        E-fold values.
    H : array-like
        Hubble parameter values (in some units) corresponding to the e-folds.
    N_CMB : float
        Number of e-folds at the CMB scale.
    H_CMB : float
        Hubble parameter at the CMB scale (in the same units of H).

    Returns
    -------
    k_N : array-like
        Wavenumber as a function of e-folds and Hubble parameter.
    """
    # Calculate the wavenumber k for each N and H
    k_N = CMB_scale_k * (H / H_CMB) * jnp.exp(N - N_CMB)

    return k_N


@jit
def efolds_from_wavenumber_si_units(k, H, N_CMB, H_CMB):
    """
    Calculate the number of e-folds N as a function of the wavenumber k and
    the Hubble parameter H, normalized such that k=0.05 Mpc^-1 corresponds
    to N_CMB_to_end e-folds before the end of inflation.

    Parameters
    ----------
    k : array-like
        Wavenumber values in s^-1.
    H : array-like
        Hubble parameter values (in some units) corresponding to the k values.
    N_CMB : float
        Number of e-folds at the CMB scale.
    H_CMB : float
        Hubble parameter at the CMB scale (in the same units of H).

    Returns
    -------
    N : array-like
        Number of e-folds as a function of wavenumber and Hubble parameter.
    """
    # Calculate the number of e-folds N for each k
    N = N_CMB + jnp.log(k / (CMB_scale_k * (H / H_CMB)))

    return N


@jit
def H_from_wavenumber(k, N, H, N_CMB, H_CMB):
    """
    Calculate the Hubble parameter H as a function of wavenumber k using JAX.
    First it computes the wavenumber k as a function of the number of e-folds N
    and the Hubble parameter H, normalized such that k=0.05 Mpc^-1 corresponds
    to N_CMB_to_end e-folds before the end of inflation. Then it sorts the
    wavenumber k and Hubble parameter H values in ascending order and
    interpolates the Hubble parameter as a function of wavenumber k.

    Parameters
    ----------
    k : array-like
        Wavenumber values in s^-1 where we want the H evaluations.
    N : array-like
        E-fold values.
    H : array-like
        Hubble parameter values (in some units) corresponding to N values.
    N_CMB : float
        Number of e-folds at the CMB scale.
    H_CMB : float
        Hubble parameter at the CMB scale (in the same units of H).

    Returns
    -------
    H_k : array-like
        Hubble parameter as a function of wavenumber k.
    """

    # Calculate the wavenumber k as a function of N and H
    k_N = wavenumber_from_efolds_si_units(N, H, N_CMB, H_CMB)

    # Sort the wavenumber k and Hubble parameter H values
    sorted_indices = jnp.argsort(k_N)
    k_N_sorted = k_N[sorted_indices]
    H_N_sorted = H[sorted_indices]

    # Interpolation using JAX's interp
    H_k = jnp.interp(k, k_N_sorted, H_N_sorted)

    return H_k


# Utils for integration
@jit
def simpson_uniform(f, x):
    """
    Fully vectorized implementation of Simpson's rule for a uniform grid.
    See composite 1/3 rule in https://en.wikipedia.org/wiki/Simpson%27s_rule.
    If f is a multi-dimensional array, the integration is performed over the
    first axis.

    Parameters:
    - f: jax.numpy.ndarray, shape (N, ...)
        Array of function values. Integration is performed over the first axis.
    - x: jax.numpy.ndarray, shape (N,)
        Array of grid points.

    Returns:
    - result: jax.numpy.ndarray
        Array of integrated values.
    """
    # number of subintervals used in the integration
    N = f.shape[0] - 1

    # step size
    h = x[1] - x[0]

    # step in the sum over i
    step = 2

    # Simpson's rule
    result = (
        f[0]
        + 4 * jnp.sum(f[1:N:step], axis=0)
        + 2 * jnp.sum(f[2:N:step], axis=0)
        + f[-1]
    )

    # Multiply by h/3 to get the final result
    result *= h / 3

    # # Adjust if N is odd (last segment)
    if N % 2 == 1:
        # Subtract last interval computed in the main sum
        result -= (h / 3) * (f[-2] + f[-1])
        # Trapezoidal rule for the last interval
        result += (h / 2) * ((2 * f[-1] + f[-2]) + (f[-1] + f[-2]))
    return result


@jit
def simpson_nonuniform(f, x):
    """
    Numerical integration using Simpson's rule on a non-uniform grid.

    See composite 1/3 rule for irregularly spaced data in
    https://en.wikipedia.org/wiki/Simpson%27s_rule.

    This fully vectorized implementation is suitable for multi-dimensional
    arrays, integrating over the first axis. Assumes non-uniformly spaced grid
    points.

    Parameters:
    - f (jax.numpy.ndarray): Array of function values, shape (N, ...).
    - x (jax.numpy.ndarray): Array of grid points, shape (N,) or same as 'f'.

    Returns:
    - jax.numpy.ndarray: Integrated values.
    """

    # Number of subintervals
    N = len(x) - 1

    # Differences between consecutive x values
    h = jnp.diff(x, axis=0)

    # get the shape of f
    f_shape = f.shape

    # Step size in the sum over i
    step = 2

    # Adjusting shape for broadcasting if necessary
    if x.shape != f_shape:
        broadcast_shape = (-1,) + (1,) * (len(f_shape) - 1)
        h0 = h[:-1:step].reshape(broadcast_shape)
        h1 = h[1::step].reshape(broadcast_shape)
    else:
        h0 = h[:-1:step]
        h1 = h[1::step]

    hph = h1 + h0
    hdh = h1 / h0
    hmh = h1 * h0
    result = jnp.sum(
        (hph / 6)
        * (
            (2 - hdh) * f[:-2:step]
            + (hph**2 / hmh) * f[1:-1:step]
            + (2 - 1 / hdh) * f[2::step]
        ),
        axis=0,
    )

    # Additional computation for even N (last segment)
    if N % 2 == 1:
        h0 = h[-2]
        h1 = h[-1]
        result += f[-1] * (2 * h1**2 + 3 * h0 * h1) / (6 * (h0 + h1))
        result += f[-2] * (h1**2 + 3 * h1 * h0) / (6 * h0)
        result -= f[-3] * h1**3 / (6 * h0 * (h0 + h1))

    return result
