# Local
import jax
from jax import numpy as jnp
from jax import jit

jax.config.update("jax_enable_x64", True)

# Conversion factors
Mpc_to_m = 3.086e22  # 1 Mpc in meters
c = 299792458.0  # Speed of light in m/s
M_p = 2.176e-8  # Planck mass in kg (in natural units this would be 1)
CMB_scale = 0.05  # Mpc^-1
CMB_scale_k = CMB_scale / Mpc_to_m * c  # Mpc^-1 to s^-1


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
