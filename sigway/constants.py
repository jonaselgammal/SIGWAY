# Global
import jax
from jax import numpy as jnp

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

# CMB mean default mean values for As, ns and r from Planck 2018 (1807.06211)
CMB_means = jnp.array([3.04442188, 0.96488871, 0.0])

# CMB covariance matrix for As, ns and r from Planck 2018 (1807.06211)
CMB_cov = jnp.array(
    [
        [2.00112315e-04, 1.35106101e-05, 0.0],
        [1.35106101e-05, 1.72537423e-05, 0.0],
        [0.0, 0.0, 0.01],
    ]
)

# CMB bounds default dictionary to be used in the ms_solver.py file
CMB_BOUNDS = {"means": CMB_means, "cov": CMB_cov}
