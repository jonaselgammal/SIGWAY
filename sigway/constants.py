"""Physical constants and fixed cosmological parameters used across SIGWAY.

Single, documented source of truth for the numerical constants that were
previously scattered across ``utils.py``, ``kernels.py`` and
``binned_pzeta.py``.  Names follow the physics notation used in the literature
(e.g. ``c``, ``M_p``) rather than PEP 8 upper-case, so that formulae that use
them stay readable; constants with no canonical single-letter form
(``RD_EOS``, ``RD_SOUND_SPEED``) are upper-case.
"""

import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

__all__ = [
    "c",
    "M_p",
    "Mpc_to_m",
    "CMB_scale",
    "CMB_scale_k",
    "Omega_radiation_h2_today",
    "SM_CG_factor",
    "RD_EOS",
    "RD_SOUND_SPEED",
    "CMB_means",
    "CMB_cov",
    "CMB_BOUNDS",
]

# --- Fundamental constants ---
c = 299792458.0  # Speed of light in m/s
M_p = 2.176e-8  # Planck mass in kg (unity in natural units)

# --- Unit conversions ---
Mpc_to_m = 3.086e22  # 1 Mpc in metres

# --- Cosmological parameters ---
CMB_scale = 0.05  # CMB pivot scale in Mpc^-1
CMB_scale_k = CMB_scale / Mpc_to_m * c  # pivot scale converted to s^-1
Omega_radiation_h2_today = 4.2e-5  # Omega_r h^2 today
SM_CG_factor = 0.39  # g_* (c_g) factor for the Standard Model

# --- Kernel constants ---
# Radiation domination is the w = 1/3 special case of a constant-w fluid, for
# which the adiabatic sound speed is c_s = sqrt(w).  The RD kernel resonance
# sits at u + v = 1/c_s = sqrt(3), i.e. the (s, t) slice t = sqrt(3) - 1.
RD_EOS = 1.0 / 3.0  # radiation equation of state, w = p/rho = 1/3
RD_SOUND_SPEED = RD_EOS**0.5  # adiabatic sound speed c_s = sqrt(w) = 1/sqrt(3)

# --- CMB priors (Planck 2018, arXiv:1807.06211): [A_s, n_s, r] ---
CMB_means = jnp.array([3.04442188, 0.96488871, 0.0])
CMB_cov = jnp.array(
    [
        [2.00112315e-04, 1.35106101e-05, 0.0],
        [1.35106101e-05, 1.72537423e-05, 0.0],
        [0.0, 0.0, 0.01],
    ]
)
CMB_BOUNDS = {"means": CMB_means, "cov": CMB_cov}
