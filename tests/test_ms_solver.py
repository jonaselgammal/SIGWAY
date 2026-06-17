"""Mukhanov-Sasaki solver physics (the EOM-based P_zeta representation).

Validates the P_zeta the MS path (SingleFieldPerturbations) consumes,
independently of the SIGW integration:

* for quadratic chaotic inflation the numerically solved P_zeta(k) must have the
  analytic slow-roll tilt  n_s - 1 = -2/N  (a textbook result, no solver SR
  machinery involved);
* the full MS spectrum must reduce to the solver's slow-roll formula in the
  slow-roll regime.

The MS end-to-end Omega_GW regression and the MS-vs-analytic cross-check live in
test_omega_gw_regression / test_cross_backend.
"""

import numpy as np
import jax
import jax.numpy as jnp

from sigway.ms_solver import SingleFieldSolver
from sigway.utils import efolds_from_wavenumber_si_units, H_from_wavenumber

jax.config.update("jax_enable_x64", True)


def _V_quadratic(phi, m):
    return 0.5 * m**2 * phi**2


def _quadratic_solver():
    return SingleFieldSolver(
        _V_quadratic,
        phi0=16.0,
        pi0=0.0,
        N_CMB_to_end=55.0,
        k=jnp.geomspace(1e-4, 1e-1, 40),
    )


def test_quadratic_pzeta_tilt_matches_slow_roll():
    """Full MS P_zeta(k) tilt for V=m^2 phi^2/2 equals analytic n_s-1 = -2/N.

    n_s is convention-independent (a log-log slope), so this is a clean check of
    the perturbation solver. The two agree to ~0.1%; require |Dn_s| < 0.01.
    """
    m = 6e-6
    solver = _quadratic_solver()
    N, phi, y, h = solver.run_background((m,))
    k = jnp.geomspace(1e-4, 1e-1, 40)
    Pz = np.array(solver.run_perturbations(k, N, phi, y, h, (m,)))
    assert np.all(np.isfinite(Pz)) and np.all(Pz > 0)

    sl = slice(5, -5)  # drop grid edges
    tilt = np.polyfit(np.log(np.array(k))[sl], np.log(Pz)[sl], 1)[0]
    ns_measured = 1.0 + tilt

    # map the pivot wavenumbers to e-folds-before-end -> analytic n_s = 1 - 2/N
    Nend = float(jnp.max(N))
    N_CMB = Nend - 55.0
    H_CMB = float(jnp.interp(N_CMB, N, h))
    Hk = H_from_wavenumber(k, N, h, N_CMB, H_CMB)
    Nk = np.array(efolds_from_wavenumber_si_units(k, Hk, N_CMB, H_CMB))
    N_before_end = np.mean(Nend - Nk[sl])
    ns_slow_roll = 1.0 - 2.0 / N_before_end

    assert abs(ns_measured - ns_slow_roll) < 0.01


def test_full_pzeta_reduces_to_slow_roll():
    """In the slow-roll regime the full MS spectrum matches the SR formula.

    Compares the numerically integrated P_zeta(k) to the solver's analytic
    slow-roll P_zeta evaluated at the matching horizon-crossing e-fold; they
    agree to ~2% for a featureless large-field model.
    """
    m = 6e-6
    solver = _quadratic_solver()
    N, phi, y, h = solver.run_background((m,))
    k = jnp.geomspace(1e-4, 1e-1, 40)
    Pz_full = np.array(solver.run_perturbations(k, N, phi, y, h, (m,)))
    Pz_sr = np.array(solver.pzeta_sr(y, h, (m,)))

    Nend = float(jnp.max(N))
    N_CMB = Nend - 55.0
    H_CMB = float(jnp.interp(N_CMB, N, h))
    Hk = H_from_wavenumber(k, N, h, N_CMB, H_CMB)
    Nk = np.array(efolds_from_wavenumber_si_units(k, Hk, N_CMB, H_CMB))
    Pz_sr_at_k = np.interp(Nk, np.array(N), Pz_sr)

    assert np.median(np.abs(Pz_full / Pz_sr_at_k - 1.0)) < 0.05
