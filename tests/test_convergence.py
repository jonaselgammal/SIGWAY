"""Integration convergence in the (s, t) grids.

Confirms Omega_GW converges as the s/t resolution increases and that the default
resolution sits within tolerance of the converged value -- and documents the two
grid-resolution limitations found while building this suite:

* the example notebook's USR t-grid `logspace(-3, 3, 1000)` is ~13% high because
  pure-log spacing starves the t<~1 region (a linear-low-t grid with the *same*
  point count converges);
* the eMD t-grid under-resolves the t^4 large-V integrand for k << kmax.
"""
import os

import numpy as np
import jax.numpy as jnp

from sigway.omega_gw_jax import OmegaGWjax
from sigway.ms_solver import SingleFieldSolver
from sigway.omega_gw_ms import OmegaGWms
import _sigway_configs as C

REFDIR = os.path.join(os.path.dirname(__file__), "test_data", "reference")


def _lognormal_omega_at(f0, ns, nt_lo, nt_hi):
    p = (-2.5, -0.3010299956639812, -2.0)
    ks = 10.0 ** p[2]

    def tgrid(k, logAs, logDelta, logks):
        D = 10.0 ** logDelta
        upper = jnp.exp(4 * D) * (2 * ks / k)
        one = jnp.ones_like(k)
        t1 = jnp.linspace(1e-5 * one, 0.999 * one, nt_lo)
        t2 = jnp.geomspace(jnp.ones_like(upper), upper, nt_hi)
        return jnp.concatenate([t1, t2], axis=0)

    m = OmegaGWjax(C.pzeta_ln, jnp.linspace(0, 1, ns), tgrid,
                   f=jnp.array([f0]), norm="RD", kernel="RD",
                   upsample=False, dP_zeta="auto", jit=True)
    return float(np.array(m(jnp.array([f0]), *p))[0])


def test_lognormal_st_convergence():
    """Omega_GW converges under (s, t) refinement; the paper-default resolution
    sits within 2% of the converged value.

    Evaluated just off the exact resonance, where the result is smooth (at the
    log-divergent peak the geomspace endpoint placement adds ~1-2% noise).
    """
    f0 = 0.001
    default = _lognormal_omega_at(f0, ns=10, nt_lo=200, nt_hi=600)  # paper grid
    fine = _lognormal_omega_at(f0, ns=80, nt_lo=800, nt_hi=2400)
    finer = _lognormal_omega_at(f0, ns=160, nt_lo=1600, nt_hi=4800)
    # the two finest resolutions agree -> converged reference established
    assert abs(fine / finer - 1) < 1e-2
    # the paper-default resolution is within 2% of the converged value
    assert abs(default / finer - 1) < 0.02


def test_usr_t_spacing_matters_not_count():
    """The notebook's log t-grid is ~13% high; a linear-low-t grid (same count)
    converges. Pins both the converged value and the spacing diagnosis."""
    cfg = C.USR_CONFIG
    p = cfg["params"]
    solver = SingleFieldSolver(
        C.usr_potential, phi0=cfg["phi0"], pi0=cfg["pi0"],
        N_CMB_to_end=cfg["N_CMB_to_end"], k=jnp.array(cfg["k_solver"]))
    ref = np.load(os.path.join(REFDIR, "usr_ms.npz"))
    f = jnp.array([cfg["f"][np.argmax(ref["omega_gw"])]])  # peak frequency

    def omega(t):
        return float(np.array(OmegaGWms(
            solver, jnp.array(cfg["s"]), t, f=f, kernel="RD",
            upsample=False)(f, *p))[0])

    nf = 1
    t_log = jnp.repeat(
        jnp.expand_dims(jnp.logspace(-3, 3, 1000), -1), nf, axis=1)
    t_lin = C.usr_t_grid(nlo=200, nhi=800, nf=nf)
    t_conv = C.usr_t_grid(nlo=400, nhi=3600, nf=nf)
    o_log, o_lin, o_conv = omega(t_log), omega(t_lin), omega(t_conv)
    assert abs(o_log / o_conv - 1) > 0.05      # log grid is far from converged
    assert abs(o_lin / o_conv - 1) < 0.02      # same count, better spacing


def _emd_omega_at(k_over_kmax, nt):
    cfg = C.ANALYTIC_CONFIGS["emd_imd2rd"]
    As, kmax, etaR = cfg["params"]
    f0 = k_over_kmax * kmax / (2 * np.pi)

    def tgrid(k, As, kmax, etaR):
        return jnp.geomspace(1e-10 * jnp.ones_like(k), 2 * kmax / k, nt)

    m = OmegaGWjax(C.pzeta_heaviside, jnp.linspace(0, 1, 100), tgrid,
                   f=jnp.array([f0]), norm="CT", kernel="I_MD_to_RD",
                   upsample=False, dP_zeta="auto", jit=True)
    return float(np.array(m(jnp.array([f0]), As, kmax, etaR))[0])


def test_emd_peak_band_converged_ir_tail_not():
    """eMD: default 100-pt t-grid is converged in the peak band but the k<<kmax
    IR tail keeps changing under refinement (documented limitation)."""
    # peak band (k = 0.7 kmax): default already within 1% of refined
    near = _emd_omega_at(0.7, nt=100)
    near_fine = _emd_omega_at(0.7, nt=1600)
    assert abs(near / near_fine - 1) < 0.01
    # deep IR (k = 0.05 kmax): default is NOT converged -> moves a lot
    ir = _emd_omega_at(0.05, nt=100)
    ir_fine = _emd_omega_at(0.05, nt=3200)
    assert abs(ir / ir_fine - 1) > 0.1
