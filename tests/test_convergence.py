"""Integration convergence in the (s, t) grids.

Confirms Omega_GW converges as the s/t resolution increases and that the default
resolution sits within tolerance of the converged value -- and documents the two
grid-resolution limitations found while building this suite:

* the example notebook's USR t-grid `logspace(-3, 3, 1000)` is ~13% high because
  pure-log spacing starves the t<~1 region (a linear-low-t grid with the *same*
  point count converges);
* the eMD t-grid under-resolves the t^4 large-V integrand for k << kmax.
"""

import numpy as np
import jax.numpy as jnp

from sigway.spectrum import OmegaGW
from sigway.kernels import RadiationKernel
from sigway.perturbations import AnalyticPerturbations
import _sigway_configs as C


def _lognormal_omega_at(f0, ns, nt_lo, nt_hi):
    p = (-2.5, -0.3010299956639812, -2.0)
    ks = 10.0 ** p[2]

    def tgrid(k, logAs, logDelta, logks):
        D = 10.0**logDelta
        upper = jnp.exp(4 * D) * (2 * ks / k)
        one = jnp.ones_like(k)
        t1 = jnp.linspace(1e-5 * one, 0.999 * one, nt_lo)
        t2 = jnp.geomspace(jnp.ones_like(upper), upper, nt_hi)
        return jnp.concatenate([t1, t2], axis=0)

    m = OmegaGW(
        AnalyticPerturbations(C.pzeta_ln, ("logAs", "logDelta", "logks")),
        RadiationKernel(),
        s=jnp.linspace(0, 1, ns),
        t=tgrid,
        upsample=False,
    )
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
