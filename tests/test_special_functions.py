"""Si/Ci special-function machinery behind the instant eMD->RD kernel.

The large-V eMD kernel multiplies by  LV(x) = 4 Ci(x/2)^2 + (pi - 2 Si(x/2))^2,
precomputed once on a 10^7-point grid and read back with jnp.interp. An
interpolation table is exactly the kind of thing that silently drifts, so we
check it (and its analytic derivative) against scipy to tight tolerance, with
extra density at small/large x.

Independent reference: scipy.special.sici (and finite differences for the
derivative). These functions are jitted helpers that the refactor may relocate;
only the import line would change.
"""

import numpy as np
import jax.numpy as jnp

from scipy.special import sici

from sigway.kernels import _sici_precomp, _d_sici_precomp


def _LV_exact(x):
    si, ci = sici(x / 2.0)
    return 4.0 * ci**2 + (np.pi - 2.0 * si) ** 2


def test_sici_precomp_vs_scipy():
    """LV(x) interpolation table == scipy sici to ~1e-9 over 1e-2..1e4."""
    x = np.geomspace(1e-2, 1e4, 2000)
    got = np.array(_sici_precomp(jnp.array(x)))
    assert np.nanmax(np.abs(got / _LV_exact(x) - 1.0)) < 1e-9


def test_sici_precomp_dense_small_and_large_x():
    """Accuracy holds near the table edges (x -> small and x -> large)."""
    for x in (np.geomspace(1e-4, 1e-1, 500), np.geomspace(1e4, 1e5, 500)):
        got = np.array(_sici_precomp(jnp.array(x)))
        assert np.nanmax(np.abs(got / _LV_exact(x) - 1.0)) < 1e-8


def test_d_sici_precomp_matches_finite_difference():
    """_d_sici_precomp is d/dx of LV's argument; check vs central differences.

    (The eMD-kernel gradient uses _d_sici_precomp/_sici_precomp = d/dx ln LV.)
    """
    x = np.geomspace(1e-1, 1e3, 400)
    h = x * 1e-6
    fd = (
        np.array(_sici_precomp(jnp.array(x + h)))
        - np.array(_sici_precomp(jnp.array(x - h)))
    ) / (2 * h)
    ana = np.array(_d_sici_precomp(jnp.array(x)))
    assert np.nanmax(np.abs(ana / fd - 1.0)) < 1e-4
