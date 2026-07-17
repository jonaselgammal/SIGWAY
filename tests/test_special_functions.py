"""Si/Ci special-function machinery behind the instant eMD->RD kernel.

The large-V eMD kernel multiplies by  LV(x) = 4 Ci(x/2)^2 + (pi - 2 Si(x/2))^2,
evaluated directly with jax.scipy.special.sici. We check it against scipy to
tight tolerance, with extra density at small/large x.

Independent reference: scipy.special.sici. _sici_precomp is a jitted helper the
refactor may relocate/inline; only the import line would change.
"""

import numpy as np
import jax.numpy as jnp

from scipy.special import sici

from sigway.kernels import _sici_precomp


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
