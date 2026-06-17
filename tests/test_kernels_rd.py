"""Radiation-domination kernel physics.

These pin the closed-form oscillation-averaged transfer function used for every
RD spectrum. The reference is an *independent* numpy re-derivation of
\\overline{I^2}(u,v) from the literature (see _sigway_oracle.kernel_RD_text),
not a stored snapshot. Tolerances are ~1e-10: the two implementations agree to
~1e-11, so this bites on any real change to the kernel.

Targets stable physics (the kernel's numerical output), so it survives the
planned Kernel-class refactor; only the import line would move.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from sigway.kernels import (
    I_sq_RD,
    I_sq_RD_uv,
    I_sq_IRD_LV,
    I_sq_IRD_res,
    RadiationKernel,
    InstantEMDKernel,
)
from _sigway_oracle import kernel_RD_text

# (t, s) sample grid spanning sub- and super-resonance (t = sqrt(3)-1) regions.
_T = np.geomspace(1e-2, 1e2, 60)
_S = np.linspace(0.0, 0.97, 25)  # avoid s=1 boundary (v=0 -> 1/v^3 blow-up)
_TT, _SS = np.meshgrid(_T, _S, indexing="ij")


@pytest.mark.parametrize(
    "fn,name", [(I_sq_RD, "I_sq_RD"), (I_sq_RD_uv, "I_sq_RD_uv")]
)
def test_rd_kernel_matches_textbook(fn, name):
    """sigway's RD kernel == independent literature form to ~1e-10."""
    got = np.array(fn(jnp.array(_TT), jnp.array(_SS), k=1.0))
    ref = kernel_RD_text(_TT, _SS)
    assert np.nanmax(np.abs(got / ref - 1.0)) < 1e-10, name


def test_radiation_kernel_class():
    """RadiationKernel wraps I_sq_RD, is k-independent, carries the RD norm."""
    kern = RadiationKernel()
    got = np.array(kern.overline_Isq(jnp.array(_TT), jnp.array(_SS), 1.0))
    ref = np.array(I_sq_RD(jnp.array(_TT), jnp.array(_SS), 1.0))
    assert np.array_equal(got, ref)
    assert kern.k_dependent is False
    assert kern.param_names == ()
    assert kern.resonant_t == ()
    assert np.isclose(kern.norm(1.0), 0.39 * 4.2e-5 / 12.0, rtol=1e-12)


def test_instant_emd_kernel_class():
    """InstantEMDKernel wraps the LV + resonant cores and the eMD metadata."""
    kern = InstantEMDKernel()
    t = jnp.array([0.3, 1.0])
    s = jnp.array([0.2, 0.5])
    # only etaR is a kernel param; kmax is unused by the cores (dummy 0.0)
    assert np.array_equal(
        np.array(kern.overline_Isq(t, s, 0.02, 2000.0)),
        np.array(I_sq_IRD_LV(t, s, 0.02, 0.0, 2000.0)),
    )
    assert np.array_equal(
        np.array(kern.overline_Isq_resonant(t, s, 0.02, 2000.0)),
        np.array(I_sq_IRD_res(t, s, 0.02, 0.0, 2000.0)),
    )
    assert kern.k_dependent is True
    assert kern.param_names == ("etaR",)
    assert np.isclose(kern.resonant_t[0], np.sqrt(3.0) - 1.0)
    assert np.isclose(kern.norm(1.0), 1.0 / 12.0)  # bare/CT default


def test_kernel_norm_override_and_error():
    """norm is overridable; an unknown preset raises a clear error."""
    assert np.isclose(RadiationKernel(norm="bare").norm(1.0), 1.0 / 12.0)
    assert np.isclose(RadiationKernel(norm=2.5).norm(1.0), 2.5)
    with pytest.raises(ValueError, match="Unknown norm preset"):
        RadiationKernel(norm="nope")
