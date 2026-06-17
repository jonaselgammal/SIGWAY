"""Binned (precomputed-coefficient) model: OmegaGW-consistent interface.

The binned path returns Omega_GW directly from precomputed coefficients (no
kernel, no (s, t) integral), so it does not go through OmegaGW. These checks pin
that it nonetheless exposes the same inference interface (parameter_names,
__call__(f, *theta), jacobian) and that the adapter just forwards to the
existing template / per-bin derivative.
"""

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from sigway.binned_pzeta import Binned_P_zeta  # noqa: E402


def test_binned_inference_interface():
    model = Binned_P_zeta("binned", "Binned", nbins=10)
    n = len(model.fp)
    assert model.parameter_names == tuple(f"A_{i}" for i in range(n))

    f = np.geomspace(2e-5, 1.0, 50)
    amps = [0.0] * n  # log10 amplitudes
    # __call__ forwards to template
    assert np.array_equal(
        np.array(model(f, *amps)), np.array(model.template(f, *amps))
    )
    # jacobian stacks the per-bin derivative; column i == dtemplate_default(i)
    jac = np.array(model.jacobian(f, amps))
    assert jac.shape == (len(f), n)
    assert np.array_equal(
        jac[:, 0], np.array(model.dtemplate_default(0, f, *amps))
    )
