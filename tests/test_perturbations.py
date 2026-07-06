"""ScalarPerturbations wrappers: faithful to the underlying func / solver.

These pin the uniform P_zeta(k) interface the integrator will rely on. The
physics of each spectrum is covered elsewhere (analytic forms in the regression
suite; the MS solver in test_ms_solver) -- here we only check the wrappers don't
distort what they wrap, and that param_names is carried through.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from sigway.perturbations import (  # noqa: E402
    ScalarPerturbations,
    AnalyticPerturbations,
)
from sigway.single_field import SingleFieldPerturbations  # noqa: E402
from _sigway_configs import pzeta_ln  # noqa: E402


def test_analytic_perturbations_wraps_callable():
    """AnalyticPerturbations(func) == func, and carries param_names."""
    names = ("logAs", "logDelta", "logks")
    pert = AnalyticPerturbations(pzeta_ln, names)
    assert isinstance(pert, ScalarPerturbations)
    k = jnp.geomspace(1e-4, 1e-1, 50)
    p = (-2.5, -0.3, -2.0)
    assert np.array_equal(np.array(pert(k, *p)), np.array(pzeta_ln(k, *p)))
    assert pert.param_names == names


def _quadratic_pert():
    def V(phi, m):
        return 0.5 * m**2 * phi**2

    return SingleFieldPerturbations(V, ("m",), phi0=16.0, N_CMB_to_end=55.0)


def test_single_field_perturbations_interface():
    """The merged SingleFieldPerturbations is a non-jittable ScalarPerturbations
    whose prepare() interpolant reproduces __call__ at the solve nodes."""
    pert = _quadratic_pert()
    assert isinstance(pert, ScalarPerturbations)
    assert pert.jittable is False
    assert pert.param_names == ("m",)

    k = jnp.geomspace(1e-4, 1e-1, 20)
    m = 6e-6
    pz = np.array(pert(k, m))
    assert np.all(np.isfinite(pz)) and np.all(pz > 0)

    # prepare() solves once and returns a spline; at the nodes it matches __call__
    nodes = np.array(pert.prepare(k, m)(k))
    assert np.allclose(nodes, pz, rtol=1e-6)
