"""ScalarPerturbations wrappers: faithful to the underlying func / solver.

These pin the uniform P_zeta(k) interface the integrator will rely on. The
physics of each spectrum is covered elsewhere (analytic forms in the regression
suite; the MS solver in test_ms_solver) -- here we only check the wrappers don't
distort what they wrap, and that param_names is carried through.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from sigway.ms_solver import SingleFieldSolver  # noqa: E402
from sigway.perturbations import (  # noqa: E402
    ScalarPerturbations,
    AnalyticPerturbations,
    SingleFieldPerturbations,
)
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


def _quadratic_solver():
    def V(phi, m):
        return 0.5 * m**2 * phi**2

    return SingleFieldSolver(
        V,
        phi0=16.0,
        pi0=0.0,
        N_CMB_to_end=55.0,
        k=jnp.geomspace(1e-4, 1e-1, 20),
    )


def test_single_field_perturbations_matches_solver():
    """SingleFieldPerturbations forwards __call__ and prepare to the solver."""
    solver = _quadratic_solver()
    pert = SingleFieldPerturbations(solver, param_names=("m",))
    k = jnp.geomspace(1e-4, 1e-1, 20)
    m = 6e-6
    assert np.allclose(np.array(pert(k, m)), np.array(solver(k, m)))
    kk = jnp.geomspace(2e-4, 5e-2, 10)
    got = np.array(pert.prepare(k, m)(kk))
    ref = np.array(solver.run(k, m)(kk))
    assert np.allclose(got, ref)
    assert pert.param_names == ("m",)


def test_single_field_rejects_non_solver():
    """A bare callable must go through AnalyticPerturbations, not this class."""
    with pytest.raises(ValueError, match="SingleFieldSolver"):
        SingleFieldPerturbations(lambda k: k)
