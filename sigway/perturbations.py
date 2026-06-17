"""Scalar (curvature) perturbation power spectra P_zeta(k).

A small hierarchy that gives every representation of P_zeta a uniform interface
so the integrator no longer cares whether the spectrum is an analytic closed
form or solved from the Mukhanov-Sasaki equation:

* ``AnalyticPerturbations`` wraps a closed-form ``P_zeta(k, *params)`` callable
  (log-normal, broken power law, ...). It is itself callable, so it is a drop-in
  for the bare callables the legacy classes accept.
* ``SingleFieldPerturbations`` wraps a ``SingleFieldSolver`` and exposes the
  same ``__call__(k, *params)`` interface, plus ``prepare`` which returns the
  interpolated spectrum the integrator evaluates on its grid.

Each carries ``param_names`` so the top-level model can build a single ordered
parameter vector for inference (Phase 3).

The binned/precomputed-coefficient representation is intentionally not here: it
returns Omega_GW directly (no kernel, no (s, t) quadrature) and is handled as a
dedicated path in Phase 4.
"""

from sigway.ms_solver import SingleFieldSolver


class ScalarPerturbations:
    """Base class for curvature-perturbation power spectra P_zeta(k).

    Subclasses implement ``__call__(k, *params)`` returning P_zeta at ``k`` and
    declare ``param_names`` (the ordered names of ``*params``).
    """

    param_names = ()

    def __call__(self, k, *params):
        raise NotImplementedError


class AnalyticPerturbations(ScalarPerturbations):
    """Closed-form P_zeta(k, *params) supplied as a callable."""

    def __init__(self, func, param_names=()):
        self.func = func
        self.param_names = tuple(param_names)

    def __call__(self, k, *params):
        return self.func(k, *params)


class SingleFieldPerturbations(ScalarPerturbations):
    """P_zeta from the Mukhanov-Sasaki solver for a single-field model.

    Wraps a ``SingleFieldSolver``. ``__call__(k, *params)`` returns P_zeta
    solved directly at ``k``; ``prepare(kint, *params)`` returns a fast
    interpolated P_zeta(k) callable (one solve on ``kint``) for the
    integrator to evaluate on the dense (k u, k v) grid.
    """

    def __init__(self, solver, param_names=()):
        if not isinstance(solver, SingleFieldSolver):
            raise ValueError(
                "SingleFieldPerturbations expects a SingleFieldSolver; for a "
                "closed-form spectrum use AnalyticPerturbations."
            )
        self.solver = solver
        self.param_names = tuple(param_names)

    def __call__(self, k, *params):
        return self.solver(k, *params)

    def prepare(self, kint, *params):
        """Return an interpolated P_zeta(k) callable from a single solve."""
        return self.solver.run(kint, *params)
