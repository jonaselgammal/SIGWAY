"""Deprecated shim: the MS solver moved to :mod:`sigway.single_field`.

``SingleFieldSolver`` and its thin ``SingleFieldPerturbations`` wrapper were
merged into a single :class:`sigway.single_field.SingleFieldPerturbations`,
constructed directly from the potential.  This module re-exports the names that
survived and raises a clear error for the removed two-step construction.  It
will be removed in a future release.
"""

import warnings

from sigway.single_field import (  # noqa: F401  (re-exported for back-compat)
    SingleFieldPerturbations,
    SolverOptions,
    ConsistencyError,
)

warnings.warn(
    "sigway.ms_solver is deprecated; import from sigway.single_field instead "
    "(SingleFieldSolver + its SingleFieldPerturbations wrapper are now the "
    "single class sigway.single_field.SingleFieldPerturbations, built directly "
    "from the potential).",
    DeprecationWarning,
    stacklevel=2,
)


class SingleFieldSolver:
    """Removed. Use :class:`sigway.single_field.SingleFieldPerturbations`."""

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "SingleFieldSolver has been merged into "
            "sigway.single_field.SingleFieldPerturbations. Construct it "
            "directly from the potential, e.g. "
            "SingleFieldPerturbations(V, ('a', 'lam'), phi0=3.0), instead of "
            "wrapping a solver."
        )
