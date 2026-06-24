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

__all__ = ["ScalarPerturbations", "AnalyticPerturbations", "SingleFieldPerturbations"]


class ScalarPerturbations:
    """Base class for curvature-perturbation power spectra P_zeta(k).

    All concrete spectrum models inherit from this class and implement
    ``__call__(k, *params)`` to return P_zeta evaluated at wavenumber ``k``
    for the given ordered parameter vector ``*params``.  The class attributes
    below form the contract that [OmegaGW][sigway.spectrum.OmegaGW] and the
    integrators rely on.

    Attributes
    ----------
    param_names : tuple of str
        Ordered names of the free parameters passed as ``*params`` to
        ``__call__`` and ``prepare``.  The order must be consistent with the
        parameter vector used for inference.
    jittable : bool
        Whether ``__call__`` can be traced by JAX (``jit`` / ``grad``).
        Analytic spectra set this to ``True``; the Mukhanov-Sasaki solver sets
        it to ``False`` because it uses SciPy splines internally, so the
        integrator runs that code path eagerly.
    nonsmooth_params : tuple of str
        Subset of ``param_names`` whose partial derivatives must be computed
        with finite differences rather than JAX autodiff.  These are parameters
        that enter a discontinuous operation such as a Heaviside step function
        or an integration-limit cutoff (e.g. a hard ``k_max``).
        [OmegaGW][sigway.spectrum.OmegaGW] uses central differences for these
        parameters when building the Jacobian.
    """

    param_names = ()
    # Whether P_zeta can be evaluated inside jit/grad. Analytic spectra are
    # jittable; the MS solver is not (it uses scipy splines internally), so the
    # integrator runs that path eagerly.
    jittable = True
    # Parameters whose derivative needs finite differences rather than autodiff
    # because they enter a step / integration limit (e.g. a heaviside cutoff
    # kmax). OmegaGW.jacobian uses central differences for these.
    nonsmooth_params = ()

    def __call__(self, k, *params):
        """Evaluate P_zeta at wavenumber ``k``.

        Parameters
        ----------
        k : array_like
            Wavenumber(s) at which to evaluate the primordial power spectrum.
        *params : float
            Free parameters in the order specified by ``param_names``.

        Returns
        -------
        array_like
            P_zeta evaluated at ``k``.

        Raises
        ------
        NotImplementedError
            Always — subclasses must override this method.
        """
        raise NotImplementedError

    def prepare(self, kint, *params):
        """Return a 1-arg ``P_zeta(k)`` callable with ``params`` baked in.

        The integrator calls this once before evaluating P_zeta on its dense
        ``(k*u, k*v)`` grid.  The default implementation simply closes over
        ``__call__``; the Mukhanov-Sasaki subclass overrides this to solve the
        system once on ``kint`` and return a fast interpolant, avoiding
        repeated solves on every grid point.

        Parameters
        ----------
        kint : array_like
            Wavenumber grid on which the integrator will sample P_zeta.
            Passed to the MS solver when overridden; ignored by the default.
        *params : float
            Free parameters in the order specified by ``param_names``.

        Returns
        -------
        callable
            A single-argument function ``f(k)`` returning P_zeta at ``k``
            with the given ``params`` already applied.
        """
        return lambda k: self(k, *params)


class AnalyticPerturbations(ScalarPerturbations):
    """Closed-form P_zeta(k, *params) supplied as a Python callable.

    This is the lightest-weight way to define a primordial power spectrum: pass
    any function ``func(k, *params)`` and a matching tuple of parameter names.
    The result is a fully compliant [ScalarPerturbations][sigway.perturbations.ScalarPerturbations]
    subclass that is JAX-traceable (``jittable = True``) provided ``func``
    itself uses JAX-compatible operations.

    Parameters
    ----------
    func : callable
        A function with signature ``func(k, *params) -> array_like`` returning
        P_zeta at wavenumber ``k`` for the given parameter values.  Must be
        JAX-compatible if autodiff or ``jit`` compilation is required.
    param_names : tuple of str, optional
        Ordered names of the free parameters expected by ``func``.  Defaults
        to an empty tuple (no free parameters).
    nonsmooth_params : tuple of str, optional
        Subset of ``param_names`` for which JAX autodiff is invalid (e.g.
        parameters that enter a Heaviside cutoff).  Defaults to an empty
        tuple.  [OmegaGW][sigway.spectrum.OmegaGW] will use central finite
        differences for these parameters.

    Attributes
    ----------
    func : callable
        The wrapped callable as supplied at construction time.
    param_names : tuple of str
        Ordered parameter names.
    nonsmooth_params : tuple of str
        Parameters requiring finite-difference derivatives.

    Examples
    --------
    Wrap a log-normal primordial power spectrum:

    >>> import jax.numpy as jnp
    >>> def lognormal_pzeta(k, A, k_star, sigma):
    ...     return A * jnp.exp(-jnp.log(k / k_star)**2 / (2 * sigma**2))
    >>> pzeta = AnalyticPerturbations(
    ...     lognormal_pzeta,
    ...     param_names=("A", "k_star", "sigma"),
    ... )
    >>> pzeta(1.0, 1e-2, 1.0, 0.5)  # doctest: +SKIP
    """

    def __init__(self, func, param_names=(), nonsmooth_params=()):
        self.func = func
        self.param_names = tuple(param_names)
        self.nonsmooth_params = tuple(nonsmooth_params)

    def __call__(self, k, *params):
        """Evaluate the wrapped callable at ``k``.

        Parameters
        ----------
        k : array_like
            Wavenumber(s) at which to evaluate P_zeta.
        *params : float
            Free parameters in the order specified by ``param_names``,
            forwarded directly to ``func``.

        Returns
        -------
        array_like
            ``func(k, *params)``.
        """
        return self.func(k, *params)


class SingleFieldPerturbations(ScalarPerturbations):
    """P_zeta from the Mukhanov-Sasaki solver for a single-field inflation model.

    Wraps a [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver] and exposes
    the same ``__call__`` / ``prepare`` interface as every other
    [ScalarPerturbations][sigway.perturbations.ScalarPerturbations] subclass,
    so the integrator does not need to know that the spectrum comes from a
    numerical ODE solver rather than a closed-form expression.

    Because [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver] uses SciPy
    splines internally, P_zeta cannot be traced by JAX; ``jittable`` is
    therefore ``False`` and the integrator evaluates this path eagerly.

    Parameters
    ----------
    solver : SingleFieldSolver
        A configured [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver]
        instance.  Passing anything else raises ``ValueError``.
    param_names : tuple of str, optional
        Ordered names of the free parameters forwarded to the solver.  Defaults
        to an empty tuple.

    Attributes
    ----------
    solver : SingleFieldSolver
        The wrapped [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver].
    param_names : tuple of str
        Ordered parameter names.
    jittable : bool
        Always ``False`` for this class.

    Raises
    ------
    ValueError
        If ``solver`` is not an instance of
        [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver].
    """

    jittable = False  # uses scipy splines internally -> integrate eagerly

    def __init__(self, solver, param_names=()):
        if not isinstance(solver, SingleFieldSolver):
            raise ValueError(
                "SingleFieldPerturbations expects a SingleFieldSolver; for a "
                "closed-form spectrum use AnalyticPerturbations."
            )
        self.solver = solver
        self.param_names = tuple(param_names)

    def __call__(self, k, *params):
        """Evaluate P_zeta by solving the Mukhanov-Sasaki equation at ``k``.

        Delegates directly to the wrapped solver.  This is the point-by-point
        path; for bulk evaluation on the integrator grid use ``prepare``.

        Parameters
        ----------
        k : array_like
            Wavenumber(s) at which to evaluate P_zeta.
        *params : float
            Free parameters in the order specified by ``param_names``,
            forwarded to the solver.

        Returns
        -------
        array_like
            P_zeta at ``k``.
        """
        return self.solver(k, *params)

    def prepare(self, kint, *params):
        """Solve once on ``kint`` and return a fast interpolated P_zeta(k).

        Rather than re-running the Mukhanov-Sasaki ODE for every point on the
        integrator's ``(k*u, k*v)`` grid, this method delegates to
        ``solver.run`` which solves the system on the supplied ``kint`` grid
        and returns a SciPy-spline interpolant.  The integrator then evaluates
        the cheap interpolant at each grid point.

        Parameters
        ----------
        kint : array_like
            Wavenumber grid on which the ODE is solved.
        *params : float
            Free parameters forwarded to the solver.

        Returns
        -------
        callable
            A single-argument interpolant ``f(k)`` for P_zeta.
        """
        return self.solver.run(kint, *params)
