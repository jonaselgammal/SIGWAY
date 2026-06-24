r"""Primordial curvature power spectrum $\mathcal{P}_\zeta(k)$ — interface and implementations.

During inflation, quantum fluctuations of the curvature perturbation $\zeta$ seed all
large-scale structure.  Their statistical properties are encoded in the primordial power
spectrum $\mathcal{P}_\zeta(k)$, defined by

$$\langle \zeta_\mathbf{k}\, \zeta_{\mathbf{k}'} \rangle
= \frac{2\pi^2}{k^3}\,\mathcal{P}_\zeta(k)\,(2\pi)^3\,\delta(\mathbf{k}+\mathbf{k}').$$

This module provides a small class hierarchy so that every model of
$\mathcal{P}_\zeta(k)$ — whether an analytic formula or a numerical solution of the
Mukhanov-Sasaki equation — exposes the same interface to the gravitational-wave
integrator:

* [ScalarPerturbations][sigway.perturbations.ScalarPerturbations] — abstract base
  class that defines the contract.
* [AnalyticPerturbations][sigway.perturbations.AnalyticPerturbations] — wraps any
  closed-form $\mathcal{P}_\zeta(k,\theta)$ supplied as a Python callable.
* [SingleFieldPerturbations][sigway.perturbations.SingleFieldPerturbations] — obtains
  $\mathcal{P}_\zeta(k)$ by numerically solving the Mukhanov-Sasaki equation for a
  given inflationary potential via [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver].

The binned / precomputed-coefficient representation is intentionally absent: it returns
$\Omega_\mathrm{GW}$ directly (bypassing the kernel and $(s,t)$ quadrature) and is
handled as a dedicated path elsewhere.
"""

from sigway.ms_solver import SingleFieldSolver

__all__ = ["ScalarPerturbations", "AnalyticPerturbations", "SingleFieldPerturbations"]


class ScalarPerturbations:
    r"""Abstract base class for the primordial curvature power spectrum $\mathcal{P}_\zeta(k)$.

    Every spectrum model in SIGWAY inherits from this class and implements
    ``__call__(k, *params)`` to return $\mathcal{P}_\zeta$ evaluated at wavenumber $k$
    for a given set of model parameters $\theta$.  The class attributes below form
    the contract that [OmegaGW][sigway.spectrum.OmegaGW] and the gravitational-wave
    integrator rely on.

    Attributes
    ----------
    param_names : tuple of str
        Ordered names of the free model parameters $\theta$ passed as ``*params`` to
        ``__call__`` and ``prepare``.  The order must be consistent with the parameter
        vector used for inference.  For example, ``("logAs", "logks")`` means the first
        positional argument after $k$ is the log-amplitude and the second is the
        log-peak scale.
    jittable : bool
        Whether $\mathcal{P}_\zeta(k,\theta)$ can be evaluated inside the compiled,
        differentiable code path (JAX).  Analytic spectra set this to ``True``.  The
        Mukhanov-Sasaki solver sets it to ``False`` because it calls SciPy ODE routines
        internally and must therefore be run separately, before the main integration.
    nonsmooth_params : tuple of str
        Subset of ``param_names`` whose partial derivatives cannot be obtained by
        automatic differentiation.  This arises when a parameter enters a sharp feature
        — for example, a hard ultraviolet cutoff $k_{\max}$ implemented as a Heaviside
        step function, or a parameter that sets an integration limit.  At a
        discontinuity, automatic differentiation returns zero or ``nan`` rather than the
        correct finite-difference slope.  [OmegaGW][sigway.spectrum.OmegaGW] uses
        central finite differences for these parameters when building the Jacobian.
        Leave as an empty tuple if all parameters enter the spectrum smoothly.
    """

    param_names = ()
    # Analytic spectra can be evaluated inside the compiled, differentiable code path;
    # the Mukhanov-Sasaki solver cannot (it calls SciPy ODE routines internally),
    # so it is run separately before the main integration.
    jittable = True
    # Parameters that enter a sharp feature (step function, hard cutoff k_max, etc.)
    # need finite differences for their derivatives; OmegaGW.jacobian uses central
    # differences for these.
    nonsmooth_params = ()

    def __call__(self, k, *params):
        r"""Evaluate $\mathcal{P}_\zeta$ at wavenumber $k$.

        Parameters
        ----------
        k : array_like
            Comoving wavenumber(s) $k$, in the units used throughout the model
            ($\mathrm{s}^{-1}$), at which to evaluate the primordial power
            spectrum.
        *params : float
            Model parameters $\theta$ in the order given by ``param_names``.

        Returns
        -------
        array_like
            $\mathcal{P}_\zeta(k,\theta)$ evaluated at the supplied $k$.

        Raises
        ------
        NotImplementedError
            Always — concrete subclasses must override this method.
        """
        raise NotImplementedError

    def prepare(self, kint, *params):
        r"""Return a single-argument $\mathcal{P}_\zeta(k)$ with parameters fixed.

        The gravitational-wave integrator calls this once before evaluating
        $\mathcal{P}_\zeta$ on its dense $(k u,\, k v)$ grid.  The default
        implementation simply closes over ``__call__`` with the given ``params``.
        [SingleFieldPerturbations][sigway.perturbations.SingleFieldPerturbations]
        overrides this to solve the Mukhanov-Sasaki equation once on ``kint`` and
        return a fast spline interpolant, avoiding a full ODE solve at every grid
        point.

        Parameters
        ----------
        kint : array_like
            Wavenumber grid on which the integrator will sample $\mathcal{P}_\zeta$.
            Forwarded to the Mukhanov-Sasaki solver when overridden; ignored by the
            default analytic implementation.
        *params : float
            Model parameters $\theta$ in the order given by ``param_names``.

        Returns
        -------
        callable
            A single-argument function ``f(k)`` returning $\mathcal{P}_\zeta(k)$
            with $\theta$ already applied.
        """
        return lambda k: self(k, *params)


class AnalyticPerturbations(ScalarPerturbations):
    r"""Closed-form primordial power spectrum $\mathcal{P}_\zeta(k,\theta)$.

    The lightest-weight way to define a spectrum model: supply any function
    ``func(k, *params)`` that computes $\mathcal{P}_\zeta$ analytically together
    with the ordered names of its free parameters.  The result is a fully
    compliant [ScalarPerturbations][sigway.perturbations.ScalarPerturbations] object
    that can be evaluated inside the compiled, differentiable code path, provided
    ``func`` itself uses JAX-compatible operations (e.g. ``jax.numpy``).

    Parameters
    ----------
    func : callable
        A function with signature ``func(k, *params) -> array_like`` returning
        $\mathcal{P}_\zeta$ at wavenumber $k$ for parameters $\theta$.  Use
        ``jax.numpy`` operations to keep the spectrum differentiable and fast.
    param_names : tuple of str, optional
        Ordered names of the free parameters expected by ``func``, e.g.
        ``("logAs", "logks", "sigma")``.  Defaults to an empty tuple.
    nonsmooth_params : tuple of str, optional
        Parameters in ``param_names`` that enter a sharp feature in the spectrum
        (e.g. a hard cutoff $k_{\max}$ via a Heaviside step), for which automatic
        differentiation does not give a meaningful slope.
        [OmegaGW][sigway.spectrum.OmegaGW] falls back to central finite differences
        for these.  Defaults to an empty tuple.

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
    Wrap a log-normal peak in $\mathcal{P}_\zeta(k)$:

    >>> import jax.numpy as jnp
    >>> from sigway.perturbations import AnalyticPerturbations
    >>> def pz(k, logAs, logks):
    ...     return 10.0**logAs * jnp.exp(-0.5*(jnp.log(k/10**logks)/0.3)**2)
    >>> perts = AnalyticPerturbations(pz, ("logAs", "logks"))

    If the spectrum has a hard ultraviolet cutoff $k_{\max}$ whose derivative must
    be estimated by finite differences, declare it via ``nonsmooth_params``:

    >>> perts_cut = AnalyticPerturbations(pz, ("logAs", "kmax"), nonsmooth_params=("kmax",))
    """

    def __init__(self, func, param_names=(), nonsmooth_params=()):
        self.func = func
        self.param_names = tuple(param_names)
        self.nonsmooth_params = tuple(nonsmooth_params)

    def __call__(self, k, *params):
        r"""Evaluate the wrapped callable at wavenumber $k$.

        Parameters
        ----------
        k : array_like
            Comoving wavenumber(s) $k$ at which to evaluate $\mathcal{P}_\zeta$.
        *params : float
            Model parameters $\theta$ in the order given by ``param_names``,
            forwarded directly to ``func``.

        Returns
        -------
        array_like
            $\mathcal{P}_\zeta(k,\theta)$ as returned by ``func``.
        """
        return self.func(k, *params)


class SingleFieldPerturbations(ScalarPerturbations):
    r"""Primordial power spectrum $\mathcal{P}_\zeta(k)$ from single-field inflation.

    For inflationary models where no analytic expression for $\mathcal{P}_\zeta(k)$
    is available, this class solves the Mukhanov-Sasaki equation numerically using
    [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver].  The result is exposed
    through the same ``__call__`` / ``prepare`` interface as every other
    [ScalarPerturbations][sigway.perturbations.ScalarPerturbations] subclass, so the
    gravitational-wave integrator is unaware of whether the spectrum is analytic or
    numerical.

    Because the Mukhanov-Sasaki solver calls SciPy ODE routines internally, it cannot
    be evaluated inside the compiled, differentiable code path.  ``jittable`` is
    therefore ``False``, and the integrator runs this path eagerly.  Gradients with
    respect to inflationary potential parameters must be obtained by finite differences.

    Parameters
    ----------
    solver : SingleFieldSolver
        A configured [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver] instance
        encoding the inflationary potential and integration settings.  Passing any
        other object raises ``ValueError``.
    param_names : tuple of str, optional
        Ordered names of the free parameters forwarded to the solver (e.g. potential
        coefficients).  Defaults to an empty tuple.

    Attributes
    ----------
    solver : SingleFieldSolver
        The wrapped [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver].
    param_names : tuple of str
        Ordered parameter names.
    jittable : bool
        Always ``False`` for this class; see above.

    Raises
    ------
    ValueError
        If ``solver`` is not an instance of
        [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver].
    """

    jittable = False  # Mukhanov-Sasaki solver uses SciPy ODE routines -> run eagerly

    def __init__(self, solver, param_names=()):
        if not isinstance(solver, SingleFieldSolver):
            raise ValueError(
                "SingleFieldPerturbations expects a SingleFieldSolver; for a "
                "closed-form spectrum use AnalyticPerturbations."
            )
        self.solver = solver
        self.param_names = tuple(param_names)

    def __call__(self, k, *params):
        r"""Evaluate $\mathcal{P}_\zeta$ by solving the Mukhanov-Sasaki equation at $k$.

        Delegates directly to the wrapped solver.  This is the point-by-point
        evaluation path.  For bulk evaluation on the integrator grid — where solving
        the ODE at every point separately would be prohibitively slow — use
        ``prepare`` instead.

        Parameters
        ----------
        k : array_like
            Comoving wavenumber(s) $k$ at which to evaluate $\mathcal{P}_\zeta$.
        *params : float
            Inflationary model parameters in the order given by ``param_names``,
            forwarded to the solver.

        Returns
        -------
        array_like
            $\mathcal{P}_\zeta(k)$ at the requested wavenumbers.
        """
        return self.solver(k, *params)

    def prepare(self, kint, *params):
        r"""Solve the Mukhanov-Sasaki equation once and return a fast interpolant.

        Rather than re-running the full ODE for every point on the integrator's dense
        $(k u,\, k v)$ grid, this method solves the Mukhanov-Sasaki system once on the
        supplied wavenumber grid ``kint`` and returns a SciPy spline interpolant.  The
        integrator then evaluates the cheap interpolant at each grid point.

        Parameters
        ----------
        kint : array_like
            Wavenumber grid $k_{\mathrm{int}}$ on which to solve the
            Mukhanov-Sasaki equation.  Should span the range of $k$ values required
            by the subsequent integration.
        *params : float
            Inflationary model parameters forwarded to the solver.

        Returns
        -------
        callable
            A single-argument spline interpolant ``f(k)`` approximating
            $\mathcal{P}_\zeta(k)$ across the range of ``kint``.
        """
        return self.solver.run(kint, *params)
