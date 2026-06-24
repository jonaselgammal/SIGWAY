"""
Single-field Mukhanov-Sasaki solver.

Solves the inflationary background and Mukhanov-Sasaki perturbation equations
for a single-field slow-roll (or ultra-slow-roll) model given a JAX-compatible
potential ``V(phi, *params)``. The background ODE is integrated first; an
optional consistency check against CMB observables is then performed; finally
the scalar power spectrum P_zeta(k) is computed via mode-by-mode integration
of the perturbation equations.

Public API
----------
- [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver]
- [SolverOptions][sigway.ms_solver.SolverOptions]
- [ConsistencyError][sigway.ms_solver.ConsistencyError]
"""

__all__ = ["SingleFieldSolver", "SolverOptions", "ConsistencyError"]

# Global
import jax
from jax import numpy as jnp
from jax import jit

from jax.scipy.stats import multivariate_normal

from collections import namedtuple

from diffrax import (
    diffeqsolve,
    ODETerm,
    Tsit5,
    PIDController,
    Event,
    SaveAt,
    backward_hermite_coefficients,
    # LinearInterpolation,
    CubicInterpolation,
)


from functools import partial

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FormatStrFormatter

# Local
from sigway.utils import (
    efolds_from_wavenumber_si_units,
    H_from_wavenumber,
    wavenumber_from_efolds_si_units,
)

jax.config.update("jax_enable_x64", True)


# Create an Exception class for consistency checks
class ConsistencyError(Exception):
    """Raised when a physical consistency check fails during solving.

    Typical triggers include insufficient e-folds to reach the CMB scale, or
    too few e-folds remaining to evolve perturbations across the required
    sub- and super-horizon window.
    """


@partial(jax.jit, static_argnums=(0, 1, 2, 4))
def _run_background(Ud, max_efolds, phi0, pvalues, solver_opts):
    """
    Solves the background equations for an ultra slow rolling single inflaton
    field.

    Parameters
    ----------
    Ud : function
        Function that computes the first derivative of the potential with
        respect to the field.
    max_efolds : int
        Maximum number of e-folds to integrate.
    phi0 : float
        Initial value of the inflaton field.
    pvalues : tuple
        Parameters for the potential function.
    solver_opts : dict
        Options for the differential equation solver.

    Returns
    -------
    diffrax.diffeqsolve.Solution
        The solution of the differential equations containing the evolution of
        the inflaton field, its derivative, and the Hubble parameter over the
        specified number of e-folds.
    """

    def equations(n, variables, args):
        """
        Defines the system of differential equations for the background
        evolution.

        Parameters
        ----------
        n : float
            The number of e-folds.
        variables : array-like
            The state variables of the system [x, y, h], where x is the field
            value, y is its derivative with respect to the number of e-folds,
            and h is the Hubble parameter.
        args : tuple
            Additional arguments for the function (unused here).

        Returns
        -------
        array-like
            Derivatives of the state variables.
        """
        x, y, h = variables
        derx = y
        dery = -3 * (1 - y**2 / 6) * y - Ud(x, *pvalues) / h**2
        derh = -(y**2) / 2 * h
        return jnp.array([derx, dery, derh])

    def inflation_end(t, y, *args, **kwargs):
        """
        Event function to determine the end of inflation.

        Parameters
        ----------
        t : float
            The current time or e-fold number.
        y : array-like
            The current state of the system [x, y, h].
        args : tuple
            Additional arguments for the function (unused here).

        Returns
        -------
        bool
            True if inflation has ended, False otherwise.
        """
        return y[1] ** 2 > 2

    # Initial conditions
    Ud0 = Ud(phi0, *pvalues)
    y0 = -(3 / Ud0) * (
        jnp.sqrt(1 + 2 / 3 * Ud0**2) - 1
    )  # Initial derivative wrt N of the rescaled inflaton
    h0 = (
        1 / jnp.sqrt(6) * (jnp.sqrt(1 + 2 / 3 * Ud0**2) + 1) ** (1 / 2)
    )  # Initial rescaled Hubble rate

    # Setting up the differential equation solver
    term = ODETerm(equations)
    solver = Tsit5()
    stepsize_controller = PIDController(
        rtol=solver_opts.rtol, atol=solver_opts.atol
    )
    stopping_event = Event(inflation_end)
    saveat = solver_opts.saveat

    # Solving the differential equations
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=max_efolds,
        dt0=solver_opts.dt0,
        y0=jnp.array([phi0, y0, h0]),
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=solver_opts.max_steps,
        event=stopping_event,
    )

    return sol


@partial(jax.jit, static_argnums=(0, 1, 9))
def _solve_perturbations(
    Ud, Udd, phiIn, yIn, hIn, nin, nout, lograt, pvalues, solver_opts
):
    """
    Solves the perturbation equations for an ultra slow rolling single inflaton
    field.

    Parameters
    ----------
    Ud : function
        Function that computes the first derivative of the potential with
        respect to the field.
    Udd : function
        Function that computes the second derivative of the potential with
        respect to the field.
    phiIn : float
        Initial value of the inflaton field.
    yIn : float
        Initial value of the derivative of the inflaton field with respect to
        the number of e-folds.
    hIn : float
        Initial value of the Hubble parameter.
    nin : float
        Initial number of e-folds.
    nout : float
        Final number of e-folds.
    lograt : float
        log(k/a) at nin.
    pvalues : tuple
        Parameters for the potential function.
    solver_opts : NamedTuple (SolverOptions)
        Options for the differential equation solver.

    Returns
    -------
    diffrax.diffeqsolve.Solution
        The solution of the differential equations containing the evolution of
        the perturbations over the specified number of e-folds.
    """
    # Initial conditions for the perturbations

    dPhiRin = 1.0  # Initial value of the real part of tilde Delta phi
    dPhiRpRin = -1.0  # Initial value of the derivative of the real part of
    # tilde Delta phi with respect to N
    dPhiIin = 0.0  # Initial value of the imaginary part of tilde Delta phi
    dPhiIpIin = 0.0  # Initial value of the derivative of the imaginary part of
    # tilde Delta phi with respect to N

    def equations_perturbations(n, variables, args):
        """
        Defines the system of differential equations for the perturbations.

        Parameters
        ----------
        n : float
            The number of e-folds.
        variables : array-like
            The state variables of the system [x, y, h, dPhiR, dPhipR, dPhiI,
            dPhipI], where x is the field value, y is its derivative with
            respect to the number of e-folds, h is the Hubble parameter, dPhiR
            and dPhiI are the real and imaginary parts of the perturbation
            field, and dPhipR and dPhipI are their derivatives with respect to
            the number of e-folds.
        args : tuple
            Additional arguments for the function (nin, lograt).

        Returns
        -------
        array-like
            Derivatives of the state variables.
        """
        nin, lograt = args
        x, y, h, dPhiR, dPhipR, dPhiI, dPhipI = variables
        ud = Ud(x, *pvalues)
        udd = Udd(x, *pvalues)
        derx = y
        dery = -3 * (1 - y**2 / 6) * y - ud / h**2
        derh = -(y**2) / 2 * h

        derPhiR = dPhipR
        derPhipR = -(
            (3 - y**2 / 2) * dPhipR
            + 2 * jnp.exp(lograt - n + nin) / h * dPhipI
            + ((udd + 2 * ud * y) / h**2 + 3 * y**2 - y**4 / 2) * dPhiR
            + 2 * jnp.exp(lograt - n + nin) / h * dPhiI
        )

        derPhiI = dPhipI
        derPhipI = -(
            (3 - y**2 / 2) * dPhipI
            - 2 * jnp.exp(lograt - n + nin) / h * dPhipR
            + ((udd + 2 * ud * y) / h**2 + 3 * y**2 - y**4 / 2) * dPhiI
            - 2 * jnp.exp(lograt - n + nin) / h * dPhiR
        )

        return jnp.array(
            [derx, dery, derh, derPhiR, derPhipR, derPhiI, derPhipI]
        )

    # Setting up the differential equation solver
    term = ODETerm(equations_perturbations)
    solver = Tsit5()
    stepsize_controller = PIDController(
        rtol=solver_opts.rtol, atol=solver_opts.atol
    )
    saveat = solver_opts.saveat

    # Solving the differential equations
    sol = diffeqsolve(
        term,
        solver,
        t0=nin,
        t1=nout,
        dt0=solver_opts.dt0,
        y0=jnp.array([phiIn, yIn, hIn, dPhiRin, dPhiRpRin, dPhiIin, dPhiIpIin]),
        stepsize_controller=stepsize_controller,
        args=(nin, lograt),
        saveat=saveat,
        max_steps=solver_opts.max_steps,
    )

    return sol


@partial(jax.jit, static_argnums=(0, 1, 10))
def _run_perturbations(
    Ud, Udd, phiIn, yIn, yOut, hIn, nin, nout, lograt, pvalues, solver_opts
):
    """
    Runs the perturbation solver and computes the dimensionless power spectrum
    Pzeta/V0.

    Parameters
    ----------
    Ud : function
        Function that computes the first derivative of the potential with
        respect to the field.
    Udd : function
        Function that computes the second derivative of the potential with
        respect to the field.
    phiIn : float
        Initial value of the inflaton field.
    yIn : float
        Initial value of the derivative of the inflaton field with respect to
        the number of e-folds.
    yOut : float
        Final value of the derivative of the inflaton field with respect to the
        number of e-folds.
    hIn : float
        Initial value of the Hubble parameter.
    nin : float
        Initial number of e-folds.
    nout : float
        Final number of e-folds.
    lograt : float
        log(k/a) at nin.
    pvalues : tuple
        Parameters for the potential function.
    solver_opts : NamedTuple (SolverOptions)
        Options for the differential equation solver.

    Returns
    -------
    float
        The dimensionless power spectrum Pzeta/V0.
    """
    # Solve the perturbation equations
    sol = _solve_perturbations(
        Ud, Udd, phiIn, yIn, hIn, nin, nout, lograt, pvalues, solver_opts
    )

    # Extract the real and imaginary parts of the perturbation field at the
    # final step
    deltaPhiR = sol.ys[-1][3]
    deltaPhiI = sol.ys[-1][5]
    yOut = sol.ys[-1][1]

    # Compute the dimensionless power spectrum Pzeta/V0
    Pzeta_by_V0 = (
        1
        / (4 * jnp.pi**2)
        * jnp.exp(2 * lograt)
        * (deltaPhiR**2 + deltaPhiI**2)
        / yOut**2
    )

    return Pzeta_by_V0


class SolverOptions(namedtuple(
    "SolverOptions", ["rtol", "atol", "max_steps", "dt0", "saveat"]
)):
    """Named tuple holding ODE solver configuration for diffrax.

    Used separately for the background and perturbation solvers via the
    ``background_solver_opts`` and ``perturbation_solver_opts`` constructor
    arguments of [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver].

    Attributes
    ----------
    rtol : float
        Relative tolerance for the adaptive step-size (PID) controller.
    atol : float
        Absolute tolerance for the adaptive step-size (PID) controller.
    max_steps : int
        Maximum number of ODE steps before the solver gives up.
    dt0 : float
        Initial step size in e-folds.
    saveat : diffrax.SaveAt
        Diffrax ``SaveAt`` object specifying which solution points to store.
    """


@jit
def interpolation_inner(knew, k, coeff):
    return CubicInterpolation(k, coeff).evaluate(knew)


class SingleFieldSolver:
    """Solves the Mukhanov-Sasaki equation for a single-field inflationary model.

    Given a JAX-compatible potential ``V(phi, *params)``, this class integrates
    the background Klein-Gordon + Friedmann equations and then solves the
    Mukhanov-Sasaki perturbation equations mode by mode to compute the
    dimensionful scalar power spectrum P_zeta(k).

    Calling ``solver.run(k, *params)`` returns a callable ``P_zeta(k_new)``
    that interpolates the computed spectrum.  Calling the instance directly
    (``solver(k, *params)``) returns the raw P_zeta array evaluated at ``k``.

    Parameters
    ----------
    V : callable
        Potential function ``V(phi, *params)``.  Must be JAX-traceable
        (i.e. compatible with ``jax.jit`` and ``jax.grad``).
    phi0 : float, optional
        Initial field value.  Also used as the normalisation point for the
        internal rescaled potential ``U = V / V(phi0)``.  Default ``0.0``.
    pi0 : float, optional
        Initial field momentum (currently stored but not used to set the
        background initial conditions, which are instead fixed by slow-roll
        attractor values).  Default ``0.0``.
    N_CMB_to_end : float, optional
        Number of e-folds from the CMB pivot scale to the end of inflation,
        assuming instantaneous reheating.  Default ``65.0``.
    max_efolds : float, optional
        Maximum number of e-folds for the background integration.
        Default ``1000.0``.
    cmb_bounds : dict, optional
        Dictionary with keys ``"means"`` (array of shape ``(3,)``) and
        ``"cov"`` (array of shape ``(3, 3)``) encoding Gaussian CMB
        observational constraints on
        ``[ln(10^10 P_zeta), n_s, r]``.
        Used only when ``check_consistency=True``.  Defaults to Planck-like
        values if not provided.
    check_consistency : bool, optional
        If ``True``, evaluate the log-likelihood against ``cmb_bounds`` at
        the CMB scale after the background run.  Currently informational only
        (the result is not used to gate the perturbation run).
        Default ``False``.
    N_subhorizon : float, optional
        Number of e-folds before horizon crossing at which to initialise
        each perturbation mode (Bunch-Davies vacuum initial conditions).
        Default ``3.0``.
    N_suphorizon : float, optional
        Number of e-folds after horizon crossing at which to freeze each
        perturbation mode and read off the power spectrum.  Default ``7.0``.
    k : array-like or callable or None, optional
        Wavenumber grid used for upsampled output.  Only relevant when
        ``upsample=True``.  Default ``None``.
    upsample : bool, optional
        If ``True``, interpolate the computed power spectrum onto the ``k``
        grid.  Requires ``k`` to be provided.  Default ``False``.
    background_solver_opts : dict, optional
        Override fields of the default [SolverOptions][sigway.ms_solver.SolverOptions]
        for the background ODE solver.  Defaults:
        ``rtol=1e-8``, ``atol=1e-8``, ``max_steps=100000``,
        ``dt0=1e-3``, ``saveat=SaveAt(steps=True)``.
        **Warning:** only change these if you know what you are doing.
    perturbation_solver_opts : dict, optional
        Override fields of the default [SolverOptions][sigway.ms_solver.SolverOptions]
        for the perturbation ODE solver.  Defaults:
        ``rtol=1e-6``, ``atol=1e-6``, ``max_steps=1000000``,
        ``dt0=1e-3``, ``saveat=SaveAt(t1=True)`` (only the final value is
        stored).  Setting ``saveat=SaveAt(steps=True)`` will considerably
        slow down the computation.
        **Warning:** only change these if you know what you are doing.
    error_on_fail : bool, optional
        If ``True``, raise an error when the perturbation solver fails for a
        given mode.  Default ``False``.
    """

    def __init__(
        self,
        V,
        phi0=0.0,
        pi0=0.0,
        N_CMB_to_end=65.0,
        max_efolds=1000.0,
        cmb_bounds={},
        check_consistency=False,
        N_subhorizon=3.0,
        N_suphorizon=7.0,
        k=None,
        upsample=False,
        background_solver_opts={},
        perturbation_solver_opts={},
        error_on_fail=False,
    ):
        # Ensure the potential function is JAX-compatible
        if not isinstance(V, jax.interpreters.ad.JVPTracer):
            # try:
            V = jax.jit(V)
            self.V = V

            # except:
            #     raise ValueError(
            #         "The potential V must be a jax-compatible function."
            #     )
        else:
            self.V = V

        # Set the normalized potential and its derivatives
        self.U = lambda phi, *p: self.V(phi, *p) / self.V(
            self.phi0, *p
        )  # Normalize to 1 at phi0
        self.Ud = jax.grad(self.U)
        self.Udd = jax.grad(self.Ud)

        # Set the initial conditions
        self.phi0 = phi0
        self.pi0 = pi0

        # Number of efolds from CMB to the end of inflation
        # By fixing this we assume instantaneous reheating
        self.N_CMB_to_end = N_CMB_to_end

        # Maximum number of efolds to run the background evolution
        self.max_efolds = max_efolds

        # Number of efolds to run the perturbations
        self.N_subhorizon = N_subhorizon
        self.N_suphorizon = N_suphorizon

        # Set CMB bounds
        self.check_consistency = check_consistency
        self.cmb_means = cmb_bounds.get(
            "means", jnp.array([3.04442188, 0.96488871, 0.0])
        )
        self.cmb_cov = cmb_bounds.get(
            "cov",
            jnp.array(
                [
                    [2.00112315e-04, 1.35106101e-05, 0.0],
                    [1.35106101e-05, 1.72537423e-05, 0.0],
                    [0.0, 0.0, 0.01],
                ]
            ),
        )

        if k is None:  # If k is None, no upsampling
            self.k = k
            if upsample:
                raise ValueError("If upsample is True, f cannot be None")
        elif callable(k):  # If f is a function, evaluate it
            self.k = k
            if not upsample:
                Warning(
                    "Providing k and not upsampling will result in k "
                    "being ignored."
                )
        elif isinstance(
            k, jnp.ndarray
        ):  # If f is an iterable, convert to array
            self.k = jnp.array(k)
            if not upsample:
                Warning(
                    "Providing k and not upsampling will result in k being "
                    "ignored."
                )
        else:
            raise ValueError(
                "k should be None, a callable or an iterable. Got {}".format(
                    type(k)
                )
            )
        self.upsample = upsample

        # Default solver options
        default_solver_opts = {
            "background": SolverOptions(
                rtol=1e-8,
                atol=1e-8,
                max_steps=100000,
                dt0=1e-3,
                saveat=SaveAt(steps=True),
            ),
            "perturbation": SolverOptions(
                rtol=1e-6,
                atol=1e-6,
                max_steps=1000000,
                dt0=1e-3,
                saveat=SaveAt(t1=True),
            ),
        }

        # Set background solver options
        self.background_solver_opts = default_solver_opts[
            "background"
        ]._replace(**background_solver_opts)
        invalid_opts = set(self.background_solver_opts._fields) - set(
            default_solver_opts["background"]._fields
        )
        if invalid_opts:
            raise ValueError(
                "Invalid options found in background_solver_opts:"
                f" {', '.join(invalid_opts)}"
            )

        # Set perturbation solver options
        self.perturbation_solver_opts = default_solver_opts[
            "perturbation"
        ]._replace(**perturbation_solver_opts)
        invalid_opts = set(self.perturbation_solver_opts._fields) - set(
            default_solver_opts["perturbation"]._fields
        )
        if invalid_opts:
            raise ValueError(
                "Invalid options found in perturbation_solver_opts:"
                f" {', '.join(invalid_opts)}"
            )

    def run_background(self, params):
        """Integrate the inflationary background equations.

        Solves the background Klein-Gordon and Friedmann equations from
        N = 0 until inflation ends (epsilon_H >= 1) or ``max_efolds`` is
        reached.  The internal rescaled variables are returned; in particular
        ``h`` is the Hubble parameter normalised by sqrt(V(phi0) / 3).

        Parameters
        ----------
        params : tuple
            Parameters forwarded to the potential ``V(phi, *params)``.

        Returns
        -------
        N : numpy.ndarray
            E-fold values at each saved solver step.
        phi : numpy.ndarray
            Field values at each saved step.
        y : numpy.ndarray
            Field derivative d phi / dN at each saved step.
        h : numpy.ndarray
            Rescaled Hubble parameter at each saved step.
        """
        bsol = _run_background(
            self.Ud,
            self.max_efolds,
            self.phi0,
            params,
            self.background_solver_opts,
        )
        bmask = jnp.isfinite(bsol.ts)
        N = bsol.ts[bmask]
        bsolf = bsol.ys[bmask]
        phi = bsolf[:, 0]
        y = bsolf[:, 1]
        h = bsolf[:, 2]
        return N, phi, y, h

    def run_perturbations(self, k, N, phi, y, h, params):
        """Run the perturbation solver over all wavenumbers and return P_zeta.

        Uses ``jax.vmap`` to integrate the Mukhanov-Sasaki equations for each
        mode in ``k`` in parallel.  Initial conditions (Bunch-Davies vacuum)
        are set ``N_subhorizon`` e-folds before horizon crossing; the mode
        amplitude is read off ``N_suphorizon`` e-folds after horizon crossing.

        Parameters
        ----------
        k : array-like
            Wavenumber grid (in s^-1).
        N : array-like
            E-fold array from the background evolution.
        phi : array-like
            Field values from the background evolution, same shape as ``N``.
        y : array-like
            Field derivatives d phi / dN from the background evolution.
        h : array-like
            Rescaled Hubble parameter values from the background evolution.
        params : tuple
            Parameters forwarded to the potential ``V(phi, *params)``.

        Returns
        -------
        numpy.ndarray
            Dimensionful scalar power spectrum P_zeta, shape ``(len(k),)``.

        Raises
        ------
        ConsistencyError
            If the background does not contain enough e-folds to reach the CMB
            scale, or to evolve the perturbations across the required window.
        """
        N_CMB = jnp.max(N) - self.N_CMB_to_end
        if N_CMB < 0.0:
            raise ConsistencyError("Not enough e-folds to reach the CMB scale")
        H_CMB = jnp.interp(N_CMB, N, h)
        Hk = H_from_wavenumber(k, N, h, N_CMB, H_CMB)
        Nk = efolds_from_wavenumber_si_units(k, Hk, N_CMB, H_CMB)

        # Check that we have enough e-folds to evolve the perturbations
        if jnp.min(Nk) < self.N_subhorizon:
            raise ConsistencyError(
                "Not enough e-folds to solve for perturbations"
            )

        Nin = Nk - self.N_subhorizon
        Nout = Nk + self.N_suphorizon

        phiIn = CubicSpline(N, phi)(Nin)
        yIn = CubicSpline(N, y)(Nin)
        yOut = CubicSpline(N, y)(Nout)
        hIn = CubicSpline(N, h)(Nin)
        lograt = jnp.log(Hk) + self.N_subhorizon

        # Compute the perturbations with vmap
        compute_Pzeta = jax.vmap(
            _run_perturbations,
            in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, None, None),
            out_axes=0,
        )
        Pzetas = compute_Pzeta(
            self.Ud,
            self.Udd,
            phiIn,
            yIn,
            yOut,
            hIn,
            Nin,
            Nout,
            lograt,
            params,
            self.perturbation_solver_opts,
        )
        Pzetas *= self.V(self.phi0, *params)
        return Pzetas

    def run_single_k(self, k, N, phi, y, h, params):
        """Solve the perturbation equations for a single wavenumber, saving all steps.

        Unlike `run_perturbations`, this method forces ``saveat=SaveAt(steps=True)``
        so the full time series of the mode function is available for
        inspection or plotting.

        Parameters
        ----------
        k : float
            Single wavenumber (in s^-1) to integrate.
        N : array-like
            E-fold array from the background evolution.
        phi : array-like
            Field values from the background evolution.
        y : array-like
            Field derivatives d phi / dN from the background evolution.
        h : array-like
            Rescaled Hubble parameter from the background evolution.
        params : tuple
            Parameters forwarded to the potential ``V(phi, *params)``.

        Returns
        -------
        sol : diffrax.Solution
            Full ODE solution object with shape ``(steps, 7)`` state array
            containing ``[phi, y, h, dPhiR, dPhiR', dPhiI, dPhiI']`` at each
            saved step.
        lograt : float
            log(k / aH) evaluated at the initial e-fold ``Nin``.

        Raises
        ------
        ConsistencyError
            If there are insufficient e-folds to reach the CMB scale or to
            evolve the mode across the required window.
        """
        N_CMB = jnp.max(N) - self.N_CMB_to_end
        if N_CMB < 0.0:
            raise ConsistencyError("Not enough e-folds to reach the CMB scale")
        H_CMB = jnp.interp(N_CMB, N, h)
        Hk = H_from_wavenumber(k, N, h, N_CMB, H_CMB)
        Nk = efolds_from_wavenumber_si_units(k, Hk, N_CMB, H_CMB)

        # Check that we have enough e-folds to evolve the perturbations
        if jnp.min(Nk) < self.N_subhorizon:
            raise ConsistencyError(
                "Not enough e-folds to solve for perturbations"
            )

        Nin = Nk - self.N_subhorizon
        Nout = Nk + self.N_suphorizon

        phiIn = jnp.interp(Nin, N, phi)
        yIn = jnp.interp(Nin, N, y)
        # yOut = jnp.interp(Nout, N, y)
        hIn = jnp.interp(Nin, N, h)
        lograt = jnp.log(Hk) + self.N_subhorizon

        # We need to make sure that we are saving the solution at each step.
        solver_opts = self.perturbation_solver_opts._replace(
            saveat=SaveAt(steps=True)
        )
        # Compute the perturbations with vmap
        sol = _solve_perturbations(
            self.Ud,
            self.Udd,
            phiIn,
            yIn,
            hIn,
            Nin,
            Nout,
            lograt,
            params,
            solver_opts,
        )
        return sol, lograt

    def p_at_cmb(self, N, phi, y, h, params):
        """Evaluate the Gaussian log-likelihood at the CMB pivot scale.

        Interpolates the background trajectory to the CMB scale (at
        ``N_end - N_CMB_to_end``), computes the slow-roll observables
        ``[ln(10^10 P_zeta), n_s, r]``, and evaluates the multivariate
        Gaussian log-probability against ``cmb_means`` / ``cmb_cov``.

        Parameters
        ----------
        N : array-like
            E-fold array from the background evolution.
        phi : array-like
            Field values from the background evolution.
        y : array-like
            Field derivatives d phi / dN from the background evolution.
        h : array-like
            Rescaled Hubble parameter from the background evolution.
        params : tuple
            Parameters forwarded to the potential ``V(phi, *params)``.

        Returns
        -------
        p : float
            Multivariate-Gaussian log-likelihood of the CMB observables.
        params_at_cmb : numpy.ndarray
            Array ``[ln(10^10 P_zeta), n_s, r]`` evaluated at the CMB scale.
        """
        Nend = jnp.max(N)
        N_cmb = Nend - self.N_CMB_to_end

        phi_cmb = jnp.interp(N_cmb, N, phi)
        y_cmb = jnp.interp(N_cmb, N, y)
        h_cmb = jnp.interp(N_cmb, N, h)
        epsilon_cmb = self.epsilon_h(y_cmb)
        eta_cmb = self.eta_h(phi_cmb, y_cmb, h_cmb, params)
        params_at_cmb = jnp.array(
            [
                jnp.log(1e10 * self.pzeta_sr(y_cmb, h_cmb, params)),
                self.n_s(epsilon_cmb, eta_cmb),
                self.r(epsilon_cmb),
            ]
        )

        p = multivariate_normal.logpdf(
            params_at_cmb, mean=self.cmb_means, cov=self.cmb_cov
        )
        return p, params_at_cmb

    def pzeta_sr(self, y, h, params):
        r"""Slow-roll approximation for the scalar power spectrum P_zeta.

        Computes $P_\zeta \approx V({\phi_0}) \, h^2 / (4\pi^2 \, y^2)$
        using the Hubble slow-roll parameter $\epsilon_H = y^2 / 2$.

        Parameters
        ----------
        y : float or array-like
            Rescaled field derivative d phi / dN.
        h : float or array-like
            Rescaled Hubble parameter (same shape as ``y``).
        params : tuple
            Parameters forwarded to the potential ``V(phi, *params)``.

        Returns
        -------
        float or numpy.ndarray
            Slow-roll estimate of the dimensionful scalar power spectrum
            P_zeta.
        """
        return self.V(self.phi0, *params) * h**2 / (8 * jnp.pi**2 * y**2 / 2)

    def epsilon_h(self, y):
        r"""Compute the first Hubble slow-roll parameter $\epsilon_H = y^2 / 2$.

        Parameters
        ----------
        y : float or array-like
            Rescaled field derivative d phi / dN.

        Returns
        -------
        float or numpy.ndarray
            First Hubble slow-roll parameter $\epsilon_H$.
        """
        return y**2 / 2

    def eta_h(self, phi, y, h, params):
        r"""Compute the second Hubble slow-roll parameter $\eta_H$.

        Parameters
        ----------
        phi : float or array-like
            Inflaton field value.
        y : float or array-like
            Rescaled field derivative d phi / dN (same shape as ``phi``).
        h : float or array-like
            Rescaled Hubble parameter (same shape as ``phi``).
        params : tuple
            Parameters forwarded to the potential ``V(phi, *params)``.

        Returns
        -------
        float or numpy.ndarray
            Second Hubble slow-roll parameter $\eta_H$.
        """
        eps = self.epsilon_h(y)
        return -6 + 2 * eps - self.Ud(phi, *params) * y / (eps * h**2)

    def n_s(self, epsilon_h, eta_h):
        """Compute the scalar spectral index in the slow-roll approximation.

        Parameters
        ----------
        epsilon_h : float or array-like
            First Hubble slow-roll parameter.
        eta_h : float or array-like
            Second Hubble slow-roll parameter.

        Returns
        -------
        float or numpy.ndarray
            Scalar spectral index $n_s = 1 - 2\epsilon_H - \eta_H$.
        """
        return 1 - 2 * epsilon_h - eta_h

    def r(self, epsilon_h):
        """Compute the tensor-to-scalar ratio in the slow-roll approximation.

        Parameters
        ----------
        epsilon_h : float or array-like
            First Hubble slow-roll parameter.

        Returns
        -------
        float or numpy.ndarray
            Tensor-to-scalar ratio $r = 16\epsilon_H$.
        """
        return 16 * epsilon_h

    def run(self, k, *params):
        r"""Compute P_zeta(k) and return a callable interpolant.

        Runs the background evolution, optionally checks CMB consistency, then
        solves the Mukhanov-Sasaki equations for each wavenumber in ``k``.
        The result is fitted with a cubic Hermite interpolant so that the
        returned callable can be evaluated at arbitrary wavenumbers inside the
        original ``k`` range.

        This is the entry point used by
        [SingleFieldPerturbations][sigway.perturbations.SingleFieldPerturbations].

        Parameters
        ----------
        k : array-like
            Wavenumber grid (in s^-1) at which to solve the perturbation
            equations.
        *params : float
            Scalar parameters forwarded to the potential ``V(phi, *params)``.

        Returns
        -------
        callable
            A function ``P_zeta_interpolation(k_new)`` that evaluates the
            cubic-Hermite interpolant of P_zeta at wavenumbers ``k_new``.
        """
        params = jnp.array(params)
        # Compute background evolution
        N, phi, y, h = self.run_background(params)
        # Compute likelihood at CMB scale
        # THIS IS NOT SUPPORTED YET
        if self.check_consistency:
            _ = self.p_at_cmb(N, phi, y, h, params)
        # Compute perturbations
        P_zeta = self.run_perturbations(k, N, phi, y, h, params)

        coeff = backward_hermite_coefficients(k, P_zeta)

        def P_zeta_interpolation(knew, *params):
            return interpolation_inner(knew, k, coeff)
            # return jnp.interp(knew, k, P_zeta, left=0.0, right=0.0)
            # return LinearInterpolation(k, P_zeta).evaluate(knew)

        return P_zeta_interpolation

    def __call__(self, k, *params):
        r"""Compute and return the raw P_zeta array evaluated at ``k``.

        Equivalent to `run` but returns the power-spectrum values directly
        rather than an interpolating callable.

        Parameters
        ----------
        k : array-like
            Wavenumber grid (in s^-1) at which to solve the perturbation
            equations.
        *params : float
            Scalar parameters forwarded to the potential ``V(phi, *params)``.

        Returns
        -------
        numpy.ndarray
            Dimensionful scalar power spectrum P_zeta evaluated at each
            wavenumber in ``k``.
        """
        params = jnp.array(params)
        # Compute background evolution
        N, phi, y, h = self.run_background(params)
        # Compute likelihood at CMB scale
        # THIS IS NOT SUPPORTED YET
        if self.check_consistency:
            _ = self.p_at_cmb(N, phi, y, h, params)
        # Compute perturbations
        P_zeta = self.run_perturbations(k, N, phi, y, h, params)

        return P_zeta

    def plot_evolution(self, k, params):
        """Plot the background evolution and scalar power spectrum.

        Produces a five-panel figure showing (from top to bottom):
        P_zeta (slow-roll approximation vs full Mukhanov-Sasaki result),
        slow-roll parameters epsilon_H and |eta_H|, the rescaled field phi,
        -y = -d phi / dN, and the rescaled Hubble parameter h.  A vertical
        dashed line marks the CMB pivot scale; the top panel has a secondary
        x-axis showing the corresponding wavenumber k.

        Parameters
        ----------
        k : array-like
            Wavenumber grid (in s^-1) used for the full perturbation run.
        params : tuple
            Parameters forwarded to the potential ``V(phi, *params)``.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the five-panel plot.
        """
        # Compute background evolution
        N, phi, y, h = self.run_background(params)
        Pzeta_SR = self.pzeta_sr(y, h, params)

        Nend = jnp.max(N)

        # Compute perturbations
        Pzeta = self.run_perturbations(k, N, phi, y, h, params)
        N_CMB = jnp.max(N) - self.N_CMB_to_end
        H_CMB = jnp.interp(N_CMB, N, h)
        Nk = efolds_from_wavenumber_si_units(
            k, H_from_wavenumber(k, N, h, N_CMB, H_CMB), N_CMB, H_CMB
        )

        # Compute slow-roll parameters
        epsilon = self.epsilon_h(y)
        eta = jnp.array(
            [self.eta_h(i, j, k, params) for i, j, k in zip(phi, y, h)]
        )

        # Create subplots
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 12))

        # Add a vertical line at the CMB scale
        for ax in axs:
            ax.axvline(
                N_CMB - Nend,
                color="k",
                linestyle="--",
                label="CMB scale",
                alpha=0.5,
            )

        # Plot P_zeta for background and perturbations
        axs[0].plot(
            N - Nend, Pzeta_SR, label="$\\mathcal{P}_{\\zeta}$ (SR-approx.)"
        )
        axs[0].plot(Nk - Nend, Pzeta, label="$\\mathcal{P}_{\\zeta}$ (full)")
        axs[0].set_ylabel("$\\mathcal{P}_{\\zeta}$")
        axs[0].set_yscale("log")
        axs[0].legend(loc=2)

        # Plot slow-roll parameters
        axs[1].plot(N - Nend, epsilon, label="$\\epsilon_H$")
        axs[1].plot(N - Nend, jnp.abs(eta), label="$|\\eta_H|$")
        axs[1].set_yscale("log")
        axs[1].set_ylabel("Slow-roll parameters")
        axs[1].legend(loc=2)

        # Plot phi
        axs[2].plot(N - Nend, phi, label="$x$")
        axs[2].set_ylabel("$x$")
        axs[2].legend(loc=3)

        # Plot y=dphi/dN
        axs[3].plot(N - Nend, -y, label="$-y$")
        axs[3].set_ylabel("$-y$")
        axs[3].set_yscale("log")
        axs[3].legend(loc=3)

        # Plot h
        axs[4].plot(N - Nend, h, label="$h$")
        axs[4].set_ylabel("$h$")
        axs[4].legend(loc=3)

        # Set x-axis label
        axs[-1].set_xlabel(r"$N-N_{\rm end}$")

        # Create a secondary x-axis for wavenumbers on the first subplot
        ax2 = axs[0].twiny()

        k_values = wavenumber_from_efolds_si_units(N, h, N_CMB, H_CMB)

        # Set the secondary x-axis scale to logarithmic
        ax2.set_xscale("log")

        # Use LogLocator to set logarithmic ticks
        log_locator = LogLocator(base=10.0)
        ax2.xaxis.set_major_locator(log_locator)
        ax2.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto"))
        ax2.xaxis.set_minor_formatter(NullFormatter())

        # Format the secondary x-axis tick labels
        ax2.xaxis.set_major_formatter(FormatStrFormatter("%g"))

        # Synchronize the secondary x-axis range with the primary x-axis range
        ax2.set_xlim(k_values[0], k_values[-1])

        # Label the secondary x-axis
        ax2.set_xlabel(r"$k$ [s$^{-1}$]")

        return fig

    def plot_potential(
        self, params, phi_range=None, n_points=1000, relative=False
    ):
        """Plot the inflaton potential V(phi).

        Parameters
        ----------
        params : tuple
            Parameters forwarded to the potential ``V(phi, *params)``.
        phi_range : tuple of float, optional
            ``(phi_min, phi_max)`` range for the plot.  Defaults to
            ``(0.1 * phi0, 2 * phi0)``.
        n_points : int, optional
            Number of field values at which to evaluate V.  Default ``1000``.
        relative : bool, optional
            If ``True``, plot V / V(phi0) on the y-axis and phi / phi0 on
            the x-axis.  Default ``False``.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the potential plot.
        """
        if phi_range is None:
            phi_range = (0.1 * self.phi0, 2 * self.phi0)
        phi = jnp.linspace(*phi_range, n_points)
        V = self.V(phi, *params)
        if relative:
            V /= self.V(self.phi0, *params)
            phi /= self.phi0
            labels = ["$V/V_0$", "$\\phi/\\phi_0$"]
        else:
            labels = ["$V$", "$\\phi$"]

        fig, ax = plt.subplots()
        ax.plot(phi, V)
        ax.set_xlabel(labels[1])
        ax.set_ylabel(labels[0])
        return fig


if __name__ == "__main__":

    def V(phi, a, lam, v, nfac):
        # a, lam, v, nfac = p
        b = (1 + nfac) * (
            1 - a**2 / 3 + a**2 / 3 * (9 / (2 * a**2) - 1) ** (2 / 3)
        )
        f = phi / v
        return (
            lam
            * v**4
            / 12
            * f**2
            * (6 - 4 * a * f + 3 * f**2)
            / (1 + b * f**2) ** 2
        )

    phi0 = 3.0
    pi0 = 0.0
    pvalues = jnp.array(
        [
            1 / jnp.sqrt(2) * (1 + 0.56 * 1e-2),
            1.86e-6 * (1 - 0.12),
            0.19669 * (1 + 1e-3),
            0.3 * 6.23 * 1e-5,
        ]
    )

    robbiesmodel = SingleFieldSolver(V, phi0=phi0, pi0=pi0)
    k = jnp.geomspace(1e-5, 10 ** (-2.0), 100)

    fig = robbiesmodel.plot_evolution(k, pvalues)
    plt.show()
    # plt.savefig("test.pdf")
