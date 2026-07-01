r"""
Single-field Mukhanov-Sasaki solver.

This module computes the primordial scalar power spectrum
$\mathcal{P}_\zeta(k)$ for a single-field inflationary model with potential
$V(\phi)$.  Two sets of equations are solved in sequence:

1. **Background** — the Klein-Gordon equation for the inflaton $\phi(N)$
   coupled to the Friedmann equation for the Hubble parameter $H(N)$, where
   $N$ is the number of e-folds.  Integration proceeds from the initial
   conditions $(\phi_0, \pi_0)$ until the slow-roll parameter
   $\epsilon_H \equiv \dot\phi^2/(2H^2) = 1$ (end of inflation).

2. **Perturbations** — the Mukhanov-Sasaki equation for each comoving
   wavenumber $k$ is integrated from deep inside the horizon (Bunch-Davies
   vacuum) to well after horizon crossing, yielding the amplitude of the
   curvature perturbation $\zeta_k$.  The power spectrum is then
   $\mathcal{P}_\zeta(k) = k^3 |\zeta_k|^2 / (2\pi^2)$.

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
from sigway.constants import CMB_BOUNDS
from sigway.utils import (
    efolds_from_wavenumber_si_units,
    H_from_wavenumber,
    wavenumber_from_efolds_si_units,
)

jax.config.update("jax_enable_x64", True)


class ConsistencyError(Exception):
    r"""Raised when the model fails a physical self-consistency check.

    Typical causes: the background trajectory produces too few e-folds to
    reach the CMB pivot scale, or the horizon-crossing e-fold $N_k$ is so
    early that there is insufficient room to initialise the Bunch-Davies
    vacuum $N_{\mathrm{sub}}$ e-folds before crossing.
    """


@partial(jax.jit, static_argnums=(0, 1, 2, 4))
def _run_background(Ud, max_efolds, phi0, pvalues, solver_opts):
    r"""
    Integrate the inflationary background equations from $N=0$ to inflation end.

    The state vector is $(\phi, \pi, h)$ where $\phi$ is the inflaton field,
    $\pi \equiv \mathrm{d}\phi/\mathrm{d}N$ is its velocity with respect to
    e-folds, and $h$ is the Hubble parameter rescaled by
    $\sqrt{V(\phi_0)/3}$.  Integration stops when $\pi^2 > 2$, i.e. when
    the slow-roll parameter $\epsilon_H = \pi^2/2 \geq 1$ signals the end
    of inflation.

    Parameters
    ----------
    Ud : callable
        Gradient of the rescaled potential $U(\phi) = V(\phi)/V(\phi_0)$
        with respect to $\phi$.
    max_efolds : int
        Hard upper limit on the number of e-folds integrated.
    phi0 : float
        Initial inflaton field value $\phi_0$.
    pvalues : tuple
        Extra parameters passed to the potential.
    solver_opts : SolverOptions
        Numerical tolerances and step-size settings for the ODE solver.

    Returns
    -------
    diffrax.Solution
        ODE solution; ``sol.ts`` gives the e-fold grid and ``sol.ys`` the
        state $(\phi, \pi, h)$ at each saved step.
    """

    def equations(n, variables, args):
        r"""
        Background equations of motion in e-fold time.

        The system is the Klein-Gordon equation for $\phi$ and the Friedmann
        equation for $h$, written as first-order ODEs in $N$.

        Parameters
        ----------
        n : float
            Current e-fold number $N$.
        variables : array-like
            State vector $(\phi, \pi, h)$.
        args : tuple
            Unused; required by the diffrax ODE interface.

        Returns
        -------
        array-like
            Time derivatives $(\mathrm{d}\phi/\mathrm{d}N,\,
            \mathrm{d}\pi/\mathrm{d}N,\, \mathrm{d}h/\mathrm{d}N)$.
        """
        x, y, h = variables
        derx = y
        dery = -3 * (1 - y**2 / 6) * y - Ud(x, *pvalues) / h**2
        derh = -(y**2) / 2 * h
        return jnp.array([derx, dery, derh])

    def inflation_end(t, y, *args, **kwargs):
        r"""
        Termination condition: inflation ends when $\epsilon_H \geq 1$.

        Parameters
        ----------
        t : float
            Current e-fold number (unused directly).
        y : array-like
            Current state $(\phi, \pi, h)$.
        args : tuple
            Unused.

        Returns
        -------
        bool
            ``True`` (stop) when $\pi^2 > 2$, i.e. $\epsilon_H > 1$.
        """
        return y[1] ** 2 > 2

    # Initial conditions: slow-roll attractor values for pi and h at phi0.
    # pi0 is set from the attractor relation dU/dphi = -3H pi (slow-roll).
    # h0 is the corresponding Hubble rate from the Friedmann equation.
    Ud0 = Ud(phi0, *pvalues)
    y0 = -(3 / Ud0) * (
        jnp.sqrt(1 + 2 / 3 * Ud0**2) - 1
    )  # $\pi_0$: initial field velocity on the slow-roll attractor
    h0 = (
        1 / jnp.sqrt(6) * (jnp.sqrt(1 + 2 / 3 * Ud0**2) + 1) ** (1 / 2)
    )  # $h_0$: initial rescaled Hubble rate

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
    r"""
    Integrate the Mukhanov-Sasaki equation for a single mode $k$.

    The perturbation $\Delta\phi_k$ is decomposed into real and imaginary
    parts.  Both are evolved simultaneously together with the background
    trajectory, starting from Bunch-Davies initial conditions at $N_{\rm in}$
    (deep inside the horizon) and ending at $N_{\rm out}$ (well outside the
    horizon).

    Parameters
    ----------
    Ud : callable
        First derivative $\partial U/\partial\phi$ of the rescaled potential.
    Udd : callable
        Second derivative $\partial^2 U/\partial\phi^2$ of the rescaled
        potential.
    phiIn : float
        Inflaton field value $\phi$ at $N_{\rm in}$.
    yIn : float
        Field velocity $\pi = \mathrm{d}\phi/\mathrm{d}N$ at $N_{\rm in}$.
    hIn : float
        Rescaled Hubble parameter $h$ at $N_{\rm in}$.
    nin : float
        Initial e-fold number $N_{\rm in}$ (Bunch-Davies conditions set here).
    nout : float
        Final e-fold number $N_{\rm out}$ at which the frozen mode amplitude
        is read off.
    lograt : float
        $\ln(k/aH)$ evaluated at $N_{\rm in}$; encodes how far the mode is
        inside the horizon at the start of integration.
    pvalues : tuple
        Extra parameters passed to the potential.
    solver_opts : SolverOptions
        Numerical tolerances and step-size settings for the ODE solver.

    Returns
    -------
    diffrax.Solution
        ODE solution; ``sol.ys`` has shape ``(steps, 7)`` with columns
        $(\phi, \pi, h, \mathrm{Re}\,\Delta\phi, \mathrm{Re}\,\Delta\phi',
        \mathrm{Im}\,\Delta\phi, \mathrm{Im}\,\Delta\phi')$.
    """
    # Bunch-Davies vacuum initial conditions for the mode function
    # at N_in (deep inside the horizon).  The normalisation sets the
    # standard quantum vacuum amplitude; the imaginary part starts at zero.
    dPhiRin = 1.0  # $\mathrm{Re}\,\widetilde{\Delta\phi}$ at $N_{\rm in}$
    dPhiRpRin = (
        -1.0
    )  # $\mathrm{d}(\mathrm{Re}\,\widetilde{\Delta\phi})/\mathrm{d}N$ at $N_{\rm in}$
    dPhiIin = 0.0  # $\mathrm{Im}\,\widetilde{\Delta\phi}$ at $N_{\rm in}$
    dPhiIpIin = 0.0  # $\mathrm{d}(\mathrm{Im}\,\widetilde{\Delta\phi})/\mathrm{d}N$ at $N_{\rm in}$

    def equations_perturbations(n, variables, args):
        r"""
        Mukhanov-Sasaki + background equations of motion in e-fold time.

        Evolves the background $(\phi, \pi, h)$ jointly with the real and
        imaginary parts of the rescaled field perturbation
        $\widetilde{\Delta\phi}$ and their $N$-derivatives.

        Parameters
        ----------
        n : float
            Current e-fold number $N$.
        variables : array-like
            State vector $(\phi, \pi, h,
            \mathrm{Re}\,\Delta\phi, \mathrm{Re}\,\Delta\phi',
            \mathrm{Im}\,\Delta\phi, \mathrm{Im}\,\Delta\phi')$.
        args : tuple
            ``(nin, lograt)`` — initial e-fold and $\ln(k/aH)$ at $N_{\rm in}$,
            used to track the comoving horizon ratio throughout the integration.

        Returns
        -------
        array-like
            $N$-derivatives of all seven state variables.
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
    r"""
    Integrate the Mukhanov-Sasaki equation and return $\mathcal{P}_\zeta/V_0$.

    Calls `_solve_perturbations` and then assembles the dimensionless power
    spectrum (normalised by $V_0 = V(\phi_0)$) from the frozen mode amplitude
    at $N_{\rm out}$:

    $$\frac{\mathcal{P}_\zeta}{V_0} =
      \frac{k^3}{4\pi^2\,\epsilon_H}
      \bigl|\widetilde{\Delta\phi}_k(N_{\rm out})\bigr|^2.$$

    Parameters
    ----------
    Ud : callable
        First derivative $\partial U/\partial\phi$ of the rescaled potential.
    Udd : callable
        Second derivative $\partial^2 U/\partial\phi^2$ of the rescaled
        potential.
    phiIn : float
        Inflaton field value at $N_{\rm in}$.
    yIn : float
        Field velocity $\pi$ at $N_{\rm in}$.
    yOut : float
        Field velocity $\pi$ at $N_{\rm out}$ (used for the spectrum
        amplitude; overwritten internally by the ODE output).
    hIn : float
        Rescaled Hubble parameter at $N_{\rm in}$.
    nin : float
        Initial e-fold number $N_{\rm in}$.
    nout : float
        Final e-fold number $N_{\rm out}$.
    lograt : float
        $\ln(k/aH)$ at $N_{\rm in}$.
    pvalues : tuple
        Extra parameters passed to the potential.
    solver_opts : SolverOptions
        Numerical tolerances and step-size settings for the ODE solver.

    Returns
    -------
    float
        $\mathcal{P}_\zeta(k)/V_0$, the mode contribution to the scalar
        power spectrum divided by $V(\phi_0)$.
    """
    # Solve the perturbation equations
    sol = _solve_perturbations(
        Ud, Udd, phiIn, yIn, hIn, nin, nout, lograt, pvalues, solver_opts
    )

    # Read off the frozen mode amplitude at N_out.
    # deltaPhiR, deltaPhiI are the real and imaginary parts of the
    # rescaled perturbation; yOut = pi at N_out gives epsilon_H = yOut^2/2.
    deltaPhiR = sol.ys[-1][3]
    deltaPhiI = sol.ys[-1][5]
    yOut = sol.ys[-1][1]

    # Assemble P_zeta / V0 from the mode amplitude.
    # The factor exp(2*lograt) = (k/aH)^2 at N_in converts the rescaled
    # amplitude back to physical units.
    Pzeta_by_V0 = (
        1
        / (4 * jnp.pi**2)
        * jnp.exp(2 * lograt)
        * (deltaPhiR**2 + deltaPhiI**2)
        / yOut**2
    )

    return Pzeta_by_V0


class SolverOptions(
    namedtuple("SolverOptions", ["rtol", "atol", "max_steps", "dt0", "saveat"])
):
    r"""Numerical settings for the adaptive ODE integrator (diffrax Tsit5).

    One instance is used for the background solver and a separate one for the
    perturbation solver; both are configured via the
    ``background_solver_opts`` and ``perturbation_solver_opts`` arguments of
    [SingleFieldSolver][sigway.ms_solver.SingleFieldSolver].

    Attributes
    ----------
    rtol : float
        Relative error tolerance for the adaptive (PID) step-size controller.
        Tighter values increase accuracy at the cost of more ODE steps.
    atol : float
        Absolute error tolerance for the adaptive step-size controller.
    max_steps : int
        Maximum number of ODE steps allowed before the integrator gives up.
        Increase this for potentials with sharp features or very long
        inflation.
    dt0 : float
        Initial step size in e-folds $\Delta N$.
    saveat : diffrax.SaveAt
        Specification of which solution points to store.  The default
        ``SaveAt(steps=True)`` retains every adaptive step (needed for the
        background); ``SaveAt(t1=True)`` stores only the final value (used
        for the perturbation solver to save memory).
    """


@jit
def interpolation_inner(knew, k, coeff):
    return CubicInterpolation(k, coeff).evaluate(knew)


class SingleFieldSolver:
    r"""Compute the primordial scalar power spectrum $\mathcal{P}_\zeta(k)$ for a single inflaton field.

    Given a potential $V(\phi, *\mathrm{params})$, this class:

    1. Integrates the **background** Klein-Gordon and Friedmann equations in
       e-fold time $N$ from the initial field value $\phi_0$ (on the slow-roll
       attractor) until the end of inflation ($\epsilon_H = 1$).
    2. For each comoving wavenumber $k$ in the requested grid, integrates the
       **Mukhanov-Sasaki equation** from Bunch-Davies initial conditions
       (set $N_{\rm sub}$ e-folds before horizon crossing) to well after
       horizon crossing ($N_{\rm sup}$ e-folds after), then reads off the
       frozen amplitude to obtain $\mathcal{P}_\zeta(k)$.

    The potential is normalised internally as $U(\phi) = V(\phi)/V(\phi_0)$,
    so the overall energy scale $V_0 \equiv V(\phi_0)$ is restored at the end.
    The rescaled background variables are $\pi \equiv \mathrm{d}\phi/\mathrm{d}N$
    and $h = H / \sqrt{V_0/3}$.

    The primary entry points are:

    - ``solver.run(k, *params)`` — runs the full calculation and returns a
      callable interpolant $\mathcal{P}_\zeta(k_{\rm new})$.
    - ``solver(k, *params)`` — same calculation, returns the raw
      $\mathcal{P}_\zeta$ array instead.

    Parameters
    ----------
    V : callable
        Inflaton potential $V(\phi, *\mathrm{params})$.  Must be written in
        JAX (using ``jax.numpy``) so that automatic differentiation can
        compute $V'$ and $V''$ internally.
    phi0 : float, optional
        Initial field value $\phi_0$.  Also the normalisation point:
        $U(\phi) \equiv V(\phi)/V(\phi_0)$.  Default ``0.0``.
    pi0 : float, optional
        Initial field velocity $\pi_0 = \mathrm{d}\phi/\mathrm{d}N\big|_0$.
        Currently stored for reference; the background integration always
        starts from the slow-roll attractor value derived from $V'(\phi_0)$.
        Default ``0.0``.
    N_CMB_to_end : float, optional
        Number of e-folds from the CMB pivot scale $k_* \approx 0.05\,\mathrm{Mpc}^{-1}$
        to the end of inflation, assuming instantaneous reheating.  This sets
        the absolute $k$-to-$N$ mapping.  Default ``65.0``.
    max_efolds : float, optional
        Hard upper limit on the number of e-folds integrated for the
        background.  Increase this for models with very long inflationary
        phases.  Default ``1000.0``.
    cmb_bounds : dict, optional
        Gaussian CMB observational priors on
        $[\ln(10^{10}\mathcal{P}_\zeta),\, n_s,\, r]$, supplied as a dict
        with keys ``"means"`` (shape ``(3,)``) and ``"cov"`` (shape
        ``(3, 3)``).  Used only when ``check_consistency=True``.  Defaults
        to Planck 2018 best-fit values.
    check_consistency : bool, optional
        If ``True``, compute the Gaussian log-likelihood of the slow-roll
        CMB observables against ``cmb_bounds`` after the background run.
        Informational only; does not affect the perturbation calculation.
        Default ``False``.
    N_subhorizon : float, optional
        Number of e-folds before horizon crossing ($k = aH$) at which to
        impose Bunch-Davies vacuum initial conditions on each mode.
        Default ``3.0``.
    N_suphorizon : float, optional
        Number of e-folds after horizon crossing at which to read off the
        frozen super-horizon mode amplitude.  Default ``7.0``.
    k : array-like or callable or None, optional
        Wavenumber grid (in $\mathrm{s}^{-1}$) onto which the spectrum is
        interpolated when ``upsample=True``.  Ignored otherwise.
        Default ``None``.
    upsample : bool, optional
        If ``True``, evaluate the returned power spectrum on the dense ``k``
        grid rather than on the wavenumbers passed to ``run``.  Requires
        ``k`` to be set.  Default ``False``.
    background_solver_opts : dict, optional
        Override any field of the default
        [SolverOptions][sigway.ms_solver.SolverOptions] for the background
        integrator.  Defaults: ``rtol=1e-8``, ``atol=1e-8``,
        ``max_steps=100000``, ``dt0=1e-3``, ``saveat=SaveAt(steps=True)``.
        Tighten tolerances if the background trajectory looks noisy for
        potentials with sharp features.
    perturbation_solver_opts : dict, optional
        Override any field of the default
        [SolverOptions][sigway.ms_solver.SolverOptions] for the perturbation
        integrator.  Defaults: ``rtol=1e-6``, ``atol=1e-6``,
        ``max_steps=1000000``, ``dt0=1e-3``, ``saveat=SaveAt(t1=True)``
        (only the final mode amplitude is stored, saving memory).  Setting
        ``saveat=SaveAt(steps=True)`` retains the full mode history but is
        considerably slower.
    error_on_fail : bool, optional
        If ``True``, raise an error when the perturbation integrator fails
        for a given mode.  Default ``False``.

    Examples
    --------
    Ultra-slow-roll model with a near-inflection-point potential:

    >>> import jax.numpy as jnp
    >>> from sigway.ms_solver import SingleFieldSolver
    >>> def V(phi, a, lam, v, nfac):
    ...     b = (1+nfac)*(1 - a**2/3 + a**2/3*(9/(2*a**2)-1)**(2/3))
    ...     x = phi/v
    ...     return lam*v**4/12 * x**2*(6-4*a*x+3*x**2)/(1+b*x**2)**2
    >>> solver = SingleFieldSolver(V, phi0=3.0, pi0=0.0, N_CMB_to_end=58.0,
    ...                            k=jnp.geomspace(1e-5, 10.0, 200))

    After construction, call ``solver.run(k, *params)`` to obtain a callable
    interpolant of $\mathcal{P}_\zeta(k)$, or ``solver(k, *params)`` for the
    raw spectrum array.
    """

    def __init__(
        self,
        V,
        phi0=0.0,
        pi0=0.0,
        N_CMB_to_end=65.0,
        max_efolds=1000.0,
        cmb_bounds=CMB_BOUNDS,
        check_consistency=False,
        N_subhorizon=3.0,
        N_suphorizon=7.0,
        k=None,
        upsample=False,
        background_solver_opts={},
        perturbation_solver_opts={},
        error_on_fail=False,
    ):
        # Wrap V with jit so gradients and vmapped calls compile efficiently.
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

        # Rescaled potential U(phi) = V(phi)/V(phi0) and its first two
        # derivatives, computed by automatic differentiation.
        self.U = lambda phi, *p: self.V(phi, *p) / self.V(
            self.phi0, *p
        )  # $U(\phi) = V(\phi)/V(\phi_0)$, normalised to unity at $\phi_0$
        self.Ud = jax.grad(self.U)  # $\partial U/\partial\phi$
        self.Udd = jax.grad(self.Ud)  # $\partial^2 U/\partial\phi^2$

        # Initial field conditions.
        self.phi0 = phi0  # $\phi_0$: starting field value
        self.pi0 = pi0  # $\pi_0 = \mathrm{d}\phi/\mathrm{d}N|_0$ (stored; slow-roll attractor used in practice)

        # Number of e-folds from the CMB pivot scale to the end of inflation.
        # Assuming instantaneous reheating fixes the absolute k-to-N mapping.
        self.N_CMB_to_end = N_CMB_to_end

        # Hard upper limit on the background integration.
        self.max_efolds = max_efolds

        # Window around horizon crossing used for the perturbation integration:
        # start N_subhorizon e-folds before k = aH (Bunch-Davies conditions),
        # end N_suphorizon e-folds after k = aH (super-horizon freeze-out).
        self.N_subhorizon = N_subhorizon
        self.N_suphorizon = N_suphorizon

        # CMB consistency check settings.
        self.check_consistency = check_consistency
        self.cmb_means = cmb_bounds.get("means")
        self.cmb_cov = cmb_bounds.get("cov")

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
        r"""Integrate the inflationary background equations.

        Evolves the inflaton $\phi(N)$ and Hubble parameter $H(N)$ from the
        slow-roll attractor at $\phi_0$ until $\epsilon_H \geq 1$ (end of
        inflation) or ``max_efolds`` is reached.  The returned arrays use the
        internally rescaled variables: $h = H/\sqrt{V(\phi_0)/3}$.

        Parameters
        ----------
        params : tuple
            Extra parameters forwarded to the potential $V(\phi, *\mathrm{params})$.

        Returns
        -------
        N : numpy.ndarray
            E-fold number at each saved integration step.
        phi : numpy.ndarray
            Inflaton field $\phi(N)$ at each saved step.
        y : numpy.ndarray
            Field velocity $\pi(N) = \mathrm{d}\phi/\mathrm{d}N$ at each
            saved step.
        h : numpy.ndarray
            Rescaled Hubble parameter $h(N) = H(N)/\sqrt{V(\phi_0)/3}$ at
            each saved step.
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
        r"""Solve the Mukhanov-Sasaki equation for all modes and return $\mathcal{P}_\zeta(k)$.

        For each wavenumber $k$ the integration window is
        $[N_k - N_{\rm sub},\, N_k + N_{\rm sup}]$ where $N_k$ is the horizon
        crossing e-fold ($k = a H$).  All modes are integrated simultaneously
        using ``jax.vmap``.

        Parameters
        ----------
        k : array-like
            Comoving wavenumber grid in $\mathrm{s}^{-1}$.
        N : array-like
            E-fold array from ``run_background``.
        phi : array-like
            Inflaton field values $\phi(N)$, same shape as ``N``.
        y : array-like
            Field velocity $\pi(N) = \mathrm{d}\phi/\mathrm{d}N$, same shape
            as ``N``.
        h : array-like
            Rescaled Hubble parameter $h(N)$, same shape as ``N``.
        params : tuple
            Extra parameters forwarded to the potential $V(\phi, *\mathrm{params})$.

        Returns
        -------
        numpy.ndarray
            Dimensionful scalar power spectrum $\mathcal{P}_\zeta(k)$, shape
            ``(len(k),)``.

        Raises
        ------
        ConsistencyError
            If the background trajectory is too short to reach the CMB pivot
            scale, or if the horizon-crossing e-fold $N_k$ leaves insufficient
            room to set Bunch-Davies initial conditions.
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
        r"""Solve the Mukhanov-Sasaki equation for a single mode, retaining the full history.

        Identical to ``run_perturbations`` for one wavenumber, but the
        complete mode-function trajectory
        $(\phi, \pi, h, \mathrm{Re}\,\Delta\phi, \mathrm{Re}\,\Delta\phi',
        \mathrm{Im}\,\Delta\phi, \mathrm{Im}\,\Delta\phi')$ is stored at every
        adaptive step.  Useful for diagnosing oscillations, horizon crossing,
        or super-horizon freeze-out behaviour.

        Parameters
        ----------
        k : float
            Single comoving wavenumber in $\mathrm{s}^{-1}$.
        N : array-like
            E-fold array from ``run_background``.
        phi : array-like
            Inflaton field values $\phi(N)$.
        y : array-like
            Field velocity $\pi(N) = \mathrm{d}\phi/\mathrm{d}N$.
        h : array-like
            Rescaled Hubble parameter $h(N)$.
        params : tuple
            Extra parameters forwarded to the potential $V(\phi, *\mathrm{params})$.

        Returns
        -------
        sol : diffrax.Solution
            Full ODE solution; ``sol.ys`` has shape ``(steps, 7)`` with
            columns $(\phi, \pi, h, \mathrm{Re}\,\Delta\phi,
            \mathrm{Re}\,\Delta\phi', \mathrm{Im}\,\Delta\phi,
            \mathrm{Im}\,\Delta\phi')$ at each saved step.
        lograt : float
            $\ln(k/aH)$ at the initial e-fold $N_{\rm in}$.

        Raises
        ------
        ConsistencyError
            If the background trajectory is too short to reach the CMB pivot
            scale, or if $N_k$ leaves insufficient room for Bunch-Davies
            initial conditions.
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
        r"""Evaluate the CMB log-likelihood at the pivot scale using slow-roll observables.

        Interpolates the background to $N_{\rm CMB} = N_{\rm end} - N_{\rm CMB\_to\_end}$,
        computes $[\ln(10^{10}\mathcal{P}_\zeta),\, n_s,\, r]$ in the
        slow-roll approximation, and returns their multivariate-Gaussian
        log-probability against the stored CMB priors.

        Parameters
        ----------
        N : array-like
            E-fold array from ``run_background``.
        phi : array-like
            Inflaton field values $\phi(N)$.
        y : array-like
            Field velocity $\pi(N) = \mathrm{d}\phi/\mathrm{d}N$.
        h : array-like
            Rescaled Hubble parameter $h(N)$.
        params : tuple
            Extra parameters forwarded to the potential $V(\phi, *\mathrm{params})$.

        Returns
        -------
        p : float
            Multivariate-Gaussian log-likelihood
            $\ln\mathcal{L}[\ln(10^{10}\mathcal{P}_\zeta),\, n_s,\, r]$.
        params_at_cmb : numpy.ndarray
            Array $[\ln(10^{10}\mathcal{P}_\zeta),\, n_s,\, r]$ evaluated at
            the CMB pivot scale.
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
        r"""Slow-roll estimate of the scalar power spectrum $\mathcal{P}_\zeta$.

        Uses the standard slow-roll formula
        $\mathcal{P}_\zeta \approx V(\phi_0)\,h^2 / (8\pi^2\,\epsilon_H)$
        where $\epsilon_H = \pi^2/2$.  This is fast to evaluate along the
        whole background trajectory and useful for comparison with the full
        Mukhanov-Sasaki result.

        Parameters
        ----------
        y : float or array-like
            Field velocity $\pi = \mathrm{d}\phi/\mathrm{d}N$.
        h : float or array-like
            Rescaled Hubble parameter $h$, same shape as ``y``.
        params : tuple
            Extra parameters forwarded to $V(\phi, *\mathrm{params})$.

        Returns
        -------
        float or numpy.ndarray
            Slow-roll approximation to $\mathcal{P}_\zeta$.
        """
        return self.V(self.phi0, *params) * h**2 / (8 * jnp.pi**2 * y**2 / 2)

    def epsilon_h(self, y):
        r"""First Hubble slow-roll parameter $\epsilon_H = \pi^2/2$.

        $\epsilon_H$ measures how quickly the Hubble rate is changing:
        $\epsilon_H = -\dot{H}/H^2$.  Inflation ends when $\epsilon_H = 1$.

        Parameters
        ----------
        y : float or array-like
            Field velocity $\pi = \mathrm{d}\phi/\mathrm{d}N$.

        Returns
        -------
        float or numpy.ndarray
            $\epsilon_H = \pi^2/2$.
        """
        return y**2 / 2

    def eta_h(self, phi, y, h, params):
        r"""Second Hubble slow-roll parameter $\eta_H$.

        $\eta_H$ characterises the curvature of the inflationary trajectory.
        In slow roll $|\eta_H| \ll 1$; large $|\eta_H|$ signals departures
        from slow roll (e.g. an inflection point).

        Parameters
        ----------
        phi : float or array-like
            Inflaton field value $\phi$.
        y : float or array-like
            Field velocity $\pi = \mathrm{d}\phi/\mathrm{d}N$, same shape as
            ``phi``.
        h : float or array-like
            Rescaled Hubble parameter $h$, same shape as ``phi``.
        params : tuple
            Extra parameters forwarded to $V(\phi, *\mathrm{params})$.

        Returns
        -------
        float or numpy.ndarray
            $\eta_H = -6 + 2\epsilon_H - U'(\phi)\,\pi / (\epsilon_H\,h^2)$.
        """
        eps = self.epsilon_h(y)
        return -6 + 2 * eps - self.Ud(phi, *params) * y / (eps * h**2)

    def n_s(self, epsilon_h, eta_h):
        r"""Scalar spectral index $n_s$ in the slow-roll approximation.

        A scale-invariant spectrum has $n_s = 1$; Planck measures
        $n_s \approx 0.965$.

        Parameters
        ----------
        epsilon_h : float or array-like
            First Hubble slow-roll parameter $\epsilon_H$.
        eta_h : float or array-like
            Second Hubble slow-roll parameter $\eta_H$.

        Returns
        -------
        float or numpy.ndarray
            $n_s = 1 - 2\epsilon_H - \eta_H$.
        """
        return 1 - 2 * epsilon_h - eta_h

    def r(self, epsilon_h):
        r"""Tensor-to-scalar ratio $r$ in the slow-roll approximation.

        Current CMB upper limits place $r \lesssim 0.036$ (95% CL).

        Parameters
        ----------
        epsilon_h : float or array-like
            First Hubble slow-roll parameter $\epsilon_H$.

        Returns
        -------
        float or numpy.ndarray
            $r = 16\,\epsilon_H$.
        """
        return 16 * epsilon_h

    def run(self, k, *params):
        r"""Compute $\mathcal{P}_\zeta(k)$ and return a callable interpolant.

        Runs the full background + perturbation calculation over the
        wavenumber grid ``k`` and fits a cubic-Hermite spline to the result.
        The returned callable can then be evaluated at any $k$ inside the
        original grid.  This is the entry point used by
        [SingleFieldPerturbations][sigway.perturbations.SingleFieldPerturbations].

        Parameters
        ----------
        k : array-like
            Comoving wavenumber grid in $\mathrm{s}^{-1}$ at which to solve
            the Mukhanov-Sasaki equation.
        *params : float
            Scalar potential parameters forwarded to $V(\phi, *\mathrm{params})$.

        Returns
        -------
        callable
            A function ``P_zeta_interpolation(k_new)`` that evaluates the
            cubic-Hermite interpolant of $\mathcal{P}_\zeta$ at wavenumbers
            ``k_new`` within the original ``k`` range.
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
        r"""Compute and return the raw $\mathcal{P}_\zeta$ array at ``k``.

        Identical to ``run`` but returns the power-spectrum values directly
        instead of a callable interpolant.  Convenient for quickly plotting
        or fitting the spectrum at a fixed parameter set.

        Parameters
        ----------
        k : array-like
            Comoving wavenumber grid in $\mathrm{s}^{-1}$.
        *params : float
            Scalar potential parameters forwarded to $V(\phi, *\mathrm{params})$.

        Returns
        -------
        numpy.ndarray
            Dimensionful scalar power spectrum $\mathcal{P}_\zeta(k)$
            evaluated at each wavenumber in ``k``.
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
        r"""Plot the background evolution and scalar power spectrum.

        Produces a five-panel figure (e-folds $N - N_{\rm end}$ on the
        shared x-axis) showing, from top to bottom:

        1. $\mathcal{P}_\zeta(k)$: slow-roll approximation vs. full
           Mukhanov-Sasaki result.
        2. Slow-roll parameters $\epsilon_H$ and $|\eta_H|$.
        3. Inflaton field $\phi$.
        4. Field velocity $-\pi = -\mathrm{d}\phi/\mathrm{d}N$.
        5. Rescaled Hubble parameter $h$.

        A vertical dashed line marks the CMB pivot scale; the top panel has
        a secondary x-axis labelled in $k\,[\mathrm{s}^{-1}]$.

        Parameters
        ----------
        k : array-like
            Comoving wavenumber grid in $\mathrm{s}^{-1}$ for the full
            perturbation run.
        params : tuple
            Extra parameters forwarded to the potential $V(\phi, *\mathrm{params})$.

        Returns
        -------
        matplotlib.figure.Figure
            Five-panel figure of the background and perturbation evolution.
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
        r"""Plot the inflaton potential $V(\phi)$.

        Parameters
        ----------
        params : tuple
            Extra parameters forwarded to $V(\phi, *\mathrm{params})$.
        phi_range : tuple of float, optional
            ``(phi_min, phi_max)`` field range for the plot.  Defaults to
            $(0.1\,\phi_0,\; 2\,\phi_0)$.
        n_points : int, optional
            Number of field-value samples at which to evaluate $V$.
            Default ``1000``.
        relative : bool, optional
            If ``True``, plot $V/V(\phi_0)$ vs. $\phi/\phi_0$ (dimensionless
            axes).  Default ``False``.

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
