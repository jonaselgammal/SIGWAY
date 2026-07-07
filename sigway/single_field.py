r"""
Single-field inflation: $\mathcal{P}_\zeta(k)$ via the Mukhanov-Sasaki equation.

Computes the primordial scalar power spectrum for a single-field model with
potential $V(\phi)$ by solving, in sequence, the inflationary background
(Klein-Gordon + Friedmann in e-folds, run to the end of inflation) and then the
Mukhanov-Sasaki mode equation for each comoving wavenumber (Bunch-Davies vacuum
$\to$ super-horizon freeze-out), giving
$\mathcal{P}_\zeta(k) = k^3 |\zeta_k|^2 / (2\pi^2)$.

[SingleFieldPerturbations][sigway.single_field.SingleFieldPerturbations] exposes
the result through the standard
[ScalarPerturbations][sigway.perturbations.ScalarPerturbations] interface, so
the GW integrator treats a numerically-solved spectrum exactly like an
analytic one.

Public API
----------
- [SingleFieldPerturbations][sigway.single_field.SingleFieldPerturbations]
- [SolverOptions][sigway.single_field.SolverOptions]
- [ConsistencyError][sigway.single_field.ConsistencyError]
"""

__all__ = ["SingleFieldPerturbations", "SolverOptions", "ConsistencyError"]

# Global
import jax
from jax import numpy as jnp

import warnings
from collections import namedtuple

from diffrax import (
    diffeqsolve,
    ODETerm,
    Tsit5,
    PIDController,
    Event,
    SaveAt,
    backward_hermite_coefficients,
    CubicInterpolation,
)


from functools import partial

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FormatStrFormatter

# Local
from sigway.constants import CMB_means, CMB_cov
from sigway.perturbations import ScalarPerturbations
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


def _background_rhs(phi, dphidN, h, ud):
    r"""Background equations of motion in e-fold time.

    Klein-Gordon for the inflaton plus the Friedmann equation for the rescaled
    Hubble rate ``h``, with the field velocity ``dphidN`` = $\mathrm{d}\phi/
    \mathrm{d}N$ and ``ud`` = $\partial U/\partial\phi$ at the current ``phi``.
    Returns the e-fold derivatives ``(dphi/dN, d2phi/dN2, dh/dN)`` -- shared by
    both the background-only and the joint background+perturbation systems.
    """
    d2phidN2 = -3.0 * (1.0 - dphidN**2 / 6.0) * dphidN - ud / h**2
    dhdN = -(dphidN**2) / 2.0 * h
    return dphidN, d2phidN2, dhdN


@partial(jax.jit, static_argnums=(0, 1, 2, 4))
def _evolve_background(Ud, max_efolds, phi0, pvalues, solver_opts):
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
        phi, dphidN, h = variables
        return jnp.array(_background_rhs(phi, dphidN, h, Ud(phi, *pvalues)))

    def inflation_end(t, state, *args, **kwargs):
        r"""
        Termination condition: inflation ends when $\epsilon_H \geq 1$.

        Parameters
        ----------
        t : float
            Current e-fold number (unused directly).
        state : array-like
            Current state $(\phi, \pi, h)$.
        args : tuple
            Unused.

        Returns
        -------
        bool
            ``True`` (stop) when $\pi^2 > 2$, i.e. $\epsilon_H > 1$.
        """
        return state[1] ** 2 > 2

    # Initial conditions: slow-roll attractor values for pi and h at phi0.
    # The initial field velocity is fixed by the attractor dU/dphi = -3H pi
    # (slow-roll); h0 is the corresponding Hubble rate from the Friedmann eq.
    Ud0 = Ud(phi0, *pvalues)
    dphidN0 = -(3 / Ud0) * (
        jnp.sqrt(1 + 2 / 3 * Ud0**2) - 1
    )  # initial field velocity dphi/dN on the slow-roll attractor
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
        y0=jnp.array([phi0, dphidN0, h0]),
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=solver_opts.max_steps,
        event=stopping_event,
    )

    return sol


@partial(jax.jit, static_argnums=(0, 1, 9))
def _solve_mode(
    Ud, Udd, phiIn, dphidN_in, hIn, nin, nout, lograt, pvalues, solver_opts
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
    dphidN_in : float
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
    # (dPhiR, dPhiR', dPhiI, dPhiI') = the Re/Im parts of the rescaled mode
    # perturbation and their d/dN, at N_in.
    dPhiRin = 1.0
    dPhiRpRin = -1.0
    dPhiIin = 0.0
    dPhiIpIin = 0.0

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
        phi, dphidN, h, dPhiR, dPhipR, dPhiI, dPhipI = variables
        ud = Ud(phi, *pvalues)
        udd = Udd(phi, *pvalues)
        # background part (shared with the background-only solver)
        dphi_dN, d2phidN2, dhdN = _background_rhs(phi, dphidN, h, ud)

        derPhiR = dPhipR
        derPhipR = -(
            (3 - dphidN**2 / 2) * dPhipR
            + 2 * jnp.exp(lograt - n + nin) / h * dPhipI
            + ((udd + 2 * ud * dphidN) / h**2 + 3 * dphidN**2 - dphidN**4 / 2)
            * dPhiR
            + 2 * jnp.exp(lograt - n + nin) / h * dPhiI
        )

        derPhiI = dPhipI
        derPhipI = -(
            (3 - dphidN**2 / 2) * dPhipI
            - 2 * jnp.exp(lograt - n + nin) / h * dPhipR
            + ((udd + 2 * ud * dphidN) / h**2 + 3 * dphidN**2 - dphidN**4 / 2)
            * dPhiI
            - 2 * jnp.exp(lograt - n + nin) / h * dPhiR
        )

        return jnp.array(
            [dphi_dN, d2phidN2, dhdN, derPhiR, derPhipR, derPhiI, derPhipI]
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
        y0=jnp.array(
            [phiIn, dphidN_in, hIn, dPhiRin, dPhiRpRin, dPhiIin, dPhiIpIin]
        ),
        stepsize_controller=stepsize_controller,
        args=(nin, lograt),
        saveat=saveat,
        max_steps=solver_opts.max_steps,
    )

    return sol


@partial(jax.jit, static_argnums=(0, 1, 9))
def _mode_pzeta(
    Ud, Udd, phiIn, dphidN_in, hIn, nin, nout, lograt, pvalues, solver_opts
):
    r"""
    Integrate the Mukhanov-Sasaki equation and return $\mathcal{P}_\zeta/V_0$.

    Calls `_solve_mode` and then assembles the dimensionless power
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
    dphidN_in : float
        Field velocity $\pi$ at $N_{\rm in}$.
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
    sol = _solve_mode(
        Ud, Udd, phiIn, dphidN_in, hIn, nin, nout, lograt, pvalues, solver_opts
    )

    # Read off the frozen mode amplitude at N_out.
    # deltaPhiR, deltaPhiI are the real and imaginary parts of the
    # rescaled perturbation; dphidN_out = pi at N_out gives
    # epsilon_H = dphidN_out^2/2.
    deltaPhiR = sol.ys[-1][3]
    deltaPhiI = sol.ys[-1][5]
    dphidN_out = sol.ys[-1][1]

    # Assemble P_zeta / V0 from the mode amplitude.
    # The factor exp(2*lograt) = (k/aH)^2 at N_in converts the rescaled
    # amplitude back to physical units.
    Pzeta_by_V0 = (
        1
        / (4 * jnp.pi**2)
        * jnp.exp(2 * lograt)
        * (deltaPhiR**2 + deltaPhiI**2)
        / dphidN_out**2
    )

    return Pzeta_by_V0


SolverOptions = namedtuple(
    "SolverOptions", ["rtol", "atol", "max_steps", "dt0", "saveat"]
)
SolverOptions.__doc__ = r"""Numerical settings for the adaptive ODE
integrator (diffrax Tsit5).

    One instance is used for the background solver and a separate one for the
    perturbation solver; both are configured via the
    ``background_solver_opts`` and ``perturbation_solver_opts`` arguments of
    [SingleFieldPerturbations][sigway.single_field.SingleFieldPerturbations].

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


def _merge_solver_opts(defaults, overrides, label):
    """Apply an ``overrides`` dict on top of a default ``SolverOptions``.

    Returns ``defaults`` with the given fields replaced.  Unknown keys raise a
    ``ValueError`` naming the offending option (``namedtuple._replace`` already
    rejects them; we just wrap it with a clearer message).
    """
    try:
        return defaults._replace(**(overrides or {}))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid options in {label}_solver_opts: {exc}")


class SingleFieldPerturbations(ScalarPerturbations):
    r"""Primordial scalar power spectrum $\mathcal{P}_\zeta(k)$ from
    single-field inflation.

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
    The rescaled background variables are
    $\pi \equiv \mathrm{d}\phi/\mathrm{d}N$ and $h = H / \sqrt{V_0/3}$.

    The result is exposed through the standard
    [ScalarPerturbations][sigway.perturbations.ScalarPerturbations] interface:

    - ``pert(k, *params)`` — runs the full calculation and returns the raw
      $\mathcal{P}_\zeta$ array.
    - ``pert.prepare(kint, *params)`` — solves once on ``kint`` and returns a
      callable interpolant $\mathcal{P}_\zeta(k_{\rm new})$; this is the seam
      the GW integrator uses to avoid re-solving the ODE per grid point.

    Parameters
    ----------
    V : callable
        Inflaton potential $V(\phi, *\mathrm{params})$.  Must be written in
        JAX (using ``jax.numpy``) so that automatic differentiation can
        compute $V'$ and $V''$ internally.
    param_names : tuple of str, optional
        Ordered names of the free potential parameters (the ``*params``
        forwarded to ``V``), e.g. ``("a", "lam", "v", "nfac")``.
        Default ``()``.
    phi0 : float, optional
        Initial field value $\phi_0$, and the normalisation point
        $U(\phi) \equiv V(\phi)/V(\phi_0)$.  The background always starts from
        the slow-roll attractor velocity derived from $V'(\phi_0)$.
        Default ``0.0``.
    N_CMB_to_end : float, optional
        Number of e-folds from the CMB pivot
        ($k_* \approx 0.05\,\mathrm{Mpc}^{-1}$)
        to the end of inflation, assuming instantaneous reheating; sets the
        absolute $k$-to-$N$ mapping.  Default ``65.0``.
    max_efolds : float, optional
        Hard upper limit on the number of e-folds integrated for the
        background.  Default ``1000.0``.
    N_subhorizon : float, optional
        E-folds before horizon crossing ($k = aH$) at which Bunch-Davies vacuum
        initial conditions are imposed.  Default ``3.0``.
    N_suphorizon : float, optional
        E-folds after horizon crossing at which the frozen mode amplitude is
        read off.  Default ``7.0``.
    cmb_check : bool, optional
        If ``True``, emit a ``UserWarning`` when the model's slow-roll CMB
        observables $[\ln(10^{10}\mathcal{P}_\zeta),\, n_s,\, r]$ sit
        $\geq 3\sigma$ from the Planck 2018 prior (``constants.CMB_means`` /
        ``CMB_cov``), reporting the distance.  Purely diagnostic -- it never
        affects the returned spectrum.  Default ``False``.
    background_solver_opts, perturbation_solver_opts : dict, optional
        Override any field of the default
        [SolverOptions][sigway.single_field.SolverOptions] for the background /
        perturbation ODE integrator.  Background defaults: ``rtol=atol=1e-8``,
        ``max_steps=100000``, ``dt0=1e-3``, ``saveat=SaveAt(steps=True)``;
        perturbation defaults: ``rtol=atol=1e-6``, ``max_steps=1000000``,
        ``dt0=1e-3``, ``saveat=SaveAt(t1=True)``.

    Examples
    --------
    Ultra-slow-roll model with a near-inflection-point potential:

    >>> import jax.numpy as jnp
    >>> from sigway.single_field import SingleFieldPerturbations
    >>> def V(phi, a, lam, v, nfac):
    ...     b = (1+nfac)*(1 - a**2/3 + a**2/3*(9/(2*a**2)-1)**(2/3))
    ...     x = phi/v
    ...     return lam*v**4/12 * x**2*(6-4*a*x+3*x**2)/(1+b*x**2)**2
    >>> pert = SingleFieldPerturbations(V, ("a", "lam", "v", "nfac"),
    ...                                 phi0=3.0, N_CMB_to_end=58.0)
    >>> omega = pert.prepare(jnp.geomspace(1e-5, 10.0, 200),
    ...                      0.71, 1.47e-6, 0.197, 1.87e-5)
    """

    # The Mukhanov-Sasaki solve uses SciPy/diffrax ODE routines, so it cannot
    # run inside the compiled, differentiable integrator core -> eager path.
    jittable = False

    def __init__(
        self,
        V,
        param_names=(),
        *,
        phi0=0.0,
        N_CMB_to_end=65.0,
        max_efolds=1000.0,
        N_subhorizon=3.0,
        N_suphorizon=7.0,
        cmb_check=False,
        background_solver_opts=None,
        perturbation_solver_opts=None,
    ):
        self.param_names = tuple(param_names)

        # Wrap V with jit so gradients and vmapped calls compile efficiently
        # (skip if V is already a JVPTracer, e.g. under an outer jit/grad).
        if not isinstance(V, jax.interpreters.ad.JVPTracer):
            V = jax.jit(V)
        self.V = V
        self.phi0 = phi0  # $\phi_0$: starting field value

        # Rescaled potential U(phi) = V(phi)/V(phi0) and its first two
        # derivatives, computed by automatic differentiation.
        self.U = lambda phi, *p: self.V(phi, *p) / self.V(
            self.phi0, *p
        )  # $U(\phi) = V(\phi)/V(\phi_0)$, normalised to unity at $\phi_0$
        self.Ud = jax.grad(self.U)  # $\partial U/\partial\phi$
        self.Udd = jax.grad(self.Ud)  # $\partial^2 U/\partial\phi^2$

        # Number of e-folds from the CMB pivot scale to the end of inflation
        # (instantaneous reheating fixes the absolute k-to-N mapping).
        self.N_CMB_to_end = N_CMB_to_end
        self.max_efolds = max_efolds  # hard upper limit on the background run

        # Window around horizon crossing for the perturbation integration:
        # N_subhorizon e-folds before k = aH (Bunch-Davies), N_suphorizon after.
        self.N_subhorizon = N_subhorizon
        self.N_suphorizon = N_suphorizon

        # Optional (default-off) diagnostic: warn if the model's slow-roll CMB
        # observables sit >=3 sigma from the Planck prior.  Never affects
        # output.
        self.cmb_check = cmb_check

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

        self.background_solver_opts = _merge_solver_opts(
            default_solver_opts["background"], background_solver_opts,
            "background",
        )
        self.perturbation_solver_opts = _merge_solver_opts(
            default_solver_opts["perturbation"], perturbation_solver_opts,
            "perturbation",
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
            Extra parameters forwarded to the potential
            $V(\phi, *\mathrm{params})$.

        Returns
        -------
        N : numpy.ndarray
            E-fold number at each saved integration step.
        phi : numpy.ndarray
            Inflaton field $\phi(N)$ at each saved step.
        dphidN : numpy.ndarray
            Field velocity $\pi(N) = \mathrm{d}\phi/\mathrm{d}N$ at each
            saved step.
        h : numpy.ndarray
            Rescaled Hubble parameter $h(N) = H(N)/\sqrt{V(\phi_0)/3}$ at
            each saved step.
        """
        bsol = _evolve_background(
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
        dphidN = bsolf[:, 1]
        h = bsolf[:, 2]
        return N, phi, dphidN, h

    def _horizon_window(self, k, N, phi, dphidN, h):
        r"""Bunch-Davies -> freeze-out integration window for mode(s) ``k``.

        Locates the horizon-crossing e-fold $N_k$ ($k = aH$) for each ``k``,
        brackets it with ``N_subhorizon`` e-folds before (to set Bunch-Davies
        initial conditions) and ``N_suphorizon`` after (super-horizon
        freeze-out), and interpolates the background onto $N_{\rm in}$ with a
        cubic spline.  Shared by :meth:`run_perturbations` (vectorised over the
        ``k`` grid) and :meth:`_mode_history` (a single ``k``).

        Returns ``(Nin, Nout, lograt, phiIn, dphidN_in, hIn)``.

        Raises
        ------
        ConsistencyError
            If the background is too short to reach the CMB pivot, or $N_k$
            leaves no room for the Bunch-Davies initialisation.
        """
        N_CMB = jnp.max(N) - self.N_CMB_to_end
        if N_CMB < 0.0:
            raise ConsistencyError("Not enough e-folds to reach the CMB scale")
        H_CMB = jnp.interp(N_CMB, N, h)
        Hk = H_from_wavenumber(k, N, h, N_CMB, H_CMB)
        Nk = efolds_from_wavenumber_si_units(k, Hk, N_CMB, H_CMB)
        if jnp.min(Nk) < self.N_subhorizon:
            raise ConsistencyError(
                "Not enough e-folds to solve for perturbations"
            )
        Nin = Nk - self.N_subhorizon
        Nout = Nk + self.N_suphorizon
        phiIn = CubicSpline(N, phi)(Nin)
        dphidN_in = CubicSpline(N, dphidN)(Nin)
        hIn = CubicSpline(N, h)(Nin)
        lograt = jnp.log(Hk) + self.N_subhorizon
        return Nin, Nout, lograt, phiIn, dphidN_in, hIn

    def run_perturbations(self, k, N, phi, dphidN, h, params):
        r"""Mukhanov-Sasaki solve for all modes, returning
        $\mathcal{P}_\zeta(k)$.

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
        dphidN : array-like
            Field velocity $\pi(N) = \mathrm{d}\phi/\mathrm{d}N$, same shape
            as ``N``.
        h : array-like
            Rescaled Hubble parameter $h(N)$, same shape as ``N``.
        params : tuple
            Extra parameters forwarded to the potential
            $V(\phi, *\mathrm{params})$.

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
        Nin, Nout, lograt, phiIn, dphidN_in, hIn = self._horizon_window(
            k, N, phi, dphidN, h
        )

        # Compute the perturbations with vmap
        compute_Pzeta = jax.vmap(
            _mode_pzeta,
            in_axes=(None, None, 0, 0, 0, 0, 0, 0, None, None),
            out_axes=0,
        )
        Pzetas = compute_Pzeta(
            self.Ud,
            self.Udd,
            phiIn,
            dphidN_in,
            hIn,
            Nin,
            Nout,
            lograt,
            params,
            self.perturbation_solver_opts,
        )
        Pzetas *= self.V(self.phi0, *params)
        return Pzetas

    def _mode_history(self, k, N, phi, dphidN, h, params):
        r"""Full mode-function trajectory for a single wavenumber (diagnostic).

        Like :meth:`run_perturbations` for one ``k``, but retains every
        adaptive step (``SaveAt(steps=True)``) so the complete evolution of
        $(\phi, \pi, h, \mathrm{Re}\,\Delta\phi, \ldots)$ is available for
        inspecting oscillations, horizon crossing, or freeze-out.  Returns the
        raw diffrax solution and $\ln(k/aH)$ at $N_{\rm in}$.
        """
        Nin, Nout, lograt, phiIn, dphidN_in, hIn = self._horizon_window(
            k, N, phi, dphidN, h
        )
        solver_opts = self.perturbation_solver_opts._replace(
            saveat=SaveAt(steps=True)
        )
        sol = _solve_mode(
            self.Ud, self.Udd, phiIn, dphidN_in, hIn, Nin, Nout, lograt,
            params, solver_opts,
        )
        return sol, lograt

    def cmb_observables(self, N, phi, dphidN, h, params):
        r"""Slow-roll CMB observables
        $[\ln(10^{10}\mathcal{P}_\zeta),\, n_s,\, r]$.

        Interpolates the background to the CMB pivot
        $N_{\rm CMB} = N_{\rm end} - N_{\rm CMB\_to\_end}$ and evaluates the
        three observables in the slow-roll approximation.
        """
        N_cmb = jnp.max(N) - self.N_CMB_to_end
        phi_cmb = jnp.interp(N_cmb, N, phi)
        dphidN_cmb = jnp.interp(N_cmb, N, dphidN)
        h_cmb = jnp.interp(N_cmb, N, h)
        eps = self.epsilon_h(dphidN_cmb)
        eta = self.eta_h(phi_cmb, dphidN_cmb, h_cmb, params)
        return jnp.array(
            [
                jnp.log(1e10 * self.pzeta_sr(dphidN_cmb, h_cmb, params)),
                self.n_s(eps, eta),
                self.r(eps),
            ]
        )

    def _warn_if_cmb_outlier(self, N, phi, dphidN, h, params):
        r"""Warn if the model is $\geq 3\sigma$ from the Planck CMB prior.

        Computes the Mahalanobis distance of the slow-roll observables
        $[\ln(10^{10}\mathcal{P}_\zeta),\, n_s,\, r]$ from the Planck 2018 prior
        (``constants.CMB_means``/``CMB_cov``).  If it exceeds $3\sigma$ a
        ``UserWarning`` is emitted reporting the value.  Purely diagnostic --
        it never changes the returned spectrum.  Returns the distance in sigma.
        """
        delta = self.cmb_observables(N, phi, dphidN, h, params) - CMB_means
        n_sigma = float(jnp.sqrt(delta @ jnp.linalg.solve(CMB_cov, delta)))
        if n_sigma >= 3.0:
            warnings.warn(
                f"Inflationary model is {n_sigma:.1f} sigma from the Planck "
                "2018 CMB constraints on (ln[1e10 P_zeta], n_s, r); the "
                "spectrum is still returned.",
                stacklevel=2,
            )
        return n_sigma

    def pzeta_sr(self, dphidN, h, params):
        r"""Slow-roll estimate of the scalar power spectrum $\mathcal{P}_\zeta$.

        Uses the standard slow-roll formula
        $\mathcal{P}_\zeta \approx V(\phi_0)\,h^2 / (8\pi^2\,\epsilon_H)$
        where $\epsilon_H = \pi^2/2$.  This is fast to evaluate along the
        whole background trajectory and useful for comparison with the full
        Mukhanov-Sasaki result.

        Parameters
        ----------
        dphidN : float or array-like
            Field velocity $\pi = \mathrm{d}\phi/\mathrm{d}N$.
        h : float or array-like
            Rescaled Hubble parameter $h$, same shape as ``dphidN``.
        params : tuple
            Extra parameters forwarded to $V(\phi, *\mathrm{params})$.

        Returns
        -------
        float or numpy.ndarray
            Slow-roll approximation to $\mathcal{P}_\zeta$.
        """
        return (
            self.V(self.phi0, *params)
            * h**2
            / (8 * jnp.pi**2 * dphidN**2 / 2)
        )

    def epsilon_h(self, dphidN):
        r"""First Hubble slow-roll parameter $\epsilon_H = \pi^2/2$.

        $\epsilon_H$ measures how quickly the Hubble rate is changing:
        $\epsilon_H = -\dot{H}/H^2$.  Inflation ends when $\epsilon_H = 1$.

        Parameters
        ----------
        dphidN : float or array-like
            Field velocity $\pi = \mathrm{d}\phi/\mathrm{d}N$.

        Returns
        -------
        float or numpy.ndarray
            $\epsilon_H = \pi^2/2$.
        """
        return dphidN**2 / 2

    def eta_h(self, phi, dphidN, h, params):
        r"""Second Hubble slow-roll parameter $\eta_H$.

        $\eta_H$ characterises the curvature of the inflationary trajectory.
        In slow roll $|\eta_H| \ll 1$; large $|\eta_H|$ signals departures
        from slow roll (e.g. an inflection point).

        Parameters
        ----------
        phi : float or array-like
            Inflaton field value $\phi$.
        dphidN : float or array-like
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
        eps = self.epsilon_h(dphidN)
        return -6 + 2 * eps - self.Ud(phi, *params) * dphidN / (eps * h**2)

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

    def _solve_pzeta(self, k, params):
        """Background + perturbation solve returning the raw
        $P_\\zeta(k)$ array.

        Shared core of :meth:`__call__` (which returns the array) and
        :meth:`run` (which wraps it in a spline): evolve the background once,
        optionally warn if the model is a CMB outlier, then integrate the
        Mukhanov-Sasaki modes on the ``k`` grid.
        """
        params = jnp.array(params)
        N, phi, dphidN, h = self.run_background(params)
        if self.cmb_check:
            self._warn_if_cmb_outlier(N, phi, dphidN, h, params)
        return self.run_perturbations(k, N, phi, dphidN, h, params)

    def prepare(self, kint, *params):
        r"""Solve the Mukhanov-Sasaki system once on ``kint``; return a
        $P_\zeta(k)$ spline.

        Implements the
        [ScalarPerturbations][sigway.perturbations.ScalarPerturbations]
        solve-once seam: the integrator calls this once on its internal
        wavenumber grid, then evaluates the returned cheap cubic-Hermite
        interpolant at every $(s, t)$ point -- avoiding a full ODE solve per
        grid point.

        Parameters
        ----------
        kint : array-like
            Wavenumber grid in $\mathrm{s}^{-1}$ on which to solve the
            Mukhanov-Sasaki system.
        *params : float
            Scalar potential parameters forwarded to
            $V(\phi, *\mathrm{params})$.

        Returns
        -------
        callable
            A function ``pzeta_interpolant(k_new)`` evaluating the cubic-Hermite
            interpolant of $\mathcal{P}_\zeta$ within the ``kint`` range.
        """
        P_zeta = self._solve_pzeta(kint, params)
        coeff = backward_hermite_coefficients(kint, P_zeta)

        def pzeta_interpolant(knew):
            return CubicInterpolation(kint, coeff).evaluate(knew)

        return pzeta_interpolant

    def __call__(self, k, *params):
        r"""Compute and return the raw $\mathcal{P}_\zeta$ array at ``k``.

        Like ``run`` but returns the power-spectrum values directly instead of
        a callable interpolant.  Convenient for quickly plotting or fitting the
        spectrum at a fixed parameter set.

        Parameters
        ----------
        k : array-like
            Comoving wavenumber grid in $\mathrm{s}^{-1}$.
        *params : float
            Scalar potential parameters forwarded to
            $V(\phi, *\mathrm{params})$.

        Returns
        -------
        numpy.ndarray
            Dimensionful scalar power spectrum $\mathcal{P}_\zeta(k)$
            evaluated at each wavenumber in ``k``.
        """
        return self._solve_pzeta(k, params)

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
            Extra parameters forwarded to the potential
            $V(\phi, *\mathrm{params})$.

        Returns
        -------
        matplotlib.figure.Figure
            Five-panel figure of the background and perturbation evolution.
        """
        # Compute background evolution
        N, phi, dphidN, h = self.run_background(params)
        Pzeta_SR = self.pzeta_sr(dphidN, h, params)

        Nend = jnp.max(N)

        # Compute perturbations
        Pzeta = self.run_perturbations(k, N, phi, dphidN, h, params)
        N_CMB = jnp.max(N) - self.N_CMB_to_end
        H_CMB = jnp.interp(N_CMB, N, h)
        Nk = efolds_from_wavenumber_si_units(
            k, H_from_wavenumber(k, N, h, N_CMB, H_CMB), N_CMB, H_CMB
        )

        # Compute slow-roll parameters
        epsilon = self.epsilon_h(dphidN)
        eta = jnp.array(
            [self.eta_h(i, j, k, params) for i, j, k in zip(phi, dphidN, h)]
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

        # Plot dphidN=dphi/dN
        axs[3].plot(N - Nend, -dphidN, label="$-dphidN$")
        axs[3].set_ylabel("$-dphidN$")
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
