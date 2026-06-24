"""Kernel hierarchy for the scalar-induced GW transfer function.

The module-level functions below are the JIT-able numeric cores, kept as plain
functions so the integrator can JIT them with the kernel and perturbation
callables as static arguments.  The [Kernel][sigway.kernels.Kernel] classes are
thin wrappers that select the right core, declare the integration structure
(k-dependence, resonant slices) and carry the normalisation.

Public API
----------
Helper functions:
    get_u, get_v, polynomial

Kernel classes:
    Kernel, RadiationKernel, InstantEMDKernel
"""

__all__ = [
    "get_u",
    "get_v",
    "polynomial",
    "Kernel",
    "RadiationKernel",
    "InstantEMDKernel",
]

# Global
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import sici

# Local
from sigway.utils import SM_CG_factor, Omega_radiation_h2_today

jax.config.update("jax_enable_x64", True)

# Normalisation presets: the full Omega_GW prefactor (folding in the factor of 2
# that the legacy integrator applied separately). "RD" is today's astrophysical
# value c_g * Omega_r,0 / 12; "CT"/"bare" is the dimensionless 1/12.
NORM_PRESETS = {
    "RD": SM_CG_factor / 12.0 * Omega_radiation_h2_today,
    "CT": 1.0 / 12.0,
    "bare": 1.0 / 12.0,
}


@jit
def get_u(t, s):
    """Compute the integration variable *u* from (*t*, *s*).

    Inverts Eq. (4.19) of 2501.11320: ``u = (t + s + 1) / 2``.

    Parameters
    ----------
    t : jax.Array
        Dimensionless time variable; lives in ``[0, inf)``.
    s : jax.Array
        Dimensionless momentum ratio; lives in ``[-1, 1]``.

    Returns
    -------
    jax.Array
        Array of *u* values with the same shape as the broadcast of *t* and *s*.
    """
    return (t + s + 1.0) / 2.0


@jit
def get_v(t, s):
    """Compute the integration variable *v* from (*t*, *s*).

    Inverts Eq. (4.19) of 2501.11320: ``v = (t - s + 1) / 2``.

    Parameters
    ----------
    t : jax.Array
        Dimensionless time variable; lives in ``[0, inf)``.
    s : jax.Array
        Dimensionless momentum ratio; lives in ``[-1, 1]``.

    Returns
    -------
    jax.Array
        Array of *v* values with the same shape as the broadcast of *t* and *s*.
    """
    return (t - s + 1.0) / 2.0


@jit
def polynomial(t, s):
    """Geometric factor in the GW integrand (Eq. (4.20) of 2501.11320).

    Evaluates the k-independent polynomial prefactor
    ``2 * [t(2+t)(s^2-1) / ((1-s+t)(1+s+t))]^2``.

    Parameters
    ----------
    t : jax.Array
        Dimensionless time variable; lives in ``[0, inf)``.
    s : jax.Array
        Dimensionless momentum ratio; lives in ``[-1, 1]``.

    Returns
    -------
    jax.Array
        Polynomial values with the same shape as the broadcast of *t* and *s*.
    """

    numerator = t * (2.0 + t) * (s**2 - 1.0)
    denominator = (1.0 - s + t) * (1.0 + s + t)

    return 2.0 * (numerator / denominator) ** 2


# Radiation domination all the way
@jit
def I_sq_RD_uv(t, s, k):
    r"""Oscillation-averaged RD kernel in (*u*, *v*) variables (testing only).

    Evaluates :math:`\overline{I^2_{\mathrm{RD}}(t,s,x\to\infty)}` under
    pure radiation domination using the (*u*, *v*) form of Eqs. (4.21)–(4.22)
    of 2501.11320.  The result is k-independent.

    .. note::
        This formulation is numerically less stable than ``I_sq_RD`` for large
        *t*.  It is kept for cross-validation purposes only; all production
        computations should use ``I_sq_RD``.

    Parameters
    ----------
    t : jax.Array
        Dimensionless time variable.
    s : jax.Array
        Dimensionless momentum ratio.
    k : jax.Array
        Wave-number array (unused; accepted for a uniform call signature).

    Returns
    -------
    jax.Array
        :math:`\overline{I^2_{\mathrm{RD}}}` values, same shape as the
        broadcast of *t* and *s*.
    """
    u = get_u(t, s)
    v = get_v(t, s)

    # An auxiliary factor used in several places below
    factor = u**2 + v**2 - 3.0

    # These are the terms in 4.22 of 2501.11320
    IA = 3.0 * factor / (4.0 * u**3 * v**3)
    IB = -4.0 * u * v + factor * jnp.log(
        jnp.abs((3.0 - (u + v) ** 2) / ((3.0 - (u - v) ** 2)))
    )
    IC = jnp.pi * factor * jnp.heaviside(u + v - jnp.sqrt(3), 1)

    return IA**2 * (IB**2 + IC**2) / 2.0


# Radiation domination all the way
@jit
def I_sq_RD(t, s, k):
    r"""Oscillation-averaged RD kernel in (*t*, *s*) variables (production).

    Evaluates :math:`\overline{I^2_{\mathrm{RD}}(t,s,x\to\infty)}` under
    pure radiation domination, expressed directly in (*t*, *s*) for improved
    numerical stability at large *t* (Eqs. (4.21)–(4.22) of 2501.11320).
    The result is k-independent.

    Parameters
    ----------
    t : jax.Array
        Dimensionless time variable.
    s : jax.Array
        Dimensionless momentum ratio.
    k : jax.Array
        Wave-number array (unused; accepted for a uniform call signature).

    Returns
    -------
    jax.Array
        :math:`\overline{I^2_{\mathrm{RD}}}` values, same shape as the
        broadcast of *t* and *s*.
    """

    # This is IA**2 from eq. 4.21 of 2501.11320
    prefactor = (
        288.0
        * (-5.0 + s**2 + t * (2.0 + t)) ** 2
        / ((1.0 - s + t) ** 6 * (1.0 + s + t) ** 6)
    )

    # This is IB**2 from eq. 4.21 of 2501.11320
    log_term = (
        (-1.0 + s - t) * (1.0 + s + t)
        + (
            (-5.0 + s**2 + t * (2.0 + t))
            * jnp.log(jnp.abs((-2 + t * (2.0 + t)) / (3.0 - s**2)))
        )
        / 2.0
    ) ** 2

    # This is IC**2 from eq. 4.21 of 2501.11320
    heaviside_term = (
        jnp.pi**2
        * (-5.0 + s**2 + t * (2.0 + t)) ** 2
        * jnp.heaviside(1.0 - jnp.sqrt(3.0) + t, 1)
    ) / 4.0

    return prefactor * (log_term + heaviside_term)


# Pure matter domination. This is unphysical.
# UNUSED in favour of I_MD_TO_RD
@jit
def I_sq_MD(t, s, k):
    r"""Oscillation-averaged kernel for pure matter domination (unphysical).

    Returns the constant :math:`\overline{I^2_{\mathrm{MD}}} = 18/25`,
    independent of *t*, *s*, and *k*, corresponding to the limit where all
    source modes re-enter during a matter-dominated era.

    .. note::
        This is an unphysical limiting case.  It is superseded by
        ``I_sq_IRD_LV`` / ``I_sq_IRD_res`` for realistic EMD → RD scenarios
        and is no longer used in production.

    Parameters
    ----------
    t : jax.Array
        Dimensionless time variable (unused).
    s : jax.Array
        Dimensionless momentum ratio (unused).
    k : jax.Array
        Wave-number array (unused).

    Returns
    -------
    jax.Array
        Scalar constant 18/25, broadcast to the shape of *t*.
    """
    return 18.0 / 25.0


# The Large V contribution to the early matter domination kernel contains the
# Si and Ci trigonometric integrals. These are evaluated directly with
# jax.scipy.special.sici (jit-able and differentiable; added in jax 0.8), which
# replaced a 1e7-point interpolation table that was both slower per call and
# carried a large import-time / memory cost.


@jit
def _sici_precomp(x):
    r"""Term :math:`4\,\mathrm{Ci}(x/2)^2 + (\pi - 2\,\mathrm{Si}(x/2))^2`.

    Appears in the large-V contribution to the transitioning
    (early-matter-domination → radiation) kernel.

    Parameters
    ----------
    x : jax.Array
        Argument array (typically ``k * etaR``).

    Returns
    -------
    jax.Array
        Values of the Si/Ci combination, same shape as *x*.
    """
    si, ci = sici(x / 2.0)
    return 4.0 * ci**2 + (jnp.pi - 2.0 * si) ** 2


@jit
def _d_sici_precomp(x):
    r"""Derivative of ``_sici_precomp`` with respect to *x*.

    Closed-form result using :math:`\mathrm{Si}'(y)=\sin(y)/y` and
    :math:`\mathrm{Ci}'(y)=\cos(y)/y` with :math:`y = x/2`:

    .. math::

        \frac{d}{dx}\bigl[4\,\mathrm{Ci}^2 + (\pi - 2\,\mathrm{Si})^2\bigr]
        = \frac{8\cos(x/2)\,\mathrm{Ci}(x/2)
                - 4\sin(x/2)\,(\pi - 2\,\mathrm{Si}(x/2))}{x}

    Parameters
    ----------
    x : jax.Array
        Argument array (typically ``k * etaR``).

    Returns
    -------
    jax.Array
        Derivative values, same shape as *x*.
    """
    si, ci = sici(x / 2.0)
    return (
        8.0 * jnp.cos(x / 2.0) * ci
        - 4.0 * jnp.sin(x / 2.0) * (jnp.pi - 2.0 * si)
    ) / x


# below are the two main contributions to the transitioning kernel, based on
# sudden-reheating scenarios. As they are evaluated at different t's we need to
# evaluate them separately and sum them up.


# Transition from an early matter dominated era to the RD era,
# the u ~ v >> 1 contribution, i.e. large t
@jit
def I_sq_IRD_LV(t, s, k, kmax, etaR):
    r"""Large-V contribution to the EMD → RD transitioning kernel.

    Evaluates
    :math:`\overline{I^2_{\mathrm{IRD,LV}}(t,s,k,k_{\max},\eta_R)}`,
    the :math:`u \sim v \gg 1` (large-*t*) part of the oscillation-averaged
    transfer function for the instantaneous EMD → RD transition.  The result
    depends on *k* and *etaR* only through :math:`x_R = k\,\eta_R` and on the
    ``_sici_precomp`` combination of Si/Ci functions.

    Parameters
    ----------
    t : jax.Array
        Dimensionless time variable.
    s : jax.Array
        Dimensionless momentum ratio.
    k : jax.Array
        Wave-number (Mpc\ :sup:`-1`).
    kmax : float
        Source cutoff scale (Mpc\ :sup:`-1`); controls the valid (*s*, *t*)
        domain via :math:`x_{\max,R} = k_{\max}\,\eta_R`.
    etaR : float
        Conformal time at the start of radiation domination (Mpc).

    Returns
    -------
    jax.Array
        :math:`\overline{I^2_{\mathrm{IRD,LV}}}` values, same shape as the
        broadcast of *t*, *s*, and *k*.
    """
    xR = k * etaR
    xmaxR = kmax * etaR
    xmaxR_ratio = xmaxR / xR

    # Calculate the bounds for s and t
    s_max = jnp.where(xR <= xmaxR, 1.0, 2.0 * xmaxR_ratio - 1.0)
    t_max = -s + 2.0 * xmaxR_ratio - 1.0

    # Ensure that t respects the bounds
    valid_t = jnp.logical_and(t >= 0, t <= t_max)
    valid_s = jnp.logical_and(s >= 0, s <= s_max)

    # Calculate the result only within valid regions
    result = jnp.where(
        valid_t & valid_s,
        (9.0 * t**4.0 * xR**8.0 * _sici_precomp(xR)) / 81920000.0,
        0.0,
    )
    result = (9.0 * t**4.0 * xR**8.0 * _sici_precomp(xR)) / 81920000.0

    return 4.0 * result  # the factor of 4 comes from x_R^2/(x_R-x_R/2)^2


@jit
def d_I_sq_IRD_LV(index, t, s, k, kmax, etaR):
    """Analytical gradient of the large-V EMD → RD kernel.

    Returns the partial derivative of ``I_sq_IRD_LV`` with respect to either
    *kmax* (``index=0``) or *etaR* (``index=1``).

    Notes
    -----
    The gradient with respect to *kmax* is identically zero *within* the
    kernel body: *kmax* only affects the integration domain boundaries, not the
    integrand itself.  That boundary contribution is handled separately by the
    integration function.

    Parameters
    ----------
    index : int
        Selects the differentiation target: ``0`` → *kmax*, ``1`` → *etaR*.
    t : jax.Array
        Dimensionless time variable.
    s : jax.Array
        Dimensionless momentum ratio.
    k : jax.Array
        Wave-number (Mpc\ :sup:`-1`).
    kmax : float
        Source cutoff scale (Mpc\ :sup:`-1`).
    etaR : float
        Conformal time at the start of radiation domination (Mpc).

    Returns
    -------
    jax.Array
        Gradient values, same shape as the broadcast of *t*, *s*, and *k*.
    """
    result = I_sq_IRD_LV(t, s, k, kmax, etaR)
    xR = k * etaR
    grad_etaR = k * (_d_sici_precomp(xR) / _sici_precomp(xR) + 8 / xR) * result
    # The gradient w.r.t anyting but etaR is zero
    grad_zero = jnp.zeros_like(result)
    return lax.cond(
        index == 0, lambda _: grad_zero, lambda _: grad_etaR, operand=None
    )


# the resonant contribution when u+v ~ 1/c_s, or t = sqrt(3) - 1
@jit
def I_sq_IRD_res(t, s, k, kmax, etaR):
    r"""Resonant contribution to the EMD → RD transitioning kernel.

    Evaluates
    :math:`\overline{I^2_{\mathrm{IRD,res}}(t,s,k,\eta_R)}`, the resonant
    slice of the transitioning kernel arising when
    :math:`u + v \approx 1/c_s`, i.e. at :math:`t = \sqrt{3} - 1`.  The Ci
    amplitude is approximated by ``7.97727 / xR``.

    Parameters
    ----------
    t : jax.Array
        Dimensionless time variable (should be evaluated near
        :math:`\sqrt{3} - 1` for the resonance to contribute).
    s : jax.Array
        Dimensionless momentum ratio.
    k : jax.Array
        Wave-number (Mpc\ :sup:`-1`).
    kmax : float
        Source cutoff scale (Mpc\ :sup:`-1`); accepted for call-signature
        uniformity but not used in this function.
    etaR : float
        Conformal time at the start of radiation domination (Mpc).

    Returns
    -------
    jax.Array
        :math:`\overline{I^2_{\mathrm{IRD,res}}}` values, same shape as the
        broadcast of *t*, *s*, and *k*.
    """
    fudge = 2.3
    xR = k * etaR

    num = 9 * (-5 + s**2 + 2 * t + t**2) ** 4 * xR**8
    den = 81920000 * (t - s + 1) ** 2 * (t + s + 1) ** 2
    ci_val = 7.97727 / xR
    result = fudge * (num / den) * ci_val
    return 4 * result  # the factor of 4 comes from x_R^2/(x_R-x_R/2)^2


@jit
def d_I_sq_IRD_res(index, t, s, k, kmax, etaR):
    """Analytical gradient of the resonant EMD → RD kernel.

    Returns the partial derivative of ``I_sq_IRD_res`` with respect to either
    *kmax* (``index=0``) or *etaR* (``index=1``).  The gradient with respect
    to *kmax* is identically zero; the gradient with respect to *etaR* is
    ``7 / etaR * I_sq_IRD_res``.

    Parameters
    ----------
    index : int
        Selects the differentiation target: ``0`` → *kmax*, ``1`` → *etaR*.
    t : jax.Array
        Dimensionless time variable.
    s : jax.Array
        Dimensionless momentum ratio.
    k : jax.Array
        Wave-number (Mpc\ :sup:`-1`).
    kmax : float
        Source cutoff scale (Mpc\ :sup:`-1`).
    etaR : float
        Conformal time at the start of radiation domination (Mpc).

    Returns
    -------
    jax.Array
        Gradient values, same shape as the broadcast of *t*, *s*, and *k*.
    """
    # Get the main result using the provided function
    result = I_sq_IRD_res(t, s, k, kmax, etaR)
    # The gradient w.r.t anyting but etaR is zero
    grad_zero = jnp.zeros_like(result)
    # Compute the gradient w.r.t etaR
    grad_etaR = (
        7 / etaR * result
    )  # Based on the simplified result of the derivative
    # Use lax.cond to select the derivative based on idx
    return lax.cond(
        index == 0, lambda _: grad_zero, lambda _: grad_etaR, operand=None
    )


class Kernel:
    """Base class for SIGW transfer-function kernels.

    A [Kernel][sigway.kernels.Kernel] encapsulates the oscillation-averaged
    transfer-function squared,
    :math:`\overline{I^2}(t,s,k,\ldots)`, for a particular cosmological
    history (radiation domination, EMD → RD, etc.).  Subclasses implement
    ``overline_Isq`` and, when there is a resonant contribution,
    ``overline_Isq_resonant``; they also declare the integration metadata
    (``k_dependent``, ``param_names``, ``resonant_t``).

    The ``norm(k)`` method returns the full :math:`\Omega_{\rm GW}` prefactor,
    including the factor of 2 that the legacy integrator applied separately.
    The normalisation is fixed at construction time and can be overridden via
    the *norm* argument.

    Parameters
    ----------
    norm : str, float, or callable, optional
        Normalisation specification.  A string must be one of the preset keys
        (``'RD'``, ``'CT'``, ``'bare'``); a float is used as a constant; a
        callable must accept a wave-number array *k* and return the prefactor.
        Defaults to the subclass ``_default_norm``.

    Attributes
    ----------
    k_dependent : bool
        Whether the kernel depends on *k* (and therefore requires a separate
        evaluation per *k* value).
    param_names : tuple of str
        Names of any extra kernel parameters (e.g. ``('etaR',)``).
    resonant_t : tuple of float
        Fixed *t* values at which the kernel has a resonant contribution that
        must be integrated separately.
    nonsmooth_params : tuple of str
        Kernel parameters whose gradient cannot be obtained via JAX
        auto-differentiation (finite differences required).
    norm_spec : str, float, or callable
        The normalisation specification passed at construction, stored for
        introspection.

    Raises
    ------
    ValueError
        If *norm* is a string that is not a recognised preset.
    """

    k_dependent = False
    param_names = ()
    resonant_t = ()
    # kernel params whose derivative needs finite differences (see
    # ScalarPerturbations.nonsmooth_params); none for the current kernels.
    nonsmooth_params = ()
    _default_norm = "RD"

    def __init__(self, norm=None):
        spec = self._default_norm if norm is None else norm
        if isinstance(spec, str):
            if spec not in NORM_PRESETS:
                raise ValueError(
                    "Unknown norm preset {!r}; choose from {}.".format(
                        spec, sorted(NORM_PRESETS)
                    )
                )
            value = NORM_PRESETS[spec]
            self._norm = lambda k: value
        elif callable(spec):
            self._norm = spec
        else:
            self._norm = lambda k: spec
        self.norm_spec = spec

    def norm(self, k):
        r"""Return the :math:`\Omega_{\rm GW}` prefactor evaluated at *k*.

        The value folds in the factor of 2 that the legacy integrator applied
        separately.  For the ``'RD'`` preset this equals
        ``c_g * Omega_r,0 / 12``; for ``'CT'`` / ``'bare'`` it is ``1/12``.

        Parameters
        ----------
        k : array-like
            Wave-number array (Mpc\ :sup:`-1`).

        Returns
        -------
        float or jax.Array
            Normalisation value(s) with the same shape as *k*.
        """
        return self._norm(k)

    def overline_Isq(self, t, s, k, *kparams):
        """Oscillation-averaged transfer-function squared (smooth part).

        Parameters
        ----------
        t : jax.Array
            Dimensionless time variable.
        s : jax.Array
            Dimensionless momentum ratio.
        k : jax.Array
            Wave-number (Mpc\ :sup:`-1`).
        *kparams :
            Additional kernel parameters listed in ``param_names``
            (e.g. *etaR* for [InstantEMDKernel][sigway.kernels.InstantEMDKernel]).

        Returns
        -------
        jax.Array
            :math:`\overline{I^2}` values.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def overline_Isq_resonant(self, t, s, k, *kparams):
        """Oscillation-averaged transfer-function squared (resonant slice).

        Called at the fixed *t* values listed in ``resonant_t``.  Kernels
        without a resonant contribution do not need to override this method.

        Parameters
        ----------
        t : jax.Array
            Dimensionless time variable (typically pinned to a value in
            ``resonant_t``).
        s : jax.Array
            Dimensionless momentum ratio.
        k : jax.Array
            Wave-number (Mpc\ :sup:`-1`).
        *kparams :
            Additional kernel parameters listed in ``param_names``.

        Returns
        -------
        jax.Array
            :math:`\overline{I^2_{\rm res}}` values.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses that declare ``resonant_t``.
        """
        raise NotImplementedError


class RadiationKernel(Kernel):
    """Kernel for a universe in pure radiation domination throughout.

    Uses ``I_sq_RD`` to evaluate the oscillation-averaged transfer function
    :math:`\overline{I^2_{\mathrm{RD}}(t,s,x\to\infty)}` (Eqs. (4.21)–(4.22)
    of 2501.11320).  The kernel is k-independent and has no resonant
    contribution.

    The default normalisation preset is ``'RD'`` (today's astrophysical value
    ``c_g * Omega_r,0 / 12``).

    Parameters
    ----------
    norm : str, float, or callable, optional
        Normalisation override; see [Kernel][sigway.kernels.Kernel].
        Defaults to ``'RD'``.
    """

    k_dependent = False
    param_names = ()
    _default_norm = "RD"

    def overline_Isq(self, t, s, k, *kparams):
        """Return the RD oscillation-averaged kernel via ``I_sq_RD``.

        Parameters
        ----------
        t : jax.Array
            Dimensionless time variable.
        s : jax.Array
            Dimensionless momentum ratio.
        k : jax.Array
            Wave-number (Mpc\ :sup:`-1`; unused, accepted for uniformity).
        *kparams :
            Ignored (no extra kernel parameters for this kernel).

        Returns
        -------
        jax.Array
            :math:`\overline{I^2_{\mathrm{RD}}}` values.
        """
        return I_sq_RD(t, s, k)


class InstantEMDKernel(Kernel):
    """Kernel for an instantaneous early-matter-domination → RD transition.

    Models a scenario where the universe undergoes an early matter-dominated
    era that ends abruptly at conformal time *etaR*, after which it evolves
    in radiation domination.  The oscillation-averaged transfer function has
    two distinct contributions:

    1. **Smooth (large-V) part** — integrated over the full (*s*, *t*) domain
       via ``I_sq_IRD_LV`` (``overline_Isq``).
    2. **Resonant slice** at :math:`t = \sqrt{3} - 1` (i.e. :math:`u + v =
       1/c_s`) — integrated separately via ``I_sq_IRD_res``
       (``overline_Isq_resonant``).

    The kernel is k-dependent through :math:`x_R = k\,\eta_R`.

    The transition scale *kmax* is a *source* cutoff owned by the perturbation
    object (typically a heaviside ``ScalarPerturbations``); it also sets the
    upper limit of the *t*-integration grid.  The underlying numeric cores
    (``I_sq_IRD_LV``, ``I_sq_IRD_res``) accept a *kmax* argument, but this
    kernel passes ``0.0`` because domain clipping based on *kmax* is handled
    externally.

    The default normalisation preset is ``'CT'`` (dimensionless, ``1/12``).

    Parameters
    ----------
    norm : str, float, or callable, optional
        Normalisation override; see [Kernel][sigway.kernels.Kernel].
        Defaults to ``'CT'``.

    Attributes
    ----------
    k_dependent : bool
        Always ``True`` for this kernel.
    param_names : tuple of str
        ``('etaR',)`` — *etaR* must be supplied as a kernel parameter.
    resonant_t : tuple of float
        ``(sqrt(3) - 1,)`` — the single resonant integration slice.
    """

    k_dependent = True
    param_names = ("etaR",)
    resonant_t = (3.0**0.5 - 1.0,)
    _default_norm = "CT"

    def overline_Isq(self, t, s, k, etaR):
        """Return the smooth (large-V) EMD → RD kernel via ``I_sq_IRD_LV``.

        Parameters
        ----------
        t : jax.Array
            Dimensionless time variable.
        s : jax.Array
            Dimensionless momentum ratio.
        k : jax.Array
            Wave-number (Mpc\ :sup:`-1`).
        etaR : float
            Conformal time at the start of radiation domination (Mpc).

        Returns
        -------
        jax.Array
            :math:`\overline{I^2_{\mathrm{IRD,LV}}}` values.
        """
        return I_sq_IRD_LV(t, s, k, 0.0, etaR)

    def overline_Isq_resonant(self, t, s, k, etaR):
        """Return the resonant EMD → RD kernel via ``I_sq_IRD_res``.

        Parameters
        ----------
        t : jax.Array
            Dimensionless time variable (evaluated at :math:`\sqrt{3} - 1`).
        s : jax.Array
            Dimensionless momentum ratio.
        k : jax.Array
            Wave-number (Mpc\ :sup:`-1`).
        etaR : float
            Conformal time at the start of radiation domination (Mpc).

        Returns
        -------
        jax.Array
            :math:`\overline{I^2_{\mathrm{IRD,res}}}` values.
        """
        return I_sq_IRD_res(t, s, k, 0.0, etaR)
