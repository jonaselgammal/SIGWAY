r"""Kernel hierarchy for the scalar-induced GW transfer function.

In the scalar-induced GW formalism, the energy-density spectrum
$\Omega_{\rm GW}(k)$ receives contributions from pairs of scalar (curvature)
modes with wavenumbers $p$ and $q$ that combine into a tensor mode at wavenumber
$k$. After averaging over the sub-horizon oscillations, each pair is weighted by
the **kernel** $\overline{I^2}(t, s, k)$ — the time-integrated (squared)
transfer function — which measures how efficiently that pair sources
gravitational waves during a given cosmological expansion era.

The integral is carried out in the rescaled momenta $s$ and $t$, related to the
momentum ratios $u = p/k$ and $v = q/k$ by $u=(t+s+1)/2$ and $v=(t-s+1)/2$
(inverting Eq. (4.19) of
[arXiv:2501.11320](https://arxiv.org/abs/2501.11320)). See
[Theory: the kernel](../theory/index.md#the-kernel).

This module provides the [Kernel][sigway.kernels.Kernel] classes — one per
expansion era — and the closed-form functions they evaluate.

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
from sigway.constants import (
    SM_CG_factor,
    Omega_radiation_h2_today,
    RD_SOUND_SPEED,
)

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
    r"""Recover the momentum ratio $u = p/k$ from the integration
    variables $(t, s)$.

    Inverts the change of variables in Eq. (4.19) of
    [arXiv:2501.11320](https://arxiv.org/abs/2501.11320):
    $u = (t + s + 1) / 2$, where $t = u + v - 1$ and $s = u - v$.

    Parameters
    ----------
    t : jax.Array
        Dimensionless combination $t = u + v - 1 \geq 0$, integrated over
        $[0, \infty)$.
    s : jax.Array
        Dimensionless combination $s = u - v$, integrated over $[-1, 1]$.

    Returns
    -------
    jax.Array
        The momentum ratio $u = p/k$, with the same shape as the broadcast of
        $t$ and $s$.
    """
    return (t + s + 1.0) / 2.0


@jit
def get_v(t, s):
    r"""Recover the momentum ratio $v = q/k$ from the integration
    variables $(t, s)$.

    Inverts the change of variables in Eq. (4.19) of
    [arXiv:2501.11320](https://arxiv.org/abs/2501.11320):
    $v = (t - s + 1) / 2$, where $t = u + v - 1$ and $s = u - v$.

    Parameters
    ----------
    t : jax.Array
        Dimensionless combination $t = u + v - 1 \geq 0$, integrated over
        $[0, \infty)$.
    s : jax.Array
        Dimensionless combination $s = u - v$, integrated over $[-1, 1]$.

    Returns
    -------
    jax.Array
        The momentum ratio $v = q/k$, with the same shape as the broadcast of
        $t$ and $s$.
    """
    return (t - s + 1.0) / 2.0


@jit
def polynomial(t, s):
    r"""Geometric projection factor for tensor-mode sourcing
    (Eq. (4.20) of 2501.11320).

    This $k$-independent factor arises from contracting the GW polarisation
    tensor with the stress-energy quadrupole of the two scalar modes.  In terms
    of the integration variables $(t, s)$ it reads

    $$2\left[\frac{t(2+t)(s^2-1)}{(1-s+t)(1+s+t)}\right]^2.$$

    It vanishes when $s = \pm 1$ (i.e. $u = 0$ or $v = 0$, collinear
    configuration), and is largest near $u \approx v \approx 1/\sqrt{3}$
    (the resonance).

    Parameters
    ----------
    t : jax.Array
        Dimensionless combination $t = u + v - 1 \geq 0$.
    s : jax.Array
        Dimensionless combination $s = u - v \in [-1, 1]$.

    Returns
    -------
    jax.Array
        Polynomial factor values with the same shape as the broadcast of $t$
        and $s$.
    """

    numerator = t * (2.0 + t) * (s**2 - 1.0)
    denominator = (1.0 - s + t) * (1.0 + s + t)

    return 2.0 * (numerator / denominator) ** 2


# Radiation domination all the way
@jit
def I_sq_RD_uv(t, s, k):
    r"""Oscillation-averaged radiation-domination kernel, expressed in
    $(u, v)$ variables.

    Evaluates $\overline{I^2_{\rm RD}}$ for a universe that remains in
    radiation domination from horizon re-entry to the present, using the
    $(u, v)$ form of Eqs. (4.21)–(4.22) of
    [arXiv:2501.11320](https://arxiv.org/abs/2501.11320).  Because the
    Green's function for a radiation-dominated universe has been
    oscillation-averaged
    analytically, the result is independent of $k$.

    This function is kept for cross-validation against `I_sq_RD`; at large $t$
    it is numerically less stable than the $(t, s)$ form.  All production
    calculations use `I_sq_RD` instead.

    Parameters
    ----------
    t : jax.Array
        Dimensionless combination $t = u + v - 1$.
    s : jax.Array
        Dimensionless combination $s = u - v$.
    k : jax.Array
        GW wave-number (Mpc$^{-1}$); unused, present for a uniform call
        signature across all kernel cores.

    Returns
    -------
    jax.Array
        $\overline{I^2_{\rm RD}}$ values, same shape as the broadcast of $t$
        and $s$.
    """
    u = get_u(t, s)
    v = get_v(t, s)

    # An auxiliary factor used in several places below (3 = 1/c_s^2, w = 1/3)
    factor = u**2 + v**2 - 3.0

    # These are the terms in eq. 4.22 of arXiv:2501.11320
    IA = 3.0 * factor / (4.0 * u**3 * v**3)
    IB = -4.0 * u * v + factor * jnp.log(
        jnp.abs((3.0 - (u + v) ** 2) / ((3.0 - (u - v) ** 2)))
    )
    # resonance onset u + v > 1/c_s = sqrt(3)
    IC = jnp.pi * factor * jnp.heaviside(u + v - 1.0 / RD_SOUND_SPEED, 1)

    return IA**2 * (IB**2 + IC**2) / 2.0


# Radiation domination all the way
@jit
def I_sq_RD(t, s, k):
    r"""Oscillation-averaged radiation-domination kernel, expressed in
    $(t, s)$ variables.

    Evaluates $\overline{I^2_{\rm RD}}$ for a universe in pure radiation
    domination, using the numerically stable $(t, s)$ form of
    Eqs. (4.21)–(4.22) of
    [arXiv:2501.11320](https://arxiv.org/abs/2501.11320).
    The kernel contains two pieces: a
    smooth logarithmic term (present for all $t > 0$) and a $\pi^2$ resonant
    piece that switches on when $u + v > \sqrt{3}$, i.e. when
    $1 + t > \sqrt{3}$, via a Heaviside step.  The result is independent of
    $k$ because the Green's function for radiation domination has been
    oscillation-averaged analytically.

    This is the production kernel used by
    [RadiationKernel][sigway.kernels.RadiationKernel].

    Parameters
    ----------
    t : jax.Array
        Dimensionless combination $t = u + v - 1$.
    s : jax.Array
        Dimensionless combination $s = u - v$.
    k : jax.Array
        GW wave-number (Mpc$^{-1}$); unused, present for a uniform call
        signature across all kernel cores.

    Returns
    -------
    jax.Array
        $\overline{I^2_{\rm RD}}$ values, same shape as the broadcast of $t$
        and $s$.
    """

    # This is IA**2 from eq. 4.21 of arXiv:2501.11320
    prefactor = (
        288.0
        * (-5.0 + s**2 + t * (2.0 + t)) ** 2
        / ((1.0 - s + t) ** 6 * (1.0 + s + t) ** 6)
    )

    # This is IB**2 from eq. 4.21 of arXiv:2501.11320
    # (the 3 in 3 - s**2 below is 1/c_s^2 for w = 1/3)
    log_term = (
        (-1.0 + s - t) * (1.0 + s + t)
        + (
            (-5.0 + s**2 + t * (2.0 + t))
            * jnp.log(jnp.abs((-2 + t * (2.0 + t)) / (3.0 - s**2)))
        )
        / 2.0
    ) ** 2

    # This is IC**2 from eq. 4.21 of arXiv:2501.11320
    heaviside_term = (
        jnp.pi**2
        * (-5.0 + s**2 + t * (2.0 + t)) ** 2
        * jnp.heaviside(1.0 - 1.0 / RD_SOUND_SPEED + t, 1)  # t > 1/c_s - 1
    ) / 4.0

    return prefactor * (log_term + heaviside_term)


# Pure matter domination. This is unphysical.
# UNUSED in favour of I_MD_TO_RD
@jit
def I_sq_MD(t, s, k):
    r"""Oscillation-averaged kernel for a universe in pure matter domination.

    Returns the constant $\overline{I^2_{\rm MD}} = 18/25$, independent of
    $t$, $s$, and $k$.  This is the analytic result for a universe that stays
    matter-dominated forever: the matter-era Green's function grows as a power
    law, and after oscillation-averaging the transfer function saturates at
    $18/25$.

    This is an unphysical limiting case — a realistic early-matter-dominated
    era eventually transitions to radiation domination.  For that physical
    scenario use `I_sq_IRD_LV` and `I_sq_IRD_res`.  This function is retained
    for reference and is no longer called in production.

    Parameters
    ----------
    t : jax.Array
        Dimensionless combination $t = u + v - 1$ (unused).
    s : jax.Array
        Dimensionless combination $s = u - v$ (unused).
    k : jax.Array
        GW wave-number (Mpc$^{-1}$); unused.

    Returns
    -------
    jax.Array
        Constant $18/25$, broadcast to the shape of $t$.
    """
    return 18.0 / 25.0


# The Large V contribution to the early matter domination kernel contains the
# Si and Ci trigonometric integrals. These are evaluated directly with
# jax.scipy.special.sici (jit-able and differentiable; added in jax 0.8), which
# replaced a 1e7-point interpolation table that was both slower per call and
# carried a large import-time / memory cost.


@jit
def _sici_precomp(x):
    r"""Auxiliary combination
    $4\,\mathrm{Ci}(x/2)^2 + (\pi - 2\,\mathrm{Si}(x/2))^2$.

    This combination of cosine and sine integrals arises from integrating the
    matter-era Green's function up to the EMD → RD transition time, and appears
    in the large-$V$ (large-$t$) part of the transitioning kernel.

    Parameters
    ----------
    x : jax.Array
        Argument array; in practice $x = x_R = k\,\eta_R$.

    Returns
    -------
    jax.Array
        Values of the Si/Ci combination, same shape as $x$.
    """
    si, ci = sici(x / 2.0)
    return 4.0 * ci**2 + (jnp.pi - 2.0 * si) ** 2


@jit
def _d_sici_precomp(x):
    r"""Derivative of `_sici_precomp` with respect to $x$.

    Closed-form result using $\mathrm{Si}'(y) = \sin(y)/y$ and
    $\mathrm{Ci}'(y) = \cos(y)/y$ with $y = x/2$:

    $$\frac{d}{dx}\!\left[4\,\mathrm{Ci}^2\!\left(\tfrac{x}{2}\right)
        + \left(\pi - 2\,\mathrm{Si}\!\left(\tfrac{x}{2}\right)\right)^2\right]
    = \frac{8\cos\!\left(\tfrac{x}{2}\right)
            \mathrm{Ci}\!\left(\tfrac{x}{2}\right)
            - 4\sin\!\left(\tfrac{x}{2}\right)
            \!\left(\pi - 2\,\mathrm{Si}\!\left(\tfrac{x}{2}\right)\right)}
        {x}.$$

    This is used in the analytic gradient of the large-$V$ kernel with respect
    to $\eta_R$ (via the chain rule on $x_R = k\,\eta_R$).

    Parameters
    ----------
    x : jax.Array
        Argument array; in practice $x = x_R = k\,\eta_R$.

    Returns
    -------
    jax.Array
        Derivative values, same shape as $x$.
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
    r"""Smooth (large-$V$) contribution to the instantaneous EMD → RD kernel.

    Evaluates $\overline{I^2_{\rm IRD,LV}}$, the dominant bulk piece of the
    oscillation-averaged transfer function for modes that re-enter the Hubble
    radius during the early matter-dominated era and are then amplified by the
    abrupt transition to radiation domination at conformal time $\eta_R$.

    This part of the kernel corresponds to the regime $u \sim v \gg 1$ (large
    $t$), where the mode functions have undergone many oscillations inside the
    Hubble radius before the transition.  The result depends on $k$ and
    $\eta_R$ only through the dimensionless combination $x_R = k\,\eta_R$,
    and is proportional to the Si/Ci combination `_sici_precomp`.  The
    integration domain in $(t, s)$ is bounded by $k_{\max}$ via
    $x_{\max,R} = k_{\max}\,\eta_R$.

    Used by [InstantEMDKernel][sigway.kernels.InstantEMDKernel].

    Parameters
    ----------
    t : jax.Array
        Dimensionless combination $t = u + v - 1$.
    s : jax.Array
        Dimensionless combination $s = u - v$.
    k : jax.Array
        GW wave-number (Mpc$^{-1}$).
    kmax : float
        Cutoff wave-number of the scalar power spectrum (Mpc$^{-1}$); sets the
        upper boundary of the $(t, s)$ integration domain via
        $x_{\max,R} = k_{\max}\,\eta_R$.
    etaR : float
        Conformal time at the EMD → RD transition (Mpc).

    Returns
    -------
    jax.Array
        $\overline{I^2_{\rm IRD,LV}}$ values, same shape as the broadcast of
        $t$, $s$, and $k$.
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
    r"""Analytic gradient of the smooth EMD → RD kernel with respect to
    $k_{\max}$ or $\eta_R$.

    Returns the partial derivative of `I_sq_IRD_LV` selected by *index*.
    The gradient with respect to $k_{\max}$ is identically zero inside the
    kernel body because $k_{\max}$ only shifts the integration domain
    boundaries; the corresponding boundary term is handled separately by the
    integrator.  The gradient with respect to $\eta_R$ follows from the chain
    rule on $x_R = k\,\eta_R$.

    Parameters
    ----------
    index : int
        Selects the differentiation target: ``0`` for $k_{\max}$, ``1`` for
        $\eta_R$.
    t : jax.Array
        Dimensionless combination $t = u + v - 1$.
    s : jax.Array
        Dimensionless combination $s = u - v$.
    k : jax.Array
        GW wave-number (Mpc$^{-1}$).
    kmax : float
        Cutoff wave-number of the scalar power spectrum (Mpc$^{-1}$).
    etaR : float
        Conformal time at the EMD → RD transition (Mpc).

    Returns
    -------
    jax.Array
        Gradient values, same shape as the broadcast of $t$, $s$, and $k$.
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
    r"""Resonant contribution to the instantaneous EMD → RD kernel.

    Evaluates $\overline{I^2_{\rm IRD,res}}$, the sharply peaked piece of the
    transitioning kernel that arises when $u + v = 1/c_s = \sqrt{3}$, i.e.
    when the combined momentum of the two scalar modes equals the sound horizon
    at the transition.  In integration variables this resonance sits at the
    fixed slice $t = \sqrt{3} - 1$, where the integrand is not smooth and must
    be treated separately.

    Near the resonance the $\mathrm{Ci}$ function diverges logarithmically;
    at the transition scale $x_R = k\,\eta_R$ it is approximated by
    $\mathrm{Ci}(x_R/2) \approx 7.97727 / x_R$, which captures the dominant
    behaviour for the $k$ values of interest.

    Used by [InstantEMDKernel][sigway.kernels.InstantEMDKernel] as the
    resonant integration slice declared in `resonant_t`.

    Parameters
    ----------
    t : jax.Array
        Dimensionless combination $t = u + v - 1$; this function is evaluated
        at $t = \sqrt{3} - 1$ to capture the resonance.
    s : jax.Array
        Dimensionless combination $s = u - v$.
    k : jax.Array
        GW wave-number (Mpc$^{-1}$).
    kmax : float
        Cutoff wave-number (Mpc$^{-1}$); unused here, present for a uniform
        call signature.
    etaR : float
        Conformal time at the EMD → RD transition (Mpc).

    Returns
    -------
    jax.Array
        $\overline{I^2_{\rm IRD,res}}$ values, same shape as the broadcast of
        $t$, $s$, and $k$.
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
    r"""Analytic gradient of the resonant EMD → RD kernel with respect to
    $k_{\max}$ or $\eta_R$.

    Returns the partial derivative of `I_sq_IRD_res` selected by *index*.
    The gradient with respect to $k_{\max}$ is identically zero (the resonant
    integrand does not depend on $k_{\max}$ directly).  The gradient with
    respect to $\eta_R$ follows from the power-law dependence of the kernel on
    $x_R = k\,\eta_R$: since the resonant kernel scales as $x_R^7$, the
    derivative is $7 / \eta_R$ times the kernel value.

    Parameters
    ----------
    index : int
        Selects the differentiation target: ``0`` for $k_{\max}$, ``1`` for
        $\eta_R$.
    t : jax.Array
        Dimensionless combination $t = u + v - 1$.
    s : jax.Array
        Dimensionless combination $s = u - v$.
    k : jax.Array
        GW wave-number (Mpc$^{-1}$).
    kmax : float
        Cutoff wave-number of the scalar power spectrum (Mpc$^{-1}$).
    etaR : float
        Conformal time at the EMD → RD transition (Mpc).

    Returns
    -------
    jax.Array
        Gradient values, same shape as the broadcast of $t$, $s$, and $k$.
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
    r"""Abstract base class for scalar-induced GW kernels $\overline{I^2}$.

    A kernel encapsulates $\overline{I^2}(t, s, k, \ldots)$: the
    oscillation-averaged, time-integrated transfer function squared that
    measures how efficiently a pair of scalar curvature perturbations with
    momentum ratios $u = p/k$ and $v = q/k$ sources gravitational waves at
    wave-number $k$ during a particular cosmological expansion era.  The
    energy density spectrum is

    $$\Omega_{\rm GW}(k) = \mathcal{N}(k) \int_0^\infty dt \int_{-1}^{1} ds\;
      \overline{I^2}(t, s, k)\; \mathcal{P}_\zeta(k\,u)\,
      \mathcal{P}_\zeta(k\,v),$$

    where $\mathcal{N}(k)$ is the normalisation returned by `norm` and
    $\mathcal{P}_\zeta$ is the dimensionless scalar power spectrum.

    Concrete subclasses implement `overline_Isq` (the smooth part of the
    kernel) and, when the kernel has a narrow resonant feature,
    `overline_Isq_resonant`.  They also set the class attributes
    `k_dependent`, `param_names`, and `resonant_t` to tell the integrator how
    to call them.

    Parameters
    ----------
    norm : str, float, or callable, optional
        Overall normalisation $\mathcal{N}(k)$ applied to $\Omega_{\rm GW}$.
        A string must be one of the preset keys (``'RD'``, ``'CT'``,
        ``'bare'``); a float is used as a constant; a callable must accept a
        wave-number array $k$ and return the prefactor.  The ``'RD'`` preset
        gives today's astrophysical value $c_g\,\Omega_{r,0}/12$; ``'CT'``
        and ``'bare'`` give the dimensionless $1/12$.  Defaults to the
        subclass ``_default_norm``.

    Attributes
    ----------
    k_dependent : bool
        ``True`` if $\overline{I^2}$ depends explicitly on $k$, requiring a
        separate kernel evaluation for each wave-number.
    param_names : tuple of str
        Names of any additional physical parameters the kernel requires
        beyond $(t, s, k)$, e.g. ``('etaR',)`` for the transition time.
    resonant_t : tuple of float
        Fixed $t$ values at which the kernel has a resonant feature narrow
        enough to require a dedicated integration slice.
    nonsmooth_params : tuple of str
        Kernel parameters whose gradient requires finite differences rather
        than JAX auto-differentiation.
    norm_spec : str, float, or callable
        The normalisation specification as passed at construction.

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
        r"""Return the $\Omega_{\rm GW}$ overall prefactor evaluated at $k$.

        For the ``'RD'`` preset this equals $c_g\,\Omega_{r,0}/12$, giving
        $\Omega_{\rm GW} h^2$ directly in today's radiation background.  For
        ``'CT'`` / ``'bare'`` the prefactor is $1/12$, yielding a
        dimensionless result normalised to the radiation density.

        Parameters
        ----------
        k : array-like
            GW wave-number array (Mpc$^{-1}$).

        Returns
        -------
        float or jax.Array
            Normalisation value(s) with the same shape as $k$.
        """
        return self._norm(k)

    def overline_Isq(self, t, s, k, *kparams):
        r"""Oscillation-averaged kernel $\overline{I^2}$ (smooth part).

        Returns the main, smoothly varying piece of the transfer function
        squared, integrated over the bulk of the $(t, s)$ domain.  Narrow
        resonant features at fixed $t$ values are handled separately by
        `overline_Isq_resonant`.

        Parameters
        ----------
        t : jax.Array
            Dimensionless combination $t = u + v - 1 \geq 0$.
        s : jax.Array
            Dimensionless combination $s = u - v \in [-1, 1]$.
        k : jax.Array
            GW wave-number (Mpc$^{-1}$).
        *kparams :
            Additional physical parameters listed in `param_names`
            (e.g. $\eta_R$ for
            [InstantEMDKernel][sigway.kernels.InstantEMDKernel]).

        Returns
        -------
        jax.Array
            $\overline{I^2}$ values.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def overline_Isq_resonant(self, t, s, k, *kparams):
        r"""Oscillation-averaged kernel $\overline{I^2_{\rm res}}$ at the
        resonant slice.

        Called at each fixed $t$ value listed in `resonant_t`, where the
        integrand has a narrow peak that cannot be resolved by the smooth
        quadrature grid.  Kernels without a resonance (e.g.
        [RadiationKernel][sigway.kernels.RadiationKernel]) do not need to
        override this method.

        Parameters
        ----------
        t : jax.Array
            Dimensionless combination $t = u + v - 1$, pinned to a value in
            `resonant_t` (e.g. $\sqrt{3} - 1$ for the sound-horizon
            resonance).
        s : jax.Array
            Dimensionless combination $s = u - v$.
        k : jax.Array
            GW wave-number (Mpc$^{-1}$).
        *kparams :
            Additional physical parameters listed in `param_names`.

        Returns
        -------
        jax.Array
            $\overline{I^2_{\rm res}}$ values.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses that declare `resonant_t`.
        """
        raise NotImplementedError


class RadiationKernel(Kernel):
    r"""Kernel for scalar-induced GWs produced entirely during radiation
    domination.

    Use this kernel when all relevant scalar modes re-enter the Hubble radius
    during a standard radiation-dominated era.  The underlying physics is that
    the GW source is active from Hubble re-entry until the present; once
    averaged over many oscillation cycles, the kernel takes the closed-form
    expression in Eqs. (4.21)–(4.22) of
    [arXiv:2501.11320](https://arxiv.org/abs/2501.11320).
    It has two pieces: a
    smooth logarithmic term that is always present, and a $\pi^2$ resonant
    contribution that switches on when $u + v > \sqrt{3}$ (equivalently
    $t > \sqrt{3} - 1$), where the combined scalar momentum equals the sound
    horizon.

    Because the Green's function for radiation domination is
    oscillation-averaged analytically, the kernel is independent of $k$ —
    all $k$ values share the same $(t, s)$ integrand.  There is no separate
    resonant slice to integrate.

    The default normalisation preset ``'RD'`` gives $\Omega_{\rm GW} h^2$ in
    terms of today's radiation density via $c_g\,\Omega_{r,0}/12$.

    Parameters
    ----------
    norm : str, float, or callable, optional
        Normalisation override; see [Kernel][sigway.kernels.Kernel].
        Defaults to ``'RD'``.

    Examples
    --------
    >>> from sigway.kernels import RadiationKernel
    >>> from sigway.spectrum import OmegaGW   # kernel is the 2nd argument
    """

    k_dependent = False
    param_names = ()
    _default_norm = "RD"

    def overline_Isq(self, t, s, k, *kparams):
        r"""Evaluate the radiation-domination kernel $\overline{I^2_{\rm RD}}$.

        Delegates to `I_sq_RD`, the numerically stable $(t, s)$ form of the
        oscillation-averaged kernel for pure radiation domination.

        Parameters
        ----------
        t : jax.Array
            Dimensionless combination $t = u + v - 1$.
        s : jax.Array
            Dimensionless combination $s = u - v$.
        k : jax.Array
            GW wave-number (Mpc$^{-1}$); unused, accepted for a uniform call
            signature.
        *kparams :
            Ignored (this kernel has no extra parameters).

        Returns
        -------
        jax.Array
            $\overline{I^2_{\rm RD}}$ values.
        """
        return I_sq_RD(t, s, k)


class InstantEMDKernel(Kernel):
    r"""Kernel for scalar-induced GWs produced in an early matter era with
    a sudden reheating.

    Models a universe that starts in an early matter-dominated (EMD) era
    (e.g. dominated by a pressureless oscillating field) and transitions
    instantaneously to radiation domination at conformal time $\eta_R$.
    Scalar modes that re-enter the Hubble radius before $\eta_R$ experience
    enhanced growth inside the horizon during the matter era; after the
    transition the sourcing of GWs continues in the radiation era.

    The oscillation-averaged kernel receives two distinct contributions that
    must be integrated separately:

    1. **Smooth (large-$V$) part** — the bulk contribution from modes with
       $u \sim v \gg 1$ (large $t$), evaluated via `I_sq_IRD_LV` in
       `overline_Isq`.  It depends on $k$ and $\eta_R$ through
       $x_R = k\,\eta_R$.
    2. **Resonant slice** at $t = \sqrt{3} - 1$ (i.e. $u + v = 1/c_s =
       \sqrt{3}$) — a narrow peak from the sound-horizon resonance, evaluated
       via `I_sq_IRD_res` in `overline_Isq_resonant` and declared in
       `resonant_t`.

    The cutoff wave-number $k_{\max}$, which marks the end of the enhanced
    scalar power spectrum, is a property of the perturbation object (typically
    a heaviside `ScalarPerturbations`) rather than of the kernel itself; it
    sets the upper boundary of the $(t, s)$ integration domain externally.
    Accordingly, the underlying numeric cores receive ``kmax=0.0`` here and
    domain clipping is handled by the integrator.

    The default normalisation preset ``'CT'`` gives the dimensionless ratio
    $\Omega_{\rm GW} / \Omega_r$ via the prefactor $1/12$.

    Parameters
    ----------
    norm : str, float, or callable, optional
        Normalisation override; see [Kernel][sigway.kernels.Kernel].
        Defaults to ``'CT'``.

    Attributes
    ----------
    k_dependent : bool
        Always ``True``; the kernel depends on $k$ through $x_R = k\,\eta_R$.
    param_names : tuple of str
        ``('etaR',)`` — the conformal time of the EMD → RD transition must be
        supplied as a kernel parameter when calling `overline_Isq` and
        `overline_Isq_resonant`.
    resonant_t : tuple of float
        ``(sqrt(3) - 1,)`` — the single resonant integration slice at
        $t = \sqrt{3} - 1$.
    """

    k_dependent = True
    param_names = ("etaR",)
    resonant_t = (1.0 / RD_SOUND_SPEED - 1.0,)  # t = 1/c_s - 1 (sound-horizon)
    _default_norm = "CT"

    def overline_Isq(self, t, s, k, etaR):
        r"""Evaluate the smooth (large-$V$) EMD → RD kernel
        $\overline{I^2_{\rm IRD,LV}}$.

        Delegates to `I_sq_IRD_LV` with ``kmax=0.0``: domain clipping based
        on $k_{\max}$ is applied externally by the integrator.

        Parameters
        ----------
        t : jax.Array
            Dimensionless combination $t = u + v - 1$.
        s : jax.Array
            Dimensionless combination $s = u - v$.
        k : jax.Array
            GW wave-number (Mpc$^{-1}$).
        etaR : float
            Conformal time at the EMD → RD transition (Mpc).

        Returns
        -------
        jax.Array
            $\overline{I^2_{\rm IRD,LV}}$ values.
        """
        return I_sq_IRD_LV(t, s, k, 0.0, etaR)

    def overline_Isq_resonant(self, t, s, k, etaR):
        r"""Evaluate the resonant EMD → RD kernel
        $\overline{I^2_{\rm IRD,res}}$.

        Called at the resonant slice $t = \sqrt{3} - 1$ declared in
        `resonant_t`, where $u + v = \sqrt{3} = 1/c_s$.  Delegates to
        `I_sq_IRD_res` with ``kmax=0.0``.

        Parameters
        ----------
        t : jax.Array
            Dimensionless combination $t = u + v - 1$, evaluated at
            $\sqrt{3} - 1$.
        s : jax.Array
            Dimensionless combination $s = u - v$.
        k : jax.Array
            GW wave-number (Mpc$^{-1}$).
        etaR : float
            Conformal time at the EMD → RD transition (Mpc).

        Returns
        -------
        jax.Array
            $\overline{I^2_{\rm IRD,res}}$ values.
        """
        return I_sq_IRD_res(t, s, k, 0.0, etaR)
