"""Kernel hierarchy for the scalar-induced GW transfer function.

The module-level functions below are the jit-able numeric cores, kept as plain
functions so the integrator can jit them with the kernel / perturbation
callables as static arguments. The Kernel classes are thin wrappers that select
the right core, declare the integration structure (k-dependence, resonant
slices) and carry the normalisation.
"""

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
    """
    Helper function to get u from s, t. To get this invert 4.19 of 2501.11320.
    s should be in [-1,1], t should be in [0, infty].

    Parameters:
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.

    Returns:
    - jax.numpy.ndarray
        Array of u values.
    """
    return (t + s + 1.0) / 2.0


@jit
def get_v(t, s):
    """
    Helper function to get v from s, t. To get this invert 4.19 of 2501.11320.
    s should be in [-1,1], t should be in [0, infty].

    Parameters:
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.

    Returns:
    - jax.numpy.ndarray
        Array of v values.
    """
    return (t - s + 1.0) / 2.0


@jit
def polynomial(t, s):
    """
    Polynomial term in the integrand for the Tensor power spectrum.
    See 4.20 of 2501.11320.
    Note that this term is k-independent.

    Parameters:
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.

    Returns:
    - jax.numpy.ndarray
        Array of polynomial values.
    """

    numerator = t * (2.0 + t) * (s**2 - 1.0)
    denominator = (1.0 - s + t) * (1.0 + s + t)

    return 2.0 * (numerator / denominator) ** 2


# Radiation domination all the way
@jit
def I_sq_RD_uv(t, s, k):
    r"""
    :math:`overline{I^2_{RD}(t, s, x\\to\\infty)}` assuming radiation
    domination. Note that this term is k-independent.

    This function is written in terms of u and v to match eq. 4.21, 4.22 of
    2501.11320. Notice that this is less stable numerically than I_sq_RD,
    especially for large values of t! We keep it for testing but for all
    computations use I_sq_RD.

    Parameters:
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.
    - k: jax.numpy.ndarray
        Array of k values.

    Returns:
    - jax.numpy.ndarray
        Array of :math:`overline{I^2_{RD}(t, s, x\\to\\infty)}` values.
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
    r"""
    :math:`overline{I^2_{RD}(t, s, x\\to\\infty)}` assuming radiation
    domination. Note that this term is k-independent.

    This function is written explicitly in terms of t and s.
    The output of this function matches with the output of I_sq_RD_uv,
    which is consistent with eq. 4.21, 4.22 of 2501.11320.

    Parameters:
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.
    - k: jax.numpy.ndarray
        Array of k values.

    Returns:
    - jax.numpy.ndarray
        Array of :math:`overline{I^2_{RD}(t, s, x\\to\\infty)}` values.
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
    """
    :math:`overline{I^2_{RD}(t, s)}` assuming all modes are reentering
    during the matter dominated era.

    Parameters:
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.
    - k: jax.numpy.ndarray
        Array of k values.

    Returns:
    - jax.numpy.ndarray
        Array of :math:`overline{I^2_{RD}(t, s)}` values.

    :Note:
    This function is independent of t, s and k.
    """
    return 18.0 / 25.0


# The Large V contribution to the early matter domination kernel contains the
# Si and Ci trigonometric integrals. These are evaluated directly with
# jax.scipy.special.sici (jit-able and differentiable; added in jax 0.8), which
# replaced a 1e7-point interpolation table that was both slower per call and
# carried a large import-time / memory cost.


@jit
def _sici_precomp(x):
    r"""
    Term containing Si and Ci functions in the Large V contribution to the
    transitioning kernel: :math:`4\,Ci(x/2)^2 + (\pi - 2\,Si(x/2))^2`.

    Parameters:
    - x: jax.numpy.ndarray
        Array of x values.

    Returns:
    - jax.numpy.ndarray
        Array of values.
    """
    si, ci = sici(x / 2.0)
    return 4.0 * ci**2 + (jnp.pi - 2.0 * si) ** 2


@jit
def _d_sici_precomp(x):
    """
    Derivative with respect to x of :func:`_sici_precomp`. In closed form
    (using Si'(y) = sin(y)/y, Ci'(y) = cos(y)/y with y = x/2) this is
    (8 cos(x/2) Ci(x/2) - 4 sin(x/2) (pi - 2 Si(x/2))) / x.

    Parameters:
    - x: jax.numpy.ndarray
        Array of x values.

    Returns:
    - jax.numpy.ndarray
        Array of derivative values.
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
    r"""
    :math:`overline{I^2_{\rm IRD, LV}(t, s, k, k_{\rm max}, \eta_R)}` for the
    large V contribution to the transitioning kernel from an early matter
    dominated era to radiation domination.

    Parameters:
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.
    - k: jax.numpy.ndarray
        Array of k values.
    - kmax: float
        k value at which the transition occurs.
    - etaR: float
        Conformal time at the beginning of radiation domination.

    Returns:
    - jax.numpy.ndarray
        Array of :math:`overline{I^2_{IRD}(t, s, k)}` values.
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
    """
    Compute the analytical gradient of the large V contribution to the
    transitioning kernel with respect to `kmax` or `etaR` based on `idx`.

    .. note::
    The gradient w.r.t `kmax` is zero even though the kernel depends on it
    through the integration limits. This is handled in the integration function.

    Parameters:
    - index: int
        Index of the parameter to differentiate with respect to
        (0 for kmax, 1 for etaR).
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.
    - k: jax.numpy.ndarray
        Array of k values.
    - kmax: float
        k value at which the transition occurs.
    - etaR: float
        Conformal time at the beginning of radiation domination.

    Returns:
    - jax.numpy.ndarray
        Array of gradient values.
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
    r"""
    :math:`overline{I^2_{\rm IRD, res}(t, s, k, \eta_R)}` for the resonant
    contribution to the transitioning kernel from an early matter dominated era
    to the radiation domination era.

    Parameters:
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.
    - k: jax.numpy.ndarray
        Array of k values.
    - kmax: float
        k value at which the transition occurs.
    - etaR: float
        Conformal time at the beginning of radiation domination. (???)

    Returns:
    - jax.numpy.ndarray
        Array of :math:`overline{I^2_{IRD}(t, s, k)}` values.

    :Note:
    This part of the kernel
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
    """
    Compute the analytical gradient of the resonant contribution to the
    transitioning kernel with respect to `kmax` or `etaR` based on `idx`.

    Parameters:
    - index: int
        Index of the parameter to differentiate with respect to
        (0 for kmax, 1 for etaR).
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.
    - k: jax.numpy.ndarray
        Array of k values.
    - kmax: float
        k value at which the transition occurs.
    - etaR: float
        Conformal time at the beginning of radiation domination.

    Returns:
    - jax.numpy.ndarray
        Array of gradient values.
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

    Subclasses provide ``overline_Isq`` (the oscillation-averaged transfer
    function) and declare their integration structure. ``norm(k)`` returns the
    full Omega_GW prefactor and is set at construction: a sensible default per
    kernel, overridable with a preset name, a constant, or a callable.
    """

    k_dependent = False
    param_names = ()
    resonant_t = ()
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
        """Full Omega_GW prefactor (includes the legacy factor of 2)."""
        return self._norm(k)

    def overline_Isq(self, t, s, k, *kparams):
        raise NotImplementedError

    def overline_Isq_resonant(self, t, s, k, *kparams):
        raise NotImplementedError


class RadiationKernel(Kernel):
    """Radiation-domination kernel (k-independent)."""

    k_dependent = False
    param_names = ()
    _default_norm = "RD"

    def overline_Isq(self, t, s, k, *kparams):
        return I_sq_RD(t, s, k)


class InstantEMDKernel(Kernel):
    """Instant early-matter-domination -> radiation kernel (k-dependent).

    Two contributions: a smooth large-V part integrated over (s, t) and a
    resonant slice at t = sqrt(3) - 1.

    Only ``etaR`` is a kernel parameter: this implementation depends on k and
    etaR (via xR = k * etaR), not on kmax. The transition scale kmax is the
    *source* cutoff, owned by the (heaviside) ScalarPerturbations, and also sets
    the t-grid range. (The underlying cores still take a kmax slot, which this
    implementation ignores, so we pass a dummy 0.0.)
    """

    k_dependent = True
    param_names = ("etaR",)
    resonant_t = (3.0**0.5 - 1.0,)
    _default_norm = "CT"

    def overline_Isq(self, t, s, k, etaR):
        return I_sq_IRD_LV(t, s, k, 0.0, etaR)

    def overline_Isq_resonant(self, t, s, k, etaR):
        return I_sq_IRD_res(t, s, k, 0.0, etaR)
