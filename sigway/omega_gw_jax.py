# Global
import numpy as np
from scipy.special import sici

# Local
import jax
import jax.numpy as jnp
from jax import jit, jvp, lax

jax.config.update("jax_enable_x64", True)


"""
Implementation of the OmegaGW class using jax and jit.
"""


@jit
def norm():
    """
    Prefactor in front of the integral for the Tensor power spectrum.
    Is a constant assuming radiation domination.
    """
    OMEGA_R = 4.2 * 10 ** (-5)
    CG = 0.39
    return CG / 12 * OMEGA_R


@jit
def u(t, s):
    """
    Helper function to translate between u, v and s, t.

    Parameters:
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.

    Returns:
    - jax.numpy.ndarray
        Array of u values.
    """
    return (t + s + 1) / 2


@jit
def v(t, s):
    """
    Helper function to translate between u, v and s, t.

    Parameters:
    - t: jax.numpy.ndarray
        Array of t values.
    - s: jax.numpy.ndarray
        Array of s values.

    Returns:
    - jax.numpy.ndarray
        Array of v values.
    """
    return (t - s + 1) / 2


@jit
def polynomial(t, s):
    """
    Polynomial term in the integrand for the Tensor power spectrum.
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
    return 2 * ((t * (2 + t) * (s**2 - 1)) / ((1 - s + t) * (1 + s + t))) ** 2


# Radiation domination all the way
@jit
def I_sq_RD(t, s, k):
    r"""
    :math:`overline{I^2_{RD}(t, s, x\\to\\infty)}` assuming radiation
    domination. Note that this term is k-independent.

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
    prefactor = (
        288
        * (-5 + s**2 + t * (2 + t)) ** 2
        / ((1 - s + t) ** 6 * (1 + s + t) ** 6)
    )
    log_term = (
        (-1 + s - t) * (1 + s + t)
        + (
            (-5 + s**2 + t * (2 + t))
            * jnp.log(jnp.abs((-2 + t * (2 + t)) / (3 - s**2)))
        )
        / 2.0
    ) ** 2
    heaviside_term = (
        jnp.pi**2
        * (-5 + s**2 + t * (2 + t)) ** 2
        * jnp.heaviside(1 - jnp.sqrt(3) + t, 1)
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


# We precompute the part of the Large V contribution to the early matter
# domination kernel that contains the Si and Ci trigonometric integrals as
# there is no jit-able implementation of these functions in jax.
# These functions should be computed in a large enough range for all reasonable
# applications but will return jnp.nan for values outside the range.

xvals = jnp.geomspace(1e-5, 1e5, 10000000)
si, ci = sici(xvals / 2.0)
cos = jnp.cos(xvals / 2.0)
sin = jnp.sin(xvals / 2.0)
LV = jnp.log(4.0 * ci**2.0 + (jnp.pi - 2.0 * si) ** 2)


@jit
def _sici_precomp(x):
    """
    Precomputed term containing Si and Ci functions in the Large V
    contribution to the transitioning kernel.

    Parameters:
    - x: jax.numpy.ndarray
        Array of x values.

    Returns:
    - jax.numpy.ndarray
        Array of precomputed values.
    """
    return jnp.exp(jnp.interp(x, xvals, LV, left=jnp.nan, right=jnp.nan))


d_LV = 1 / xvals * (8.0 * cos * ci - 4 * sin * (jnp.pi - 2 * si))


@jit
def _d_sici_precomp(x):
    """
    Derivative of the precomputed term containing Si and Ci functions
    in the Large V contribution to the transitioning kernel.

    Parameters:
    - x: jax.numpy.ndarray
        Array of x values.

    Returns:
    - jax.numpy.ndarray
        Array of derivative values.
    """
    res = jnp.interp(x, xvals, d_LV, left=jnp.nan, right=jnp.nan)
    return res  # jnp.interp(x, xvals, d_LV, left=jnp.nan, right=jnp.nan)


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


@jit
def simpson_uniform(f, x):
    """
    Fully vectorized implementation of Simpson's rule for a uniform grid.
    If f is a multi-dimensional array, the integration is performed over the
    first axis.

    Parameters:
    - f: jax.numpy.ndarray, shape (N, ...)
        Array of function values. Integration is performed over the first axis.
    - x: jax.numpy.ndarray, shape (N,)
        Array of grid points.

    Returns:
    - result: jax.numpy.ndarray
        Array of integrated values.
    """
    N = f.shape[0] - 1
    h = x[1] - x[0]

    # Main computation for Simpson's rule
    result = (
        f[0]
        + f[-1]
        + 2 * jnp.sum(f[2:N:2], axis=0)
        + 4 * jnp.sum(f[1:N:2], axis=0)
    )

    # Multiply by h/3
    result *= h / 3

    # Adjust if N is odd (last segment)
    if N % 2 == 0:
        result -= (
            (f[-2] + f[-1]) * h / 3
        )  # Subtract last interval computed in the main sum
        result += (h / 2) * (
            (2 * f[-1] + f[-2]) + (f[-1] + f[-2])
        )  # Trapezoidal rule for the last interval

    return result


@jit
def simpson_nonuniform(f, x):
    """
    Numerical integration using Simpson's rule on a non-uniform grid.

    This fully vectorized implementation is suitable for multi-dimensional
    arrays, integrating over the first axis. Assumes non-uniformly spaced grid
    points.

    Parameters:
    - f (jax.numpy.ndarray): Array of function values, shape (N, ...).
    - x (jax.numpy.ndarray): Array of grid points, shape (N,) or same as 'f'.

    Returns:
    - jax.numpy.ndarray: Integrated values.
    """

    N = len(x) - 1
    h = jnp.diff(x, axis=0)  # Differences between consecutive x values
    f_shape = f.shape

    # Adjusting shape for broadcasting if necessary
    if x.shape != f_shape:
        broadcast_shape = (-1,) + (1,) * (len(f_shape) - 1)
        h0 = h[:-1:2].reshape(broadcast_shape)
        h1 = h[1::2].reshape(broadcast_shape)
    else:
        h0 = h[:-1:2]
        h1 = h[1::2]

    hph = h1 + h0
    hdh = h1 / h0
    hmh = h1 * h0
    result = jnp.sum(
        (hph / 6)
        * (
            (2 - hdh) * f[:-2:2]
            + (hph**2 / hmh) * f[1:-1:2]
            + (2 - 1 / hdh) * f[2::2]
        ),
        axis=0,
    )

    # Additional computation for even N (last segment)
    if N % 2 == 1:
        h0 = h[-2]
        h1 = h[-1]
        result += f[-1] * (2 * h1**2 + 3 * h0 * h1) / (6 * (h0 + h1))
        result += f[-2] * (h1**2 + 3 * h1 * h0) / (6 * h0)
        result -= f[-3] * h1**3 / (6 * h0 * (h0 + h1))

    return result


def integrand(P_zeta, kernel, s, t, k, *params):
    """
    Integrand for the non-transitioning kernels.

    Parameters:
    - P_zeta: callable
        The primordial power spectrum function.
    - kernel: callable
        The kernel function.
    - s: jax.numpy.ndarray
        Array of s values.
    - t: jax.numpy.ndarray
        Array of t values.
    - k: jax.numpy.ndarray
        Array of k values.
    - params: tuple
        Tuple of parameters for the P_zeta function.
    """
    return (
        kernel(t, s, k)
        * polynomial(t, s)
        * P_zeta(k * u(t, s), *params)
        * P_zeta(k * v(t, s), *params)
    )


jitted_integrand = jit(integrand, static_argnums=[0, 1])


def integrand_transitioning_kernel(P_zeta, kernel, s, t, k, *params):
    """
    Integrand for the transitioning kernels. They pass the two extra arguments
    kmax and etaR to the kernels.

    Parameters:
    - P_zeta: callable
        The primordial power spectrum function.
    - kernel: callable
        The kernel function.
    - s: jax.numpy.ndarray
        Array of s values.
    - t: jax.numpy.ndarray
        Array of t values.

    Returns:
    - jax.numpy.ndarray
        Array of integrand values.
    """
    kmax = params[-2]
    etaR = params[-1]
    return (
        kernel(t, s, k, kmax, etaR)
        * polynomial(t, s)
        * P_zeta(k * u(t, s), *params)
        * P_zeta(k * v(t, s), *params)
    )


jitted_integrand_transitioning_kernel = jit(
    integrand_transitioning_kernel, static_argnums=[0, 1]
)


def d_integrand(index, P_zeta, dP_zeta, kernel, s, t, k, *params):
    """
    Derivative of the integrand with respect to a parameter for the
    non-transitioning kernels.

    Parameters:
    - index: int
        Index of the parameter to differentiate with respect to.
    - P_zeta: callable
        The primordial power spectrum function.
    - dP_zeta: callable
        Gradient of the primordial power spectrum function with
        respect to the parameters.
    - kernel: callable
        The kernel function.
    - s: jax.numpy.ndarray
        Array of s values.
    - t: jax.numpy.ndarray
        Array of t values.
    - k: jax.numpy.ndarray
        Array of k values.
    - params: tuple
        Tuple of parameters for the P_zeta function.

    Returns:
    - jax.numpy.ndarray
        Array of integrand values.
    """
    return (
        kernel(t, s, k)
        * polynomial(t, s)
        * (
            P_zeta(k * u(t, s), *params)
            * dP_zeta(P_zeta, index, k * v(t, s), *params)
            + dP_zeta(P_zeta, index, k * u(t, s), *params)
            * P_zeta(k * v(t, s), *params)
        )
    )


def d_integrand_transitioning_kernel(
    index, P_zeta, dP_zeta, kernel, d_kernel, s, t, k, *params
):
    """
    Derivative of the integrand with respect to a parameter for the
    transitioning kernels.

    Parameters:
    - index: int
        Index of the parameter to differentiate with respect to.
    - P_zeta: callable
        The primordial power spectrum function.
    - dP_zeta: callable
        Gradient of the primordial power spectrum function with
        respect to the parameters.
    - kernel: callable
        The kernel function.
    - s: jax.numpy.ndarray
        Array of s values.
    - t: jax.numpy.ndarray
        Array of t values.
    - k: jax.numpy.ndarray
        Array of k values.
    - params: tuple
        Tuple of parameters for the P_zeta function.

    Returns:
    - jax.numpy.ndarray
        Array of integrand values.
    """
    kmax = params[-2]
    etaR = params[-1]
    # Standard derivative except for etaR
    term1 = (
        kernel(t, s, k, kmax, etaR)
        * polynomial(t, s)
        * (
            P_zeta(k * u(t, s), *params)
            * dP_zeta(P_zeta, index, k * v(t, s), *params)
            + dP_zeta(P_zeta, index, k * u(t, s), *params)
            * P_zeta(k * v(t, s), *params)
        )
    )
    if index != len(params) - 1:
        return term1
    else:
        # Second term containing the derivatives of the kernel
        term2 = (
            d_kernel(1, t, s, k, kmax, etaR)
            * polynomial(t, s)
            * P_zeta(k * u(t, s), *params)
            * P_zeta(k * v(t, s), *params)
        )
        return term1 + term2


def d_integrand_transitioning_kernel_delta(
    index, P_zeta, dP_zeta, kernel, d_kernel, s, t, k, *params
):
    """
    Derivative of the integrand with respect to a parameter for the
    transitioning kernels.

    Parameters:
    - index: int
        Index of the parameter to differentiate with respect to.
    - P_zeta: callable
        The primordial power spectrum function.
    - dP_zeta: callable
        Gradient of the primordial power spectrum function with
        respect to the parameters.
    - kernel: callable
        The kernel function.
    - s: jax.numpy.ndarray
        Array of s values.
    - t: jax.numpy.ndarray
        Array of t values.
    - k: jax.numpy.ndarray
        Array of k values.
    - params: tuple
        Tuple of parameters for the P_zeta function.

    Returns:
    - jax.numpy.ndarray
        Array of integrand values.
    """
    kmax = params[-2]
    etaR = params[-1]
    # First term containing the derivatives of P_zeta
    if index == len(params) - 2:
        term1 = (
            kernel(t, s, k, kmax, etaR)
            * polynomial(t, s)
            * dP_zeta(P_zeta, index, k * v(t, s), *params) ** 2.0
        )
    else:
        term1 = (
            kernel(t, s, k, kmax, etaR)
            * polynomial(t, s)
            * (
                P_zeta(k * u(t, s), *params)
                * dP_zeta(P_zeta, index, k * v(t, s), *params)
                + dP_zeta(P_zeta, index, k * u(t, s), *params)
                * P_zeta(k * v(t, s), *params)
            )
        )
    if index != len(params) - 1:
        return term1
    else:
        # Second term containing the derivatives of the kernel
        term2 = (
            d_kernel(1, t, s, k, kmax, etaR)
            * polynomial(t, s)
            * P_zeta(k * u(t, s), *params)
            * P_zeta(k * v(t, s), *params)
        )
        return term1 + term2


def dP_zeta_auto(P_zeta, index, k, *params):
    """
    Automatic gradient of the primordial power spectrum function. This only
    works for jax functions.
    """
    args = (k, *params)
    tangent_vector = tuple(
        jnp.zeros_like(arg) if i != index + 1 else jnp.ones_like(arg)
        for i, arg in enumerate(args)
    )
    _, deriv = jvp(lambda *args: P_zeta(*args), args, tangent_vector)
    return deriv


class OmegaGWjax:
    """
    OmegaGW implementation with a fully vectorized jax implementation of
    Simpson's rule for non-uniform grids.

    Parameters:
    - P_zeta: function
        The primordial power spectrum function. This function should take k as
        the first argument, followed by the parameters it depends on. For
        optimal performance, this function should be jitted.
    - k: array-like, shape (nk,)
        The k values at which to compute the integral. This should map the
        region of for :math:`Omega_{GW}` (i.e. the sensitivity of the
        experiment). If you need a higher resolution you can use the upsample
        parameter to interpolate the k values.
    - s: array-like, shape (ns,) or function
        The s values to use in the calculation. If s is a function, it should
        take k as the first argument followed the values of the parameters of
        the P_zeta function in the same order. It should return an array-like
        of s values.
        If s is an array-like, it is assumed to stay constant for all k.
    - t: array-like, shape (nt,) or function
        The t values to use in the calculation. If t is a function, it should
        take k as the first argument followed the values of the parameters of
        the P_zeta function in the same order. It should return an array-like
        of t values.
        If t is an array-like, it is assumed to stay constant for all k.
    - Norm: callable or str, default="RD"
        The normalization function for the primordial power spectrum. If a
        callable is passed, it should take k as the first argument and return
        the normalization factor. If "RD" is passed, the normalization factor
        will be calculated using the radiation domination approximation.
    - upsample: array-like, shape (nk,), default=None
        If this parameter is not None, it should be an array-like of k values
        at which to interpolate :math:`Omega_{GW}`. This is useful if you need
        a higher resolution in the k values.
    - dP_zeta: callable or str, default="auto"
        The derivative of the primordial power spectrum function. If a callable
        is passed, it should take the primordial power spectrum function, the
        index of the parameter to differentiate with respect to, k and the
        parameters of the primordial power spectrum function. If "auto" is
        passed, the derivative will be calculated automatically using jax.
    - to_numpy: bool, default=False
        If True, the output will be converted to a numpy array. Otherwise the
        native jax array will be returned.


        .. note::
            In dblquad integration the spacing of the grid in s and t is
            handled by scipy. Thus the grid in s and t will be ignored and
            only the end points will be used.
    """

    def __init__(
        self,
        P_zeta,
        s,
        t,
        f=None,
        norm="RD",
        kernel="RD",
        upsample=False,
        dP_zeta="auto",
        to_numpy=False,
        jit=True,
        dP_zeta_has_delta=False,
        **kwargs
    ):
        # Constants
        OMEGA_R = 4.2 * 10 ** (-5)  # times h^2, otherwise ~ 8* 10**(-5)
        CG = 0.39

        self.P_zeta = P_zeta
        if dP_zeta == "auto":
            self.dP_zeta = dP_zeta_auto
        elif callable(dP_zeta):
            self.dP_zeta = dP_zeta
        else:
            self.dP_zeta = None
        self.dP_zeta_has_delta = dP_zeta_has_delta
        if callable(s):
            self.s = s
            self.fixed_s = False
        else:
            self.s = jnp.array(s)
            self.fixed_s = True
        if callable(t):
            self.t = t
            self.fixed_t = False
        else:
            self.t = jnp.array(t)
            self.fixed_t = True

        if norm == "RD":
            self.norm = lambda k: CG / 24.0 * OMEGA_R
        elif norm == "CT":
            self.norm = lambda k: 1.0 / 24.0
        elif callable(norm):
            self.norm = norm
        else:
            raise ValueError(
                "Norm should be a callable, 'RD' or 'CT'. Other "
                "norms are not supported right now."
            )

        if jit:
            integrands = [
                jitted_integrand,
                jitted_integrand_transitioning_kernel,
            ]
        else:
            integrands = [integrand, integrand_transitioning_kernel]

        d_integrands = [
            d_integrand,
            d_integrand_transitioning_kernel,
            d_integrand_transitioning_kernel_delta,
        ]

        if kernel == "RD":
            self.kernel = I_sq_RD
            self.integration_routine = self.integrate_constant_kernel
            self.d_integration_routine = self.d_integrate_constant_kernel
            self.integrand = integrands[0]
            self.d_integrand = d_integrands[0]
        elif kernel == "I_MD_to_RD":
            self.kernel = [I_sq_IRD_LV, I_sq_IRD_res]
            self.d_kernel = [d_I_sq_IRD_LV, d_I_sq_IRD_res]
            self.integration_routine = self.integrate_transitioning_kernel
            self.d_integration_routine = self.d_integrate_transitioning_kernel
            self.integrand = integrands[1]
            if dP_zeta_has_delta:
                self.d_integrand = d_integrands[2]
            else:
                self.d_integrand = d_integrands[1]
        elif callable(kernel):
            self.kernel = kernel
            self.integration_routine = self.integrate_constant_kernel
            self.d_integration_routine = self.d_integrate_constant_kernel
            self.integrand = integrands[0]
            self.d_integrand = d_integrands[0]
        elif hasattr(kernel, "__iter__"):
            if len(kernel) == 2:
                self.kernel = kernel
                self.integration_routine = self.integrate_transitioning_kernel
                self.integrand = integrands[1]
            else:
                raise ValueError(
                    "If kernel is an iterable, it should have "
                    "length 2. Got length {}".format(len(kernel))
                )
        else:
            raise ValueError(
                "Supported kernels are 'RD', 'I_MD_to_RD', a callable or an "
                "iterable containing exactly 2 kernel functions. Got {}".format(
                    kernel
                )
            )

        if f is None:  # If k is None, no upsampling
            self.f = f
            if upsample:
                raise ValueError("If upsample is True, f cannot be None")
        elif callable(f):  # If f is a function, evaluate it
            self.f = f
            if not upsample:
                Warning(
                    "Providing f and not upsampling will result in f being "
                    "ignored."
                )
        elif isinstance(
            f, jnp.ndarray
        ):  # If f is an iterable, convert to array
            self.f = jnp.array(f)
            if not upsample:
                Warning(
                    "Providing f and not upsampling will result in f being "
                    "ignored."
                )
        else:
            raise ValueError(
                "f should be None, a callable or an iterable. Got {}".format(
                    type(f)
                )
            )
        self.upsample = upsample
        self.to_numpy = to_numpy

    def upsample_k(self, fvec_new, fvec, omega_gw):
        """
        Function to upsample the result to a higher resolution in k.

        Parameters:
        - fvec: jax.numpy.ndarray
            Array of frequency values to upsample to.
        - omega_gw: jax.numpy.ndarray
            Array of :math:`Omega_{GW}` values.

        Returns:
        - jax.numpy.ndarray
            Array of upsampled :math:`Omega_{GW}` values.
        """
        # Function to upsample the result to a higher resolution in k
        if self.upsample:
            res = jnp.interp(fvec_new, fvec, omega_gw)
        else:
            res = omega_gw

        if not self.to_numpy:
            return res
        else:
            return np.array(res)

    def __call__(self, fvec, *params):
        """
        Compute the :math:`Omega_{GW}` values.

        Parameters:
        - fvec: jax.numpy.ndarray
            Array of frequency values to compute :math:`Omega_{GW}` at.
        - params: tuple
            Tuple of parameters for the P_zeta function.

        Returns:
        - jax.numpy.ndarray
            Array of :math:`Omega_{GW}` values.
        """

        # setting k for evaluation
        kvec_full_resolution = jnp.copy(jnp.array(fvec)) * 2 * jnp.pi
        if self.upsample:
            if callable(self.f):
                fvec = self.f(*params)
            else:
                fvec = self.f
            kvec = jnp.array(fvec) * 2 * jnp.pi
        else:
            kvec = kvec_full_resolution
        # Setting s
        if self.fixed_s:
            s = self.s
        else:
            s = self.s(kvec, *params)
        # Setting t
        if self.fixed_t:
            t = self.t
        else:
            t = self.t(kvec, *params)

        res = self.integration_routine(self.P_zeta, s, t, kvec, *params)

        out = (
            2
            * self.norm(kvec_full_resolution)
            * self.upsample_k(kvec_full_resolution, kvec, res)
        )

        return out

    def integrate_constant_kernel(self, P_zeta, s, t, kvec, *params):
        r"""
        Compute the integral for the non-transitioning kernels.

        Parameters:
        - s: jax.numpy.ndarray
            Array of s values. Needs to be one-dimensional.
        - t: jax.numpy.ndarray
            Array of t values. Needs to be two-dimensional with
            shape (Nt, len(kvec)).
        - kvec: jax.numpy.ndarray
            Array of k values. Needs to be one-dimensional.
        - params: tuple
            Tuple of parameters for the P_zeta function.

        Returns:
        - jax.numpy.ndarray
            Array of :math:`\overline{\mathcal{P}_h}` values.
        """
        integrand_values = self.integrand(
            P_zeta,
            self.kernel,
            s[:, None, None],
            t[None, :, :],
            kvec[None, None, :],
            *params
        )

        s_integrated = simpson_uniform(integrand_values, s)
        return simpson_nonuniform(s_integrated, t)

    def integrate_transitioning_kernel(self, P_zeta, s, t, kvec, *params):
        r"""
        Compute the integral for the transitioning kernels.

        Parameters:
        - s: jax.numpy.ndarray
            Array of s values. Needs to be one-dimensional.
        - t: jax.numpy.ndarray
            Array of t values. Needs to be two-dimensional with
            shape (Nt, len(kvec)).
        - kvec: jax.numpy.ndarray
            Array of k values. Needs to be one-dimensional.
        - params: tuple
            Tuple of parameters for the P_zeta function.

        Returns:
        - jax.numpy.ndarray
            Array of :math:`\overline{\mathcal{P}_h}` values.
        """
        lV_values = self.integrand(
            P_zeta,
            self.kernel[0],
            s[:, None, None],
            t[None, :, :],
            kvec[None, None, :],
            *params
        )
        lV_integrated = simpson_uniform(lV_values, s)
        lV_integrated = simpson_nonuniform(lV_integrated, t)
        t_res = jnp.sqrt(3) - 1.0
        res_values = self.integrand(
            P_zeta, self.kernel[1], s[:, None], t_res, kvec[None, :], *params
        )
        res_integrated = simpson_uniform(res_values, s)

        return lV_integrated + res_integrated

    def d_integrate(self, index, fvec, *params):
        # setting k for evaluation
        kvec_full_resolution = jnp.copy(jnp.array(fvec)) * 2 * jnp.pi
        if self.upsample:
            if callable(self.f):
                fvec = self.f(*params)
            else:
                fvec = self.f
            kvec = jnp.array(fvec) * 2 * jnp.pi
        else:
            kvec = kvec_full_resolution
        # Setting s
        if self.fixed_s:
            s = self.s
        else:
            s = self.s(kvec, *params)
        # Setting t
        if self.fixed_t:
            t = self.t
        else:
            t = self.t(kvec, *params)

        res = self.d_integration_routine(
            index, self.P_zeta, self.dP_zeta, s, t, kvec, *params
        )
        out = (
            2
            * self.norm(kvec_full_resolution)
            * self.upsample_k(kvec_full_resolution, kvec, res)
        )

        return out

    def d_integrate_constant_kernel(
        self, index, P_zeta, dP_zeta, s, t, kvec, *params
    ):
        """
        Compute the derivative of :math:`Omega_{GW}` values with respect
        to params[index].

        Parameters:
        - index: int
            Index of the parameter to differentiate with respect to.
        - fvec: jax.numpy.ndarray
            Array of frequency values to compute :math:`Omega_{GW}` at.
        - params: tuple
            Tuple of parameters for the P_zeta function.

        Returns:
        - jax.numpy.ndarray
            Array of :math:`Omega_{GW}` values.
        """

        integrand_values = self.d_integrand(
            index,
            P_zeta,
            dP_zeta,
            self.kernel,
            s[:, None, None],
            t[None, :, :],
            kvec[None, None, :],
            *params
        )

        s_integrated = simpson_uniform(integrand_values, s)
        return simpson_nonuniform(s_integrated, t)

    def d_integrate_transitioning_kernel(
        self, index, P_zeta, dP_zeta, s, t, kvec, *params
    ):
        """
        Compute the derivative of :math:`Omega_{GW}` values with respect
        to params[index].

        Parameters:
        - index: int
            Index of the parameter to differentiate with respect to.
        - fvec: jax.numpy.ndarray
            Array of frequency values to compute :math:`Omega_{GW}` at.
        - params: tuple
            Tuple of parameters for the P_zeta function.

        Returns:
        - jax.numpy.ndarray
            Array of :math:`Omega_{GW}` values.
        """

        kmax = params[-2]
        _ = params[-1]

        # First the large V contribution which is a bit more complicated

        # For kmax we need to fix t because of the theta functions in the
        # kernel and the derivative of  the resonant contribution is zero
        if index == len(params) - 2 and self.dP_zeta_has_delta:
            t = (-kvec[None, :] + 2 * kmax - kvec[None, :] * s[:, None]) / kvec[
                None, :
            ]
            lV_values = self.d_integrand(
                index,
                P_zeta,
                dP_zeta,
                self.kernel[0],
                self.d_kernel[0],
                s[:, None],
                t,
                kvec[None, :],
                *params
            )
            lV_values = lV_values * jnp.where(t >= 0, 1, 0)
            lV_integrated = simpson_uniform(lV_values, s)
            t = jnp.sqrt(3) - 1.0
            s = (-kvec + 2 * kmax - kvec * t) / kvec
            res_values = self.d_integrand(
                index,
                P_zeta,
                dP_zeta,
                self.kernel[1],
                self.d_kernel[1],
                s,
                t,
                kvec,
                *params
            )
            res_values = (
                res_values
                * jnp.where(2 * kmax - kvec * (1 + t) >= 0, 1, 0)
                * jnp.where((1 - s >= 0) & (s + 1 >= 0), 1, 0)
            )

            return 2 / kvec * (lV_integrated + res_values)

        # For all other derivatives we just proceed as usual
        else:
            lV_values = self.d_integrand(
                index,
                P_zeta,
                dP_zeta,
                self.kernel[0],
                self.d_kernel[0],
                s[:, None, None],
                t[None, :, :],
                kvec[None, None, :],
                *params
            )
            lV_integrated = simpson_uniform(lV_values, s)
            lV_integrated = simpson_nonuniform(lV_integrated, t)
            t_res = jnp.sqrt(3) - 1.0
            res_values = self.d_integrand(
                index,
                P_zeta,
                dP_zeta,
                self.kernel[1],
                self.d_kernel[1],
                s[:, None],
                t_res,
                kvec[None, :],
                *params
            )
            res_integrated = simpson_uniform(res_values, s)
            return lV_integrated + res_integrated

    def d_integrate_transitioning_kernel_analytical(
        self, index, P_zeta, s, t, fvec, *params
    ):

        kvec = jnp.array(fvec) * 2 * jnp.pi

        As = params[-3]
        kmax = params[-2]
        etaR = params[-1]

        if index == len(params) - 2:  # kmax
            t1 = (
                -kvec[None, :] + 2 * kmax - kvec[None, :] * s[:, None]
            ) / kvec[None, :]
            lV_values = I_sq_IRD_LV(
                t1, s[:, None], kvec[None, :], kmax, etaR
            ) * polynomial(t1, s[:, None])
            lV_values = lV_values * jnp.where(
                t1 >= 0, 1, 0
            )  # * jnp.where( kvec[None, :]*s[:, None] >= 0, 1, 0)
            lV_integrated = simpson_uniform(lV_values, s)
            lV_integrated = lV_integrated * As**2 * 2 * 2 / kvec

            t2 = jnp.sqrt(3) - 1.0
            s2 = (-kvec + 2 * kmax - kvec * t2) / kvec
            res_values = I_sq_IRD_res(t2, s2, kvec, kmax, etaR) * polynomial(
                t2, s2
            )
            res_values = (
                res_values
                * jnp.where(2 * kmax - kvec * (1 + t2) >= 0, 1, 0)
                * jnp.where((1 - s2 >= 0) & (s2 + 1 >= 0), 1, 0)
            )
            res_values = res_values * As**2 * 2 * 2 / kvec
            res = lV_integrated + res_values

            return self.norm(kvec) * res

    # UNUSED
    def integrate_single_k(self, f, *params):
        k = jnp.array(f) * 2 * jnp.pi

        if self.fixed_s:
            s = self.s
        else:
            s = self.s(self.k, *params)
        if self.fixed_t:
            t = self.t
        else:
            t = self.t(k, *params)

        # TOOD: Need to change this to allow for different spacings in s and t
        integrand_values = integrand(
            self.P_zeta,
            self.kernel,
            s[:, None, None],
            t[None, :, None],
            k,
            *params
        )

        s_integrated = simpson_uniform(integrand_values, s)
        res = simpson_nonuniform(s_integrated, t)

        out = 2 * self.norm(k) * res

        return jnp.squeeze(out)
