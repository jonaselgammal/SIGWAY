"""
Numerical and unit-conversion helpers used across SIGWAY.

Provides conversions between the comoving wavenumber $k$, the number of
inflationary e-folds $N$, and the Hubble rate $H$, together with composite
Simpson quadrature routines for uniform and non-uniform grids.
"""

__all__ = [
    "wavenumber_from_efolds_si_units",
    "efolds_from_wavenumber_si_units",
    "H_from_wavenumber",
    "simpson_uniform",
    "simpson_uniform_even",
    "simpson_uniform_odd",
    "simpson_nonuniform",
    "simpson_nonuniform_even",
    "simpson_nonuniform_odd",
    "do_broadcasting",
    "no_broadcasting",
    "interpolate_spectrum",
]

# Global
import jax
from jax import jit
from jax import numpy as jnp

# Local
from sigway.constants import CMB_scale_k

# enable 64-bit precision for JAX computations
jax.config.update("jax_enable_x64", True)


# Utils to convert between e-folds, wavenumber, and Hubble parameter
@jit
def wavenumber_from_efolds_si_units(N, H, N_CMB, H_CMB):
    r"""
    Convert inflationary e-folds $N$ and Hubble rate $H$ to comoving
    wavenumber $k$.

    Uses the horizon-crossing relation
    $k = k_\mathrm{CMB}\,(H/H_\mathrm{CMB})\,e^{N - N_\mathrm{CMB}}$,
    where $k_\mathrm{CMB} = 0.05\,\mathrm{Mpc}^{-1}$ converted to SI units
    ($\mathrm{s}^{-1}$).

    Parameters
    ----------
    N : array_like
        Inflationary e-fold values $N$.
    H : array_like
        Hubble parameter values in s^-1 evaluated at the corresponding $N$.
    N_CMB : float
        E-fold value $N_\mathrm{CMB}$ at which the CMB pivot scale crosses
        the horizon.
    H_CMB : float
        Hubble parameter $H_\mathrm{CMB}$ in s^-1 at the CMB pivot scale.

    Returns
    -------
    k_N : jnp.ndarray
        Comoving wavenumber $k$ in s^-1, with the same shape as `N` and `H`.
    """
    # Calculate the wavenumber k for each N and H
    k_N = CMB_scale_k * (H / H_CMB) * jnp.exp(N - N_CMB)

    return k_N


@jit
def efolds_from_wavenumber_si_units(k, H, N_CMB, H_CMB):
    r"""
    Convert comoving wavenumber $k$ and Hubble rate $H$ to inflationary
    e-folds $N$.

    Inverts the horizon-crossing relation used in
    `wavenumber_from_efolds_si_units`:
    $N = N_\mathrm{CMB} +
    \ln\!\left[k\,/\,(k_\mathrm{CMB}\,H/H_\mathrm{CMB})\right]$.

    Parameters
    ----------
    k : array_like
        Comoving wavenumber values $k$ in s^-1.
    H : array_like
        Hubble parameter values in s^-1 evaluated at the corresponding $k$.
    N_CMB : float
        E-fold value $N_\mathrm{CMB}$ at which the CMB pivot scale crosses
        the horizon.
    H_CMB : float
        Hubble parameter $H_\mathrm{CMB}$ in s^-1 at the CMB pivot scale.

    Returns
    -------
    N : jnp.ndarray
        Number of e-folds $N$, with the same shape as `k` and `H`.
    """
    # Calculate the number of e-folds N for each k
    N = N_CMB + jnp.log(k / (CMB_scale_k * (H / H_CMB)))

    return N


@jit
def H_from_wavenumber(k, N, H, N_CMB, H_CMB):
    r"""
    Interpolate the Hubble rate $H$ as a function of comoving wavenumber $k$.

    First maps the inflationary trajectory $(N, H)$ to a wavenumber array via
    `wavenumber_from_efolds_si_units`, sorts it in ascending order, then
    linearly interpolates $H$ at the requested $k$ values.

    Parameters
    ----------
    k : array_like
        Comoving wavenumber values $k$ in s^-1 at which $H$ is evaluated.
    N : array_like
        Inflationary e-fold array describing the background trajectory.
    H : array_like
        Hubble parameter values in s^-1 corresponding to `N`.
    N_CMB : float
        E-fold value $N_\mathrm{CMB}$ at which the CMB pivot scale crosses
        the horizon.
    H_CMB : float
        Hubble parameter $H_\mathrm{CMB}$ in s^-1 at the CMB pivot scale.

    Returns
    -------
    H_k : jnp.ndarray
        Hubble parameter $H(k)$ in s^-1, with the same shape as `k`.
    """

    # Calculate the wavenumber k as a function of N and H
    k_N = wavenumber_from_efolds_si_units(N, H, N_CMB, H_CMB)

    # Sort the wavenumber k and Hubble parameter H values
    sorted_indices = jnp.argsort(k_N)
    k_N_sorted = k_N[sorted_indices]
    H_N_sorted = H[sorted_indices]

    # Interpolation using JAX's interp
    H_k = jnp.interp(k, k_N_sorted, H_N_sorted)

    return H_k


# Util for simpson integration
@jax.jit
def simpson_uniform_even(f, h):
    r"""
    Composite Simpson 1/3 rule for an even number of intervals.

    Applies the standard composite Simpson formula
    $\int f\,\mathrm{d}x \approx
    \tfrac{h}{3}(f_0 + 4f_1 + 2f_2 + \cdots + 4f_{N-1} + f_N)$
    assuming $N$ (number of intervals) is even.  Called internally by
    `simpson_uniform`; use that function directly unless you need fine-grained
    control.

    Parameters
    ----------
    f : jnp.ndarray, shape (N+1, ...)
        Function values on a uniform grid.  Integration is over the first axis.
    h : jnp.ndarray, shape (1,)
        Uniform step size $h = x_1 - x_0$.

    Returns
    -------
    result : jnp.ndarray, shape (...)
        Approximate integral $\int f\,\mathrm{d}x$ over the remaining axes.
    """

    # Number of intervals
    N = f.shape[0] - 1

    return (
        f[0]
        + 4 * jnp.sum(f[1:N:2], axis=0)
        + 2 * jnp.sum(f[2 : N - 1 : 2], axis=0)
        + f[N]
    ) * (h / 3)


# Util for simpson integration
@jax.jit
def simpson_uniform_odd(f, h):
    r"""
    Composite Simpson 1/3 rule for an odd number of intervals.

    Handles the case where the total number of intervals $N$ is odd by
    treating the last (unpaired) interval with a higher-order correction
    formula and applying the standard composite Simpson rule to the remaining
    even number of intervals.  Called internally by `simpson_uniform`.

    Parameters
    ----------
    f : jnp.ndarray, shape (N+1, ...)
        Function values on a uniform grid.  Integration is over the first axis.
    h : jnp.ndarray, shape (1,)
        Uniform step size $h = x_1 - x_0$.

    Returns
    -------
    result : jnp.ndarray, shape (...)
        Approximate integral $\int f\,\mathrm{d}x$ over the remaining axes.
    """

    # factor to adjust the last interval
    result = h * (f[-1] * 5 / 12 + f[-2] * 2 / 3 - f[-3] * 1 / 12)

    # Number of intervals (reduced by one to make it even)
    N = f.shape[0] - 2

    return result + (
        f[0]
        + 4 * jnp.sum(f[1:N:2], axis=0)
        + 2 * jnp.sum(f[2 : N - 1 : 2], axis=0)
        + f[N]
    ) * (h / 3)


simpson_uniform_fun_list = [simpson_uniform_even, simpson_uniform_odd]


@jit
def simpson_uniform(f, x):
    r"""
    Composite Simpson quadrature on a uniform grid.

    Implements the composite Simpson 1/3 rule (see
    https://en.wikipedia.org/wiki/Simpson%27s_rule).  Integration is
    always performed over the **first axis** of `f`; any remaining axes are
    treated as independent integrals and are preserved in the output shape.

    When the number of intervals $N = \mathrm{len}(x) - 1$ is odd (i.e.
    `len(x)` is even), the last interval is handled with the higher-order
    correction from the irregularly-spaced variant so that the overall
    method remains fourth-order accurate.  The grid is assumed uniform;
    no uniformity check is performed for performance reasons.

    Parameters
    ----------
    f : jnp.ndarray, shape (N+1, ...)
        Function values at the grid points.  The first axis must match
        ``len(x)``.
    x : jnp.ndarray, shape (N+1,)
        Uniformly spaced grid points.  Only the spacing
        $h = x_1 - x_0$ is used internally.

    Returns
    -------
    result : jnp.ndarray, shape (...)
        Approximate integral $\int f(x)\,\mathrm{d}x$.  A scalar is
        returned when `f` is one-dimensional.

    Examples
    --------
    Integrate $\sin(x)$ from $0$ to $\pi$ (exact value: $2$):

    >>> import jax.numpy as jnp
    >>> from sigway.utils import simpson_uniform
    >>> x = jnp.linspace(0, jnp.pi, 101)
    >>> f = jnp.sin(x)
    >>> float(simpson_uniform(f, x))
    2.0000000108...
    """
    # Number of intervals
    N = f.shape[0] - 1

    # interval spacing
    h = jnp.array([x[1] - x[0]])

    # choose the different branch
    result = jax.lax.switch((N % 2), simpson_uniform_fun_list, f, h)

    return result


def no_broadcasting(f, h):
    r"""
    Return step-size pairs for non-uniform Simpson without broadcasting.

    Extracts alternating consecutive step-size pairs $(h_{2i}, h_{2i+1})$
    from `h` without reshaping, for use when `f` and `x` have the same
    shape (i.e. `x` is already broadcast-compatible with `f`).

    Parameters
    ----------
    f : jnp.ndarray
        Function values array (used only to determine dimensionality;
        not modified).
    h : jnp.ndarray, shape (N,)
        Array of consecutive step sizes $h_i = x_{i+1} - x_i$.

    Returns
    -------
    h0 : jnp.ndarray
        Even-indexed step sizes $h_{0}, h_{2}, h_{4}, \ldots$
    h1 : jnp.ndarray
        Odd-indexed step sizes $h_{1}, h_{3}, h_{5}, \ldots$
    """
    step = 2

    h0 = h[:-1:step]
    h1 = h[1::step]

    return h0, h1


def do_broadcasting(f, h):
    """
    Return step-size pairs for non-uniform Simpson with broadcasting.

    Extracts alternating consecutive step-size pairs $(h_{2i}, h_{2i+1})$
    from `h` and reshapes them so they broadcast over all trailing axes of
    `f`.  Used when `x` is one-dimensional but `f` is multi-dimensional.

    Parameters
    ----------
    f : jnp.ndarray, shape (N+1, ...)
        Function values array.  Its number of dimensions determines the
        reshape target.
    h : jnp.ndarray, shape (N,)
        Array of consecutive step sizes $h_i = x_{i+1} - x_i$.

    Returns
    -------
    h0 : jnp.ndarray, shape (N//2, 1, 1, ...)
        Even-indexed step sizes reshaped for broadcasting against `f`.
    h1 : jnp.ndarray, shape (N//2, 1, 1, ...)
        Odd-indexed step sizes reshaped for broadcasting against `f`.
    """
    step = 2

    broadcast_shape = (-1,) + (1,) * (len(f.shape) - 1)

    h0 = h[:-1:step].reshape(broadcast_shape)
    h1 = h[1::step].reshape(broadcast_shape)

    return h0, h1


broadcasting_list = [no_broadcasting, do_broadcasting]


def simpson_nonuniform_even(f, h, result):
    """
    Pass-through for non-uniform Simpson when the interval count is even.

    When the number of intervals $N$ is even, all intervals are already
    covered by the main vectorised sum in `simpson_nonuniform` and no
    correction is needed.  This function simply returns `result` unchanged,
    serving as the even-branch of the ``jax.lax.switch`` inside
    `simpson_nonuniform`.

    Parameters
    ----------
    f : jnp.ndarray
        Function values (unused; present for a uniform call signature).
    h : jnp.ndarray
        Step-size array (unused; present for a uniform call signature).
    result : jnp.ndarray
        Partial integral accumulated by `simpson_nonuniform`.

    Returns
    -------
    result : jnp.ndarray
        The input `result` unchanged.
    """
    return result


def simpson_nonuniform_odd(f, h, result):
    r"""
    Correction term for non-uniform Simpson when the interval count is odd.

    Adds the contribution of the last unpaired interval using the
    irregularly-spaced three-point correction formula:

    $\Delta I = f_{N}\,\frac{2h_1^2 + 3h_0 h_1}{6(h_0+h_1)}
              + f_{N-1}\,\frac{h_1^2 + 3h_1 h_0}{6h_0}
              - f_{N-2}\,\frac{h_1^3}{6h_0(h_0+h_1)}$

    where $h_0 = x_{N-1}-x_{N-2}$ and $h_1 = x_N - x_{N-1}$.  Called as
    the odd-branch of the ``jax.lax.switch`` inside `simpson_nonuniform`.

    Parameters
    ----------
    f : jnp.ndarray, shape (N+1, ...)
        Function values on the non-uniform grid.
    h : jnp.ndarray, shape (N,)
        Consecutive step sizes $h_i = x_{i+1} - x_i$.
    result : jnp.ndarray, shape (...)
        Partial integral from `simpson_nonuniform` covering all but the
        last interval.

    Returns
    -------
    result : jnp.ndarray, shape (...)
        Corrected integral including the final interval.
    """
    h0 = h[-2]
    h1 = h[-1]
    result += f[-1] * (2 * h1**2 + 3 * h0 * h1) / (6 * (h0 + h1))
    result += f[-2] * (h1**2 + 3 * h1 * h0) / (6 * h0)
    result -= f[-3] * h1**3 / (6 * h0 * (h0 + h1))

    return result


simpson_nonuniform_list = [simpson_nonuniform_even, simpson_nonuniform_odd]


@jit
def simpson_nonuniform(f, x):
    r"""
    Composite Simpson quadrature on a non-uniform grid.

    Implements the composite irregularly-spaced Simpson 1/3 rule (see
    https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data).
    Integration is always performed over the **first axis** of `f`; any
    remaining axes are treated as independent integrals and are preserved
    in the output.

    When the number of intervals $N = \mathrm{len}(x) - 1$ is odd, the
    final unpaired interval is handled by `simpson_nonuniform_odd` using a
    three-point correction that maintains fourth-order accuracy.

    Parameters
    ----------
    f : jnp.ndarray, shape (N+1, ...)
        Function values at the grid points.  The first axis must match
        ``len(x)``.
    x : jnp.ndarray, shape (N+1,) or shape (N+1, ...)
        Grid point coordinates.  When `x` has the same shape as `f`, step
        sizes are used without reshaping; when `x` is one-dimensional,
        step sizes are broadcast over the trailing axes of `f`.

    Returns
    -------
    result : jnp.ndarray, shape (...)
        Approximate integral $\int f(x)\,\mathrm{d}x$.  A scalar is
        returned when `f` is one-dimensional.
    """

    # Number of subintervals
    N = len(x) - 1

    # Differences between consecutive x values
    h = jnp.diff(x, axis=0)

    # get the shape of f
    f_shape = f.shape

    # Step size in the sum over i
    step = 2

    # Adjusting shape for broadcasting if necessary.
    # x.shape and f_shape are static under jit, so this Python conditional is
    # jit-safe. jax.lax.switch must NOT be used here: its two branches return
    # arrays of different shapes, which lax.switch forbids (it traces every
    # branch and requires matching output types). That broke every N-D call
    # into the integrator; see no_broadcasting / do_broadcasting below.
    if x.shape != f_shape:
        h0, h1 = do_broadcasting(f, h)
    else:
        h0, h1 = no_broadcasting(f, h)

    hph = h1 + h0
    hdh = h1 / h0
    hmh = h1 * h0

    # get the result
    result = jnp.sum(
        (hph / 6)
        * (
            (2 - hdh) * f[:-2:step]
            + (hph**2 / hmh) * f[1:-1:step]
            + (2 - 1 / hdh) * f[2::step]
        ),
        axis=0,
    )

    # # Additional computation for even N (last segment) if necessary
    return jax.lax.switch((N % 2), simpson_nonuniform_list, f, h, result)


def interpolate_spectrum(x_out, x_in, y_in, mode="linear"):
    r"""Resample a spectrum ``y_in`` from ``x_in`` onto ``x_out``.

    Used by [OmegaGW][sigway.spectrum.OmegaGW] to evaluate the expensive
    integrand on a coarse internal grid and interpolate it onto the requested
    frequencies (the ``interp_grid`` feature).

    Parameters
    ----------
    x_out : jax.Array
        Query points to interpolate onto (e.g. the call-time wavenumbers).
    x_in : jax.Array
        Monotonically increasing sample points ``y_in`` is defined at.
    y_in : jax.Array
        Spectrum values sampled at ``x_in``.
    mode : {"linear", "loglog"}, optional
        ``"linear"`` (default) interpolates ``y_in`` linearly in ``x``.
        ``"loglog"`` interpolates ``log(y_in)`` linearly in ``log(x)``, which
        is exact for a power law.  Non-positive ``y_in`` entries (a sharp
        cutoff, an underflowed tail, or round-off noise) are floored to the
        smallest positive float before the log, so ``"loglog"`` never returns
        ``nan`` / ``-inf``; it stays traceable (no value branching) for use
        under ``jax`` transformations such as ``jacfwd``.

    Returns
    -------
    jax.Array
        The interpolated spectrum evaluated at ``x_out``, same shape as
        ``x_out``.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"linear"`` or ``"loglog"``.
    """
    if mode == "loglog":
        floor = jnp.finfo(y_in.dtype).tiny
        log_y = jnp.log(jnp.maximum(y_in, floor))
        return jnp.exp(jnp.interp(jnp.log(x_out), jnp.log(x_in), log_y))
    if mode == "linear":
        return jnp.interp(x_out, x_in, y_in)
    raise ValueError(
        "mode must be 'linear' or 'loglog', got {!r}.".format(mode)
    )
