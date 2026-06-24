"""Integration strategies that turn a Kernel + ScalarPerturbations into the
(un-normalised) tensor power spectrum integral.

The numerical method is a *strategy*: ``SimpsonIntegrator`` is the current
fixed-grid (s, t) Simpson quadrature; an ``FFTIntegrator`` (convolution-based,
for non-Gaussianities) is a future sibling. The Gaussian "two P_zeta factors"
assumption lives here, in SimpsonIntegrator, not in the base, so the FFT/NG
strategy can replace it without touching the kernels or perturbations.

Each integrand core is written once as ``*_impl`` and also exposed jit-ed (the
kernel/perturbation callables are static arguments). Jit-able perturbations
(analytic spectra) run through the jit-ed core -- it compiles once and re-runs
without retracing when only the parameter values change at fixed shapes. The MS
solver is not jit-able (scipy splines), so that path runs the plain ``*_impl``
eagerly.
"""

import jax
import jax.numpy as jnp
from jax import jit

from sigway.kernels import get_u, get_v, polynomial
from sigway.utils import simpson_uniform, simpson_nonuniform

jax.config.update("jax_enable_x64", True)

__all__ = ["Integrator", "SimpsonIntegrator"]


def _kint(kvec, u, v):
    """Wavenumber grid spanning the (k u, k v) range a P_zeta is sampled on.

    Only used by perturbations whose ``prepare`` interpolates (the MS solver);
    analytic spectra ignore it.
    """
    lo = jnp.minimum(u.min(), v.min()) * kvec.min()
    hi = jnp.maximum(u.max(), v.max()) * kvec.max()
    return jnp.geomspace(lo, hi, 100)


def _simpson_constant_impl(pzeta, kernel_fn, s, t, kvec, theta_pz, theta_k):
    """Double (s, t) Simpson integral for a smooth kernel. Returns (nk,)."""
    ss = s[:, None, None]
    tt = t[None, :, :]
    k = kvec[None, None, :]
    u = get_u(tt, ss)
    v = get_v(tt, ss)
    pz = pzeta.prepare(_kint(kvec, u, v), *theta_pz)
    integ = (
        kernel_fn(tt, ss, k, *theta_k)
        * polynomial(tt, ss)
        * pz(k * u)
        * pz(k * v)
    )
    return simpson_nonuniform(simpson_uniform(integ, s), t)


_simpson_constant = jit(_simpson_constant_impl, static_argnums=(0, 1))


def _simpson_transitioning_impl(
    pzeta, kern_smooth, kern_res, t_res, s, t, kvec, theta_pz, theta_k
):
    """Smooth (s, t) part plus a resonant 1-D (s) slice at fixed t = t_res."""
    ss = s[:, None, None]
    tt = t[None, :, :]
    k = kvec[None, None, :]
    u = get_u(tt, ss)
    v = get_v(tt, ss)
    pz = pzeta.prepare(_kint(kvec, u, v), *theta_pz)
    lv = (
        kern_smooth(tt, ss, k, *theta_k)
        * polynomial(tt, ss)
        * pz(k * u)
        * pz(k * v)
    )
    lv_int = simpson_nonuniform(simpson_uniform(lv, s), t)

    s2 = s[:, None]
    kr = kvec[None, :]
    ur = get_u(t_res, s2)
    vr = get_v(t_res, s2)
    rv = (
        kern_res(t_res, s2, kr, *theta_k)
        * polynomial(t_res, s2)
        * pz(kr * ur)
        * pz(kr * vr)
    )
    return lv_int + simpson_uniform(rv, s)


_simpson_transitioning = jit(
    _simpson_transitioning_impl, static_argnums=(0, 1, 2, 3)
)


class Integrator:
    """Abstract base class for SIGW emission integrators.

    An integrator encapsulates the numerical quadrature strategy that turns a
    [Kernel][sigway.kernels.Kernel] and a scalar-perturbation spectrum into the
    un-normalised tensor power spectrum integral evaluated on a wavenumber grid.

    Subclasses must override `integrate`.  The overall normalisation factor
    (``kernel.norm``), frequency-to-wavenumber conversion, and any upsampling
    are handled by [OmegaGW][sigway.spectrum.OmegaGW], not here.

    Methods
    -------
    integrate(kernel, pzeta, kvec, theta_pz, theta_k)
        Evaluate the un-normalised tensor power spectrum on ``kvec``.
        Must be implemented by every concrete subclass.
    """

    def integrate(self, kernel, pzeta, kvec, theta_pz, theta_k):
        raise NotImplementedError


class SimpsonIntegrator(Integrator):
    """Fixed-grid (s, t) Simpson quadrature integrator (Gaussian two-P_zeta form).

    Evaluates the scalar-induced GW tensor power spectrum integral using
    composite Simpson quadrature over the two rescaled internal momenta *s* and
    *t*.  The integration variable *s* is dimensionless and lies in [0, 1];
    *t* runs from 0 to infinity and parameterises the momentum transfer along
    the other axis.  The physical momenta (u, v) relate to (s, t) by

    .. code-block:: none

        u = (t + s + 1) / 2
        v = (t - s + 1) / 2

    The integrand at each grid point is

    .. code-block:: none

        kernel(t, s, k) * polynomial(t, s) * P_zeta(k*u) * P_zeta(k*v)

    where ``polynomial(t, s)`` is the geometric Jacobian factor from the
    (u, v) -> (s, t) change of variables (see [polynomial][sigway.kernels.polynomial]),
    and the two ``P_zeta`` factors embody the Gaussian/two-point assumption.

    If the kernel declares a ``resonant_t`` value, an additional 1-D Simpson
    integral over *s* at the fixed resonant *t* slice is added on top of the
    regular 2-D integral.

    JIT compilation: for perturbation spectra whose ``jittable`` attribute is
    ``True`` (analytic spectra), the quadrature core is JIT-compiled so that
    repeated calls with different parameter values but identical array shapes
    do not retrace.  Perturbations backed by the Mukhanov-Sasaki solver are not
    JIT-compilable and run the same core eagerly.

    Parameters
    ----------
    s : array_like or callable
        Quadrature nodes for the *s* momentum variable, lying in [0, 1].
        May be a fixed 1-D array or a callable ``s(kvec, *theta)`` that
        returns a 1-D array, where ``theta`` is the full concatenated
        parameter vector ``(*theta_pz, *theta_k)``.  A callable allows
        the *s* grid to adapt to the current wavenumber range or parameter
        values.
    t : array_like or callable
        Quadrature nodes for the *t* momentum variable, lying in [0, ∞).
        May be a fixed 1-D array, a fixed 2-D array of shape ``(nt, nk)``
        providing a separate node sequence for each wavenumber, or a callable
        ``t(kvec, *theta)`` returning either a 1-D or 2-D array.  When a
        1-D array (or callable returning one) is supplied it is broadcast to
        shape ``(nt, nk)`` internally so that the non-uniform Simpson rule
        over *t* can handle k-dependent node positions.

    Attributes
    ----------
    s : array_like or callable
        The *s*-grid as supplied at construction time.
    t : array_like or callable
        The *t*-grid as supplied at construction time.

    Methods
    -------
    integrate(kernel, pzeta, kvec, theta_pz, theta_k)
        Evaluate the un-normalised tensor power spectrum on ``kvec``.
    """

    def __init__(self, s, t):
        self.s = s
        self.t = t

    def _grids(self, kvec, theta):
        s = self.s(kvec, *theta) if callable(self.s) else self.s
        t = self.t(kvec, *theta) if callable(self.t) else self.t
        t = jnp.asarray(t)
        if t.ndim == 1:
            # a single t grid shared across all k -> broadcast to (nt, nk)
            t = jnp.broadcast_to(
                t[:, None], (t.shape[0], jnp.asarray(kvec).shape[0])
            )
        return jnp.asarray(s), t

    def integrate(self, kernel, pzeta, kvec, theta_pz, theta_k):
        """Evaluate the un-normalised tensor power spectrum on a wavenumber grid.

        Resolves the (s, t) grids (calling them if they are callables), selects
        the appropriate quadrature core based on the kernel's ``resonant_t``
        attribute and the perturbation spectrum's ``jittable`` flag, and returns
        the result of the double Simpson integral.

        Parameters
        ----------
        kernel : Kernel
            A [Kernel][sigway.kernels.Kernel] instance that provides
            ``overline_Isq``, optionally ``overline_Isq_resonant`` and
            ``resonant_t``.
        pzeta : ScalarPerturbations
            A scalar-perturbation spectrum object with a ``prepare`` method and
            a ``jittable`` attribute.  ``prepare`` is called with a helper
            wavenumber grid and ``theta_pz`` to return a callable
            ``P_zeta(k_array)``.
        kvec : array_like, shape (nk,)
            Wavenumber grid on which the integral is evaluated.  Units must be
            consistent with those used by ``kernel`` and ``pzeta``.
        theta_pz : tuple
            Parameter values forwarded to the perturbation spectrum.
        theta_k : tuple
            Parameter values forwarded to the kernel.

        Returns
        -------
        jax.Array, shape (nk,)
            Un-normalised tensor power spectrum values at each wavenumber in
            ``kvec``.  The caller ([OmegaGW][sigway.spectrum.OmegaGW]) applies
            the normalisation factor ``kernel.norm`` separately.
        """
        theta = (*theta_pz, *theta_k)
        s, t = self._grids(kvec, theta)
        jittable = getattr(pzeta, "jittable", True)
        if kernel.resonant_t:
            core = (
                _simpson_transitioning
                if jittable
                else _simpson_transitioning_impl
            )
            return core(
                pzeta,
                kernel.overline_Isq,
                kernel.overline_Isq_resonant,
                kernel.resonant_t[0],
                s,
                t,
                kvec,
                theta_pz,
                theta_k,
            )
        core = _simpson_constant if jittable else _simpson_constant_impl
        return core(pzeta, kernel.overline_Isq, s, t, kvec, theta_pz, theta_k)
