"""Integration strategies that turn a Kernel + ScalarPerturbations into the
(un-normalised) tensor power spectrum integral.

The numerical method is a *strategy*: ``SimpsonIntegrator`` is the current
fixed-grid (s, t) Simpson quadrature; an ``FFTIntegrator`` (convolution-based,
for non-Gaussianities) is a future sibling. The Gaussian "two P_zeta factors"
assumption lives here, in SimpsonIntegrator, not in the base, so the FFT/NG
strategy can replace it without touching the kernels or perturbations.

The numeric cores are module-level jit-ed functions (the kernel/perturbation
callables are static arguments), so the analytic path compiles once and re-runs
without retracing when only the parameter values change at fixed array shapes.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from sigway.kernels import get_u, get_v, polynomial
from sigway.utils import simpson_uniform, simpson_nonuniform

jax.config.update("jax_enable_x64", True)


@partial(jit, static_argnums=(0, 1))
def _simpson_constant(pzeta, kernel_fn, s, t, kvec, theta_pz, theta_k):
    """Double (s, t) Simpson integral for a k-(in)dependent smooth kernel.

    s: (ns,), t: (nt, nk), kvec: (nk,). Returns (nk,).
    """
    ss = s[:, None, None]
    tt = t[None, :, :]
    k = kvec[None, None, :]
    u = get_u(tt, ss)
    v = get_v(tt, ss)
    integ = (
        kernel_fn(tt, ss, k, *theta_k)
        * polynomial(tt, ss)
        * pzeta(k * u, *theta_pz)
        * pzeta(k * v, *theta_pz)
    )
    return simpson_nonuniform(simpson_uniform(integ, s), t)


@partial(jit, static_argnums=(0, 1, 2, 3))
def _simpson_transitioning(
    pzeta, kern_smooth, kern_res, t_res, s, t, kvec, theta_pz, theta_k
):
    """Smooth (s, t) part plus a resonant 1-D (s) slice at fixed t = t_res."""
    ss = s[:, None, None]
    tt = t[None, :, :]
    k = kvec[None, None, :]
    u = get_u(tt, ss)
    v = get_v(tt, ss)
    lv = (
        kern_smooth(tt, ss, k, *theta_k)
        * polynomial(tt, ss)
        * pzeta(k * u, *theta_pz)
        * pzeta(k * v, *theta_pz)
    )
    lv_int = simpson_nonuniform(simpson_uniform(lv, s), t)

    s2 = s[:, None]
    kr = kvec[None, :]
    ur = get_u(t_res, s2)
    vr = get_v(t_res, s2)
    rv = (
        kern_res(t_res, s2, kr, *theta_k)
        * polynomial(t_res, s2)
        * pzeta(kr * ur, *theta_pz)
        * pzeta(kr * vr, *theta_pz)
    )
    return lv_int + simpson_uniform(rv, s)


class Integrator:
    """Base class for emission integrators (the numerical method).

    Subclasses implement ``integrate(kernel, pzeta, kvec, theta_pz, theta_k)``
    returning the un-normalised tensor power spectrum on ``kvec``. The
    normalisation (``kernel.norm``) and the f<->k / upsampling orchestration are
    applied by the top-level ``OmegaGW`` model, not here.
    """

    def integrate(self, kernel, pzeta, kvec, theta_pz, theta_k):
        raise NotImplementedError


class SimpsonIntegrator(Integrator):
    """Fixed-grid (s, t) Simpson quadrature (Gaussian, two P_zeta factors).

    ``s`` and ``t`` are either arrays or callables ``grid(kvec, *theta)`` (theta
    is the full ordered parameter vector). For a kernel that declares
    ``resonant_t`` the resonant slice is added.
    """

    def __init__(self, s, t):
        self.s = s
        self.t = t

    def _grids(self, kvec, theta):
        s = self.s(kvec, *theta) if callable(self.s) else self.s
        t = self.t(kvec, *theta) if callable(self.t) else self.t
        return jnp.asarray(s), jnp.asarray(t)

    def integrate(self, kernel, pzeta, kvec, theta_pz, theta_k):
        theta = (*theta_pz, *theta_k)
        s, t = self._grids(kvec, theta)
        if kernel.resonant_t:
            return _simpson_transitioning(
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
        return _simpson_constant(
            pzeta, kernel.overline_Isq, s, t, kvec, theta_pz, theta_k
        )
