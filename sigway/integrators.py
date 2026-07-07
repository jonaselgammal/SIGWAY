r"""Numerical integration strategies for scalar-induced gravitational waves.

This module performs the double integral over the two internal momenta $s$ and
$t$ that appears in the standard expression for the SIGW tensor power spectrum.
All physics assumptions about the primordial curvature power spectrum
$\mathcal{P}_\zeta$ live here (specifically the Gaussian, two-point assumption
expressed as two $\mathcal{P}_\zeta$ factors in the integrand), keeping the
kernel and perturbation modules free of quadrature details.

[SimpsonIntegrator][sigway.integrators.SimpsonIntegrator] is the primary
concrete strategy: it applies composite Simpson quadrature on user-supplied
$s$ and $t$ grids.

See [Theory: the $(s,t)$
reparameterisation](../theory/index.md#in-the-codes-st-variables)
for how $(s,t)$ relate to the physical momenta and the form of the integrand.
"""

import jax
import jax.numpy as jnp
from jax import jit

from sigway.kernels import get_u, get_v, polynomial
from sigway.utils import simpson_uniform, simpson_nonuniform

jax.config.update("jax_enable_x64", True)

__all__ = ["Integrator", "SimpsonIntegrator"]


def _kint(kvec, u, v):
    """Wavenumber grid spanning the range $ku$ to $kv$ needed by the spectrum.

    Used only when the scalar-perturbation spectrum is backed by a numerical
    solver (e.g. Mukhanov-Sasaki) and needs an interpolation grid; analytic
    spectra ignore this helper.
    """
    lo = jnp.minimum(u.min(), v.min()) * kvec.min()
    hi = jnp.maximum(u.max(), v.max()) * kvec.max()
    return jnp.geomspace(lo, hi, 100)


def _simpson_constant_impl(pzeta, kernel_fn, s, t, kvec, theta_pz, theta_k):
    """2-D Simpson integral over $(s, t)$ for a smooth (non-resonant) kernel.

    Returns an array of shape ``(nk,)`` containing the un-normalised tensor
    power spectrum at each wavenumber.
    """
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
    r"""2-D integral for the smooth kernel plus a 1-D resonant slice at
    fixed $t$.

    Used when a kernel (e.g. the constant equation-of-state kernel near a
    resonant feature) contributes a delta-like peak at a specific value
    $t = t_\mathrm{res}$.  The smooth 2-D part and the resonant 1-D part are
    evaluated separately and summed.
    """
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
    r"""Abstract base class for SIGW double-integral strategies.

    An integrator performs the numerical double integral over the two rescaled
    internal momenta $s \in [0, 1]$ and $t \in [0, \infty)$ that defines the
    scalar-induced gravitational-wave tensor power spectrum.  Given a
    [Kernel][sigway.kernels.Kernel] (which encodes the radiation-transfer
    physics) and a primordial scalar power spectrum
    [ScalarPerturbations][sigway.perturbations.ScalarPerturbations], it returns
    the un-normalised tensor power spectrum on a wavenumber grid.

    The global normalisation prefactor (``kernel.norm``), the conversion between
    frequency and wavenumber, and any output upsampling are applied by
    [OmegaGW][sigway.spectrum.OmegaGW] after calling this method.

    Subclasses must override ``integrate``.

    Methods
    -------
    integrate(kernel, pzeta, kvec, theta_pz, theta_k)
        Evaluate the un-normalised tensor power spectrum on ``kvec``.
        Must be implemented by every concrete subclass.
    """

    def integrate(self, kernel, pzeta, kvec, theta_pz, theta_k):
        raise NotImplementedError


class SimpsonIntegrator(Integrator):
    r"""2-D Simpson quadrature over the $(s, t)$ momentum plane (Gaussian
    spectrum).

    This integrator performs the double integral that enters the scalar-induced
    gravitational-wave energy density.  It assumes a **Gaussian** primordial
    curvature perturbation, so the integrand contains the product of two copies
    of the dimensionless scalar power spectrum $\mathcal{P}_\zeta$:

    $$
    \overline{I^2}(k) = \int_0^1 ds \int_0^\infty dt \;
        \overline{I^2}(t, s, k) \cdot J(t, s)
        \cdot \mathcal{P}_\zeta(ku) \cdot \mathcal{P}_\zeta(kv),
    $$

    where the dimensionless internal momenta are defined by

    $$
    u = \frac{t + s + 1}{2}, \qquad v = \frac{t - s + 1}{2},
    $$

    and $J(t, s)$ is the Jacobian of the $(u, v) \to (s, t)$ change of
    variables (see [polynomial][sigway.kernels.polynomial]).

    The integral is discretised on user-supplied grids for $s$ and $t$ and
    evaluated with composite Simpson quadrature.  Grid choice matters
    physically: the integrand peaks near $t \approx 1$, so the $t$-grid should
    be **linear** below 1 (to resolve the peak) and **geometric** above 1 (to
    cover the slowly-decaying tail efficiently).  A few hundred points total
    is typically sufficient.

    If the [Kernel][sigway.kernels.Kernel] declares a resonant feature at a
    specific value $t_\mathrm{res}$ (``kernel.resonant_t`` is set), an
    additional 1-D Simpson integral over $s$ at that fixed $t$ slice is
    computed and added to the 2-D result.

    Parameters
    ----------
    s : array_like or callable
        Quadrature nodes for the momentum variable $s \in [0, 1]$.
        May be a fixed 1-D array or a callable ``s(kvec, *theta)`` returning
        a 1-D array, where ``theta`` collects all physical parameters
        ``(*theta_pz, *theta_k)``.  A callable is useful when the required
        $s$-resolution depends on the spectral peak position.
    t : array_like or callable
        Quadrature nodes for the momentum variable $t \in [0, \infty)$.
        May be a fixed 1-D array, a 2-D array of shape ``(nt, nk)`` providing
        a separate node sequence per wavenumber, or a callable
        ``t(kvec, *theta)`` returning either shape.  When a 1-D array is
        provided it is broadcast internally to ``(nt, nk)`` so that the
        non-uniform Simpson rule over $t$ applies consistently at every $k$.
        A $k$-dependent grid is useful for tracking a spectral feature whose
        relevant $t$-range shifts with $k$.

    Attributes
    ----------
    s : array_like or callable
        The $s$-grid as supplied at construction time.
    t : array_like or callable
        The $t$-grid as supplied at construction time.

    Methods
    -------
    integrate(kernel, pzeta, kvec, theta_pz, theta_k)
        Evaluate the un-normalised tensor power spectrum on ``kvec``.

    Examples
    --------
    Build a `SimpsonIntegrator` with a hybrid $t$-grid that is linear near the
    integrand peak ($t \lesssim 1$) and geometric in the tail ($t \gtrsim 1$):

    >>> import jax.numpy as jnp
    >>> from sigway.integrators import SimpsonIntegrator
    >>> s = jnp.linspace(0.0, 1.0, 10)
    >>> t = jnp.concatenate([jnp.linspace(1e-5, 0.999, 200),
    ...                      jnp.geomspace(1.0, 1e3, 800)])
    >>> integ = SimpsonIntegrator(s, t)   # or pass s=, t= straight to OmegaGW
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
        r"""Evaluate the un-normalised tensor power spectrum
        $\overline{I^2}(k)$.

        Resolves the $s$ and $t$ grids (evaluating them as callables if
        needed),
        then performs the composite Simpson quadrature over both momentum
        variables and returns the result on the requested wavenumber grid.

        For analytic primordial spectra (``pzeta.jittable = True``) the
        quadrature is compiled at first call and reused for subsequent calls
        with
        different physical parameters at the same array shapes, making parameter
        scans fast.  Spectra computed by the Mukhanov-Sasaki solver run the same
        quadrature without compilation.

        Parameters
        ----------
        kernel : Kernel
            A [Kernel][sigway.kernels.Kernel] instance providing the
            radiation-transfer function $\overline{I^2}(t, s, k)$.  If the
            kernel has a resonant feature it additionally provides
            ``overline_Isq_resonant`` and ``resonant_t``.
        pzeta : ScalarPerturbations
            The primordial scalar power spectrum
            $\mathcal{P}_\zeta(k)$.  Its ``prepare`` method returns a
            callable that evaluates $\mathcal{P}_\zeta$ on an arbitrary
            wavenumber array.
        kvec : array_like, shape (nk,)
            Wavenumber grid at which $\overline{I^2}(k)$ is evaluated.
            Units must be consistent with those used by ``kernel`` and
            ``pzeta``.
        theta_pz : tuple
            Physical parameter values for the scalar power spectrum (e.g.
            amplitude, tilt, peak scale).
        theta_k : tuple
            Physical parameter values for the kernel (e.g. equation-of-state
            parameter $w$).

        Returns
        -------
        jax.Array, shape (nk,)
            Un-normalised tensor power spectrum values $\overline{I^2}(k)$ at
            each wavenumber in ``kvec``.  [OmegaGW][sigway.spectrum.OmegaGW]
            multiplies by ``kernel.norm`` and the standard prefactors to produce
            $\Omega_\mathrm{GW}(f)$.
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
