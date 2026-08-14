r"""Shared foundation for the kernel package: geometry, normalisation, ABC.

This module holds the pieces every concrete kernel builds on:

* the $(t, s) \leftrightarrow (u, v)$ coordinate helpers
  ([get_u][sigway.kernels.get_u], [get_v][sigway.kernels.get_v]) and the
  geometric projection factor ([polynomial][sigway.kernels.polynomial]);
* the $\Omega_{\rm GW}$ normalisation presets and the
  [resolve_norm_preset][sigway.kernels.resolve_norm_preset] helper that turns a
  norm specification into a callable;
* the abstract [Kernel][sigway.kernels.Kernel] base class.

The closed-form era kernels live one-per-module alongside their
[Kernel][sigway.kernels.Kernel] subclass (``radiation``, ``matter``,
``instant_emd``); everything is re-exported from ``sigway.kernels``.
"""

import jax
from jax import jit

from sigway.constants import (
    SM_CG_factor,
    Omega_radiation_h2_today,
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


def resolve_norm_preset(spec):
    r"""Turn a normalisation spec into a uniform ``k -> value`` callable.

    Wraps a preset key (from :data:`NORM_PRESETS`: ``'RD'``, ``'CT'``,
    ``'bare'``) or a constant float as ``lambda k: value``, and returns a
    ``k``-dependent callable unchanged, so every kernel's norm shares one
    ``(k)`` signature. Raises ``ValueError`` on an unknown preset string.
    """
    if isinstance(spec, str):
        if spec not in NORM_PRESETS:
            raise ValueError(
                "Unknown norm preset {!r}; choose from {}.".format(
                    spec, sorted(NORM_PRESETS)
                )
            )
        value = NORM_PRESETS[spec]
        return lambda k: value
    if callable(spec):
        return spec
    return lambda k: spec


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
        self._norm = resolve_norm_preset(spec)
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
