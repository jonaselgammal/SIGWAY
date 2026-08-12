r"""Radiation-domination kernel and its closed-form cores.

Provides [RadiationKernel][sigway.kernels.RadiationKernel] for scalar-induced
GWs sourced entirely during a radiation-dominated era, and the
oscillation-averaged closed forms it evaluates
(Eqs. (4.21)–(4.22) of [arXiv:2501.11320](https://arxiv.org/abs/2501.11320)):
the production $(t, s)$ form [I_sq_RD][sigway.kernels.I_sq_RD] and the
$(u, v)$ cross-check form [I_sq_RD_uv][sigway.kernels.I_sq_RD_uv].
"""

import jax.numpy as jnp
from jax import jit

from sigway.constants import RD_SOUND_SPEED
from sigway.kernels.base import Kernel, get_u, get_v


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
