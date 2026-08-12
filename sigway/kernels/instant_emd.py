r"""Instantaneous early-matter → radiation transition kernel.

Provides [InstantEMDKernel][sigway.kernels.InstantEMDKernel] for a universe
that begins in an early matter-dominated (EMD) era and reheats instantaneously
to radiation domination at conformal time $\eta_R$, together with its two
closed-form contributions: the smooth large-$V$ bulk
[I_sq_IRD_LV][sigway.kernels.I_sq_IRD_LV] and the sound-horizon resonant slice
[I_sq_IRD_res][sigway.kernels.I_sq_IRD_res].  These are evaluated at different
$t$ and summed, so they live in separate cores.
"""

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import sici

from sigway.constants import RD_SOUND_SPEED
from sigway.kernels.base import Kernel


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
    and is proportional to the Si/Ci combination
    $4\,\mathrm{Ci}(x_R/2)^2 + (\pi - 2\,\mathrm{Si}(x_R/2))^2$.

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
        Unused; accepted for a uniform kernel-core signature.  The $k_{\max}$
        cutoff of the scalar power spectrum bounds the $(t, s)$ domain via the
        perturbation spectrum, applied by the integrator, not here.
    etaR : float
        Conformal time at the EMD → RD transition (Mpc).

    Returns
    -------
    jax.Array
        $\overline{I^2_{\rm IRD,LV}}$ values, same shape as the broadcast of
        $t$, $s$, and $k$.
    """
    xR = k * etaR
    # Si/Ci combination 4 Ci(xR/2)^2 + (pi - 2 Si(xR/2))^2, evaluated
    # directly with jax.scipy.special.sici (jit-able, differentiable).
    si, ci = sici(xR / 2.0)
    sici_factor = 4.0 * ci**2 + (jnp.pi - 2.0 * si) ** 2
    result = (9.0 * t**4.0 * xR**8.0 * sici_factor) / 81920000.0
    return 4.0 * result  # the factor of 4 comes from x_R^2/(x_R-x_R/2)^2


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
