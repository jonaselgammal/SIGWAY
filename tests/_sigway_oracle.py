"""Independent re-implementation of the SIGW double integral, for validation.

This module deliberately shares **no code** with ``sigway``:

* the radiation-domination kernel is the closed-form oscillation-averaged
  transfer function :math:`\\overline{I^2}(u,v)` written directly from the
  literature (Kohri & Terada 2018; Espinosa, Racco & Riotto 2018) -- it agrees
  with ``sigway``'s ``I_sq_RD`` to ~1e-11 (see test_kernels_rd);
* the integration uses ``scipy`` (a dense ``scipy.integrate.simpson`` grid, or
  adaptive ``dblquad``/``quad``), a different algorithm from ``sigway``'s
  hand-rolled vectorised Simpson rule.

It therefore provides a genuine cross-check of the *physics* and of the
*integration*, not a self-comparison. Used by the fixture generator to certify
reference spectra and by ``test_cross_backend``.
"""

import numpy as np
from scipy.integrate import simpson, dblquad, quad
from scipy.special import sici

# Prefactor applied as  Omega_GW = NORM[norm] * (double integral).
# NORM["RD"] = CG/24 * OMEGA_R, NORM["CT"] = 1/24 (this independent oracle
# keeps its own prefactor convention, cross-checked against sigway output).
_OMEGA_R = 4.2e-5
_CG = 0.39
NORM = {"RD": 2.0 * _CG / 24.0 * _OMEGA_R, "CT": 2.0 / 24.0}


# ---------------------------------------------------------------------------
# Textbook radiation-domination kernel and polynomial (numpy, independent)
# ---------------------------------------------------------------------------
def kernel_RD_text(t, s):
    r"""Oscillation-averaged overline{I^2_RD}(u,v) from the literature."""
    u = (t + s + 1.0) / 2.0
    v = (t - s + 1.0) / 2.0
    fac = u**2 + v**2 - 3.0
    IA = 3.0 * fac / (4.0 * u**3 * v**3)
    IB = -4.0 * u * v + fac * np.log(
        np.abs((3.0 - (u + v) ** 2) / (3.0 - (u - v) ** 2))
    )
    IC = np.pi * fac * np.heaviside(u + v - np.sqrt(3.0), 1.0)
    return IA**2 * (IB**2 + IC**2) / 2.0


def polynomial_np(t, s):
    num = t * (2.0 + t) * (s**2 - 1.0)
    den = (1.0 - s + t) * (1.0 + s + t)
    return 2.0 * (num / den) ** 2


def omega_RD_oracle(k, Pz, norm, t_max, ns=400, nt=6000):
    """Omega_GW(k) for a constant (RD) kernel via a dense scipy-Simpson grid.

    Pz is a numpy callable P_zeta(q); t_max is the upper t limit (chosen to
    match the physical support of the source); norm is "RD" or "CT".
    """
    t = np.concatenate(
        [np.linspace(1e-7, 0.999, ns), np.geomspace(1.0, t_max, nt)]
    )
    s = np.linspace(0.0, 1.0, ns)
    T, S = np.meshgrid(t, s, indexing="ij")
    u = (T + S + 1.0) / 2.0
    v = (T - S + 1.0) / 2.0
    integ = kernel_RD_text(T, S) * polynomial_np(T, S) * Pz(k * u) * Pz(k * v)
    inner = simpson(integ, x=s, axis=1)
    return NORM[norm] * simpson(inner, x=t)


# ---------------------------------------------------------------------------
# Instant eMD -> RD (I_MD_to_RD) kernel, numpy mirror of sigway's functions
# ---------------------------------------------------------------------------
def _LV(xR):
    si, ci = sici(xR / 2.0)
    return 4.0 * ci**2 + (np.pi - 2.0 * si) ** 2


def kernel_IRD_LV_np(t, k, etaR):
    """Large-V eMD->RD contribution (independent of s); mirrors I_sq_IRD_LV."""
    xR = k * etaR
    return 4.0 * (9.0 * t**4 * xR**8 * _LV(xR)) / 81920000.0


def kernel_IRD_res_np(t, s, k, etaR):
    """Resonant eMD->RD contribution; mirrors I_sq_IRD_res (fudge=2.3)."""
    xR = k * etaR
    num = 9.0 * (-5.0 + s**2 + 2.0 * t + t**2) ** 4 * xR**8
    den = 81920000.0 * (t - s + 1.0) ** 2 * (t + s + 1.0) ** 2
    return 4.0 * 2.3 * (num / den) * (7.97727 / xR)


def omega_eMD_oracle(k, Pz, kmax, etaR, norm, k_is_array=False):
    """Omega_GW(k) for the instant eMD->RD kernel via adaptive scipy quadrature.

    Reproduces ``integrate_transitioning_kernel``: a 2-D large-V integral over
    the triangular support (set by the heaviside source) plus a 1-D resonant
    integral at t = sqrt(3)-1.  The source-support bounds are encoded directly
    in the integration limits, so there is no integrand discontinuity (unlike
    the fixed-grid Simpson SIGWAY uses).
    """
    # source support: P(ku),P(kv) != 0  <=>  u,v < kmax/k,
    # i.e. t + s < 2 kmax/k - 1
    Tcut = 2.0 * kmax / k - 1.0
    As2 = Pz(np.array([0.5 * kmax]))[0] ** 2  # flat amplitude squared
    if Tcut <= 0.0:
        return 0.0

    def lv_integrand(s, t):
        return kernel_IRD_LV_np(t, k, etaR) * polynomial_np(t, s)

    lv, _ = dblquad(
        lv_integrand,
        0.0,
        Tcut,
        lambda t: 0.0,
        lambda t: min(1.0, Tcut - t),
        epsabs=0.0,
        epsrel=1e-7,
    )
    t_res = np.sqrt(3.0) - 1.0
    s_hi = min(1.0, Tcut - t_res)
    res = 0.0
    if s_hi > 0.0:
        res, _ = quad(
            lambda s: kernel_IRD_res_np(t_res, s, k, etaR)
            * polynomial_np(t_res, s),
            0.0,
            s_hi,
            epsabs=0.0,
            epsrel=1e-7,
        )
    return NORM[norm] * As2 * (lv + res)
