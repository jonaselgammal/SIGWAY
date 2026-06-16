"""Derivatives: hand-coded kernel gradients and the end-to-end derivative path.

The eMD kernel carries *hand-written* analytic gradients w.r.t. etaR (used by
the Fisher pipeline), and the spectrum derivative d_integrate uses jax jvp
through P_zeta. Both are checked against finite differences -- the independent
reference -- which catches interpolation-gradient artifacts and algebra slips.

To compare like with like, the end-to-end test uses a *fixed* (s, t) grid so the
finite difference does not also perturb the integration grid (the analytic
derivative treats the grid as constant).
"""
import numpy as np
import jax.numpy as jnp

from sigway.omega_gw_jax import (
    OmegaGWjax, I_sq_IRD_LV, d_I_sq_IRD_LV, I_sq_IRD_res, d_I_sq_IRD_res,
)
import _sigway_configs as C

_KMAX, _ETAR = 0.06, 2000.0


def test_d_I_sq_IRD_LV_wrt_etaR():
    """Hand-coded large-V eMD gradient d/d(etaR) matches finite differences."""
    t = jnp.array([0.3, 1.0, 3.0])
    s = jnp.array([0.1, 0.4, 0.7])
    k = 0.02
    h = _ETAR * 1e-6
    fd = (np.array(I_sq_IRD_LV(t, s, k, _KMAX, _ETAR + h))
          - np.array(I_sq_IRD_LV(t, s, k, _KMAX, _ETAR - h))) / (2 * h)
    ana = np.array(d_I_sq_IRD_LV(1, t, s, k, _KMAX, _ETAR))  # index 1 = etaR
    assert np.allclose(ana, fd, rtol=1e-5)


def test_d_I_sq_IRD_res_wrt_etaR():
    """Hand-coded resonant eMD gradient d/d(etaR) matches finite differences."""
    t = jnp.sqrt(3.0) - 1.0
    s = jnp.array([0.1, 0.4, 0.7])
    k = 0.02
    h = _ETAR * 1e-6
    fd = (np.array(I_sq_IRD_res(t, s, k, _KMAX, _ETAR + h))
          - np.array(I_sq_IRD_res(t, s, k, _KMAX, _ETAR - h))) / (2 * h)
    ana = np.array(d_I_sq_IRD_res(1, t, s, k, _KMAX, _ETAR))
    assert np.allclose(ana, fd, rtol=1e-5)


def test_d_I_sq_IRD_LV_kmax_is_zero():
    """Documented: the LV kernel gradient w.r.t kmax is zero (kmax enters only
    via integration limits, handled in the integrator)."""
    t = jnp.array([0.3, 1.0])
    s = jnp.array([0.2, 0.5])
    grad = np.array(d_I_sq_IRD_LV(0, t, s, 0.02, _KMAX, _ETAR))
    assert np.allclose(grad, 0.0)


def test_d_integrate_bpl_matches_finite_difference():
    """Spectrum derivative w.r.t every broken-power-law parameter == central FD.

    Uses a fixed t-grid so FD and the (fixed-grid) analytic derivative agree.
    """
    cfg = C.ANALYTIC_CONFIGS["bpl_rd"]
    p = list(cfg["params"])
    f = np.geomspace(1e-4, 1e-1, 12)
    # freeze the parameter-dependent t-grid at the fiducial parameters
    t_fixed = cfg["t"](jnp.array(f) * 2 * jnp.pi, *p)
    m = OmegaGWjax(cfg["pzeta"], jnp.array(cfg["s"]), t_fixed, f=jnp.array(f),
                   norm="RD", kernel="RD", upsample=False, dP_zeta="auto",
                   jit=True)
    for idx in range(len(p)):
        ana = np.array(m.d_integrate(idx, jnp.array(f), *p))
        h = 1e-5 * max(abs(p[idx]), 1.0)
        pp = list(p)
        pp[idx] += h
        pm = list(p)
        pm[idx] -= h
        fd = (np.array(m(jnp.array(f), *pp))
              - np.array(m(jnp.array(f), *pm))) / (2 * h)
        good = np.abs(ana) > np.nanmax(np.abs(ana)) * 1e-6
        assert np.allclose(ana[good], fd[good], rtol=2e-4), f"param {idx}"


def test_amplitude_derivative_closed_form():
    """d Omega / d(logAs) = 2 ln10 * Omega (since Omega ~ 10^(2 logAs))."""
    cfg = C.ANALYTIC_CONFIGS["lognormal_rd"]
    p = list(cfg["params"])
    f = np.geomspace(1e-4, 1e-2, 10)
    t_fixed = cfg["t"](jnp.array(f) * 2 * jnp.pi, *p)
    m = OmegaGWjax(cfg["pzeta"], jnp.array(cfg["s"]), t_fixed, f=jnp.array(f),
                   norm="RD", kernel="RD", upsample=False, dP_zeta="auto",
                   jit=True)
    og = np.array(m(jnp.array(f), *p))
    dlog = np.array(m.d_integrate(0, jnp.array(f), *p))  # d/d logAs
    good = og > og.max() * 1e-6
    assert np.allclose(dlog[good], 2 * np.log(10) * og[good], rtol=1e-6)
