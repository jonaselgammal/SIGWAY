"""Derivatives: hand-coded kernel gradients and the end-to-end jacobian.

The eMD kernel carries *hand-written* analytic gradients w.r.t. etaR (used by
the Fisher pipeline); the spectrum jacobian uses jax.jacfwd through OmegaGW.
Both are checked against finite differences -- the independent reference --
which catches interpolation-gradient artifacts and algebra slips.

The end-to-end test uses a *fixed* (s, t) grid so the finite difference does not
also perturb the integration grid.
"""

import numpy as np
import jax.numpy as jnp

from sigway.spectrum import OmegaGW
from sigway.kernels import (
    RadiationKernel,
    I_sq_IRD_LV,
    d_I_sq_IRD_LV,
    I_sq_IRD_res,
    d_I_sq_IRD_res,
)
from sigway.perturbations import AnalyticPerturbations
import _sigway_configs as C

_KMAX, _ETAR = 0.06, 2000.0


def test_d_I_sq_IRD_LV_wrt_etaR():
    """Hand-coded large-V eMD gradient d/d(etaR) matches finite differences."""
    t = jnp.array([0.3, 1.0, 3.0])
    s = jnp.array([0.1, 0.4, 0.7])
    k = 0.02
    h = _ETAR * 1e-6
    fd = (
        np.array(I_sq_IRD_LV(t, s, k, _KMAX, _ETAR + h))
        - np.array(I_sq_IRD_LV(t, s, k, _KMAX, _ETAR - h))
    ) / (2 * h)
    ana = np.array(d_I_sq_IRD_LV(1, t, s, k, _KMAX, _ETAR))  # index 1 = etaR
    assert np.allclose(ana, fd, rtol=1e-5)


def test_d_I_sq_IRD_res_wrt_etaR():
    """Hand-coded resonant eMD gradient d/d(etaR) matches finite differences."""
    t = jnp.sqrt(3.0) - 1.0
    s = jnp.array([0.1, 0.4, 0.7])
    k = 0.02
    h = _ETAR * 1e-6
    fd = (
        np.array(I_sq_IRD_res(t, s, k, _KMAX, _ETAR + h))
        - np.array(I_sq_IRD_res(t, s, k, _KMAX, _ETAR - h))
    ) / (2 * h)
    ana = np.array(d_I_sq_IRD_res(1, t, s, k, _KMAX, _ETAR))
    assert np.allclose(ana, fd, rtol=1e-5)


def test_d_I_sq_IRD_LV_kmax_is_zero():
    """Documented: the LV kernel gradient w.r.t kmax is zero (kmax enters only
    via integration limits, handled in the integrator)."""
    t = jnp.array([0.3, 1.0])
    s = jnp.array([0.2, 0.5])
    grad = np.array(d_I_sq_IRD_LV(0, t, s, 0.02, _KMAX, _ETAR))
    assert np.allclose(grad, 0.0)


def test_jacobian_bpl_matches_finite_difference():
    """OmegaGW.jacobian == central FD for every broken-power-law parameter.

    Uses a fixed t-grid so FD and the (fixed-grid) jacfwd derivative agree.
    """
    cfg = C.ANALYTIC_CONFIGS["bpl_rd"]
    p = list(cfg["params"])
    f = np.geomspace(1e-4, 1e-1, 12)
    t_fixed = cfg["t"](jnp.array(f) * 2 * jnp.pi, *p)  # freeze grid
    m = OmegaGW(
        AnalyticPerturbations(cfg["pzeta"], C._PZ_NAMES["bpl_rd"]),
        RadiationKernel(),
        s=jnp.array(cfg["s"]),
        t=t_fixed,
        upsample=False,
    )
    J = np.array(m.jacobian(f, jnp.array(p)))
    for idx in range(len(p)):
        h = 1e-5 * max(abs(p[idx]), 1.0)
        pp = list(p)
        pp[idx] += h
        pm = list(p)
        pm[idx] -= h
        fd = (
            np.array(m(jnp.array(f), *pp)) - np.array(m(jnp.array(f), *pm))
        ) / (2 * h)
        col = J[:, idx]
        good = np.abs(col) > np.nanmax(np.abs(col)) * 1e-6
        assert np.allclose(col[good], fd[good], rtol=2e-4), f"param {idx}"


def test_emd_jacobian_handles_kmax():
    """OmegaGW.jacobian handles the eMD grid-limit param kmax.

    jacfwd is correct for the smooth params (As, etaR) but wrong for kmax (it
    moves the heaviside cutoff / t-grid limit); the model marks kmax nonsmooth
    so its column falls back to central differences. The full jacobian must then
    match a full central-FD jacobian in the well-resolved band.
    """
    cfg = C.ANALYTIC_CONFIGS["emd_imd2rd"]
    p = cfg["params"]
    f = cfg["f"]
    m = C.build_model("emd_imd2rd")
    th = jnp.array(p)
    J = np.array(m.jacobian(f, th))
    og = np.array(m(jnp.array(f), *p))
    good = og > og.max() * 1e-2  # well-resolved peak band
    for i in range(len(p)):
        h = 1e-5 * max(abs(float(th[i])), 1.0)
        fd = (
            np.array(m(jnp.array(f), *th.at[i].add(h)))
            - np.array(m(jnp.array(f), *th.at[i].add(-h)))
        ) / (2 * h)
        assert np.allclose(J[good, i], fd[good], rtol=2e-3), f"param {i}"
