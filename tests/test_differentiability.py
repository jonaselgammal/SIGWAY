"""Differentiability: the end-to-end OmegaGW.jacobian.

The spectrum jacobian uses jax.jacfwd through OmegaGW, with a finite-difference
fallback for non-smooth (grid-limit) parameters.  Both branches are checked
against central finite differences -- the independent reference -- which catches
interpolation-gradient artifacts and the heaviside-cutoff case.

The tests use a *fixed* (s, t) grid so the finite difference does not also
perturb the integration grid.
"""

import numpy as np
import jax.numpy as jnp

from sigway.spectrum import OmegaGW
from sigway.kernels import RadiationKernel
from sigway.perturbations import AnalyticPerturbations
import _sigway_configs as C


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
