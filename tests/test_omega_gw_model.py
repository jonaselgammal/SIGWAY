"""Inference API of the OmegaGW composition model.

The end-to-end physics regression (incl. MS) lives in test_omega_gw_regression;
here we exercise the parameter API: ordered parameter vector, name-collision
error, keyword routing, the jacfwd jacobian vs finite differences, and that the
analytic path does not retrace under changing parameter values (jit contract).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import _sigway_configs as C  # noqa: E402
from sigway.spectrum import OmegaGW  # noqa: E402
from sigway.kernels import RadiationKernel, Kernel  # noqa: E402
from sigway.perturbations import AnalyticPerturbations  # noqa: E402
from sigway.integrators import _simpson_constant  # noqa: E402


def test_parameter_names_order():
    """theta order is perturbation params then kernel params."""
    model = C.build_model("emd_imd2rd")
    assert model.parameter_names == ("As", "kmax", "etaR")


def test_keyword_equals_positional():
    """Keyword routing matches positional theta in parameter_names order."""
    model = C.build_model("emd_imd2rd")
    cfg = C.ANALYTIC_CONFIGS["emd_imd2rd"]
    As, kmax, etaR = cfg["params"]
    f = jnp.array(cfg["f"])
    pos = np.array(model(f, As, kmax, etaR))
    kw = np.array(model(f, As=As, kmax=kmax, etaR=etaR))
    assert np.array_equal(pos, kw)


def test_parameter_name_collision_raises():
    """A name shared by perturbations and kernel is rejected clearly."""

    class ClashKernel(Kernel):
        param_names = ("As",)

        def overline_Isq(self, t, s, k, As):
            return k * 0.0

    with pytest.raises(ValueError, match="collision"):
        OmegaGW(
            AnalyticPerturbations(C.pzeta_heaviside2, ("As", "kmax")),
            ClashKernel(),
            s=jnp.array([0.0, 1.0]),
            t=jnp.ones((3, 2)),
        )


def _fixed_grid_lognormal():
    cfg = C.ANALYTIC_CONFIGS["lognormal_rd"]
    p = cfg["params"]
    f = np.geomspace(1e-4, 1e-2, 10)
    t_fixed = cfg["t"](jnp.array(f) * 2 * jnp.pi, *p)  # freeze grid
    model = OmegaGW(
        AnalyticPerturbations(cfg["pzeta"], C._PZ_NAMES["lognormal_rd"]),
        RadiationKernel(),
        s=jnp.array(cfg["s"]),
        t=t_fixed,
        upsample=False,
    )
    return model, f, p


def test_jacobian_matches_finite_difference():
    """jacfwd Jacobian == central FD for a smooth (RD) config; and the
    log-amplitude column equals the closed form 2 ln10 * Omega_GW."""
    model, f, p = _fixed_grid_lognormal()
    J = np.array(model.jacobian(f, jnp.array(p)))
    og = np.array(model(f, *p))
    good = og > og.max() * 1e-6
    for i in range(len(p)):
        h = 1e-5 * max(abs(p[i]), 1.0)
        pp = list(p)
        pp[i] += h
        pm = list(p)
        pm[i] -= h
        fd = (np.array(model(f, *pp)) - np.array(model(f, *pm))) / (2 * h)
        assert np.allclose(J[good, i], fd[good], rtol=1e-4)
    assert np.allclose(J[good, 0], 2 * np.log(10) * og[good], rtol=1e-6)


def test_analytic_path_does_not_retrace():
    """Changing only theta (fixed shapes) reuses the compiled jit core."""
    model, f, p = _fixed_grid_lognormal()
    model(f, *p)  # compile once
    n = _simpson_constant._cache_size()
    model(f, p[0] + 0.3, p[1], p[2])  # different theta, same shapes
    assert _simpson_constant._cache_size() == n
