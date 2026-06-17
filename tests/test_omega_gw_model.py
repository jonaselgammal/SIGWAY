"""The new composition model: OmegaGW(kernel, perturbations, integrator).

Validates that the new user-facing path reproduces the validated reference
spectra (so the refactor preserves the physics), and exercises the inference
API: ordered parameter vector, name-collision error, keyword routing, the
jacfwd jacobian vs finite differences, and that the analytic path does not
retrace under changing parameter values (the jit contract).
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import _sigway_configs as C  # noqa: E402
from sigway.spectrum import OmegaGW  # noqa: E402
from sigway.kernels import (  # noqa: E402
    RadiationKernel,
    InstantEMDKernel,
    Kernel,
)
from sigway.perturbations import AnalyticPerturbations  # noqa: E402
from sigway.integrators import _simpson_constant  # noqa: E402

REFDIR = os.path.join(os.path.dirname(__file__), "test_data", "reference")

_NAMES = {
    "bpl_rd": ("logA", "alpha", "beta", "gamma", "logks"),
    "lognormal_rd": ("logAs", "logDelta", "logks"),
    "osc_multifield_rd": ("log10A", "log10ks", "delta", "eta_L", "F"),
}


def _heaviside2(k, As, kmax):
    return jnp.heaviside(kmax - k, 1.0) * As


def _build(name):
    cfg = C.ANALYTIC_CONFIGS[name]
    if name == "emd_imd2rd":
        pert = AnalyticPerturbations(_heaviside2, ("As", "kmax"))
        kern = InstantEMDKernel()
    else:
        pert = AnalyticPerturbations(cfg["pzeta"], _NAMES[name])
        kern = RadiationKernel()
    model = OmegaGW(
        pert,
        kern,
        s=jnp.array(cfg["s"]),
        t=cfg["t"],
        f=jnp.array(cfg["f"]),
        upsample=True,
    )
    return model, cfg


@pytest.mark.parametrize(
    "name", ["bpl_rd", "lognormal_rd", "osc_multifield_rd", "emd_imd2rd"]
)
def test_omega_gw_reproduces_fixture(name):
    """New OmegaGW path reproduces the validated reference spectrum."""
    ref = np.load(os.path.join(REFDIR, f"{name}.npz"))
    model, cfg = _build(name)
    got = np.array(model(jnp.array(cfg["f"]), *ref["params"]))
    peak = np.nanmax(ref["omega_gw"])
    np.testing.assert_allclose(
        got, ref["omega_gw"], rtol=1e-6, atol=peak * 1e-10
    )


def test_parameter_names_order():
    """theta order is perturbation params then kernel params."""
    model, _ = _build("emd_imd2rd")
    assert model.parameter_names == ("As", "kmax", "etaR")


def test_keyword_equals_positional():
    """Keyword routing matches positional theta in parameter_names order."""
    model, cfg = _build("emd_imd2rd")
    As, kmax, etaR = cfg["params"]
    pos = np.array(model(jnp.array(cfg["f"]), As, kmax, etaR))
    kw = np.array(model(jnp.array(cfg["f"]), As=As, kmax=kmax, etaR=etaR))
    assert np.array_equal(pos, kw)


def test_parameter_name_collision_raises():
    """A name shared by perturbations and kernel is rejected clearly."""

    class ClashKernel(Kernel):
        param_names = ("As",)

        def overline_Isq(self, t, s, k, As):
            return k * 0.0

    with pytest.raises(ValueError, match="collision"):
        OmegaGW(
            AnalyticPerturbations(_heaviside2, ("As", "kmax")),
            ClashKernel(),
            s=jnp.array([0.0, 1.0]),
            t=jnp.ones((3, 2)),
        )


def test_jacobian_matches_finite_difference():
    """jacfwd Jacobian == central FD for a smooth (RD) config; and the
    log-amplitude column equals the closed form 2 ln10 * Omega_GW."""
    cfg = C.ANALYTIC_CONFIGS["lognormal_rd"]
    p = cfg["params"]
    f = np.geomspace(1e-4, 1e-2, 10)
    t_fixed = cfg["t"](jnp.array(f) * 2 * jnp.pi, *p)  # freeze grid
    model = OmegaGW(
        AnalyticPerturbations(cfg["pzeta"], _NAMES["lognormal_rd"]),
        RadiationKernel(),
        s=jnp.array(cfg["s"]),
        t=t_fixed,
        upsample=False,
    )
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
    cfg = C.ANALYTIC_CONFIGS["lognormal_rd"]
    p = cfg["params"]
    f = np.geomspace(1e-4, 1e-2, 10)
    t_fixed = cfg["t"](jnp.array(f) * 2 * jnp.pi, *p)
    model = OmegaGW(
        AnalyticPerturbations(cfg["pzeta"], _NAMES["lognormal_rd"]),
        RadiationKernel(),
        s=jnp.array(cfg["s"]),
        t=t_fixed,
        upsample=False,
    )
    model(f, *p)  # compile once
    n = _simpson_constant._cache_size()
    model(f, p[0] + 0.3, p[1], p[2])  # different theta, same shapes
    assert _simpson_constant._cache_size() == n
