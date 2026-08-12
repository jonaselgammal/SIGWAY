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


def test_one_dimensional_t_grid_broadcasts():
    """A 1-D t grid is broadcast across k (matches the explicit 2-D form)."""
    cfg = C.ANALYTIC_CONFIGS["lognormal_rd"]
    p = cfg["params"]
    f = np.geomspace(1e-4, 1e-2, 8)
    names = C._PZ_NAMES["lognormal_rd"]
    t1d = jnp.geomspace(1e-5, 1e3, 400)
    t2d = jnp.broadcast_to(t1d[:, None], (t1d.shape[0], f.shape[0]))
    m1 = OmegaGW(
        AnalyticPerturbations(cfg["pzeta"], names),
        RadiationKernel(),
        s=jnp.array(cfg["s"]),
        t=t1d,
    )
    m2 = OmegaGW(
        AnalyticPerturbations(cfg["pzeta"], names),
        RadiationKernel(),
        s=jnp.array(cfg["s"]),
        t=t2d,
    )
    assert np.allclose(np.array(m1(f, *p)), np.array(m2(f, *p)))


def test_analytic_path_does_not_retrace():
    """Changing only theta (fixed shapes) reuses the compiled jit core."""
    model, f, p = _fixed_grid_lognormal()
    model(f, *p)  # compile once
    n = _simpson_constant._cache_size()
    model(f, p[0] + 0.3, p[1], p[2])  # different theta, same shapes
    assert _simpson_constant._cache_size() == n


def _powerlaw_model(interp_grid=None, interp="linear"):
    """RD kernel + a pure power-law P_zeta on a fixed (s, t) grid.

    RadiationKernel is k-independent, so a power-law P_zeta(k) = A k^n makes the
    un-normalised integral an exact power law res(k) ∝ k^(2n) -- the case where
    log-log interpolation is exact and linear interpolation is not.
    """
    def pz(k, logA, n):
        return 10.0**logA * k**n

    return OmegaGW(
        AnalyticPerturbations(pz, ("logA", "n")),
        RadiationKernel(),
        s=jnp.linspace(0.0, 1.0, 17),
        t=jnp.geomspace(1e-2, 1e2, 200),
        interp_grid=interp_grid,
        interp=interp,
    )


def test_interp_loglog_exact_for_power_law():
    """interp='loglog' reproduces the direct spectrum for a power law; linear
    (the other mode) is visibly worse on the same coarse grid."""
    p = (-1.0, -0.7)  # logA, n  ->  res(k) ∝ k^(-1.4)
    grid = np.geomspace(1e-3, 1e0, 12)
    # dense off-grid frequencies strictly inside the grid (no extrapolation)
    f = np.geomspace(grid[0] * 1.05, grid[-1] * 0.95, 60)

    truth = np.array(_powerlaw_model()(jnp.array(f), *p))
    loglog = np.array(
        _powerlaw_model(interp_grid=grid, interp="loglog")(jnp.array(f), *p)
    )
    linear = np.array(
        _powerlaw_model(interp_grid=grid, interp="linear")(jnp.array(f), *p)
    )

    err_loglog = np.max(np.abs(loglog / truth - 1.0))
    err_linear = np.max(np.abs(linear / truth - 1.0))
    assert err_loglog < 1e-8            # exact up to float / integrator noise
    assert err_linear > 1e-3            # linear cannot follow the power law
    assert err_linear > 100 * err_loglog


def test_interp_grid_none_ignores_interp_mode():
    """Without interp_grid the spectrum is integrated directly; interp is a
    no-op, so 'linear' and 'loglog' constructions agree exactly."""
    p = (-1.0, -0.7)
    f = np.geomspace(1e-3, 1e0, 20)
    a = np.array(_powerlaw_model(interp="linear")(jnp.array(f), *p))
    b = np.array(_powerlaw_model(interp="loglog")(jnp.array(f), *p))
    np.testing.assert_array_equal(a, b)


def test_interp_mode_validated():
    """An unknown interp mode is rejected at construction."""
    with pytest.raises(ValueError, match="interp must be"):
        _powerlaw_model(interp="log")


def test_interp_loglog_safe_when_res_hits_zero():
    """loglog must not produce nan/-inf when the un-normalised spectrum is
    exactly zero on some grid nodes (log(0) territory). A P_zeta with a sharp
    low-k cutoff zeroes the lowest interp_grid node; the output must stay
    finite and non-negative."""
    grid = np.geomspace(1e-3, 1e0, 12)
    # kcut chosen so every internal momentum u*k at the lowest node sits below
    # the cutoff (u_max = (t_max+s_max+1)/2 ~ 51 on the grid below) -> res == 0
    # there, while higher nodes keep support.
    kcut = 51.0 * (grid[0] * 2 * np.pi) * 1.5

    def pz(k, logA):
        return 10.0**logA * jnp.where(k > kcut, 1.0, 0.0)

    def build(**kw):
        return OmegaGW(
            AnalyticPerturbations(pz, ("logA",)),
            RadiationKernel(),
            s=jnp.linspace(0.0, 1.0, 17),
            t=jnp.geomspace(1e-2, 1e2, 200),
            **kw,
        )

    # confirm the guard is actually exercised: some grid nodes are exactly 0,
    # but not all (norm > 0, so Omega == 0 iff res == 0).
    direct = np.array(build()(jnp.array(grid), -1.0))
    assert np.any(direct == 0.0)
    assert np.any(direct > 0.0)

    f = np.geomspace(grid[0] * 1.05, grid[-1] * 0.95, 40)
    out = np.array(build(interp_grid=grid, interp="loglog")(jnp.array(f), -1.0))
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
