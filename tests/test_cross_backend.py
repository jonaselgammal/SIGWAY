"""Cross-backend / cross-representation agreement.

* sigway's fixed-grid jax Simpson integrator vs the independent numpy/scipy
  oracle (adaptive quadrature / dense Simpson, textbook kernels) -- agreement
  within the documented integration error proves the result is right, not just
  reproducible.
* OmegaGWms (P_zeta from the Mukhanov-Sasaki solver) vs OmegaGWjax fed the same
  P_zeta interpolant on the same grids -- checks the MS P_zeta plumbing against
  the generic integrator to tight tolerance.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from sigway.omega_gw_jax import OmegaGWjax, get_u, get_v
from sigway.ms_solver import SingleFieldSolver
from sigway.omega_gw_ms import OmegaGWms
import _sigway_configs as C
import _sigway_oracle as oracle


def _sig_rd(name):
    cfg = C.ANALYTIC_CONFIGS[name]
    m = OmegaGWjax(
        cfg["pzeta"], jnp.array(cfg["s"]), cfg["t"], f=jnp.array(cfg["f"]),
        norm=cfg["norm"], kernel=cfg["kernel"], upsample=True,
        dP_zeta="auto", jit=True,
    )
    return cfg, np.array(m(jnp.array(cfg["f"]), *cfg["params"]))


@pytest.mark.parametrize(
    "name,tol,kappa_lo,kappa_hi",
    [
        ("lognormal_rd", 0.03, 0.05, 1.6),
        ("bpl_rd", 0.05, 0.05, 30.0),
    ],
)
def test_rd_spectrum_vs_oracle(name, tol, kappa_lo, kappa_hi):
    """SIGWAY RD spectrum agrees with the independent numpy oracle in-band."""
    cfg, og = _sig_rd(name)
    Pz = cfg["pzeta_np"](*cfg["params"])
    ks = cfg["ks"]
    k = cfg["f"] * 2 * np.pi
    band = (k > kappa_lo * ks) & (k < kappa_hi * ks) & (og > og.max() * 1e-3)
    fb = cfg["f"][band]
    for f0 in fb[np.unique(np.linspace(0, len(fb) - 1, 6).astype(int))]:
        kk = f0 * 2 * np.pi
        tmax = float(np.array(cfg["t"](jnp.array([kk]), *cfg["params"])).max())
        orc = oracle.omega_RD_oracle(
            kk, Pz, cfg["norm"], t_max=tmax, ns=300, nt=5000)
        assert abs(float(np.interp(f0, cfg["f"], og)) / orc - 1.0) < tol


def test_emd_spectrum_vs_oracle():
    """eMD->RD spectrum agrees with the adaptive-quadrature oracle in the
    well-resolved 0.5-1.0 kmax band (the IR tail is grid-limited; see
    test_convergence)."""
    cfg, og = _sig_rd("emd_imd2rd")
    As, kmax, etaR = cfg["params"]
    Pz = cfg["pzeta_np"](*cfg["params"])
    for r in (0.5, 0.6, 0.7, 0.85, 0.95, 1.0):
        f0 = r * kmax / (2 * np.pi)
        orc = oracle.omega_eMD_oracle(r * kmax, Pz, kmax, etaR, cfg["norm"])
        assert abs(float(np.interp(f0, cfg["f"], og)) / orc - 1.0) < 0.05


def test_omega_gw_ms_matches_jax_same_pzeta():
    """OmegaGWms == OmegaGWjax fed the identical MS P_zeta on identical grids.

    Isolates the MS-specific P_zeta extraction: both integrate the same source
    over the same (s, t, f), so they must agree to integration round-off.
    """
    cfg = C.USR_CONFIG
    p = cfg["params"]
    solver = SingleFieldSolver(
        C.usr_potential, phi0=cfg["phi0"], pi0=cfg["pi0"],
        N_CMB_to_end=cfg["N_CMB_to_end"], k=jnp.array(cfg["k_solver"]),
    )
    s = jnp.array(cfg["s"])
    t = C.usr_t_grid(nf=len(cfg["f"]))
    integ_ms = OmegaGWms(solver, s, t, f=jnp.array(cfg["f"]), kernel="RD",
                         upsample=True)
    og_ms = np.array(integ_ms(jnp.array(cfg["f"]), *p))

    # Build the same P_zeta interpolant OmegaGWms uses internally (kint spans
    # min..max of k*u, k*v) and feed it to the generic OmegaGWjax integrator.
    kvec = jnp.array(cfg["f"]) * 2 * jnp.pi
    uv = jnp.array([get_u(t[None, :, :], s[:, None, None]),
                    get_v(t[None, :, :], s[:, None, None])])
    kint = jnp.geomspace(float(jnp.min(kvec) * jnp.min(uv)),
                         float(jnp.max(kvec) * jnp.max(uv)), 100)
    pzc = solver.run(kint, *p)

    def pz_wrap(k, *params):
        return pzc(k)

    integ_jax = OmegaGWjax(pz_wrap, s, t, f=jnp.array(cfg["f"]), norm="RD",
                           kernel="RD", upsample=True, jit=False)
    og_jax = np.array(integ_jax(jnp.array(cfg["f"]), *p))
    np.testing.assert_allclose(og_ms, og_jax, rtol=1e-4,
                               atol=np.nanmax(og_ms) * 1e-10)
