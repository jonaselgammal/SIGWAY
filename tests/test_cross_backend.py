"""Cross-backend / cross-representation agreement (through OmegaGW).

* OmegaGW's fixed-grid jax Simpson integrator vs the independent numpy/scipy
  oracle (adaptive quadrature / dense Simpson, textbook kernels) -- agreement
  within the documented integration error proves the result is right, not just
  reproducible.
* The MS path (SingleFieldPerturbations, run eagerly) vs the same MS P_zeta
  interpolant fed as an AnalyticPerturbations (the jit path) -- checks the
  MS-specific prepare/eager plumbing against the generic integrator.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from sigway.spectrum import OmegaGW
from sigway.kernels import RadiationKernel, get_u, get_v
from sigway.perturbations import AnalyticPerturbations
from sigway.single_field import SingleFieldPerturbations
import _sigway_configs as C
import _sigway_oracle as oracle


def _sig_rd(name):
    cfg = C.ANALYTIC_CONFIGS[name]
    m = C.build_model(name)
    return cfg, np.array(m(jnp.array(cfg["f"]), *cfg["params"]))


@pytest.mark.parametrize(
    "name,tol,kappa_lo,kappa_hi",
    [
        ("lognormal_rd", 0.03, 0.05, 1.6),
        ("bpl_rd", 0.05, 0.05, 30.0),
    ],
)
def test_rd_spectrum_vs_oracle(name, tol, kappa_lo, kappa_hi):
    """OmegaGW RD spectrum agrees with the independent numpy oracle in-band."""
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
            kk, Pz, cfg["norm"], t_max=tmax, ns=300, nt=5000
        )
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


def test_ms_path_matches_same_pzeta_analytic():
    """MS path (eager) == the same MS P_zeta interpolant fed analytically (jit).

    Isolates the MS-specific prepare/eager plumbing: both integrate the same
    source over the same (s, t, f), so they must agree to integration round-off.
    """
    cfg = C.USR_CONFIG
    p = cfg["params"]
    names = ("a", "lam", "v", "nfac")
    ms_pert = SingleFieldPerturbations(
        C.usr_potential,
        names,
        phi0=cfg["phi0"],
        N_CMB_to_end=cfg["N_CMB_to_end"],
    )
    s = jnp.array(cfg["s"])
    t = C.usr_t_grid(nf=len(cfg["f"]))
    f = jnp.array(cfg["f"])

    ms_model = OmegaGW(
        ms_pert,
        RadiationKernel(),
        s=s,
        t=t,
        f=f,
        upsample=True,
    )
    og_ms = np.array(ms_model(f, *p))

    # same interpolant the integrator builds internally (kint over k*u, k*v)
    uv = jnp.array(
        [
            get_u(t[None, :, :], s[:, None, None]),
            get_v(t[None, :, :], s[:, None, None]),
        ]
    )
    kint = jnp.geomspace(
        float(jnp.min(f * 2 * jnp.pi) * jnp.min(uv)),
        float(jnp.max(f * 2 * jnp.pi) * jnp.max(uv)),
        100,
    )
    pzc = ms_pert.prepare(kint, *p)
    analytic = OmegaGW(
        AnalyticPerturbations(lambda k, *q: pzc(k), names),
        RadiationKernel(),
        s=s,
        t=t,
        f=f,
        upsample=True,
    )
    og_an = np.array(analytic(f, *p))
    np.testing.assert_allclose(
        og_ms, og_an, rtol=1e-4, atol=np.nanmax(og_ms) * 1e-10
    )
