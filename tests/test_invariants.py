"""Physics invariants and analytic limits of the public Omega_GW(f) surface.

Each test encodes an expectation that holds independently of sigway's internals
(a scaling law, a resonance location, a causal slope, a source-support cutoff),
so they survive the planned refactor and fail on real physics regressions.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from sigway.omega_gw_jax import OmegaGWjax
import _sigway_configs as C


def _model(cfg, f, s=None):
    return OmegaGWjax(
        cfg["pzeta"], jnp.array(s if s is not None else cfg["s"]), cfg["t"],
        f=jnp.array(f), norm=cfg["norm"], kernel=cfg["kernel"],
        upsample=True, dP_zeta="auto", jit=True,
    )


@pytest.mark.parametrize(
    "name,kind", [
        ("bpl_rd", "log"),
        ("lognormal_rd", "log"),
        ("osc_multifield_rd", "log"),
        ("emd_imd2rd", "lin"),
    ]
)
def test_amplitude_bilinearity(name, kind):
    """Omega_GW is bilinear in P_zeta amplitude: A -> lambda*A gives lambda^2.

    The integrand is kernel*poly*P_zeta(ku)*P_zeta(kv), and the paper P_zeta
    forms are linear in their amplitude, so this is algebraically exact -- the
    rtol 1e-10 catches any change that breaks the quadratic-in-source structure.
    """
    cfg = C.ANALYTIC_CONFIGS[name]
    m = _model(cfg, cfg["f"])
    p = list(cfg["params"])
    base = np.array(m(jnp.array(cfg["f"]), *p))
    lam = 3.0
    if kind == "log":
        p[0] = p[0] + np.log10(lam)
    else:
        p[0] = p[0] * lam
    scaled = np.array(m(jnp.array(cfg["f"]), *p))
    good = base > np.nanmax(base) * 1e-8
    assert np.nanmax(np.abs(scaled[good] / base[good] / lam**2 - 1.0)) < 1e-10


def test_lognormal_peak_at_resonance():
    """A narrow log-normal peaks at the resonance k = 2/sqrt(3) k_*.

    For a Dirac source the RD spectrum diverges at k = 2 k_*/sqrt(3).
    A finite (narrow) log-normal peaks just below it; require within 4e-3
    which we tested by hand to be fine for a peak with Delta = 1e-2 or sharper.
    """
    cfg = C.ANALYTIC_CONFIGS["lognormal_rd"]
    p = (-2.0, -2.0, -2.0)  # narrow: logDelta = -1  (Delta = 0.1)
    ks = 10.0 ** p[2]
    kpk_expected = 2.0 / np.sqrt(3.0) * ks
    f = np.geomspace(0.6 * kpk_expected, 1.4 * kpk_expected, 300) / (2 * np.pi)
    m = _model(cfg, f, s=np.linspace(0.0, 1.0, 40))
    og = np.array(m(jnp.array(f), *p))
    k_peak = f[np.argmax(og)] * 2 * np.pi
    assert abs(k_peak / kpk_expected - 1.0) < 4e-3


def test_lognormal_ir_causal_tail():
    """Deep-IR tail follows the causal law Omega ~ k^3 ln^2(k_*/k) for pure RD.

    The local log-log slope is therefore < 3 and *increases* toward 3 as k->0
    (slope = 3 - 2/ln(k_*/k)); a pure power law (or a wrong index like 2 or 4)
    would give a constant/incorrect slope. We require the deep slope in (2.2, 3)
    and steeper deeper-in (monotone toward 3).
    """
    cfg = C.ANALYTIC_CONFIGS["lognormal_rd"]
    p = (-2.0, -0.5, -2.0)
    ks = 10.0 ** p[2]

    # Custom uncapped t-grid so the deep-IR source (peak at t ~ 2 k_*/k) is
    # captured; the shipped t_ln caps t at 1e5*k_*, which truncates k << k_*.
    def t_deep(k, logAs, logDelta, logks):
        D = 10.0 ** logDelta
        upper = jnp.exp(4 * D) * (8 * ks / k)
        one = jnp.ones_like(k)
        t1 = jnp.linspace(1e-5 * one, 0.999 * one, 200)
        t2 = jnp.geomspace(jnp.ones_like(upper), upper, 900)
        return jnp.concatenate([t1, t2], axis=0)

    f = np.geomspace(1e-7, 0.2 * ks / (2 * np.pi), 60)
    m = OmegaGWjax(
        cfg["pzeta"], jnp.linspace(0.0, 1.0, 20), t_deep, f=jnp.array(f),
        norm="RD", kernel="RD", upsample=True, dP_zeta="auto", jit=True,
    )
    og = np.array(m(jnp.array(f), *p))
    k = f * 2 * np.pi
    ok = (og > 0) & np.isfinite(og)
    lk, lo = np.log(k[ok]), np.log(og[ok])
    n = len(lk)
    slope_deep = np.polyfit(lk[: n // 2], lo[: n // 2], 1)[0]
    slope_shallow = np.polyfit(lk[n // 2:], lo[n // 2:], 1)[0]
    assert 2.2 < slope_deep < 3.0          # causal tail with log enhancement
    assert slope_deep > slope_shallow      # steepens toward 3 as k -> 0


def test_emd_source_cutoff():
    """Flat+cutoff source (k<kmax) yields a GW spectrum suppressed above ~kmax.

    The induced spectrum has support up to 2 kmax but, for this flat source,
    falls steeply past the peak; require Omega(2 kmax)/Omega_peak < 1e-7.
    Basically after that the spectrum should be more or less exact 0.
    """
    cfg = C.ANALYTIC_CONFIGS["emd_imd2rd"]
    As, kmax, etaR = cfg["params"]
    m = _model(cfg, cfg["f"])
    og = np.array(m(jnp.array(cfg["f"]), *cfg["params"]))
    peak = np.nanmax(og)
    f_2kmax = 2 * kmax / (2 * np.pi)
    assert np.interp(f_2kmax, cfg["f"], og) / peak < 1e-7
