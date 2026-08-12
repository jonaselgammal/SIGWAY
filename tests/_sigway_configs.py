"""Single-source-of-truth definitions of the *paper* SIGW configurations.

These are the parameter sets that produced the published figures (extracted from
``insert_name_here/examples/template/*`` and the in-repo example notebooks
``examples/template_pipeline.ipynb`` / ``examples/sigw_pipeline.ipynb``).

Both the reference generator (``scripts/generate_reference_spectra.py``)
and the regression tests import this module, so a test can never silently use a
configuration that differs from the one used to generate its stored fixture.

For every analytic power spectrum we provide *two* implementations of the
identical closed form:

* a ``jax``/``jit`` version (``pzeta_*``) consumed by ``OmegaGW`` (via
  ``AnalyticPerturbations``, wired up in ``build_model``);
* a plain ``numpy`` factory (``pzeta_*_np``) consumed by the independent
  integration oracle in :mod:`_sigway_oracle`.

The names and underscore prefix keep pytest from collecting this file as tests.
"""

from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from sigway.spectrum import OmegaGW
from sigway.perturbations import AnalyticPerturbations
from sigway.kernels import RadiationKernel, InstantEMDKernel
from sigway.single_field import SingleFieldPerturbations

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Primordial power spectra  P_zeta(k)  (jax + numpy mirrors)
# ---------------------------------------------------------------------------
@jit
def pzeta_bpl(k, logA, alpha, beta, gamma, logks):
    """Broken power law (template_pipeline.ipynb)."""
    A = 10.0**logA
    ks = 10.0**logks
    return (
        A
        * (alpha + beta) ** gamma
        / (
            beta * (k / ks) ** (-alpha / gamma)
            + alpha * (k / ks) ** (beta / gamma)
        )
        ** gamma
    )


def pzeta_bpl_np(logA, alpha, beta, gamma, logks):
    A = 10.0**logA
    ks = 10.0**logks

    def P(k):
        return (
            A
            * (alpha + beta) ** gamma
            / (
                beta * (k / ks) ** (-alpha / gamma)
                + alpha * (k / ks) ** (beta / gamma)
            )
            ** gamma
        )

    return P


@jit
def pzeta_ln(k, logAs, logDelta, logks):
    """Log-normal peak (template/lognormal)."""
    As = 10.0**logAs
    Delta = 10.0**logDelta
    ks = 10.0**logks
    return (
        As
        / ((2 * jnp.pi) ** 0.5)
        / Delta
        * jnp.exp(-(1 / (2 * Delta**2)) * (jnp.log(k / ks)) ** 2)
    )


def pzeta_ln_np(logAs, logDelta, logks):
    As = 10.0**logAs
    Delta = 10.0**logDelta
    ks = 10.0**logks

    def P(k):
        return (
            As
            / ((2 * np.pi) ** 0.5)
            / Delta
            * np.exp(-(1 / (2 * Delta**2)) * (np.log(k / ks)) ** 2)
        )

    return P


@jit
def pzeta_heaviside(k, As, kmax, etaR):
    """Flat spectrum with a sharp cutoff at kmax (early_matter_domination)."""
    return jnp.heaviside(kmax - k, 1.0) * As


def pzeta_heaviside_np(As, kmax, etaR):
    def P(k):
        return np.heaviside(kmax - k, 1.0) * As

    return P


@jit
def pzeta_osc(k, log10A, log10ks, delta, eta_L, F):
    """Multifield oscillatory spectrum (oscillations_multifield, F=1)."""
    kappa = k / 10**log10ks
    P0 = 10**log10A
    p_env = (
        kappa
        * P0
        * jnp.exp(-2.0 * eta_L * delta)
        * jnp.exp(2.0 * jnp.sqrt((2.0 - kappa) * kappa) * eta_L * delta)
        / (4.0 * (2.0 - kappa) * kappa)
    )
    penv_kappa = p_env / kappa
    cos_term = jnp.cos(2.0 * jnp.exp(-delta / 2.0) * eta_L * kappa)
    sin_term = jnp.sqrt((2.0 - kappa) * kappa) * jnp.sin(
        2.0 * jnp.exp(-delta / 2.0) * eta_L * kappa
    )
    pz = (
        penv_kappa
        * (1.0 + (kappa - 1.0) * cos_term + sin_term)
        * jnp.heaviside(2.0 - kappa, 1.0)
    )
    p_env = jnp.nan_to_num(p_env, nan=0.0)
    pz = jnp.nan_to_num(pz, nan=0.0)
    return (1 - F) * p_env + F * pz


def pzeta_osc_np(log10A, log10ks, delta, eta_L, F):
    def P(k):
        kappa = k / 10**log10ks
        P0 = 10**log10A
        # avoid sqrt of negatives outside support; masked by heaviside anyway
        rad = np.clip((2.0 - kappa) * kappa, 0.0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_env = (
                kappa
                * P0
                * np.exp(-2.0 * eta_L * delta)
                * np.exp(2.0 * np.sqrt(rad) * eta_L * delta)
                / (4.0 * (2.0 - kappa) * kappa)
            )
            penv_kappa = p_env / kappa
            cos_term = np.cos(2.0 * np.exp(-delta / 2.0) * eta_L * kappa)
            sin_term = np.sqrt(rad) * np.sin(
                2.0 * np.exp(-delta / 2.0) * eta_L * kappa
            )
            pz = (
                penv_kappa
                * (1.0 + (kappa - 1.0) * cos_term + sin_term)
                * np.heaviside(2.0 - kappa, 1.0)
            )
        p_env = np.nan_to_num(p_env, nan=0.0)
        pz = np.nan_to_num(pz, nan=0.0)
        return (1 - F) * p_env + F * pz

    return P


# ---------------------------------------------------------------------------
# Parameter-dependent t integration grids (verbatim from the paper runs)
# ---------------------------------------------------------------------------
def t_bpl(k, logA, alpha, beta, gamma, logks):
    ks = 10.0**logks
    upper = jnp.min(
        jnp.array([10 ** (6.0 / beta) / (k / ks), 1e4 / (k / ks)]), axis=0
    )
    upper = jnp.where(upper > 1e5 * ks, 1e5 * ks, upper)
    t1 = jnp.linspace(1e-6 * jnp.ones_like(k), 0.999 * jnp.ones_like(k), 100)
    t2 = jnp.geomspace(jnp.ones_like(upper), upper, 300)
    return jnp.concatenate([t1, t2], axis=0)


def t_ln(k, logAs, logDelta, logks):
    ks = 10.0**logks
    Delta = 10.0**logDelta
    upper = jnp.exp(4 * Delta) * (2 * ks / k)
    upper = jnp.where(upper > 1e5 * ks, 1e5 * ks, upper)
    t1 = jnp.linspace(1e-5 * jnp.ones_like(k), 0.999 * jnp.ones_like(k), 200)
    t2 = jnp.geomspace(jnp.ones_like(upper), upper, 600)
    return jnp.concatenate([t1, t2], axis=0)


def t_emd(k, As, kmax, etaR):
    tmax = 2 * kmax / k
    return jnp.geomspace(1e-10 * jnp.ones_like(k), tmax, 100)


def t_osc(k, log10A, log10ks, delta, eta_L, F):
    ks = 10.0**log10ks
    upper = jnp.exp(5 * delta) * (2 * ks / k)
    upper = jnp.where(upper > 1e5 * ks, 1e5 * ks, upper)
    t1 = jnp.linspace(1e-5 * jnp.ones_like(k), 0.999 * jnp.ones_like(k), 100)
    t2 = jnp.geomspace(jnp.ones_like(upper), upper, 300)
    return jnp.concatenate([t1, t2], axis=0)


# ---------------------------------------------------------------------------
# Analytic-P_zeta configuration registry
# ---------------------------------------------------------------------------
# Each entry holds the P_zeta closed form (+ a numpy mirror for the oracle),
# the integration grids and fiducial params; build_model() turns it into an
# OmegaGW. The norm/kernel string fields are kept for the oracle's bookkeeping.
ANALYTIC_CONFIGS = {
    "bpl_rd": dict(
        pzeta=pzeta_bpl,
        pzeta_np=pzeta_bpl_np,
        t=t_bpl,
        s=np.linspace(0.0, 1.0, 10),
        f=np.geomspace(1e-5, 1.0, 200),
        params=(-2.71028683, 3.10673309, 0.22154827, 1.25417717, -1.58150632),
        norm="RD",
        kernel="RD",
        ks=10.0**-1.58150632,
        doc="Broken power law, radiation domination (template_pipeline.ipynb)",
    ),
    "lognormal_rd": dict(
        pzeta=pzeta_ln,
        pzeta_np=pzeta_ln_np,
        t=t_ln,
        s=np.linspace(0.0, 1.0, 10),
        f=np.geomspace(1e-5, 1e-1, 200),
        params=(-2.5, -0.3010299956639812, -2.0),
        norm="RD",
        kernel="RD",
        ks=10.0**-2.0,
        doc="Log-normal peak, radiation domination (template/lognormal)",
    ),
    "emd_imd2rd": dict(
        pzeta=pzeta_heaviside,
        pzeta_np=pzeta_heaviside_np,
        t=t_emd,
        s=np.linspace(0.0, 1.0, 100),
        f=np.geomspace(2.1e-9, 5e-2, 350),
        params=(2.1e-9, 0.06, 2000.0),
        norm="CT",
        kernel="I_MD_to_RD",
        ks=0.06,  # kmax acts as the characteristic scale
        doc="Flat+cutoff source, instant eMD->RD kernel (eMD)",
    ),
    "osc_multifield_rd": dict(
        pzeta=pzeta_osc,
        pzeta_np=pzeta_osc_np,
        t=t_osc,
        s=np.linspace(0.0, 1.0, 10),
        f=np.geomspace(1e-5, 1e-1, 200),
        params=(-1.5, -1.5, 0.5, 14.0, 1.0),
        norm="RD",
        kernel="RD",
        ks=10.0**-1.5,
        doc="Multifield oscillations, RD (template/oscillations_multifield)",
    ),
}


# ---------------------------------------------------------------------------
# Mukhanov-Sasaki (USR inflation) configuration  (sigw_pipeline.ipynb)
# ---------------------------------------------------------------------------
def usr_potential(phi, a, lam, v, nfac):
    """Quasi-inflection-point single-field potential used for the USR run."""
    b = (1 + nfac) * (1 - a**2 / 3 + a**2 / 3 * (9 / (2 * a**2) - 1) ** (2 / 3))
    f = phi / v
    return (
        lam
        * v**4
        / 12
        * f**2
        * (6 - 4 * a * f + 3 * f**2)
        / (1 + b * f**2) ** 2
    )


USR_CONFIG: dict[str, Any] = dict(
    phi0=3.0,
    pi0=0.0,
    N_CMB_to_end=58.0,
    k_solver=np.geomspace(1e-5, 10.0, 200),
    params=(
        0.7122360088466501,
        1.47312e-06,
        0.19688844475447168,
        1.8690166574800342e-05,
    ),
    f=np.geomspace(1e-5, 1.0, 200),
    s=np.linspace(0.0, 1.0, 10),
    kernel="RD",
    doc="Ultra-slow-roll single-field inflation via Mukhanov-Sasaki solver",
)


def usr_t_grid(nlo=200, nhi=800, nf=200):
    """Broadcast (nlo+nhi, nf) t-grid for the USR run.

    NOTE: the example notebook uses ``t = jnp.logspace(-3, 3, 1000)``, but pure
    log spacing starves the t<~1 region and leaves the integral ~13% above the
    converged value.  This linear-low-t + geometric-high-t grid (same total
    point count, 1000) reproduces the converged spectrum to <0.4% -- it is the
    spacing, not the count, that matters.  See test_convergence::test_usr_*.
    """
    t = jnp.concatenate(
        [jnp.linspace(1e-5, 0.999, nlo), jnp.geomspace(1.0, 1e3, nhi)]
    )
    return jnp.repeat(jnp.expand_dims(t, axis=-1), nf, axis=-1)


# ---------------------------------------------------------------------------
# Canonical model construction (the new OmegaGW composition API).
# Single source of truth shared by the regression / invariant / cross-backend
# / convergence tests so they all exercise the same construction.
# ---------------------------------------------------------------------------
_PZ_NAMES = {
    "bpl_rd": ("logA", "alpha", "beta", "gamma", "logks"),
    "lognormal_rd": ("logAs", "logDelta", "logks"),
    "osc_multifield_rd": ("log10A", "log10ks", "delta", "eta_L", "F"),
}


@jit
def pzeta_heaviside2(k, As, kmax):
    """Flat spectrum with a cutoff at kmax; kmax is a P_zeta (source) param."""
    return jnp.heaviside(kmax - k, 1.0) * As


def perturbations_for(name):
    """The ScalarPerturbations object for a config (analytic or MS)."""
    if name == "usr_ms":
        cfg = USR_CONFIG
        return SingleFieldPerturbations(
            usr_potential,
            ("a", "lam", "v", "nfac"),
            phi0=cfg["phi0"],
            N_CMB_to_end=cfg["N_CMB_to_end"],
        )
    if name == "emd_imd2rd":
        return AnalyticPerturbations(
            pzeta_heaviside2, ("As", "kmax"), nonsmooth_params=("kmax",)
        )
    cfg = ANALYTIC_CONFIGS[name]
    return AnalyticPerturbations(cfg["pzeta"], _PZ_NAMES[name])


def kernel_for(name):
    """The Kernel object for a config (carries its own normalisation)."""
    if name == "emd_imd2rd":
        return InstantEMDKernel()
    return RadiationKernel()


def build_model(name):
    """Build the canonical OmegaGW model for a named config."""
    if name == "usr_ms":
        cfg = USR_CONFIG
        return OmegaGW(
            perturbations_for(name),
            kernel_for(name),
            s=jnp.array(cfg["s"]),
            t=usr_t_grid(nf=len(cfg["f"])),
            interp_grid=jnp.array(cfg["f"]),
        )
    cfg = ANALYTIC_CONFIGS[name]
    return OmegaGW(
        perturbations_for(name),
        kernel_for(name),
        s=jnp.array(cfg["s"]),
        t=cfg["t"],
        interp_grid=jnp.array(cfg["f"]),
    )
