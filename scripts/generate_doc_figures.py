"""Generate light- and dark-mode, transparent SVG figures for the docs example
pages. Run with the env python:

    python scripts/generate_doc_figures.py

Figures land in docs/assets/images/figures/{name}_{light,dark}.svg and are embedded
in the markdown example pages with Material's #only-light / #only-dark suffixes.
"""
import os, warnings
warnings.simplefilter("ignore")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sigway.spectrum import OmegaGW
from sigway.kernels import RadiationKernel, InstantEMDKernel
from sigway.perturbations import AnalyticPerturbations
from sigway.single_field import SingleFieldPerturbations
from sigway.binned_pzeta import Binned_P_zeta

OUT = "docs/assets/images/figures"
os.makedirs(OUT, exist_ok=True)

THEMES = {
    "light": dict(fg="#20305f", cycle=["#2a6c9c", "#c23047", "#20305f"]),
    "dark":  dict(fg="#c7cde4", cycle=["#6aa8f0", "#ec7a89", "#c7cde4"]),
}

def render(plotfn, base):
    for theme, cfg in THEMES.items():
        rc = {
            "figure.facecolor": "none", "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "axes.edgecolor": cfg["fg"], "axes.labelcolor": cfg["fg"],
            "xtick.color": cfg["fg"], "ytick.color": cfg["fg"],
            "text.color": cfg["fg"],
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=cfg["cycle"]),
            "font.size": 12,
        }
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(6.6, 3.8))
            plotfn(ax)
            fig.savefig(f"{OUT}/{base}_{theme}.svg", transparent=True,
                        bbox_inches="tight")
            plt.close(fig)
    print("rendered", base)

# ---- simple example data ----
def pzeta_lognormal(k, logAs, logDelta, logks):
    As, Delta, ks = 10.0**logAs, 10.0**logDelta, 10.0**logks
    return As/(jnp.sqrt(2*jnp.pi)*Delta) * jnp.exp(-0.5/Delta**2*jnp.log(k/ks)**2)

params = (-2.5, -0.30103, -2.0)
k = jnp.geomspace(1e-4, 1e0, 400)
pz = pzeta_lognormal(k, *params)

s = jnp.linspace(0.0, 1.0, 10)
t = jnp.concatenate([jnp.linspace(1e-5, 0.999, 200), jnp.geomspace(1.0, 1e3, 800)])
model = OmegaGW(AnalyticPerturbations(pzeta_lognormal,
                ("logAs", "logDelta", "logks")), RadiationKernel(), s=s, t=t)
f = jnp.geomspace(1e-5, 1e-1, 200)
omega = model(f, *params)

render(lambda ax: (ax.loglog(k, pz), ax.set_xlabel("$k$"),
                   ax.set_ylabel(r"$\mathcal{P}_\zeta(k)$")), "simple_pzeta")
render(lambda ax: (ax.loglog(f, omega), ax.set_xlabel("$f$ [Hz]"),
                   ax.set_ylabel(r"$\Omega_{\mathrm{GW}}\,h^2$")), "simple_omega")

# ---- advanced: single-field P_zeta ----
def usr_potential(phi, a, lam, v, nfac):
    b = (1 + nfac) * (1 - a**2/3 + a**2/3 * (9/(2*a**2) - 1)**(2/3))
    x = phi/v
    return lam*v**4/12 * x**2 * (6 - 4*a*x + 3*x**2) / (1 + b*x**2)**2

usr_pert = SingleFieldPerturbations(usr_potential, ("a", "lam", "v", "nfac"),
                                    phi0=3.0, N_CMB_to_end=58.0)
usr_params = (0.71224, 1.47312e-06, 0.19689, 1.86902e-05)
kk = jnp.geomspace(1e-5, 10.0, 200)
pz_usr = usr_pert(kk, *usr_params)
render(lambda ax: (ax.loglog(kk, pz_usr), ax.set_xlabel("$k$"),
                   ax.set_ylabel(r"$\mathcal{P}_\zeta(k)$")), "adv_pzeta_usr")

# ---- advanced: binned ----
binned = Binned_P_zeta("binned", "Binned", nbins=100)
amps = [-4.0]*len(binned.parameter_names); amps[50] = 0.0
f_b = jnp.geomspace(2e-5, 1.0, 80)
om_b = binned(f_b, *amps)
render(lambda ax: (ax.loglog(f_b, om_b), ax.set_ylim(1e-14, 1e-6),
                   ax.set_xlabel("$f$ [Hz]"),
                   ax.set_ylabel(r"$\Omega_{\mathrm{GW}}\,h^2$")), "adv_binned")

# ---- advanced: eMD ----
def pzeta_flat_cutoff(k, As, kmax):
    return jnp.heaviside(kmax - k, 1.0) * As
def t_emd(k, As, kmax, etaR):
    return jnp.geomspace(1e-10*jnp.ones_like(k), 2*kmax/k, 100)
emd_model = OmegaGW(AnalyticPerturbations(pzeta_flat_cutoff, ("As", "kmax"),
                    nonsmooth_params=("kmax",)), InstantEMDKernel(),
                    s=jnp.linspace(0.0, 1.0, 100), t=t_emd)
f_emd = jnp.geomspace(2.1e-9, 5e-2, 350)
om_emd = emd_model(f_emd, As=2.1e-9, kmax=0.06, etaR=2000.0)
render(lambda ax: (ax.loglog(f_emd, om_emd), ax.set_xlabel("$f$ [Hz]"),
                   ax.set_ylabel(r"$\Omega_{\mathrm{GW}}\,h^2$")), "adv_emd")

# ---- advanced: sensitivities ----
jac = model.jacobian(f, list(params))
def plot_sens(ax):
    for i, name in enumerate(model.parameter_names):
        ax.semilogx(f, jac[:, i]/omega, label=name)
    ax.axhline(0, lw=0.6, color="#8a8fa3")
    ax.set_xlabel("$f$ [Hz]")
    ax.set_ylabel(r"$\partial \ln \Omega_{\mathrm{GW}} / \partial\theta$")
    ax.legend(frameon=False)
render(plot_sens, "adv_sensitivity")
print("done")
