# sigway.single_field

Solve a model of inflation to get the primordial spectrum $\mathcal{P}_\zeta(k)$, rather
than writing it down by hand.

## The Mukhanov–Sasaki equation

During inflation the comoving curvature perturbation $\zeta$ is most cleanly evolved
through the gauge-invariant **Mukhanov–Sasaki (MS) variable** $v_k = z\,\zeta_k$, where
$z = a\,\dot\phi/H$. Each Fourier mode obeys a forced-oscillator equation,

$$
\frac{\mathrm{d}^2 v_k}{\mathrm{d}\eta^2}
+ \left(k^2 - \frac{1}{z}\frac{\mathrm{d}^2 z}{\mathrm{d}\eta^2}\right) v_k = 0 ,
$$

with $\eta$ the conformal time. Deep inside the horizon ($k \gg aH$) the $k^2$ term
dominates and the mode oscillates like a flat-space (Bunch–Davies) vacuum; as the mode
crosses the horizon the $z''/z$ term takes over and $\zeta_k$ **freezes** to a constant.
The curvature power spectrum is read off from the frozen amplitude,

$$
\mathcal{P}_\zeta(k) = \frac{k^3}{2\pi^2}\,\bigl|\zeta_k\bigr|^2
= \frac{k^3}{2\pi^2}\,\left|\frac{v_k}{z}\right|^2 .
$$

## How `SingleFieldPerturbations` works

For a given potential $V(\phi)$ it proceeds in three steps (the code integrates in e-folds
$N=\ln a$ rather than conformal time):

1. **Background** — integrate the inflaton equation of motion together with the Friedmann
   equation to obtain $\phi(N)$ and $H(N)$, and hence $z(N)$.
2. **Modes** — for each comoving wavenumber $k$, evolve the MS equation from a sub-horizon
   Bunch–Davies initial condition (a few e-folds before horizon crossing) through to well
   after horizon crossing, where $\zeta_k$ has frozen.
3. **Spectrum** — evaluate $\mathcal{P}_\zeta(k)$ from the frozen modes.

This turns a *model of inflation* — e.g. an ultra-slow-roll quasi-inflection-point
potential — directly into the small-scale $\mathcal{P}_\zeta$ that sources scalar-induced
GWs.
[SingleFieldPerturbations][sigway.single_field.SingleFieldPerturbations] exposes the
standard [ScalarPerturbations][sigway.perturbations.ScalarPerturbations] interface, so you
can feed it straight to an [OmegaGW][sigway.spectrum.OmegaGW] model.

## Solver

::: sigway.single_field.SingleFieldPerturbations
    options:
      heading_level: 3

## Options & errors

::: sigway.single_field.SolverOptions
    options:
      heading_level: 3
::: sigway.single_field.ConsistencyError
    options:
      heading_level: 3
