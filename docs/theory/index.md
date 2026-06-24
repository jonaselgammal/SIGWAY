# Theory & modelling

A compact review of the physics SIGWAY implements, and how the equations map onto the
code. Conventions follow [arXiv:2501.11320](https://arxiv.org/abs/2501.11320); see it for
full derivations and the explicit kernel expressions.

## Scalar-induced gravitational waves

Cosmological perturbation theory decouples scalar, vector and tensor modes only at
**linear** order. At **second order** the Einstein equations couple them: a pair of
first-order scalar (curvature) perturbations sources a tensor mode. If the primordial
curvature power $\mathcal{P}_\zeta$ is strongly enhanced on some small scale — say by a
feature in inflation — the tensor modes induced when those scales re-enter the horizon
form a sizeable **stochastic gravitational-wave background** today. Because the source is
quadratic in the scalars, the signal is **second order in $\mathcal{P}_\zeta$** and peaks
at a frequency set by the enhancement scale.

Splitting the metric tensor perturbation into Fourier modes $h_{\mathbf k}$, the
transverse-traceless part obeys a forced oscillator equation,

$$
h_{\mathbf k}'' + 2\mathcal{H}\,h_{\mathbf k}' + k^2 h_{\mathbf k} = 4\,S_{\mathbf k},
$$

where $' = \partial_\eta$, $\mathcal H = a'/a$, and the source $S_{\mathbf k}$ is a
convolution **quadratic in the first-order scalar potential** $\Phi$ (itself fixed by
$\zeta$ and an era-dependent transfer function). Solving with the Green's function and
projecting onto the GW energy density gives the result below.

## The master formula

Averaging over the sub-horizon oscillations, the induced **tensor power spectrum** is a
double integral over two internal momenta, $u=|\mathbf k-\mathbf q|/k$ and $v=q/k$:

$$
\overline{\mathcal{P}_h(\eta,k)} =
4\int_0^{\infty}\!\mathrm{d}v \int_{|1-v|}^{1+v}\!\mathrm{d}u\;
\left[\frac{4v^2-\left(1+v^2-u^2\right)^2}{4uv}\right]^{2}
\overline{I^{2}(u,v,k\eta)}\;
\mathcal{P}_\zeta(uk)\,\mathcal{P}_\zeta(vk).
$$

The square bracket is the geometric (polarisation) factor; $\overline{I^2}$ is the
**kernel** — the time-integrated transfer function for the source era (next section). The
GW energy-density fraction follows from

$$
\Omega_{\mathrm{GW}}(k,\eta)=\frac{1}{24}\left(\frac{k}{\mathcal H(\eta)}\right)^{2}
\overline{\mathcal{P}_h(\eta,k)},
$$

which during radiation domination becomes time-independent (the source decays). Redshifted
to today,

$$
\Omega_{\mathrm{GW},0}(k)\,h^2 = c_g\,\Omega_{r,0}h^2\,\Omega_{\mathrm{GW}}(k),
\qquad
c_g=\frac{g_*(T)}{g_*^{0}}\left(\frac{g_{*s}^{0}}{g_{*s}(T)}\right)^{4/3},
$$

with $\Omega_{r,0}h^2\approx4.2\times10^{-5}$. These constants are part of the
normalisation carried by the kernel object in the code.

### In the code's $(s,t)$ variables

`SimpsonIntegrator` does not integrate over $(u,v)$ directly. Inverting Eq. (4.19) of the
paper, it substitutes

$$u=\frac{t+s+1}{2},\qquad v=\frac{t-s+1}{2},$$

which maps the triangular $(u,v)$ domain onto a **rectangle**, $t\in[0,\infty)$,
$s\in[0,1]$, with Jacobian $\mathrm{d}u\,\mathrm{d}v=\tfrac12\,\mathrm{d}s\,\mathrm{d}t$. The
polarisation/geometric factor becomes the rational "polynomial" of Eq. (4.20),

$$\mathcal{N}(t,s)=2\left[\frac{t\,(2+t)\,(s^2-1)}{(1-s+t)\,(1+s+t)}\right]^2,$$

so the quantity SIGWAY actually evaluates is the **2-D Simpson sum**

$$\overline{\mathcal{P}_h(k)} = \int_0^{\infty}\!\mathrm{d}t \int_0^{1}\!\mathrm{d}s\;
\mathcal{N}(t,s)\,\overline{I^2(u,v,k\eta)}\;
\mathcal{P}_\zeta(uk)\,\mathcal{P}_\zeta(vk),$$

over exactly the `s` and `t` grids you hand to the model — which is why grid choice (next
section, and the [components tour](../examples/advanced_example.md)) matters.

## The kernel

$\overline{I^2(u,v,x)}$ encodes how efficiently a scalar configuration $(u,v)$ sources GWs
in a given expansion history.

- **Radiation domination** (`RadiationKernel`): in the late-time limit $\overline{I^2}$ has
  a **closed form**, $k$-independent, with a logarithmic term and a **resonant**
  contribution (a $\pi^2$ piece switched on for $u+v>\sqrt3$, where the scalar sound speed
  matches the GW phase velocity).
- **Instant eMD → radiation** (`InstantEMDKernel`): an early matter-dominated era ending in
  a *sudden* transition gives a different, **$k$-dependent** kernel (through
  $x_R=k\,\eta_R$), with a smooth large-momentum part and a separate resonant slice. The
  transition time $\eta_R$ becomes a model parameter.

## Beyond the Gaussian approximation

The master formula above is the **Gaussian** result. To see why, note that the GW
two-point function is sourced by a **four-point function** of $\zeta$ (since
$h\sim\zeta^2$):

$$
\langle h\,h\rangle \;\sim\; \langle \zeta^2\,\zeta^2\rangle
=\underbrace{\langle\zeta\zeta\rangle\langle\zeta\zeta\rangle}_{\text{disconnected}}
\;+\;\underbrace{\langle\zeta\zeta\zeta\zeta\rangle_{c}}_{\text{connected}} .
$$

If $\zeta$ is Gaussian, Wick's theorem keeps only the **disconnected** part, which is
exactly the $\mathcal{P}_\zeta(uk)\,\mathcal{P}_\zeta(vk)$ product in the double integral.
**Primordial non-Gaussianity** adds the **connected** piece — the curvature
**trispectrum** $T_\zeta$, and (through the quadratic source) terms built from the
**bispectrum** $B_\zeta$:

$$
\Omega_{\mathrm{GW}} = \Omega_{\mathrm{GW}}^{(G)}\!\left[\mathcal{P}_\zeta^2\right]
\;+\;\Omega_{\mathrm{GW}}^{(\mathrm{NG})}\!\left[B_\zeta,\,T_\zeta\right].
$$

For local-type non-Gaussianity,
$\zeta=\zeta_g+\tfrac{3}{5}f_{\mathrm{NL}}\!\left(\zeta_g^2-\langle\zeta_g^2\rangle\right)+\dots$,
the extra contributions scale as $f_{\mathrm{NL}}^2$ (plus Gaussian–NG cross terms) and can
**dominate** the signal when the enhancement is itself produced non-linearly. SIGWAY
currently evaluates the **Gaussian** piece; the non-Gaussian contributions are an active
research direction and a target of the extensions.

This perturbative organisation (order by order in $B_\zeta$, $T_\zeta$, …) becomes unwieldy
when $f_{\mathrm{NL}}$ is large and many $n$-point functions contribute. A complementary,
**fully non-perturbative** route works on a *lattice*: Piccoli (2026),
[*A Fast Method to Compute Scalar Induced Gravitational Waves on a Lattice with Primordial
Non-Gaussianities*](https://arxiv.org/abs/2605.27231), generates real-space realisations of
$\zeta$ with **arbitrary statistics** and measures $\Omega_{\mathrm{GW}}$ directly from them
— so it is agnostic to the form (local, non-local, …) and captures non-Gaussianity **at all
orders** at once. The cost is tamed by decomposing the non-separable source kernel into
$\sim\!50$ *separable* terms via an eigenmode decomposition, each evaluated with FFT
convolutions; this drops the scaling from $\mathcal{O}(N^6)$ direct integration to
$\mathcal{O}(N_\alpha N^3 \log N)$, yielding non-Gaussian SIGW spectra in seconds on a GPU
(public code: `FLAN-SIGW`). It is the natural cross-check of the perturbative result when
non-Gaussianity is strong.

## From the formulas to the code

| In the formula | In SIGWAY |
|---|---|
| $\mathcal{P}_\zeta(k)$ | the **perturbations** object (`AnalyticPerturbations`, `SingleFieldPerturbations`, `Binned_P_zeta`) |
| $\overline{I^2}$ and the normalisation | the **kernel** (`RadiationKernel`, `InstantEMDKernel`) |
| the $\mathrm{d}u\,\mathrm{d}v$ double integral | the **integrator** (`SimpsonIntegrator`) over rescaled variables $(s,t)$ |

The code integrates over rescaled momenta $s\in[0,1]$ and $t\in[0,\infty)$ — an affine map
of the triangular $(u,v)$ domain onto a rectangle (the exact substitution is in the paper).
A few numerical points worth keeping in mind:

- **float64 throughout** — the integrand spans many decades.
- **$t$-grid spacing** dominates accuracy: *linear below 1, geometric above*. The grid may
  be a callable $t(k,\theta)$ so its range tracks the source (see the
  [components tour](../examples/advanced_example.ipynb)).
- **Differentiability** — being JAX, `model.jacobian` returns
  $\partial\Omega_{\mathrm{GW}}/\partial\theta$ for Fisher forecasts (finite differences for
  non-smooth parameters such as a cutoff).
