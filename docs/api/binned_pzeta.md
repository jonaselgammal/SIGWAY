# sigway.binned_pzeta

A **model-independent** spectrum: $\mathcal{P}_\zeta(k)$ is represented as free
$\log_{10}$ amplitudes in $k$-bins. `Binned_P_zeta` returns $\Omega_{\mathrm{GW}}$ directly
from precomputed coefficients (no kernel, no $(s,t)$ integral) but exposes the same
interface as [OmegaGW][sigway.spectrum.OmegaGW].

## Model

::: sigway.binned_pzeta.Binned_P_zeta
    options:
      heading_level: 3

## Helper functions

::: sigway.binned_pzeta.compute_omega_gw
    options:
      heading_level: 3
::: sigway.binned_pzeta.compute_domega_gw
    options:
      heading_level: 3
::: sigway.binned_pzeta.upsample_f
    options:
      heading_level: 3
::: sigway.binned_pzeta.upsample_f_binned
    options:
      heading_level: 3
::: sigway.binned_pzeta.upsample_f_linear
    options:
      heading_level: 3
