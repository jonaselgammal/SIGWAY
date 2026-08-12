r"""Kernel hierarchy for the scalar-induced GW transfer function.

In the scalar-induced GW formalism, the energy-density spectrum
$\Omega_{\rm GW}(k)$ receives contributions from pairs of scalar (curvature)
modes with wavenumbers $p$ and $q$ that combine into a tensor mode at wavenumber
$k$. After averaging over the sub-horizon oscillations, each pair is weighted by
the **kernel** $\overline{I^2}(t, s, k)$ — the time-integrated (squared)
transfer function — which measures how efficiently that pair sources
gravitational waves during a given cosmological expansion era.

The integral is carried out in the rescaled momenta $s$ and $t$, related to the
momentum ratios $u = p/k$ and $v = q/k$ by $u=(t+s+1)/2$ and $v=(t-s+1)/2$
(inverting Eq. (4.19) of
[arXiv:2501.11320](https://arxiv.org/abs/2501.11320)). See
[Theory: the kernel](../theory/index.md#the-kernel).

This package provides the [Kernel][sigway.kernels.Kernel] classes — one per
expansion era, one module each — and the closed-form functions they evaluate.
The shared foundation (geometry helpers, normalisation presets, the abstract
base class) lives in [base][sigway.kernels.base].

Public API
----------
Geometry helpers:
    get_u, get_v, polynomial

Normalisation:
    resolve_norm_preset, NORM_PRESETS

Kernel classes:
    Kernel, RadiationKernel, InstantEMDKernel, PureMDKernel

Closed-form cores (evaluated by the kernels above):
    I_sq_RD, I_sq_RD_uv, I_sq_MD, I_sq_IRD_LV, I_sq_IRD_res
"""

from sigway.kernels.base import (
    NORM_PRESETS,
    Kernel,
    get_u,
    get_v,
    polynomial,
    resolve_norm_preset,
)
from sigway.kernels.radiation import (
    I_sq_RD,
    I_sq_RD_uv,
    RadiationKernel,
)
from sigway.kernels.matter import (
    I_sq_MD,
    PureMDKernel,
)
from sigway.kernels.instant_emd import (
    I_sq_IRD_LV,
    I_sq_IRD_res,
    InstantEMDKernel,
)

__all__ = [
    # geometry helpers
    "get_u",
    "get_v",
    "polynomial",
    # normalisation
    "resolve_norm_preset",
    "NORM_PRESETS",
    # kernel classes
    "Kernel",
    "RadiationKernel",
    "InstantEMDKernel",
    "PureMDKernel",
    # closed-form cores
    "I_sq_RD",
    "I_sq_RD_uv",
    "I_sq_MD",
    "I_sq_IRD_LV",
    "I_sq_IRD_res",
]
