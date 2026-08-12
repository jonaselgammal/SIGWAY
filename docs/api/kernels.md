# sigway.kernels

The **kernel** is the time-integrated transfer function $\overline{I^2}$ — how efficiently
a scalar configuration sources gravitational waves in a given expansion era. Swap the
kernel to change the cosmology.

## Kernel classes

### Base class

::: sigway.kernels.Kernel
    options:
      heading_level: 4

### Radiation domination

::: sigway.kernels.RadiationKernel
    options:
      heading_level: 4

### Early matter domination

::: sigway.kernels.InstantEMDKernel
    options:
      heading_level: 4

### Pure matter domination (reference)

::: sigway.kernels.PureMDKernel
    options:
      heading_level: 4

## Helper functions

These are used internally by the kernels and the integrator; most users never call them
directly.

### Internal-momentum geometry

The change of variables $u=(t+s+1)/2$, $v=(t-s+1)/2$ and the geometric factor.

::: sigway.kernels.get_u
    options:
      heading_level: 4
::: sigway.kernels.get_v
    options:
      heading_level: 4
::: sigway.kernels.polynomial
    options:
      heading_level: 4

### Normalisation

Resolve a norm specification (preset name, constant, or callable) into the
$\Omega_{\rm GW}$ prefactor applied by each kernel.

::: sigway.kernels.resolve_norm_preset
    options:
      heading_level: 4

### Kernel cores (per era)

The closed-form transfer functions each kernel evaluates.

::: sigway.kernels.I_sq_RD
    options:
      heading_level: 4
::: sigway.kernels.I_sq_RD_uv
    options:
      heading_level: 4
::: sigway.kernels.I_sq_MD
    options:
      heading_level: 4
::: sigway.kernels.I_sq_IRD_LV
    options:
      heading_level: 4
::: sigway.kernels.I_sq_IRD_res
    options:
      heading_level: 4
