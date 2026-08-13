# sigway.utils

Numerical and unit-conversion helpers used across SIGWAY.

## Unit & cosmology conversions

Relate comoving wavenumber $k$, e-folds $N$, and the Hubble rate $H$.

::: sigway.utils.wavenumber_from_efolds_si_units
    options:
      heading_level: 3
::: sigway.utils.efolds_from_wavenumber_si_units
    options:
      heading_level: 3
::: sigway.utils.H_from_wavenumber
    options:
      heading_level: 3

## Spectrum interpolation

Resample a spectrum onto new frequencies, linearly or in log–log space
(backs the ``interp_grid`` feature of [OmegaGW][sigway.spectrum.OmegaGW]).

::: sigway.utils.interpolate_spectrum
    options:
      heading_level: 3

## Simpson integration

Composite Simpson quadrature on uniform and non-uniform grids.

::: sigway.utils.simpson_uniform
    options:
      heading_level: 3
::: sigway.utils.simpson_uniform_even
    options:
      heading_level: 3
::: sigway.utils.simpson_uniform_odd
    options:
      heading_level: 3
::: sigway.utils.simpson_nonuniform
    options:
      heading_level: 3
::: sigway.utils.simpson_nonuniform_even
    options:
      heading_level: 3
::: sigway.utils.simpson_nonuniform_odd
    options:
      heading_level: 3

## Broadcasting

::: sigway.utils.do_broadcasting
    options:
      heading_level: 3
::: sigway.utils.no_broadcasting
    options:
      heading_level: 3
