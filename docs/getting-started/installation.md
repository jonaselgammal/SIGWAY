# Installation

SIGWAY is a pure-Python package built on JAX. There is no PyPI release yet, so install
it **from source**.

## Requirements

- Python ≥ 3.10
- [`jax`](https://jax.readthedocs.io), [`diffrax`](https://docs.kidger.site/diffrax/),
  `numpy`, `scipy`, `matplotlib` (pulled in automatically)

## Install

Clone the repository and install in editable mode:

```bash
git clone https://github.com/jonaselgammal/SIGWAY.git
cd SIGWAY
pip install -e .
```

Editable (`-e`) means changes to the source take effect without reinstalling — handy
while the package is under active development.

!!! tip "Use float64"
    SIGW integrals span many orders of magnitude. Enable double precision **before** any
    other JAX call, at the top of your script or notebook:

    ```python
    import jax
    jax.config.update("jax_enable_x64", True)
    ```

## Check it worked

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sigway.spectrum import OmegaGW
from sigway.kernels import RadiationKernel
from sigway.perturbations import AnalyticPerturbations

model = OmegaGW(
    AnalyticPerturbations(lambda k, A: A * jnp.ones_like(k), ("A",)),
    RadiationKernel(),
    s=jnp.linspace(0.0, 1.0, 10),
    t=jnp.geomspace(1e-4, 1e2, 200),
)
print(model.parameter_names)        # ('A',)
```

If that prints `('A',)` you're ready for the [simple example](../examples/simple_example.ipynb).
