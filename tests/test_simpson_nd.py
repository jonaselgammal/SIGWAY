"""Simpson integrators, incl. the N-D contract the integrator relies on.

`simpson_nonuniform` must integrate along axis 0 of an N-D array of samples --
that is exactly how OmegaGW(jax/ms) integrate over t for a whole frequency
vector at once. A regression (commit 9b01838, "utils now GPU compatible")
replaced the shape dispatch with `jax.lax.switch`, whose branches return
mismatched shapes, so *every* multi-dimensional call raised. These tests pin the
N-D contract so that regression cannot return silently.

Independent reference: scipy.integrate.simpson, applied column-by-column.
"""
import numpy as np
import jax.numpy as jnp
from scipy.integrate import simpson as sp_simpson

from sigway.utils import simpson_uniform, simpson_nonuniform


def test_simpson_nonuniform_1d():
    """1-D non-uniform Simpson matches scipy (odd and even sample counts)."""
    for n in (101, 100):
        x = jnp.geomspace(1.0, 10.0, n)
        y = x**3
        assert np.isclose(
            float(simpson_nonuniform(y, x)),
            sp_simpson(np.array(y), x=np.array(x)),
            rtol=1e-10,
        )


def test_simpson_nonuniform_nd_x1d():
    """N-D samples (nt, nk) integrated along axis 0 with a shared 1-D t grid.

    This is the broadcasting case the lax.switch regression broke.
    """
    x = np.geomspace(1.0, 20.0, 121)
    cols = np.array([1.0, 2.5, 7.0, 0.3])
    f = (x[:, None] ** 2) * cols[None, :]
    got = np.array(simpson_nonuniform(jnp.array(f), jnp.array(x)))
    ref = np.array([sp_simpson(f[:, j], x=x) for j in range(f.shape[1])])
    assert np.allclose(got, ref, rtol=1e-10)


def test_simpson_nonuniform_nd_x2d():
    """Per-column t grids: x has the same shape as f (integrator path)."""
    nt, nk = 121, 4
    x = np.stack([np.geomspace(1.0, 10.0 + j, nt) for j in range(nk)], axis=1)
    f = x**3
    got = np.array(simpson_nonuniform(jnp.array(f), jnp.array(x)))
    ref = np.array([sp_simpson(f[:, j], x=x[:, j]) for j in range(nk)])
    assert np.allclose(got, ref, rtol=1e-10)


def test_simpson_uniform_nd():
    """Uniform Simpson over axis 0 of an (n, k) array vs scipy per column."""
    for n in (101, 100):  # odd / even sample counts
        x = np.linspace(0.0, 1.0, n)
        cols = np.array([1.0, 3.0, 0.5])
        f = (x[:, None] ** 3) * cols[None, :]
        got = np.array(simpson_uniform(jnp.array(f), jnp.array(x)))
        ref = np.array([sp_simpson(f[:, j], x=x) for j in range(f.shape[1])])
        assert np.allclose(got, ref, rtol=1e-9)
