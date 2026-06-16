"""Radiation-domination kernel physics.

These pin the closed-form oscillation-averaged transfer function used for every
RD spectrum. The reference is an *independent* numpy re-derivation of
\\overline{I^2}(u,v) from the literature (see _sigway_oracle.kernel_RD_text),
not a stored snapshot. Tolerances are ~1e-10: the two implementations agree to
~1e-11, so this bites on any real change to the kernel.

Targets stable physics (the kernel's numerical output), so it survives the
planned Kernel-class refactor; only the import line would move.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from sigway.omega_gw_jax import I_sq_RD, I_sq_RD_uv
from _sigway_oracle import kernel_RD_text

# (t, s) sample grid spanning sub- and super-resonance (t = sqrt(3)-1) regions.
_T = np.geomspace(1e-2, 1e2, 60)
_S = np.linspace(0.0, 0.97, 25)  # avoid s=1 boundary (v=0 -> 1/v^3 blow-up)
_TT, _SS = np.meshgrid(_T, _S, indexing="ij")


@pytest.mark.parametrize(
    "fn,name", [(I_sq_RD, "I_sq_RD"), (I_sq_RD_uv, "I_sq_RD_uv")]
)
def test_rd_kernel_matches_textbook(fn, name):
    """sigway's RD kernel == independent literature form to ~1e-10."""
    got = np.array(fn(jnp.array(_TT), jnp.array(_SS), k=1.0))
    ref = kernel_RD_text(_TT, _SS)
    assert np.nanmax(np.abs(got / ref - 1.0)) < 1e-10, name


def test_rd_kernel_nonnegative():
    """The kernel is |overline I|^2 >= 0 (a squared transfer function)."""
    got = np.array(I_sq_RD(jnp.array(_TT), jnp.array(_SS), k=1.0))
    assert np.all(got >= 0.0)


def test_rd_kernel_even_in_s():
    """I_sq_RD depends on s only through s^2, so it must be even in s."""
    a = np.array(I_sq_RD(jnp.array(_TT), jnp.array(_SS), k=1.0))
    b = np.array(I_sq_RD(jnp.array(_TT), jnp.array(-_SS), k=1.0))
    assert np.allclose(a, b, rtol=1e-12, atol=0.0)


def test_rd_kernel_k_independent():
    """The RD kernel is k-independent; varying k must not change it."""
    a = np.array(I_sq_RD(jnp.array(_TT), jnp.array(_SS), k=1.0))
    b = np.array(I_sq_RD(jnp.array(_TT), jnp.array(_SS), k=137.0))
    assert np.array_equal(a, b)
