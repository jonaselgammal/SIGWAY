# Global
import jax
import jax.numpy as jnp

# Local
from sigway.ms_solver import SingleFieldSolver
from sigway.omega_gw_jax import OmegaGWjax, get_u, get_v


jax.config.update("jax_enable_x64", True)

"""
Implementation of the OmegaGW class using jax and jit. Used when integrating a
Mukhanov-Sasaki solver.
"""


class OmegaGWms(OmegaGWjax):
    r"""
    OmegaGW implementation with a fully vectorized jax implementation of
    Simpson's rule for non-uniform grids.

    Parameters:
    - P_zeta: function
        The primordial power spectrum function. This function should take k as
        the first argument, followed by the parameters it depends on. For
        optimal performance, this function should be jitted.
    - k: array-like, shape (nk,)
        The k values at which to compute the integral. This should map the
        region of for :math:`\Omega_{GW}` (i.e. the sensitivity of the
        experiment). If you need a higher resolution you can use the upsample
        parameter to interpolate the k values.
    - s: array-like, shape (ns,) or function
        The s values to use in the calculation. If s is a function, it should
        take k as the first argument followed the values of the parameters of
        the P_zeta function in the same order. It should return an array-like
        of s values.
        If s is an array-like, it is assumed to stay constant for all k.
    - t: array-like, shape (nt,) or function
        The t values to use in the calculation. If t is a function, it should
        take k as the first argument followed the values of the parameters of
        the P_zeta function in the same order. It should return an array-like
        of t values.
        If t is an array-like, it is assumed to stay constant for all k.
    - norm: callable or str, default="RD"
        The normalization function for the primordial power spectrum. If a
        callable is passed, it should take k as the first argument and return
        the normalization factor. If "RD" is passed, the normalization factor
        will be calculated using the radiation domination approximation.
    - upsample: array-like, shape (nk,), default=None
        If this parameter is not None, it should be an array-like of k values
        at which to interpolate :math:`\Omega_{GW}`. This is useful if you need
        a higher resolution in the k values.
    - dP_zeta: function, default=None
        Will be ignored for this implementation since there is no derivative
        version of the Mukhanov-Sasaki solver.

        .. note::
            In dblquad integration the spacing of the grid in s and t is
            handled by scipy. Thus the grid in s and t will be ignored and
            only the end points will be used.
    """

    def __init__(
        self,
        P_zeta,
        s,
        t,
        f=None,
        norm="RD",
        kernel="RD",
        upsample=False,
        dP_zeta=None,
        jit=False,
    ):
        if not isinstance(P_zeta, SingleFieldSolver):
            raise ValueError(
                "Pzeta should be an instance of SingleFieldSolver. If your "
                "Pzeta is a function or a callable, you should use one of the "
                " other OmegaGW classes."
            )
        super().__init__(
            P_zeta,
            s,
            t,
            f=f,
            norm=norm,
            kernel=kernel,
            upsample=upsample,
            dP_zeta=None,
            jit=jit,
        )

    def __call__(self, fvec, *params):
        r"""
        Compute the :math:`\Omega_{GW}` values.

        Parameters:
        - fvec: jax.numpy.ndarray
            Array of frequency values to compute :math:`\Omega_{GW}` at.
        - params: tuple
            Tuple of parameters for the P_zeta function.

        Returns:
        - jax.numpy.ndarray
            Array of :math:`\Omega_{GW}` values.
        """

        # setting k for evaluation
        kvec_full_resolution = jnp.copy(jnp.array(fvec)) * 2 * jnp.pi
        if self.upsample:
            if callable(self.f):
                fvec = self.f(*params)
            else:
                fvec = self.f
            kvec = jnp.array(fvec) * 2 * jnp.pi
        else:
            kvec = kvec_full_resolution
        # Setting s
        if self.fixed_s:
            s = self.s
        else:
            s = self.s(kvec, *params)
        # Setting t
        if self.fixed_t:
            t = self.t
        else:
            t = self.t(kvec, *params)
        # If no k is provided for P_zeta it's calculated in the max
        # range of s and t
        if self.P_zeta.k is None or not self.P_zeta.upsample:
            uv = jnp.array(
                [
                    get_u(t[None, :, :], s[:, None, None]),
                    get_v(t[None, :, :], s[:, None, None]),
                ]
            )
            mink = jnp.min(kvec) * jnp.min(uv)
            maxk = jnp.max(kvec) * jnp.max(uv)
            kint = jnp.geomspace(mink, maxk, 100)
        else:
            kint = self.P_zeta.k

        P_zeta_interp = self.P_zeta.run(kint, *params)

        res = self.integration_routine(P_zeta_interp, s, t, kvec, *params)

        out = (
            2
            * self.norm(kvec_full_resolution)
            * self.upsample_k(kvec_full_resolution, kvec, res)
        )

        return out
