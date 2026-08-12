r"""Pure matter-domination kernel (unphysical reference case).

Provides the constant oscillation-averaged core
[I_sq_MD][sigway.kernels.I_sq_MD] and its thin
[PureMDKernel][sigway.kernels.PureMDKernel] wrapper.  A universe that stays
matter-dominated forever is not physical — a realistic early-matter era
transitions to radiation domination (see
[InstantEMDKernel][sigway.kernels.InstantEMDKernel]).  These are retained for
reference and cross-checks, not production use.
"""

from jax import jit

from sigway.kernels.base import Kernel


# Pure matter domination. This is unphysical.
# UNUSED in favour of I_MD_TO_RD
@jit
def I_sq_MD(t, s, k):
    r"""Oscillation-averaged kernel for a universe in pure matter domination.

    Returns the constant $\overline{I^2_{\rm MD}} = 18/25$, independent of
    $t$, $s$, and $k$.  This is the analytic result for a universe that stays
    matter-dominated forever: the matter-era Green's function grows as a power
    law, and after oscillation-averaging the transfer function saturates at
    $18/25$.

    This is an unphysical limiting case — a realistic early-matter-dominated
    era eventually transitions to radiation domination.  For that physical
    scenario use `I_sq_IRD_LV` and `I_sq_IRD_res`.  This function is retained
    for reference and is no longer called in production.

    Parameters
    ----------
    t : jax.Array
        Dimensionless combination $t = u + v - 1$ (unused).
    s : jax.Array
        Dimensionless combination $s = u - v$ (unused).
    k : jax.Array
        GW wave-number (Mpc$^{-1}$); unused.

    Returns
    -------
    jax.Array
        Constant $18/25$, broadcast to the shape of $t$.
    """
    return 18.0 / 25.0


class PureMDKernel(Kernel):
    r"""Kernel for a universe in pure (eternal) matter domination.

    A thin wrapper around the constant core `I_sq_MD`: the
    oscillation-averaged transfer function saturates at $\overline{I^2_{\rm
    MD}} = 18/25$ for all $(t, s, k)$.  Like `I_sq_MD` this is an **unphysical**
    limiting case — a realistic early-matter era transitions to radiation
    domination, for which
    [InstantEMDKernel][sigway.kernels.InstantEMDKernel] should be used
    instead.  It is provided for reference and cross-checks.

    The default normalisation preset ``'CT'`` gives the dimensionless ratio
    $\Omega_{\rm GW} / \Omega_r$ via the prefactor $1/12$.

    Parameters
    ----------
    norm : str, float, or callable, optional
        Normalisation override; see [Kernel][sigway.kernels.Kernel].
        Defaults to ``'CT'``.
    """

    k_dependent = False
    param_names = ()
    _default_norm = "CT"

    def overline_Isq(self, t, s, k, *kparams):
        r"""Evaluate the pure matter-domination kernel
        $\overline{I^2_{\rm MD}} = 18/25$.

        Delegates to `I_sq_MD`, the constant oscillation-averaged core.

        Parameters
        ----------
        t : jax.Array
            Dimensionless combination $t = u + v - 1$ (unused).
        s : jax.Array
            Dimensionless combination $s = u - v$ (unused).
        k : jax.Array
            GW wave-number (Mpc$^{-1}$); unused.
        *kparams :
            Ignored (this kernel has no extra parameters).

        Returns
        -------
        jax.Array
            Constant $18/25$.
        """
        return I_sq_MD(t, s, k)
