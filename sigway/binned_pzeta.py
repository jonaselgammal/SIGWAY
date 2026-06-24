# Global
__all__ = [
    "Binned_P_zeta",
    "compute_omega_gw",
    "compute_domega_gw",
    "upsample_f",
    "upsample_f_binned",
    "upsample_f_linear",
]

import os
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax import config
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)


@jit
def compute_omega_gw(C_mij, A):
    r"""Contract the precomputed coefficient tensor with the bin amplitudes.

    Evaluates $\Omega_{\mathrm{GW}}[m] = \sum_{i,j} C_{mij}\, A_i A_j$ on the
    internal frequency grid using a single tensor contraction.

    Parameters
    ----------
    C_mij : jaxlib.xla_extension.ArrayImpl
        Bilinear coefficient tensor of shape ``(knum, pnum, pnum)``.
    A : jaxlib.xla_extension.ArrayImpl
        Linear bin amplitudes, shape ``(pnum,)``.  These are
        $A_i = 10^{\theta_i}$, **not** the log10 values.

    Returns
    -------
    jaxlib.xla_extension.ArrayImpl
        Un-normalised $\Omega_{\mathrm{GW}}$ on the internal frequency grid,
        shape ``(knum,)``.
    """
    A_outer = jnp.outer(A, A)
    return jnp.tensordot(C_mij, A_outer, axes=([1, 2], [0, 1]))
    # return jnp.einsum("mij,ij", C_mij, A_outer)


@jit
def compute_domega_gw(C_mij, i, A):
    r"""Partial derivative of the GW spectrum with respect to one bin amplitude.

    Evaluates $\partial \Omega_{\mathrm{GW}}[m] / \partial A_i =
    2 \sum_j C_{mij}\, A_j$ on the internal frequency grid.

    Parameters
    ----------
    C_mij : jaxlib.xla_extension.ArrayImpl
        Bilinear coefficient tensor of shape ``(knum, pnum, pnum)``.
    i : int
        Index of the bin amplitude being differentiated (0-based).
    A : jaxlib.xla_extension.ArrayImpl
        Linear bin amplitudes $A_j = 10^{\theta_j}$, shape ``(pnum,)``.

    Returns
    -------
    jaxlib.xla_extension.ArrayImpl
        $\partial \Omega_{\mathrm{GW}} / \partial A_i$ on the internal
        frequency grid, shape ``(knum,)``.
    """
    return 2 * jnp.sum(C_mij[:, :, i] * A, axis=1)


@jit
def upsample_f(f, fk, Omega_GW):
    r"""Interpolate $\Omega_{\mathrm{GW}}$ to a new frequency grid (log–log).

    Performs log-linear interpolation: both the frequency axis and
    $\Omega_{\mathrm{GW}}$ are treated in $\log_{10}$, so the result is
    power-law accurate between grid points.  Frequencies outside ``fk``
    are extrapolated to $10^{-30}$ (effectively zero).

    Parameters
    ----------
    f : array_like
        Target frequencies at which to evaluate $\Omega_{\mathrm{GW}}$.
    fk : array_like
        Internal frequency grid (must be sorted ascending).
    Omega_GW : array_like
        $\Omega_{\mathrm{GW}}$ values on ``fk``.

    Returns
    -------
    jaxlib.xla_extension.ArrayImpl
        $\Omega_{\mathrm{GW}}$ interpolated to ``f``, same shape as ``f``.
    """
    return 10.0 ** (
        jnp.interp(f, fk, jnp.log10(Omega_GW), left=-30.0, right=-30.0)
    )


@jit
def upsample_f_binned(f, fk, Omega_GW):
    r"""Resample $\Omega_{\mathrm{GW}}$ with nearest-bin (step) interpolation.

    Holds $\Omega_{\mathrm{GW}}$ constant within each bin of ``fk``, so the
    result is a piecewise-constant (staircase) function.  Useful when the
    output should preserve the bin structure rather than blend between bins.

    Parameters
    ----------
    f : array_like
        Target frequencies.
    fk : array_like
        Internal frequency-bin edges (sorted ascending).
    Omega_GW : array_like
        $\Omega_{\mathrm{GW}}$ values, one per bin in ``fk``.

    Returns
    -------
    jaxlib.xla_extension.ArrayImpl
        $\Omega_{\mathrm{GW}}$ at each frequency in ``f``, same shape as ``f``.
    """
    # keep Omega_GW constant between the fk
    bin_indices = jnp.digitize(f, fk) - 1
    bin_indices = jnp.clip(bin_indices, 0, len(Omega_GW) - 1)
    return Omega_GW[bin_indices]


@jit
def upsample_f_linear(f, fk, Omega_GW):
    r"""Interpolate $\Omega_{\mathrm{GW}}$ to a new frequency grid (linear).

    Plain linear interpolation — no logarithm taken on either axis.  Used
    for derivatives, where $\Omega_{\mathrm{GW}}$ can pass through zero and
    a log-space interpolation would be undefined.

    Parameters
    ----------
    f : array_like
        Target frequencies.
    fk : array_like
        Internal frequency grid (sorted ascending).
    Omega_GW : array_like
        $\Omega_{\mathrm{GW}}$ values on ``fk``.

    Returns
    -------
    jaxlib.xla_extension.ArrayImpl
        $\Omega_{\mathrm{GW}}$ linearly interpolated to ``f``, same shape as
        ``f``.
    """
    return jnp.interp(f, fk, Omega_GW)


class Binned_P_zeta:
    r"""Model-independent gravitational-wave spectrum from a binned primordial
    curvature power spectrum $\mathcal{P}_\zeta(k)$.

    **Physical idea.**  Instead of assuming a fixed shape for
    $\mathcal{P}_\zeta(k)$ (e.g. a log-normal or power law), this class
    represents it as free, independent amplitudes in $n$ logarithmically spaced
    wavenumber bins.  Each bin $i$ carries a single free parameter
    $\theta_i \equiv \log_{10} A_i$, so the linear amplitude in that bin is
    $A_i = 10^{\theta_i}$.  The resulting scalar-induced GW energy-density
    spectrum is

    $$\Omega_{\mathrm{GW}}(f_m) = \mathcal{N}(f_m)
       \sum_{i,\,j} C_{mij}\, A_i A_j,$$

    where $C_{mij}$ are precomputed bilinear coefficients that fold in the
    GW kernel and the $(s,t)$ transfer-function integral once and for all.
    Evaluating the spectrum then reduces to a single tensor contraction — much
    cheaper than computing the full kernel integral at every likelihood call.
    This makes the class well suited for **model-independent reconstruction**
    of $\mathcal{P}_\zeta(k)$ from GW data, and for computing Fisher forecasts
    without committing to a spectral shape.

    **Interface.**  The public API is identical to
    [OmegaGW][sigway.spectrum.OmegaGW]: ``parameter_names`` returns the
    ordered parameter tuple, the instance is callable as ``model(f, *theta)``
    to obtain $\Omega_{\mathrm{GW}}(f)$, and ``jacobian(f, theta)`` returns
    the gradient matrix needed by Fisher/MCMC codes.

    Attributes
    ----------
    fk : numpy.ndarray
        Output frequency grid (in units of $k = 2\pi f$) on which the
        coefficient tensor $C_{mij}$ is tabulated.
    fp : numpy.ndarray
        Wavenumber grid of the $\mathcal{P}_\zeta$ bins (centres of the $n$
        input bins).
    C_mij : jaxlib.xla_extension.ArrayImpl
        Precomputed bilinear coefficient tensor, shape
        ``(knum, pnum, pnum)``.  Entry $C_{mij}$ encodes the contribution of
        bin pair $(i,j)$ to the GW spectrum at output frequency $m$.
    knum : int
        Number of output frequency bins (length of ``fk``).
    pnum : int
        Number of $\mathcal{P}_\zeta$ bins (length of ``fp``).
    parameterNames : dict
        Ordered mapping of parameter names to their LaTeX labels,
        e.g. ``{'A_0': '$A_0$', 'A_1': '$A_1$', ...}``.
    norm : callable
        Normalisation prefactor $\mathcal{N}(k)$ applied to
        $\Omega_{\mathrm{GW}}$; set at construction time (see ``norm``
        parameter of ``__init__``).
    d1 : callable
        Alias for ``dtemplate_default``.  Provided so that external inference
        code that calls ``model.d1(index, f, *theta)`` works without change.

    Examples
    --------
    Excite a single bin near the middle of the spectrum and evaluate
    $\Omega_{\mathrm{GW}}(f)$:

    >>> import jax.numpy as jnp
    >>> from sigway.binned_pzeta import Binned_P_zeta
    >>> model = Binned_P_zeta("binned", "Binned", nbins=100)
    >>> amps = [-4.0] * len(model.parameter_names)   # log10 amplitudes
    >>> amps[50] = 0.0                               # excite one bin
    >>> omega = model(jnp.geomspace(2e-5, 1.0, 80), *amps)
    """

    def __init__(
        self,
        model_name,
        model_label,
        nbins=50,
        path_to_C=None,
        norm="RD",
        backend="jax",
    ):
        r"""Load precomputed bilinear coefficients and set up the spectral model.

        Parameters
        ----------
        model_name : str
            Short internal identifier for the model (stored as
            ``self._model``), used by logging and output routines.
        model_label : str
            Human-readable display name for the model (stored as
            ``self._model_label``), e.g. used in plot legends.
        nbins : int, optional
            Number of logarithmically spaced $\mathcal{P}_\zeta(k)$ bins.
            Precomputed coefficient files ship with the package for
            ``nbins`` in ``{10, 15, 20, 30, 40, 50, 100, 200}``.  Choosing
            more bins gives finer spectral resolution at the cost of a larger
            parameter space.  Ignored when ``path_to_C`` is provided.
            Default is ``50``.
        path_to_C : str or None, optional
            Path to a custom coefficient file (plain text, SIGWAY format)
            containing $C_{mij}$, $f_k$, and $f_p$.  When ``None`` the file
            bundled with the package for the requested ``nbins`` is used.
            Default is ``None``.
        norm : str or callable, optional
            Overall normalisation prefactor $\mathcal{N}$ applied to
            $\Omega_{\mathrm{GW}}$.

            * ``'RD'`` (default) — standard radiation-domination prefactor
              $C_G \Omega_R \approx 0.39 \times 4.2 \times 10^{-5}$, appropriate
              for GW production deep in the radiation era.
            * A callable ``norm(k) -> float`` for a custom
              frequency-dependent normalisation.
        backend : str, optional
            Computation backend.  Currently only ``'jax'`` is supported.
            Default is ``'jax'``.

        Raises
        ------
        FileNotFoundError
            If no coefficient file is found for the requested ``nbins`` (when
            ``path_to_C`` is ``None``), or if the path given by ``path_to_C``
            does not exist.
        """
        # Constants
        OMEGA_R = 4.2 * 10 ** (-5)
        CG = 0.39

        # Load the coefficients
        if path_to_C is None:
            try:
                data = np.loadtxt(
                    os.path.join(
                        os.path.dirname(__file__),
                        "binning_coefficients",
                        f"tabkpp{nbins}.txt",
                    )
                )
            except FileExistsError:
                raise FileNotFoundError(
                    f"No coefficients found for nbins={nbins}. "
                    "Please change  the number to one where the coefficients "
                    "have been precomputed for."
                )
        else:
            try:
                data = np.loadtxt(path_to_C)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No coefficients found in {path_to_C}. "
                    "Your specified path_to_C seems to be incorrect."
                )
        self.knum = int(data[0])
        self.pnum = int(data[1])
        self.fk = data[2 : 2 + self.knum]
        self.fp = data[2 + self.knum : 2 + self.knum + self.pnum]
        C_mij = data[2 + self.knum + self.pnum :]
        self.C_mij = jnp.array(C_mij.reshape((self.knum, self.pnum, self.pnum)))

        # Essentially pointless
        self._model = model_name
        self._model_label = model_label
        self.parameterNames = {
            f"A_{i}": f"$A_{i}$" for i in range(len(self.fp))
        }

        self.parameterLabels = list(self.parameterNames.values())
        if norm == "RD":
            self.norm = lambda k: CG * OMEGA_R
        if norm == "CT":
            self.norm = lambda k: 1.0
        elif callable(norm):
            self.norm = norm

        self.d1 = self.dtemplate_default

    def template(self, fvec, *A):
        r"""Evaluate $\Omega_{\mathrm{GW}}(f)$ on an arbitrary frequency grid.

        Converts the supplied $\log_{10}$-amplitudes $\theta_i$ to linear
        amplitudes $A_i = 10^{\theta_i}$, contracts them against the
        precomputed tensor $C_{mij}$ to obtain $\Omega_{\mathrm{GW}}$ on the
        internal grid ``fk``, then log-linearly interpolates to the requested
        output frequencies.  The result is multiplied by the normalisation
        prefactor ``self.norm``.

        Parameters
        ----------
        fvec : array_like
            Frequencies in Hz at which $\Omega_{\mathrm{GW}}$ is evaluated.
        *A : float
            $\log_{10}$-amplitudes $\theta_0, \theta_1, \ldots, \theta_{n-1}$
            of the $\mathcal{P}_\zeta$ bins.  The linear amplitude in bin $i$
            is $10^{A_i}$.  The number of values must equal ``self.pnum``.

        Returns
        -------
        jaxlib.xla_extension.ArrayImpl
            $\Omega_{\mathrm{GW}}$ evaluated at each frequency in ``fvec``,
            same shape as ``fvec``.
        """
        kvec = fvec * 2 * jnp.pi
        A = 10.0 ** (jnp.array(A))
        omega = compute_omega_gw(self.C_mij, A)
        return self.norm(kvec) * upsample_f(kvec, self.fk, omega)

    def dtemplate_default(self, index, fvec, *A):
        r"""Partial derivative of $\Omega_{\mathrm{GW}}$ with respect to one
        bin $\log_{10}$-amplitude.

        Evaluates $\partial \Omega_{\mathrm{GW}} / \partial \theta_{\text{index}}$
        analytically.  Because $A_i = 10^{\theta_i}$ and the spectrum is
        bilinear in the $A_i$, the chain rule gives
        $2 \ln(10)\, A_i \sum_j C_{mij}\, A_j$; the factor $\ln(10) A_i$ is
        absorbed into the linear-amplitude convention used here.  The result is
        interpolated **linearly** (not log-linearly) to ``fvec`` to avoid
        problems where the derivative passes through zero.

        Parameters
        ----------
        index : int
            0-based index of the $\mathcal{P}_\zeta$ bin parameter $\theta_i$
            being differentiated (must be in $[0,\, n)$).
        fvec : array_like
            Frequencies in Hz at which the derivative is evaluated.
        *A : float
            $\log_{10}$-amplitudes $\theta_0, \ldots, \theta_{n-1}$ of all
            $\mathcal{P}_\zeta$ bins (same convention as ``template``).

        Returns
        -------
        jaxlib.xla_extension.ArrayImpl
            $\partial \Omega_{\mathrm{GW}} / \partial \theta_{\text{index}}$
            at each frequency in ``fvec``, same shape as ``fvec``.
        """
        kvec = fvec * 2 * jnp.pi
        A = 10.0 ** (jnp.array(A))
        domega = compute_domega_gw(self.C_mij, index, A)
        return self.norm(kvec) * upsample_f_linear(
            kvec, self.fk, domega
        )  # To avoid log(0)

    # --- OmegaGW-consistent interface (binned is its own dedicated path: it
    # returns Omega_GW directly from precomputed coefficients, with no kernel
    # or (s, t) integral, so it does not go through OmegaGW). ---
    @property
    def parameter_names(self):
        r"""Ordered names of the $\mathcal{P}_\zeta$ bin parameters.

        Returns a tuple ``('A_0', 'A_1', ..., 'A_{n-1}')`` where $n$ is the
        number of bins.  The ordering matches the expected sequence of
        $\theta_i$ values in ``__call__`` and ``jacobian``, and is consistent
        with the ``parameter_names`` property of
        [OmegaGW][sigway.spectrum.OmegaGW].

        Returns
        -------
        tuple of str
            Parameter names for the $\log_{10}$-amplitude of each
            $\mathcal{P}_\zeta$ bin.
        """
        return tuple(self.parameterNames)

    def __call__(self, f, *theta):
        r"""Evaluate $\Omega_{\mathrm{GW}}(f)$ from the bin $\log_{10}$-amplitudes.

        Thin wrapper around ``template`` that satisfies the standard callable
        interface shared with [OmegaGW][sigway.spectrum.OmegaGW]: pass a
        frequency array and the $n$ bin parameters $\theta_i$, receive the GW
        energy-density spectrum.

        Parameters
        ----------
        f : array_like
            Frequencies in Hz at which $\Omega_{\mathrm{GW}}$ is evaluated.
        *theta : float
            $\log_{10}$-amplitudes $\theta_0, \theta_1, \ldots, \theta_{n-1}$
            of the $\mathcal{P}_\zeta$ bins.

        Returns
        -------
        jaxlib.xla_extension.ArrayImpl
            $\Omega_{\mathrm{GW}}(f)$ evaluated at each frequency in ``f``.
        """
        return self.template(f, *theta)

    def jacobian(self, f, theta):
        r"""Full Jacobian of $\Omega_{\mathrm{GW}}$ with respect to the bin
        $\log_{10}$-amplitudes $\theta$.

        Calls ``dtemplate_default`` for every bin index and stacks the
        results into a matrix, following the same interface as
        [OmegaGW][sigway.spectrum.OmegaGW].  The Jacobian is used by Fisher
        forecasting and gradient-based samplers.

        Parameters
        ----------
        f : array_like
            Frequencies in Hz at which the Jacobian is evaluated.
        theta : array_like
            $\log_{10}$-amplitudes of all $\mathcal{P}_\zeta$ bins,
            shape ``(pnum,)``.

        Returns
        -------
        jaxlib.xla_extension.ArrayImpl
            Array of shape ``(len(f), pnum)`` where entry ``[m, i]`` is
            $\partial \Omega_{\mathrm{GW}}(f_m) / \partial \theta_i$.
        """
        return jnp.stack(
            [self.dtemplate_default(i, f, *theta) for i in range(len(theta))],
            axis=-1,
        )


if __name__ == "__main__":

    model_name = "log-normal-in-Pz"
    model_label = "Log Normal in Pz"
    model = Binned_P_zeta(model_name, model_label, Norm="RD")
    f = np.geomspace(2e-5, 1.0, 100)
    plt.figure()
    plt.loglog(f, model.template(f, jnp.ones_like(model.fp)))
    for i in range(len(model.fp)):
        plt.loglog(
            f, model.dtemplate_default(i, f, jnp.ones_like(model.fp)), alpha=0.5
        )
    plt.grid()
    plt.show()
