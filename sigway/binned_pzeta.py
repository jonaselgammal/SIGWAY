# Global
__all__ = ["Binned_P_zeta"]

import os
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax import config
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)


@jit
def compute_omega_gw(C_mij, A):
    A_outer = jnp.outer(A, A)
    return jnp.tensordot(C_mij, A_outer, axes=([1, 2], [0, 1]))
    # return jnp.einsum("mij,ij", C_mij, A_outer)


@jit
def compute_domega_gw(C_mij, i, A):
    return 2 * jnp.sum(C_mij[:, :, i] * A, axis=1)


@jit
def upsample_f(f, fk, Omega_GW):
    return 10.0 ** (
        jnp.interp(f, fk, jnp.log10(Omega_GW), left=-30.0, right=-30.0)
    )


@jit
def upsample_f_binned(f, fk, Omega_GW):
    # keep Omega_GW constant between the fk
    bin_indices = jnp.digitize(f, fk) - 1
    bin_indices = jnp.clip(bin_indices, 0, len(Omega_GW) - 1)
    return Omega_GW[bin_indices]


@jit
def upsample_f_linear(f, fk, Omega_GW):
    return jnp.interp(f, fk, Omega_GW)


class Binned_P_zeta:
    """Model-independent (template-free) gravitational-wave spectrum from a
    binned primordial power spectrum.

    The primordial curvature power spectrum P_zeta is parameterised as
    free log10-amplitudes ``A_0, A_1, ..., A_{n-1}`` in ``n`` logarithmically
    spaced k-bins.  The corresponding scalar-induced GW energy-density spectrum
    Omega_GW(f) is computed directly from a set of precomputed bilinear
    coefficients ``C_mij`` (one per output frequency bin ``m`` and pair of
    P_zeta bins ``i, j``) via

        Omega_GW[m] = norm(f_m) * sum_{i,j} C_mij * A_i * A_j,

    where ``A_i = 10^theta_i``.  This bypasses the kernel and (s, t) integral
    used by [OmegaGW][sigway.spectrum.OmegaGW], making evaluation cheap once
    the coefficients have been loaded.

    The public interface mirrors [OmegaGW][sigway.spectrum.OmegaGW]:
    ``parameter_names`` returns an ordered tuple of parameter names, calling
    the instance as ``model(f, *theta)`` returns Omega_GW(f), and
    ``jacobian(f, theta)`` returns the gradient with respect to the bin
    log-amplitudes.

    Attributes
    ----------
    fk : numpy.ndarray
        Frequency grid on which ``C_mij`` is defined (output frequencies).
    fp : numpy.ndarray
        Frequency grid of the P_zeta bins (input frequencies).
    C_mij : jaxlib.xla_extension.ArrayImpl
        Precomputed bilinear coefficient tensor of shape
        ``(len(fk), len(fp), len(fp))``.
    knum : int
        Number of output frequency bins (length of ``fk``).
    pnum : int
        Number of P_zeta bins (length of ``fp``).
    parameterNames : dict
        Ordered mapping of parameter names to LaTeX labels,
        e.g. ``{'A_0': '$A_0$', ...}``.
    norm : callable
        Normalisation function applied to Omega_GW; set at construction time.
    d1 : callable
        Alias for ``dtemplate_default``, provided for compatibility with
        external inference code.
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
        """Initialise ``Binned_P_zeta`` by loading precomputed coefficients.

        Parameters
        ----------
        model_name : str
            Internal name of the model, stored as ``self._model``.
        model_label : str
            Human-readable label for the model, stored as
            ``self._model_label``.
        nbins : int, optional
            Number of P_zeta bins.  The coefficient file must have been
            precomputed for this value; shipped options are 10, 20, 30, 40,
            50, 100, and 200.  Ignored when ``path_to_C`` is provided.
            Default is ``50``.
        path_to_C : str or None, optional
            Path to a custom coefficient file containing ``C_mij``, ``fk``,
            and ``fp``.  When ``None`` the file bundled with the package for
            the requested ``nbins`` is used.  Default is ``None``.
        norm : str or callable, optional
            Normalisation applied to Omega_GW.  Pass ``'RD'`` (default) for
            the standard radiation-domination prefactor
            ``C_G * Omega_R ~ 0.39 * 4.2e-5``; pass a callable
            ``norm(k) -> float`` for a custom frequency-dependent
            normalisation.
        backend : str, optional
            Computation backend.  Currently only ``'jax'`` is supported.
            Default is ``'jax'``.

        Raises
        ------
        FileNotFoundError
            If no coefficient file exists for the requested ``nbins`` (when
            ``path_to_C`` is ``None``) or if the path given by ``path_to_C``
            does not point to an existing file.
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
        """Evaluate Omega_GW on an arbitrary frequency grid.

        Converts the supplied log10-amplitudes to linear amplitudes, contracts
        them against the precomputed coefficient tensor ``C_mij`` to obtain
        Omega_GW on the internal frequency grid ``fk``, then interpolates
        (log-linearly in both axes) to the requested output frequencies.  The
        result is multiplied by the normalisation function ``self.norm``.

        Parameters
        ----------
        fvec : array_like
            Output frequencies in Hz at which Omega_GW is evaluated.
        *A : float
            Log10-amplitudes of the P_zeta bins, i.e. ``theta_i`` such that
            the linear amplitude in bin ``i`` is ``10**A[i]``.  The number of
            values must equal ``self.pnum``.

        Returns
        -------
        jaxlib.xla_extension.ArrayImpl
            Omega_GW evaluated at each frequency in ``fvec``, same shape as
            ``fvec``.
        """
        kvec = fvec * 2 * jnp.pi
        A = 10.0 ** (jnp.array(A))
        omega = compute_omega_gw(self.C_mij, A)
        return self.norm(kvec) * upsample_f(kvec, self.fk, omega)

    def dtemplate_default(self, index, fvec, *A):
        """Partial derivative of Omega_GW with respect to a single bin
        log-amplitude.

        Computes ``d Omega_GW / d theta_index`` analytically from the
        coefficient tensor via ``2 * sum_j C_mij[:,j,index] * A_j`` (linear
        amplitudes), then interpolates linearly to ``fvec`` and applies the
        normalisation.  Linear (not log-linear) interpolation is used here to
        avoid issues with zero crossings in the derivative.

        Parameters
        ----------
        index : int
            Index of the bin parameter with respect to which the derivative is
            taken (0-based, must be in ``[0, pnum)``).
        fvec : array_like
            Output frequencies in Hz.
        *A : float
            Log10-amplitudes of all P_zeta bins (same convention as
            ``template``).

        Returns
        -------
        jaxlib.xla_extension.ArrayImpl
            ``d Omega_GW / d theta_index`` evaluated at each frequency in
            ``fvec``, same shape as ``fvec``.
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
        """Ordered tuple of bin log-amplitude parameter names.

        Returns a tuple of strings such as ``('A_0', 'A_1', ..., 'A_{n-1}')``
        where ``n`` equals ``self.pnum``.  The ordering matches the expected
        order of ``*theta`` in ``__call__`` and ``jacobian``, and is identical
        in spirit to the ``parameter_names`` property of
        [OmegaGW][sigway.spectrum.OmegaGW].

        Returns
        -------
        tuple of str
            Parameter names for the bin log-amplitudes.
        """
        return tuple(self.parameterNames)

    def __call__(self, f, *theta):
        """Evaluate Omega_GW(f) from bin log-amplitudes.

        Thin wrapper around ``template`` that provides the callable interface
        expected by inference code (equivalent to
        [OmegaGW][sigway.spectrum.OmegaGW]``.__call__``).

        Parameters
        ----------
        f : array_like
            Frequencies in Hz at which Omega_GW is evaluated.
        *theta : float
            Log10-amplitudes of the P_zeta bins (``A_0, A_1, ..., A_{n-1}``).

        Returns
        -------
        jaxlib.xla_extension.ArrayImpl
            Omega_GW evaluated at each frequency in ``f``.
        """
        return self.template(f, *theta)

    def jacobian(self, f, theta):
        """Gradient of Omega_GW with respect to all bin log-amplitudes.

        Calls ``dtemplate_default`` for each bin index and stacks the results
        along the last axis, producing the full Jacobian matrix.  This matches
        the ``jacobian`` interface of [OmegaGW][sigway.spectrum.OmegaGW].

        Parameters
        ----------
        f : array_like
            Frequencies in Hz at which the Jacobian is evaluated.
        theta : array_like
            Log10-amplitudes of all P_zeta bins, shape ``(pnum,)``.

        Returns
        -------
        jaxlib.xla_extension.ArrayImpl
            Array of shape ``(len(f), pnum)`` where entry ``[m, i]`` is
            ``d Omega_GW(f_m) / d theta_i``.
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
