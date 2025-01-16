# Global
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
    def __init__(
        self,
        model_name,
        model_label,
        nbins=50,
        path_to_C=None,
        norm="RD",
        backend="jax",
    ):
        """
        Just a mini wrapper to make the Omega_GW class compatible with
        SGWBinner. This seems to be all that's required.
        For binned P_zeta. we need to pass the C_mij, fk, and fp.

        Parameters
        ----------
        model_name : str
            the name of the model.
        model_label : str
            the label of the model.
        nbins : int, optional (default=50)
            the number of bins to be used (the grid they are computed on is
            `nbins`x`nbins`x`nbins`) which corresponds to the two internal
            momenta and the external momentum. Currently available options are
            10, 20, 30, 40, 50, 100 and 200.
            If `path_to_C` is not None, this parameter is ignored.
        path_to_C : str or None, optional (default=None)
            file name of the coefficent file containing C_mij, fk, and fp.
            If None the pre-computed values are used.
        norm : str or callable, optional (default='RD')
            the normalisation to use. If a string, must be 'RD'.
            If a callable, must be a function that takes a frequency and
            returns a normalisation.
        backend : str
            the backend to use. Currently always 'jax'.
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
        kvec = fvec * 2 * jnp.pi
        A = 10.0 ** (jnp.array(A))
        omega = compute_omega_gw(self.C_mij, A)
        return self.norm(kvec) * upsample_f(kvec, self.fk, omega)

    def dtemplate_default(self, index, fvec, *A):
        kvec = fvec * 2 * jnp.pi
        A = 10.0 ** (jnp.array(A))
        domega = compute_domega_gw(self.C_mij, index, A)
        return self.norm(kvec) * upsample_f_linear(
            kvec, self.fk, domega
        )  # To avoid log(0)


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
