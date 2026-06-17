"""Top-level user-facing model: Omega_GW(f) from a kernel + perturbations.

``OmegaGW`` composes a :class:`~sigway.kernels.Kernel`, a
:class:`~sigway.perturbations.ScalarPerturbations` and an
:class:`~sigway.integrators.Integrator` (default Simpson). It owns the
inference API: a single ordered ``parameter_names`` vector (perturbation params
then kernel params, with a clear error on name collision), a
``__call__(f, *theta)``
that routes theta to the right component and applies ``kernel.norm``, and a
``jacobian`` (jax.jacfwd) for Fisher forecasts.

``__call__`` is a thin wrapper: theta is the only traced input, everything else
(kernel, perturbations, integrator, grids, norm) is static, so the analytic path
compiles once and re-runs without retracing at fixed array shapes.
"""

import jax
import jax.numpy as jnp

from sigway.integrators import SimpsonIntegrator


class OmegaGW:
    """Scalar-induced GW spectrum model.

    Parameters
    ----------
    perturbations : ScalarPerturbations
        The curvature power spectrum P_zeta(k, *pz_params).
    kernel : Kernel
        The transfer-function kernel (carries its own normalisation).
    integrator : Integrator, optional
        Integration strategy; defaults to ``SimpsonIntegrator(s, t)`` built from
        ``s`` and ``t`` if those are given instead.
    s, t : array or callable, optional
        Convenience: if ``integrator`` is not given, a SimpsonIntegrator is
        built from these grids (callables receive ``(kvec, *theta)``).
    f : array, optional
        Target frequencies (used only when ``upsample`` is True).
    upsample : bool, optional
        If True, integrate on ``f`` and interpolate onto the call frequencies.
    """

    def __init__(
        self,
        perturbations,
        kernel,
        integrator=None,
        s=None,
        t=None,
        f=None,
        upsample=False,
    ):
        if integrator is None:
            if s is None or t is None:
                raise ValueError(
                    "Provide either an integrator or both s and t grids."
                )
            integrator = SimpsonIntegrator(s, t)
        self.perturbations = perturbations
        self.kernel = kernel
        self.integrator = integrator
        self.f = None if f is None else jnp.asarray(f)
        self.upsample = upsample

        pz_names = tuple(perturbations.param_names)
        k_names = tuple(kernel.param_names)
        collisions = set(pz_names) & set(k_names)
        if collisions:
            raise ValueError(
                "Parameter name collision between perturbations and kernel: "
                "{}. Rename so every parameter has a single owner.".format(
                    sorted(collisions)
                )
            )
        self.parameter_names = pz_names + k_names
        self._n_pz = len(pz_names)

    def _split(self, theta):
        return tuple(theta[: self._n_pz]), tuple(theta[self._n_pz :])

    def __call__(self, f, *theta, **kw):
        if kw:
            if theta:
                raise ValueError(
                    "Pass parameters positionally or by keyword, not both."
                )
            theta = tuple(kw[name] for name in self.parameter_names)
        theta_pz, theta_k = self._split(theta)

        kvec_full = jnp.asarray(f) * 2 * jnp.pi
        if self.upsample:
            kvec = self.f * 2 * jnp.pi
        else:
            kvec = kvec_full
        res = self.integrator.integrate(
            self.kernel, self.perturbations, kvec, theta_pz, theta_k
        )
        if self.upsample:
            res = jnp.interp(kvec_full, kvec, res)
        return self.kernel.norm(kvec_full) * res

    def jacobian(self, f, theta):
        """Jacobian d Omega_GW(f) / d theta via forward-mode autodiff.

        Correct for parameters that enter the integrand smoothly. Parameters
        that move the integration limits (e.g. an eMD cutoff kmax) need the
        dedicated derivative path and are not handled here.
        """
        theta = jnp.asarray(theta)
        return jax.jacfwd(lambda th: self(f, *th))(theta)
