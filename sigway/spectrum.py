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
        if self.upsample and self.f is None:
            raise ValueError("upsample=True requires 'f' to be provided at construction.")
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
        # params whose jacobian column needs finite differences (step / limit)
        self._nonsmooth = tuple(
            getattr(perturbations, "nonsmooth_params", ())
        ) + tuple(getattr(kernel, "nonsmooth_params", ()))

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

    def jacobian(self, f, theta, fd_params=None):
        """Jacobian d Omega_GW(f) / d theta.

        Smooth parameters use forward-mode autodiff (jax.jacfwd). Parameters
        that enter a step or an integration limit (e.g. an eMD cutoff kmax)
        cannot be autodiffed correctly, so their column is computed with central
        finite differences. ``fd_params`` (names) overrides which parameters get
        the finite-difference treatment; by default it is the union of the
        perturbation's and kernel's ``nonsmooth_params``.

        Not available for the MS solver path (it is not differentiable).
        """
        if not getattr(self.perturbations, "jittable", True):
            raise ValueError(
                "OmegaGW.jacobian is not available for non-jittable perturbations "
                "(e.g. SingleFieldPerturbations)."
            )
        theta = jnp.asarray(theta)
        jac = jax.jacfwd(lambda th: self(f, *th))(theta)
        fd = self._nonsmooth if fd_params is None else fd_params
        for name in fd:
            i = self.parameter_names.index(name)
            h = 1e-5 * max(abs(float(theta[i])), 1.0)
            col = self(f, *theta.at[i].add(h)) - self(f, *theta.at[i].add(-h))
            jac = jac.at[:, i].set(col / (2.0 * h))
        return jac
