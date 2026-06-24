"""Top-level user-facing model: Omega_GW(f) from a kernel + perturbations.

[OmegaGW][sigway.spectrum.OmegaGW] composes a [Kernel][sigway.kernels.Kernel],
a [ScalarPerturbations][sigway.perturbations.ScalarPerturbations] and an
[Integrator][sigway.integrators.Integrator] (default Simpson). It owns the
inference API: a single ordered ``parameter_names`` vector (perturbation params
then kernel params, with a clear error on name collision), a
``__call__(f, *theta)``
that routes theta to the right component and applies ``kernel.norm``, and a
``jacobian`` (jax.jacfwd) for Fisher forecasts.

``__call__`` is a thin wrapper: theta is the only traced input, everything else
(kernel, perturbations, integrator, grids, norm) is static, so the analytic path
compiles once and re-runs without retracing at fixed array shapes.
"""

__all__ = ["OmegaGW"]

import jax
import jax.numpy as jnp

from sigway.integrators import SimpsonIntegrator


class OmegaGW:
    """Scalar-induced GW spectrum model.

    Composes a [Kernel][sigway.kernels.Kernel],
    a [ScalarPerturbations][sigway.perturbations.ScalarPerturbations], and an
    [Integrator][sigway.integrators.Integrator] into a callable that evaluates
    Omega_GW(f) for a given parameter vector. Exposes a unified
    ``parameter_names`` tuple (perturbation parameters first, then kernel
    parameters) and a ``jacobian`` method for Fisher-matrix forecasts.

    Parameters
    ----------
    perturbations : ScalarPerturbations
        The curvature power spectrum P_zeta(k, *pz_params).
    kernel : Kernel
        The transfer-function kernel (carries its own normalisation).
    integrator : Integrator, optional
        Integration strategy; defaults to a
        [SimpsonIntegrator][sigway.integrators.SimpsonIntegrator] built from
        ``s`` and ``t`` if those are provided instead.
    s : array or callable, optional
        First integration-grid argument passed to
        [SimpsonIntegrator][sigway.integrators.SimpsonIntegrator] when
        ``integrator`` is not given. Callables receive ``(kvec, *theta)``.
    t : array or callable, optional
        Second integration-grid argument passed to
        [SimpsonIntegrator][sigway.integrators.SimpsonIntegrator] when
        ``integrator`` is not given. Callables receive ``(kvec, *theta)``.
    f : array, optional
        Fixed frequency grid used for integration when ``upsample`` is True.
        Results are then interpolated onto the call frequencies.
    upsample : bool, optional
        If True, integrate on the fixed ``f`` grid and interpolate onto the
        frequencies passed at call time. Requires ``f`` to be set.

    Attributes
    ----------
    perturbations : ScalarPerturbations
        The curvature power spectrum object supplied at construction.
    kernel : Kernel
        The transfer-function kernel object supplied at construction.
    integrator : Integrator
        The integration strategy (either supplied directly or built from
        ``s`` and ``t``).
    f : jax.Array or None
        Fixed frequency grid for upsampling, or None.
    upsample : bool
        Whether upsampling is active.
    parameter_names : tuple of str
        Ordered parameter names: perturbation parameters followed by kernel
        parameters. No duplicates are allowed; a ``ValueError`` is raised on
        name collision at construction time.
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
            raise ValueError(
                "upsample=True requires 'f' to be provided at construction."
            )
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
        """Evaluate Omega_GW at the given frequencies.

        Parameters are routed to the perturbation and kernel components in the
        order defined by ``parameter_names``. The kernel's normalisation factor
        is applied to the integration result before returning.

        Parameters
        ----------
        f : array-like
            Frequencies in Hz at which to evaluate Omega_GW.
        *theta : float
            Model parameters in the order given by ``parameter_names``
            (perturbation parameters first, then kernel parameters). Mutually
            exclusive with keyword arguments.
        **kw : float
            Alternative to positional ``*theta``: supply parameters by name.
            All names in ``parameter_names`` must be provided; extras raise a
            ``ValueError``. Mutually exclusive with positional ``*theta``.

        Returns
        -------
        jax.Array
            Omega_GW evaluated at each frequency in ``f``, shape ``(len(f),)``.

        Raises
        ------
        ValueError
            If both positional and keyword parameters are supplied, or if the
            keyword arguments do not match ``parameter_names`` exactly.
        """
        if kw:
            if theta:
                raise ValueError(
                    "Pass parameters positionally or by keyword, not both."
                )
            extra = set(kw) - set(self.parameter_names)
            missing = set(self.parameter_names) - set(kw)
            if extra or missing:
                raise ValueError(
                    "Keyword parameters must match parameter_names exactly; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}."
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
        """Compute the Jacobian d Omega_GW(f) / d theta.

        Smooth parameters use forward-mode autodiff (``jax.jacfwd``). Parameters
        that enter a step function or an integration limit (e.g. an eMD cutoff
        kmax) cannot be differentiated correctly by autodiff, so their column is
        replaced by a central finite-difference estimate. The set of
        finite-difference parameters defaults to the union of
        ``perturbations.nonsmooth_params`` and ``kernel.nonsmooth_params``;
        pass ``fd_params`` to override this.

        Not available when ``perturbations`` is not JAX-jittable (e.g. the MS
        solver path).

        Parameters
        ----------
        f : array-like
            Frequencies in Hz, shape ``(N,)``.
        theta : array-like
            Parameter vector in the order given by ``parameter_names``,
            shape ``(len(parameter_names),)``.
        fd_params : sequence of str, optional
            Names of parameters to differentiate with central finite differences
            instead of autodiff. Defaults to the union of
            ``perturbations.nonsmooth_params`` and ``kernel.nonsmooth_params``.

        Returns
        -------
        jax.Array
            Jacobian matrix of shape ``(N, len(parameter_names))``, where entry
            ``[i, j]`` is d Omega_GW(f[i]) / d theta[j].

        Raises
        ------
        ValueError
            If ``perturbations`` is not JAX-jittable (e.g. the MS solver path).
        """
        if not getattr(self.perturbations, "jittable", True):
            raise ValueError(
                "OmegaGW.jacobian is not available for non-jittable "
                "perturbations (e.g. SingleFieldPerturbations)."
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
