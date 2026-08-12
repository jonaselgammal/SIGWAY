r"""Top-level model for the scalar-induced GW energy-density spectrum.

[OmegaGW][sigway.spectrum.OmegaGW] is the main entry point for computing
$\Omega_{\mathrm{GW}}(f)$ from a primordial scalar power spectrum
$\mathcal{P}_\zeta(k)$.  It combines a
[Kernel][sigway.kernels.Kernel],
a [ScalarPerturbations][sigway.perturbations.ScalarPerturbations], and an
[Integrator][sigway.integrators.Integrator] (Simpson's rule by default).

Calling the model returns $\Omega_{\mathrm{GW}}(f)$ for a given set of
parameters.  The first call is slower because the calculation is compiled;
subsequent calls are fast — convenient when scanning over a parameter grid.
"""

__all__ = ["OmegaGW"]

import jax
import jax.numpy as jnp

from sigway.integrators import SimpsonIntegrator


class OmegaGW:
    r"""Scalar-induced gravitational-wave energy-density spectrum
    $\Omega_{\mathrm{GW}}(f)$.

    Combines a primordial scalar power spectrum $\mathcal{P}_\zeta(k)$, a
    transfer kernel and a numerical integrator. The induced tensor power
    spectrum is the double integral over the rescaled internal momenta
    $s\in[0,1]$ and $t\in[0,\infty)$,

    $$
    \overline{\mathcal{P}_h(k)} =
    \int_0^{\infty}\!\mathrm{d}t \int_0^{1}\!\mathrm{d}s\;
    \mathcal{N}(t,s)\,\overline{I^2(u,v,k\eta)}\;
    \mathcal{P}_\zeta(u k)\,\mathcal{P}_\zeta(v k),
    $$

    with $u=(t+s+1)/2$, $v=(t-s+1)/2$, the geometric factor $\mathcal{N}(t,s)$
    ([polynomial][sigway.kernels.polynomial]) and the kernel $\overline{I^2}$.
    `OmegaGW` applies the kernel normalisation and converts this to the
    energy-density spectrum $\Omega_{\mathrm{GW}}(f)$ at $f = k/2\pi$. See
    [Theory: the master formula](../theory/index.md#the-master-formula) and the
    [$(s,t)$ reparameterisation](../theory/index.md#in-the-codes-st-variables).

    Parameters
    ----------
    perturbations : ScalarPerturbations
        The primordial curvature power spectrum $\mathcal{P}_\zeta(k)$ together
        with its free parameters (e.g. amplitude, peak scale).  See
        [AnalyticPerturbations][sigway.perturbations.AnalyticPerturbations].
    kernel : Kernel
        The transfer kernel $\overline{I^2(u,v,k\eta)}$ that encodes the
        cosmological background during GW production (radiation domination,
        early matter domination, …) and carries the overall normalisation. See
        [RadiationKernel][sigway.kernels.RadiationKernel] and
        [Theory: the kernel](../theory/index.md#the-kernel).
    integrator : Integrator, optional
        Numerical integration strategy over $(s, t)$.  When omitted, a
        [SimpsonIntegrator][sigway.integrators.SimpsonIntegrator] is built
        automatically from the ``s`` and ``t`` grids supplied below.
    s : array-like or callable, optional
        Grid of $s$ values passed to
        [SimpsonIntegrator][sigway.integrators.SimpsonIntegrator] when
        ``integrator`` is not provided.  A callable receives
        ``(k, *theta)`` and returns the grid dynamically.
    t : array-like or callable, optional
        Grid of $t$ values passed to
        [SimpsonIntegrator][sigway.integrators.SimpsonIntegrator] when
        ``integrator`` is not provided.  A callable receives
        ``(k, *theta)`` and returns the grid dynamically.
    interp_grid : array-like, optional
        A fixed internal frequency grid (Hz).  When provided, the (expensive)
        integration is performed only on this grid and the result is
        interpolated onto the frequencies requested at call time — useful for
        speeding up repeated evaluations at many different, finer frequency
        arrays.  When ``None`` (the default) the spectrum is integrated
        directly at the call-time frequencies and no interpolation happens.
    interp : {"linear", "loglog"}, optional
        Interpolation used when ``interp_grid`` is set.  ``"linear"`` (the
        default) interpolates $\Omega_{\mathrm{GW}}$ linearly in $f$;
        ``"loglog"`` interpolates $\log\Omega_{\mathrm{GW}}$ linearly in
        $\log f$, which is exact for a power law — pass a ``geomspace``
        ``interp_grid`` to interpolate the spectrum in log–log space.

    Attributes
    ----------
    perturbations : ScalarPerturbations
        The primordial power spectrum object.
    kernel : Kernel
        The transfer kernel object.
    integrator : Integrator
        The numerical integrator (built from ``s``/``t`` if not supplied
        directly).
    interp_grid : jax.Array or None
        Internal frequency grid the integration runs on, or ``None`` when the
        spectrum is integrated directly at the call-time frequencies.
    interp : str
        Interpolation mode (``"linear"`` or ``"loglog"``) used when
        ``interp_grid`` is set.
    parameter_names : tuple of str
        All free parameters of the model in evaluation order: perturbation
        parameters first, then kernel parameters.  Parameters must have
        unique names; a ``ValueError`` is raised at construction if any
        name appears in both components.

    Examples
    --------
    Compute $\Omega_{\mathrm{GW}}(f)$ for a log-normal peak in
    $\mathcal{P}_\zeta$ centred near $k_* = 10^{-2}\,\mathrm{s}^{-1}$:

    >>> import jax.numpy as jnp
    >>> from sigway.spectrum import OmegaGW
    >>> from sigway.kernels import RadiationKernel
    >>> from sigway.perturbations import AnalyticPerturbations
    >>> def pz(k, logAs, logks):
    ...     return 10.0**logAs * jnp.exp(-0.5*(jnp.log(k/10**logks)/0.3)**2)
    >>> model = OmegaGW(AnalyticPerturbations(pz, ("logAs", "logks")),
    ...                 RadiationKernel(),
    ...                 s=jnp.linspace(0, 1, 10),
    ...                 t=jnp.geomspace(1e-4, 1e3, 800))
    >>> omega = model(jnp.geomspace(1e-5, 1e-1, 200), -2.0, -2.0)
    """

    def __init__(
        self,
        perturbations,
        kernel,
        integrator=None,
        s=None,
        t=None,
        interp_grid=None,
        interp="linear",
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
        self.interp_grid = (
            None if interp_grid is None else jnp.asarray(interp_grid)
        )
        if interp not in ("linear", "loglog"):
            raise ValueError(
                "interp must be 'linear' or 'loglog', got {!r}.".format(interp)
            )
        self.interp = interp
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
        r"""Return $\Omega_{\mathrm{GW}}(f)$ at the requested frequencies.

        Each parameter value is forwarded automatically to the perturbation
        spectrum or the kernel, according to the order in ``parameter_names``.
        The kernel's normalisation prefactor is applied before returning.

        The first call compiles the integration loop; subsequent calls with
        the same array shapes run at full speed without recompilation.

        Parameters
        ----------
        f : array-like
            Frequencies $f$ in Hz at which to evaluate $\Omega_{\mathrm{GW}}$.
        *theta : float
            Model parameters in the order given by ``parameter_names``
            (perturbation parameters first, then kernel parameters).
            Cannot be used together with keyword arguments.
        **kw : float
            Alternative to positional ``*theta``: pass parameters by name.
            All names in ``parameter_names`` must be supplied; extra names
            raise a ``ValueError``.  Cannot be used together with ``*theta``.

        Returns
        -------
        jax.Array
            $\Omega_{\mathrm{GW}}$ evaluated at each frequency in ``f``,
            shape ``(len(f),)``.

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
        if self.interp_grid is not None:
            kvec = self.interp_grid * 2 * jnp.pi
        else:
            kvec = kvec_full
        res = self.integrator.integrate(
            self.kernel, self.perturbations, kvec, theta_pz, theta_k
        )
        if self.interp_grid is not None:
            if self.interp == "loglog":
                res = jnp.exp(
                    jnp.interp(
                        jnp.log(kvec_full), jnp.log(kvec), jnp.log(res)
                    )
                )
            else:
                res = jnp.interp(kvec_full, kvec, res)
        return self.kernel.norm(kvec_full) * res

    def jacobian(self, f, theta, fd_params=None):
        r"""Compute the parameter derivatives
        $\partial\Omega_{\mathrm{GW}}(f_i)/\partial\theta_j$.

        Returns the Jacobian matrix needed to build a Fisher information matrix
        for forecasting parameter constraints.  For a detector with noise power
        spectral density $S_n(f)$, the Fisher matrix element is

        $$
        F_{jk} = \sum_i
        \frac{\partial\Omega_i}{\partial\theta_j}
        \frac{\partial\Omega_i}{\partial\theta_k}
        / S_n(f_i)^2.
        $$

        Derivatives with respect to smooth parameters are computed by
        automatic differentiation.  Parameters that appear inside a step
        function or as an integration boundary (e.g. a sharp cutoff scale
        $k_{\max}$ in an early-matter-domination scenario) are not smooth and
        would give wrong derivatives from automatic differentiation; their
        columns are replaced by a central finite-difference estimate instead.
        The set of such non-smooth parameters is read automatically from
        ``perturbations.nonsmooth_params`` and ``kernel.nonsmooth_params``;
        use ``fd_params`` to override.

        Not available when the perturbation spectrum requires a full numerical
        mode-function integration (e.g.
        [SingleFieldPerturbations][sigway.single_field.SingleFieldPerturbations]).

        Parameters
        ----------
        f : array-like
            Frequencies $f$ in Hz, shape ``(N,)``.
        theta : array-like
            Parameter vector $\theta$ in the order given by
            ``parameter_names``, shape ``(P,)`` where $P$ is the total number
            of free parameters.
        fd_params : sequence of str, optional
            Names of parameters whose derivative column should be computed by
            central finite differences rather than automatic differentiation.
            Defaults to the union of ``perturbations.nonsmooth_params`` and
            ``kernel.nonsmooth_params``.

        Returns
        -------
        jax.Array
            Jacobian matrix of shape ``(N, P)``, where entry ``[i, j]`` is
            $\partial\Omega_{\mathrm{GW}}(f_i)/\partial\theta_j$.

        Raises
        ------
        ValueError
            If the perturbation object does not support automatic
            differentiation (e.g.
            [SingleFieldPerturbations][sigway.single_field.SingleFieldPerturbations]).
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
