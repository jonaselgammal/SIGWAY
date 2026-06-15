# Global
import numpy as np

# Local
from sigway.omega_gw_jax import OmegaGWjax

"""
Omega_GW for a smooth, finite-speed early-matter -> radiation transition,
backed by the PoltergeistNet C++ pipeline (Pearce et al. arXiv:2311.12340).

Unlike the analytic "RD"/"I_MD_to_RD" kernels in omega_gw_jax, the kernel here
is solved numerically (background + perturbations + Green's functions), so the
transition sharpness ``beta_tilde`` is a genuine inference parameter that
reshapes the kernel -- not just an amplitude/tilt of P_zeta.

Design (mirrors OmegaGWjax conventions):
  * Construction fixes only the *model setup* and *numerical precision*:
    the P_zeta callable, the (s, t) integration resolution, k_max, output
    convention. NO physics parameter values live here.
  * ``__call__(f, *params)`` takes the inference parameters. ``f`` is the
    EXTERNAL LISA frequency [Hz]; ``params`` is the flat, ordered vector the
    sampler varies, split as ``(*kernel_params, *pz_params)`` where the kernel
    params are named by ``kernel_param_names`` (default just ``beta_tilde``)
    and the rest are forwarded to the P_zeta callable.

Two k's, kept strictly separate:
  * f (external) -> k_out (external GW wavenumber); handled here.
  * P_zeta(k) receives the INTERNAL source momenta k1, k2 (tilde units),
    evaluated once inside C++ (serial, GIL-safe -- never per-node).

The C++ owns the full (s, t) integral and every physical factor, so NO SIGWAY
"RD" normalisation is applied; ``output`` just selects which C++ field to
return.

Requires the compiled ``poltergeist_cpp`` extension on the import path.
"""

try:
    import poltergeist_cpp as _pc
except ImportError as exc:  # pragma: no cover - surfaced at construction
    _pc = None
    _import_error = exc


_ALLOWED_KERNEL_PARAMS = ("beta_tilde", "T_R_GeV", "k_max_tilde")
_OUTPUT_FIELDS = {"present_day": "OmegaGW0",
                  "source": "OmegaGW_etac",
                  "legacy": "OmegaGW"}


class OmegaGWPoltergeist(OmegaGWjax):
    r"""
    PoltergeistNet-backed Omega_GW for SIGWAY.

    Parameters
    ----------
    P_zeta : callable  (REQUIRED)
        Primordial spectrum ``P_zeta(k, *pz_params) -> array``. ``k`` is the
        INTERNAL source wavenumber in dimensionless ``tilde`` units
        (k_tilde = k_phys[Mpc^-1] * eta_R). Vectorised numpy/JAX is fine; it is
        evaluated once per spectrum, inside C++. There is intentionally no
        built-in power-law default -- pass one explicitly if you want it, e.g.
        ``lambda k, A_s, n_s: A_s * (k / k_pivot_tilde) ** (n_s - 1)``.
    s : int, default 5
        Number of Gauss-Legendre nodes in the s-integral. Precision handle.
    t : int, default 40
        Number of GL nodes per near-pole t-region. Precision handle. (The far
        tail uses ``n_t_far``.) Arbitrary (s, t) arrays/callables are NOT
        accepted -- see the module notes for why.
    n_t_far : int, default 20
        GL nodes in the t tail beyond the resonance.
    kernel_param_names : tuple of str, default ("beta_tilde",)
        Which kernel parameters are inferred, and their order at the front of
        ``__call__``'s ``*params``. Must include "beta_tilde"; may also include
        "T_R_GeV" and/or "k_max_tilde". Any kernel param not listed is held
        fixed at its construction value.
    T_R_GeV : float, default 100.0
        Reheating temperature; sets eta_R (the f <-> k_out map). Used as the
        fixed value when "T_R_GeV" is not in ``kernel_param_names``.
    k_max_tilde : float, default 450.0
        Nonlinear cutoff (Heaviside on P_zeta).
    kappa_tol : float, default 1e-2
        Source-mode clustering tolerance; lower = more solves = more accurate.
    bg_n_points : int, default 40000
        Background grid density.
    output : {"present_day", "source", "legacy"}, default "present_day"
        Which C++ field to return.
    Omega_r0_h2, g_eta_c : float
        Present-day transfer constants.
    f : array-like or callable or None ; upsample : bool
        Optional SIGWAY upsampling target (frequencies) and switch.
    cache_gamma_R : bool, default True
        Solve the beta-independent calibration anchor once at construction.
    """

    def __init__(
        self,
        P_zeta,
        s=5,
        t=40,
        *,
        n_t_far=20,
        kernel_param_names=("beta_tilde",),
        T_R_GeV=100.0,
        k_max_tilde=450.0,
        kappa_tol=1e-2,
        bg_n_points=40000,
        output="present_day",
        Omega_r0_h2=4.16e-5,
        g_eta_c=106.75,
        f=None,
        upsample=False,
        cache_gamma_R=True,
        **kwargs,
    ):
        if _pc is None:
            raise ImportError(
                "poltergeist_cpp extension not importable. Build it with "
                "`cmake -S . -B build -DBUILD_PYTHON=ON ...` and put the .so on "
                "the path."
            ) from _import_error

        if not callable(P_zeta):
            raise TypeError(
                "P_zeta must be a callable P_zeta(k_tilde, *pz_params) -> array. "
                "There is no built-in power-law default; define one explicitly."
            )

        # s, t are integration-resolution handles (GL node counts) for this
        # backend, not integration grids: the (s, t) range is k-dependent
        # (t in [0, 2 k_max/k - 1]) and straddles the pole at t = sqrt(3) - 1,
        # so a fixed array cannot be correct across k. See module notes.
        if not (isinstance(s, (int, np.integer)) and isinstance(t, (int, np.integer))):
            raise TypeError(
                "For this backend s and t are GL node counts (ints), not "
                "arrays/callables. Pass arbitrary grids is unsupported (the "
                "C++ builds its own pole-aware, k-dependent grid)."
            )

        self.ns_nodes = int(s)
        self.n_t_pole = int(t)
        self.n_t_far = int(n_t_far)

        # Base-class housekeeping. norm="CT" yields the inert lambda k: 1/24,
        # which we never apply (the C++ carries all physical factors). The base
        # wants real s/t arrays; pass dummies since the C++ owns the grid.
        # dP_zeta=None -> SGWBinner uses finite differences.
        super().__init__(
            P_zeta,
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            f=f,
            norm="CT",
            kernel="RD",
            upsample=upsample,
            dP_zeta=None,
            jit=False,
        )

        bad = [nm for nm in kernel_param_names if nm not in _ALLOWED_KERNEL_PARAMS]
        if bad:
            raise ValueError(
                f"unknown kernel_param_names {bad}; allowed: {_ALLOWED_KERNEL_PARAMS}"
            )
        if "beta_tilde" not in kernel_param_names:
            raise ValueError("kernel_param_names must include 'beta_tilde'")
        self.kernel_param_names = tuple(kernel_param_names)

        if output not in _OUTPUT_FIELDS:
            raise ValueError(
                f"output must be one of {tuple(_OUTPUT_FIELDS)}; got {output!r}"
            )
        self.output = output

        self.T_R_GeV = float(T_R_GeV)
        self.k_max_tilde = float(k_max_tilde)
        self.kappa_tol = float(kappa_tol)
        self.bg_n_points = int(bg_n_points)
        self.Omega_r0_h2 = float(Omega_r0_h2)
        self.g_eta_c = float(g_eta_c)

        # Scalar k[Mpc^-1] -> f[Hz] factor (linear); constants single-sourced
        # from C++, array scaling done in numpy.
        self._hz_per_k_Mpc = _pc.k_to_freq_Hz(1.0)
        self._gamma_R = _pc.solve_gamma_R_tilde() if cache_gamma_R else -1.0

    # ---- external frequency <-> external k_out (tilde) -------------------
    # NB these convert EXTERNAL quantities only; the internal source momenta
    # k1, k2 that P_zeta sees are formed inside C++.
    def freq_to_k_out_tilde(self, f_Hz, T_R_GeV=None):
        """External LISA frequency [Hz] -> external GW wavenumber k_out (tilde)."""
        T_R = self.T_R_GeV if T_R_GeV is None else float(T_R_GeV)
        f_Hz = np.asarray(f_Hz, dtype=float)
        k_Mpc = f_Hz / self._hz_per_k_Mpc
        return k_Mpc * _pc.T_R_to_eta_R(T_R)

    def k_out_tilde_to_freq(self, k_out_tilde, T_R_GeV=None):
        """External GW wavenumber k_out (tilde) -> LISA frequency [Hz]."""
        T_R = self.T_R_GeV if T_R_GeV is None else float(T_R_GeV)
        k_out_tilde = np.asarray(k_out_tilde, dtype=float)
        k_Mpc = k_out_tilde / _pc.T_R_to_eta_R(T_R)
        return k_Mpc * self._hz_per_k_Mpc

    # ---- evaluation ------------------------------------------------------
    def _split_params(self, params):
        """Split the flat *params into (kernel dict, P_zeta param tuple)."""
        nk = len(self.kernel_param_names)
        if len(params) < nk:
            raise TypeError(
                f"expected at least {nk} kernel params {self.kernel_param_names} "
                f"before the P_zeta params; got {len(params)} total"
            )
        kdict = dict(zip(self.kernel_param_names, params[:nk]))
        return kdict, tuple(params[nk:])

    def __call__(self, fvec, *params):
        r"""
        Compute :math:`\Omega_{GW}` at external LISA frequencies.

        Parameters
        ----------
        fvec : array-like
            External present-day GW frequencies [Hz].
        params : tuple
            Inference vector, ordered ``(*kernel_params, *pz_params)`` per
            ``kernel_param_names``. e.g. with the default
            ``kernel_param_names=("beta_tilde",)`` and a power-law
            ``P_zeta(k, A_s, n_s)``: ``model(f, beta_tilde, A_s, n_s)``.

        Returns
        -------
        numpy.ndarray
            Omega_GW at each frequency (the selected ``output`` field; no
            SIGWAY norm applied).
        """
        kdict, pz_params = self._split_params(params)
        beta_tilde = float(kdict["beta_tilde"])
        T_R = float(kdict.get("T_R_GeV", self.T_R_GeV))
        k_max = float(kdict.get("k_max_tilde", self.k_max_tilde))

        # f (external) -> k_out (external GW wavenumber, tilde) -- done here.
        fvec = np.asarray(fvec, dtype=float)
        k_out_tilde = self.freq_to_k_out_tilde(fvec, T_R)

        # P_zeta sees the INTERNAL source momenta (passed by C++) in tilde units.
        def pz(k):
            return np.asarray(self.P_zeta(k, *pz_params), dtype=float)

        r = _pc.compute_spectrum(
            beta_tilde,
            np.ascontiguousarray(k_out_tilde),
            k_max_tilde=k_max,
            gamma_R_tilde=self._gamma_R,
            p_zeta=pz,
            ns_nodes=self.ns_nodes,
            n_t_pole=self.n_t_pole,
            n_t_far=self.n_t_far,
            kappa_tol=self.kappa_tol,
            bg_n_points=self.bg_n_points,
            Omega_r0_h2=self.Omega_r0_h2,
            g_eta_c=self.g_eta_c,
        )
        omega = r[_OUTPUT_FIELDS[self.output]]

        if self.upsample and self.f is not None:
            f_target = self.f(*params) if callable(self.f) else self.f
            omega = np.interp(np.asarray(f_target, dtype=float), fvec, omega)
        return omega
