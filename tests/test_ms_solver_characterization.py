"""Characterization (golden-value) tests for the Mukhanov-Sasaki solver.

These lock the *current* numerical behaviour of ``SingleFieldSolver`` so the
planned refactor (merging the solver into ``SingleFieldPerturbations``, deduping
``run``/``__call__`` and ``run_perturbations``/``_mode_history``, moving the EOM
cores to staticmethods, extracting the diagnostics/plots) can be shown to be
behaviour-preserving. They are snapshots, not physics assertions -- the physics
lives in ``test_ms_solver.py`` and ``test_omega_gw_regression.py``.

Two anchor potentials, both already validated elsewhere in the suite:
  * quadratic chaotic ``V = m^2 phi^2 / 2`` -- featureless slow roll;
  * ultra-slow-roll (USR) inflection-point model -- the enhanced-peak case,
    which exercises the non-trivial dynamics where a refactor bug would hide.

Golden fixture: ``tests/test_data/ms_solver_golden.npz``. To regenerate after an
*intended* behaviour change, run this file as a script:
    python tests/test_ms_solver_characterization.py
and eyeball the diff before committing the new .npz.
"""
import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from sigway.single_field import SingleFieldPerturbations
import _sigway_configs as C

jax.config.update("jax_enable_x64", True)

_GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "test_data", "ms_solver_golden.npz"
)

# Per-key (rtol, atol). Tight for the deterministic solve; looser only for the
# interpolant-at-nodes and the raw trajectory endpoint.
_TOL = {
    "pzeta": (1e-8, 0.0),
    "call": (1e-8, 0.0),
    "run_nodes": (1e-6, 0.0),
    "Nend": (1e-6, 0.0),
    "phi_a": (1e-6, 0.0),
    "y_a": (1e-6, 0.0),
    "h_a": (1e-6, 0.0),
    "eps_a": (1e-6, 0.0),
    "eta_a": (1e-6, 0.0),
    "psr_a": (1e-6, 0.0),
    "ns_c": (1e-6, 0.0),
    "r_c": (1e-6, 0.0),
    "eps_c": (1e-6, 0.0),
    "eta_c": (1e-6, 0.0),
    "single_final": (1e-6, 1e-14),
    "lograt": (1e-8, 0.0),
    "k_rep": (0.0, 0.0),
}


def _V_quadratic(phi, m):
    return 0.5 * m**2 * phi**2


def _build_configs():
    quad = dict(
        name="quadratic",
        solver=SingleFieldPerturbations(
            _V_quadratic, ("m",), phi0=16.0, N_CMB_to_end=55.0,
        ),
        k=jnp.geomspace(1e-4, 1e-1, 40),
        params=(6e-6,),
    )
    usr = dict(
        name="usr",
        solver=SingleFieldPerturbations(
            C.usr_potential, ("a", "lam", "v", "nfac"),
            phi0=C.USR_CONFIG["phi0"],
            N_CMB_to_end=C.USR_CONFIG["N_CMB_to_end"],
        ),
        k=jnp.array(C.USR_CONFIG["k_solver"]),
        params=tuple(C.USR_CONFIG["params"]),
    )
    return {c["name"]: c for c in (quad, usr)}


def _compute(cfg):
    """All characterized quantities for one config (mirrors the generator)."""
    s, k, params = cfg["solver"], cfg["k"], cfg["params"]
    p = jnp.array(params)
    N, phi, y, h = s.run_background(p)

    pzeta = np.asarray(s.run_perturbations(k, N, phi, y, h, p))
    call = np.asarray(s(k, *params))
    run_nodes = np.asarray(s.prepare(k, *params)(k))

    Nend = float(jnp.max(N))
    anchors = jnp.array([Nend - 10.0, Nend - 30.0])
    phi_a = np.asarray(jnp.interp(anchors, N, phi))
    y_a = np.asarray(jnp.interp(anchors, N, y))
    h_a = np.asarray(jnp.interp(anchors, N, h))

    N_CMB = Nend - s.N_CMB_to_end
    phi_c = float(jnp.interp(N_CMB, N, phi))
    y_c = float(jnp.interp(N_CMB, N, y))
    h_c = float(jnp.interp(N_CMB, N, h))
    eps_c = float(s.epsilon_h(y_c))
    eta_c = float(s.eta_h(phi_c, y_c, h_c, p))

    eps_a = np.asarray(s.epsilon_h(jnp.asarray(y_a)))
    eta_a = np.asarray([float(s.eta_h(pa, ya, ha, p))
                        for pa, ya, ha in zip(phi_a, y_a, h_a)])
    psr_a = np.asarray(s.pzeta_sr(jnp.asarray(y_a), jnp.asarray(h_a), p))

    k_rep = float(np.asarray(k)[len(np.asarray(k)) // 2])
    sol, lograt = s._mode_history(k_rep, N, phi, y, h, p)
    ys = np.asarray(sol.ys)
    valid = np.all(np.isfinite(ys), axis=1)
    single_final = ys[valid][-1][[1, 3, 5]]

    return dict(
        pzeta=pzeta, call=call, run_nodes=run_nodes,
        Nend=np.asarray(Nend), phi_a=phi_a, y_a=y_a, h_a=h_a,
        eps_a=eps_a, eta_a=eta_a, psr_a=psr_a,
        ns_c=np.asarray(float(s.n_s(eps_c, eta_c))),
        r_c=np.asarray(float(s.r(eps_c))),
        eps_c=np.asarray(eps_c), eta_c=np.asarray(eta_c),
        single_final=single_final, lograt=np.asarray(float(lograt)),
        k_rep=np.asarray(k_rep),
    )


@pytest.fixture(scope="module")
def computed():
    """Solve both configs once (expensive) and reuse across tests."""
    return {name: _compute(cfg) for name, cfg in _build_configs().items()}


@pytest.fixture(scope="module")
def golden():
    if not os.path.exists(_GOLDEN_PATH):
        pytest.skip(
            "ms_solver_golden.npz missing; regenerate with "
            "`python tests/test_ms_solver_characterization.py`"
        )
    return np.load(_GOLDEN_PATH)


@pytest.mark.parametrize("name", ["quadratic", "usr"])
def test_ms_solver_golden(name, computed, golden):
    """Every characterized quantity matches the stored golden snapshot."""
    got = computed[name]
    for key, val in got.items():
        rtol, atol = _TOL[key]
        ref = golden[f"{name}__{key}"]
        assert np.all(np.isfinite(val)), f"{name}/{key} not finite"
        np.testing.assert_allclose(
            val, ref, rtol=rtol, atol=atol,
            err_msg=f"{name}/{key} drifted from golden",
        )


@pytest.mark.parametrize("name", ["quadratic", "usr"])
def test_call_equals_run_perturbations(name, computed):
    """Invariant the run/__call__ dedup preserves: same array both ways."""
    got = computed[name]
    np.testing.assert_allclose(got["call"], got["pzeta"], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("name", ["quadratic", "usr"])
def test_run_interpolant_reproduces_nodes(name, computed):
    """run() spline evaluated at the solve nodes returns run_perturbations."""
    got = computed[name]
    np.testing.assert_allclose(
        got["run_nodes"], got["pzeta"], rtol=1e-6, atol=0.0
    )


def _regenerate():
    os.makedirs(os.path.dirname(_GOLDEN_PATH), exist_ok=True)
    flat = {}
    for name, cfg in _build_configs().items():
        for key, val in _compute(cfg).items():
            flat[f"{name}__{key}"] = val
    np.savez(_GOLDEN_PATH, **flat)
    print(f"wrote {_GOLDEN_PATH} ({len(flat)} arrays)")


if __name__ == "__main__":
    _regenerate()
