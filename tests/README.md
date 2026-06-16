# SIGWAY test suite

This suite is the **safety net for the planned composition refactor** (separate
`Kernel` / `PowerSpectrum` / `Integrator` classes). It therefore targets
**physics and the public `Omega_GW(f)`-given-config surface**, not internal
helpers that will move. Every test encodes an *independent* expectation (a closed
form, an analytic limit, a scaling law, or an independent integrator) — never
"runs / finite / positive".

Run with the project interpreter and x64:

```
python -m pytest tests/        # 53 tests
```

## Layout

| file | what it protects | independent reference | key tolerance |
|---|---|---|---|
| `test_kernels_rd.py` | RD transfer function `I_sq_RD`/`I_sq_RD_uv`; parity, non-negativity, k-independence | textbook `overline{I^2}(u,v)` (numpy) | rtol 1e-10 |
| `test_special_functions.py` | Si/Ci table + derivative behind the eMD kernel | `scipy.special.sici`, finite differences | rtol 1e-9 / 1e-4 |
| `test_simpson_nd.py` | 1-D **and N-D** Simpson contract (the regression the fix repairs) | `scipy.integrate.simpson` | rtol 1e-10 |
| `test_omega_gw_regression.py` | end-to-end `Omega_GW(f)` for 4 analytic configs + USR/MS vs validated fixtures | stored, oracle-validated fixtures | rtol 1e-4 |
| `test_invariants.py` | A² bilinearity; resonance peak `2/sqrt3 k_*`; IR `k^3 ln^2` causal tail; eMD source cutoff | analytic limits / scaling laws | 1e-10 / 5% / slope / 1e-4 |
| `test_cross_backend.py` | SIGWAY vs numpy/scipy oracle; `OmegaGWms` vs `OmegaGWjax` on the same P_ζ | independent integrator | 3–5% / rtol 1e-4 |
| `test_convergence.py` | `(s,t)` convergence; default within tolerance; documents two grid pathologies | self-convergence | 1–2% |
| `test_differentiability.py` | hand-coded eMD `etaR` gradients and `d_integrate` vs finite differences | central differences / closed form | rtol 1e-4–1e-5 |
| `test_ms_solver.py` | MS P_ζ tilt = slow-roll `n_s-1=-2/N`; full P_ζ → slow-roll limit | analytic slow roll | 0.01 in n_s / 5% |
| `test_omega_gw_jax.py`, `test_utils.py` | pre-existing snapshot/scipy checks (kept) | — | — |

Helpers (underscore-prefixed, not collected): `_sigway_configs.py` (the paper
P_ζ forms, t-grids, parameters — single source of truth shared with the fixture
generator) and `_sigway_oracle.py` (the independent numpy/scipy integrator).

## Reference fixtures

`test_data/reference/*.npz` + `MANIFEST.md`. Regenerate with:

```
python scripts/generate_reference_spectra.py
```

Each fixture is **validated against the independent oracle** at generation time
(the script asserts the max relative error is within the documented tolerance and
records the method in `MANIFEST.md`), so a fixture can never be an unvalidated
snapshot. Validation tolerances (2–5%) exceed 0.1% only because of the resonance
log-divergence and the finite paper s/t grids; the regression comparison itself
is tight (rtol 1e-4).

## The `simpson_nonuniform` fix

On `1-add-tests`, commit `9b01838` ("utils now GPU compatible") replaced the
shape dispatch in `simpson_nonuniform` with `jax.lax.switch`, whose branches
return mismatched shapes. `lax.switch` traces all branches and requires equal
output types, so **every N-D call raised** — i.e. the entire public `OmegaGW…`
integration path was broken on this branch (it works on `main`). The shapes are
static under `jit`, so the fix restores `main`'s plain `if x.shape != f_shape:`
dispatch (committed separately). `test_simpson_nd.py` pins the N-D contract so
the regression cannot return silently.

## Coverage gaps (intentional)

- **Constant-w kernel** (`I_sq_w`, Legendre) and **poltergeist** finite-transition
  kernel — not present on `1-add-tests` (constant-w is on an older fork;
  `omega_gw_poltergeist.py` is only on the `poltergeist` branch). The
  `w → 1/3 → RD` and instant-vs-finite-transition limits are **deferred** until
  those land here. (The `poltergeist_cpp` extension imports, but its sigway
  wrapper does not exist on this branch.)
- **eMD IR/UV tails**: the paper's 100-pt geomspace t-grid under-resolves the
  `t^4` large-V integrand for `k << kmax` (and the steep `k > kmax` fall-off).
  The eMD fixture is validated only in the `0.5–1.0 kmax` band; the limitation is
  demonstrated (not hidden) by `test_convergence.test_emd_…`.
- **Oscillatory tail** of the multifield config (`kappa ≳ 0.7`) is grid-sensitive;
  its physics cross-check is scoped to the smooth envelope, while the regression
  test still pins the full spectrum.
- **USR notebook t-grid**: `logspace(-3,3,1000)` is ~13% under-resolved; the
  fixture uses a converged linear-low-t grid (same point count). Documented in
  `test_convergence.test_usr_t_spacing_matters_not_count`.
- **`Binned_P_zeta`**: uses precomputed coefficients and a different
  normalisation convention; not yet cross-checked against the integrator/oracle.
