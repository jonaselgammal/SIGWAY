# Reference Omega_GW spectra
Regenerate with `python scripts/generate_reference_spectra.py`.
Each `.npz` holds `f`, `omega_gw`, `params`, the validation arrays (`val_f/val_sigway/val_oracle/val_relerr`) and a JSON `meta`.
Every spectrum is validated against an **independent** numpy/scipy oracle (`tests/_sigway_oracle.py`) sharing no code with `sigway`.

| fixture | physics | validation method | max rel err | tol |
|---|---|---|---|---|
| `bpl_rd` | Broken power law, radiation domination (template_pipeline.ipynb) | dense scipy-Simpson, textbook RD kernel (omega_RD_oracle) | 2.98e-02 | 5% |
| `lognormal_rd` | Log-normal peak, radiation domination (template/lognormal) | dense scipy-Simpson, textbook RD kernel (omega_RD_oracle) | 2.24e-02 | 3% |
| `osc_multifield_rd` | Multifield oscillations, RD (template/oscillations_multifield) | dense scipy-Simpson, textbook RD kernel (omega_RD_oracle) | 1.16e-02 | 3% |
| `emd_imd2rd` | Flat+cutoff source, instant eMD->RD kernel (eMD) | scipy dblquad/quad, eMD kernels (peak band) | 2.23e-03 | 5% |
| `usr_ms` | Ultra-slow-roll single-field inflation via Mukhanov-Sasaki solver | numpy oracle fed the MS-solver P_zeta(k) (independent) | 3.56e-03 | 2% |
