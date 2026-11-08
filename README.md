# sbi_qpm — Solar-System acceleration from quasar proper motions

Simulation-based inference (SBI) of the Solar-System barycentre acceleration
**g = (gx, gy, gz)** from the dipole of the quasar proper-motion field, using
vector spherical harmonics (VSH) as the summary statistic and neural posterior
estimation (SNPE, `sbi`).

The acceleration of the observer induces a secular aberration drift: an
apparent **E-mode (spheroidal) ℓ = 1 dipole** in the proper motions of distant
quasars, of order 5 µas/yr. The package fits the proper-motion maps with
spin-1 harmonics, simulates mock skies with noise matched to the real
catalogue, and learns the posterior on the low-ℓ multipoles from which **g**
is read off.

Catalogues: **Quaia** (G < 20) and **Gaia DR3 QSO candidates**.

---

## Paper

This code accompanies:

> Phu Huy Nguyen, Calum Murray. *Observing the Peculiar Acceleration of our
> Solar System with Quasar Proper Motions.* Astronomy & Astrophysics, 2026,
> 714, A263. [doi:10.1051/0004-6361/202660989](https://www.aanda.org/10.1051/0004-6361/202660989)

If you use this code, please cite the paper.

---

## Pipeline

```
catalogue (ra, dec, pmra, pmdec, σ, ρ)
   │  data.compute_pixel_stats         HEALPix (nside 64) inverse-variance means, |b| > galcut mask
   ▼
pixel maps  ──►  forward_model.WeightedSHT     weighted least-squares VSH fit (normal matrix + cached LU)
   │                                               → x_obs  (real E/B alm vector)
   ▼
prior  θ ~ sbi_setup.create_prior_{full,det}
   │  sbi_setup.simulator_{full,det}    θ → alm → alm2map_spin → + correlated pixel noise → VSH fit → x
   ▼
SNPE on (θ, x) pairs  →  posterior p(θ | x_obs)
   │  geometry.SkyFunction.s2g           E-mode ℓ = 1 (s10, Re s11, Im s11) → (gx, gy, gz)
   ▼
g samples  →  geometry.direction_summary (RA/Dec, l/b, 68/95 % contours)
           →  diagnostics (marginal coverage, TARP)
```

### Models

| Model | θ | dim θ | summary x | dim x |
|---|---|---|---|---|
| `full` | alm ℓ ≤ 3 (E+B, real) + Cℓ_E, Cℓ_B for ℓ = 4–10 | 30 + 14 = **44** | VSH fit to ℓ ≤ 10 | **240** |
| `det`  | alm ℓ ≤ 3 only | **30** | VSH fit to ℓ ≤ 3 | **30** |

In `full`, the ℓ = 4–10 modes are drawn as Gaussian random fields with power
Cℓ, so unmodelled small-scale structure/systematics are marginalised instead of
leaking into the dipole. Units: alm in mas/yr, Cℓ in (mas/yr)².

---

## Installation

```bash
cd sbi_qpm
pip install -e .              # core: numpy scipy pandas healpy astropy torch sbi pymaster corner matplotlib
pip install -e ".[notebooks]" # + tarp, gaiaunlimited, jupyter
pip install -e ".[test]"      # + pytest
```

`requirements.txt` is a full environment freeze kept for reference. The
results were produced with Python 3.12,
`sbi` 0.23.3, `torch` 2.5.1, `healpy` 1.19 and `pymaster` 2.6.

### Data and results paths

The input catalogues are **not** included in this repository. Put the files
listed below in `data/` at the repository root. The notebooks find them through
`DATA_DIR = Path('../data')` in their first code cell; change that line to use
another folder. Run the notebooks from `notebooks/` (the Jupyter default), as
all paths are relative to it.

`gaia_quaia_sbi_cases.ipynb` writes to the repository root (`REPO`, set in its
imports cell, section 0). It caches everything it computes in
`results/gaia_quaia_cases_v3/` (`CACHE_DIR`): simulations, trained posteriors
(`*.pt`), posterior samples, `summary.csv` and `direction_chi2.csv`. Figures go
to `output/`. Re-running reuses the cache; delete a file to force that step to
recompute. The full grid is 12 × 100 000 simulations, so keep the cache once it
is built. The cache is not part of the repository.

| File | Used by | Content |
|---|---|---|
| `quaia_G20.0_with_pmra_pmdec_corr.csv` | all | Quaia G < 20 with pmra/pmdec, errors, correlation, redshift |
| `gaia_qso_edata_5_params.csv` | all | Gaia DR3 QSO candidates, 5-parameter solutions |
| `gaia_scanning.csv` | `power_spectrum` | Gaia scanning-law / m10 map |
| `gaia_faint_stars_{90,180,270,360}.csv` | `power_spectrum` | faint-star proper-motion samples |

---

## Package modules (`sbi_qpm/`)

| Module | Main functions / classes | Purpose |
|---|---|---|
| `data.py` | `load_gaia_qso`, `compute_pixel_stats`, `compute_zbinned_pixel_stats`, `make_galactic_mask` | Load catalogue, bin to HEALPix with full 2×2 error covariance, galactic cut, redshift binning |
| `pixelization.py` | `Pixelize`, `pixelize_full_covariance` | Lower-level pixel binning (inverse-variance weighting with pmra–pmdec correlation) |
| `forward_model.py` | `WeightedSHT` | Noise-weighted spin-1 VSH least-squares fit. `build_lu(lmax)` caches the LU of the normal matrix AᵀC⁻¹A once, so every simulation is a single `solve_fast` (no refactorisation) |
| `sbi_setup.py` | `n_real_params`, `build_full_alm_complex`, `build_det_alm_complex` | θ ↔ healpy complex alm; ℓ > lmax_sig drawn from Cℓ |
| | `precompute_noise_cholesky`, `generate_pixel_noise` | Per-pixel correlated (pmra, pmdec) noise from the real errors |
| | `create_prior_full`, `create_prior_det` | BoxUniform priors (alm ∈ [−alm_bound, alm_bound], Cℓ ∈ [0, Cl_bound]), priors are taken from `plot_power_spectrum` |
| | `simulator_full`, `simulator_det` | Forward models θ → x. `simulator_full` returns `(x, pmra_map, pmdec_map)`; `simulator_det` returns `x` |
| | `build_zbinned_solvers`, `precompute_zbinned_noise` | Per-redshift-bin solver and noise set-up |
| `geometry.py` | `SkyFunction` (`s2g`, `g2s`, `vec2dir`, `dir2vec`), `icrs_to_galactic`, `direction_summary` | Dipole coefficients ↔ g vector, sky direction and credible regions |
| `diagnostics.py` | `marginal_coverage`, `run_tarp_test`, `extract_g_tarp`, `extract_g_theta` | Posterior calibration (rank coverage, TARP on the g-dipole) |
| `power_spectrum_analysis.py` | `make_bins`, `make_spin1_field`, `make_spin0_field`, `apodize_mask`, `compute_power_spectrum`, `estimate_noise_covariance`, `evaluate_cross_spectrum`, `bandpower_to_full_cl` | NaMaster pseudo-Cℓ (incl. ℓ = 1), noise covariance from simulations, SNR per ℓ and cross-spectra with systematics templates |
| `plotting.py` | `plot_mollweide`, `plot_corner`, `plot_power_spectrum`, `plot_snr_per_ell`, `plot_tarp_ecp`, … | Plot helpers |

### Minimal example

```python
import functools, numpy as np, healpy as hp, torch
from sbi.inference import SNPE, simulate_for_sbi
from sbi_qpm.data import load_gaia_qso, compute_pixel_stats
from sbi_qpm.forward_model import WeightedSHT
from sbi_qpm.sbi_setup import (n_real_params, precompute_noise_cholesky,
                               create_prior_det, simulator_det)
from sbi_qpm.geometry import SkyFunction

nside, lmax_sig, lmax_fit = 64, 3, 10
pix = compute_pixel_stats(..., nside=nside, galcut=20)          # see notebooks for the arguments
# empty pixels carry NaN errors/correlations: zero them before building the solver
sig_ra  = np.where(np.isfinite(pix['sig_pmra_wm']),  pix['sig_pmra_wm'],  0.0)
sig_dec = np.where(np.isfinite(pix['sig_pmdec_wm']), pix['sig_pmdec_wm'], 0.0)
corr    = np.nan_to_num(pix['pm_corr_wm'], nan=0.0)

solver = WeightedSHT(nside, pix['mask'], pix['pmra_wm'], pix['pmdec_wm'],
                     sig_ra, sig_dec, corr)
solver.build_lu(lmax_sig)
x_obs = solver.solve_fast(pix['pmra_wm'], pix['pmdec_wm'], lmax_sig)

npix = hp.nside2npix(nside)
L11, L21, L22, valid = precompute_noise_cholesky(sig_ra, sig_dec, corr, pix['mask'], npix)
n_alm = n_real_params(lmax_sig)                                  # 30
prior = create_prior_det(n_alm, alm_bound=50e-3)
sim = functools.partial(simulator_det, solver=solver, L11=L11, L21=L21, L22=L22,
                        valid_pix=valid, npix=npix, lmax_sig=lmax_sig,
                        lmax_fit=lmax_fit, n_alm_l3=n_alm)
```

Then train SNPE on `simulate_for_sbi(sim, prior, N)`, sample the posterior at
`x_obs`, and convert the E-mode ℓ = 1 coefficients to **g** with
`SkyFunction().s2g`. The notebooks contain the full version (ensembles, caching,
diagnostics).

---

## Notebooks (`notebooks/`)

| Notebook | What it does | Key outputs (`notebooks/output/`) |
|---|---|---|
| `quaia_sbi.ipynb` | **Reference single-case pipeline** (steps 1–10): data → pixel maps → VSH x_obs → `full` and `det` SNPE → g posterior. The only notebook that runs the **calibration checks** (marginal coverage, TARP). Switch catalogue by changing `path` in cell 4 (Quaia ↔ Gaia). | `corner_full_alm.pdf`, `corner_full_cl.pdf`, `tarp_ecp_*.pdf`, `acceleration_comparison_v2.pdf`, `gxyz_posterior_samples_quaia.npz` |
| `gaia_quaia_sbi_cases.ipynb` | **Main results grid:** {Gaia, Quaia} × galactic cut {0°, 20°, 30°} × {`full`, `det`}, 100 000 simulations each. Reports g, its sky direction (ICRS and galactic) and a direction χ² between cases. Resume-safe cache of simulations, networks and samples. | `gxyz_cases_gaia_quaia.pdf`, `direction_cases_gaia_quaia.pdf`, `gxyz_corner_{catalog}_galcut_{cut}.pdf`; `summary.csv`, `direction_chi2.csv`, `gxyz_samples.npz` in `CACHE_DIR` |
| `quaia_sbi_zbins.ipynb` | g **versus redshift**: the full pipeline per quantile z-bin of Quaia (5 bins), consistency χ² and linear-fit tests. A real acceleration must be z-independent. Resume-safe (`output/zbins/zbin_{i}_v2.npz`). | `zbins/gxyz_vs_z_quaia_v2.pdf`, `zbins/gmag_vs_z_quaia_v2.pdf`, `zbins/corner_zbins.pdf`, `zbins/chi2_consistency_*.pdf`, `zbins/linear_fit_*.pdf` |
| `power_spectrum.ipynb` | **Systematics check** with NaMaster: auto-spectra of the quasar and stellar PM fields (spin-1), stellar-density and Gaia scanning-law (spin-0) maps; cross-spectra and SNR per ℓ with noise from simulations. | `snr_per_ell_*.pdf`, `cross_spectra_panel.pdf`, `snr_cross_spectra_*.pdf`, `scanning.pdf` |

Suggested order: `quaia_sbi` (understand + validate) → `gaia_quaia_sbi_cases`
(results) → `quaia_sbi_zbins` and `power_spectrum` (robustness).

**Cost:** `gaia_quaia_sbi_cases` runs 12 × 100 000 simulations; budget hours,
and keep its cache directory.

---

## Tests

```bash
pytest            # from the sbi_qpm root; ~6 s
```

| File | Covers |
|---|---|
| `test_sbi_setup.py` | alm builders, noise, priors (dims/bounds/support), `simulator_full`/`det` (shape, dtype, finiteness), **injection–recovery**: with negligible noise the VSH fit returns the input alm, z-binned helpers |
| `test_forward_model.py` | `WeightedSHT` fit |
| `test_data.py`, `test_pixelization.py` | Pixel statistics, masks, z-binning |
| `test_geometry.py` | `s2g`/`g2s` round trip, directions, coordinate transforms |
| `test_diagnostics.py` | `marginal_coverage` (TARP is not unit-tested) |
| `test_power_spectrum_analysis.py` | Binning from ℓ = 1, band-power interpolation |
| `test_imports.py` | Every name the notebooks import from `sbi_qpm` exists |
