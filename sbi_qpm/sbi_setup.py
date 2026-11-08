import math
import numpy as np
import healpy as hp
import torch
from sbi import utils as sbi_utils


def n_real_params(lmax):
    """Number of real alm parameters (E+B) for spin-1, ell starts at 1."""
    return 2 * ((lmax + 1) + (math.comb(lmax + 2, 2) - (lmax + 1)) * 2) - 2


def build_full_alm_complex(theta_l3, Cl_E_4_to_10, Cl_B_4_to_10, lmax_sig, lmax_fit):
    """Build complex (E_alm, B_alm) for lmax_fit.

    l <= lmax_sig : deterministic, from theta_l3.
    l > lmax_sig  : Gaussian random draws from Cl_E / Cl_B.
      m=0: a_{l,0} ~ N(0, Cl)
      m>0: Re, Im  ~ N(0, Cl/2)
    """
    n_alm_fit = hp.Alm.getsize(lmax_fit)
    n_alm_sig = hp.Alm.getsize(lmax_sig)
    half = n_real_params(lmax_sig) // 2

    E_r, B_r = theta_l3[:half], theta_l3[half:]

    E_sig = np.zeros(n_alm_sig, dtype=complex)
    B_sig = np.zeros(n_alm_sig, dtype=complex)
    E_sig[1:lmax_sig + 1] = E_r[0:lmax_sig]
    B_sig[1:lmax_sig + 1] = B_r[0:lmax_sig]
    E_sig[lmax_sig + 1:] = E_r[lmax_sig::2] + 1j * E_r[lmax_sig + 1::2]
    B_sig[lmax_sig + 1:] = B_r[lmax_sig::2] + 1j * B_r[lmax_sig + 1::2]

    E_full = np.zeros(n_alm_fit, dtype=complex)
    B_full = np.zeros(n_alm_fit, dtype=complex)

    for l in range(1, lmax_sig + 1):
        for m in range(0, l + 1):
            i_s = hp.Alm.getidx(lmax_sig, l, m)
            i_f = hp.Alm.getidx(lmax_fit, l, m)
            E_full[i_f] = E_sig[i_s]
            B_full[i_f] = B_sig[i_s]

    for k, l in enumerate(range(lmax_sig + 1, lmax_fit + 1)):
        cl_E = max(float(Cl_E_4_to_10[k]), 0.0)
        cl_B = max(float(Cl_B_4_to_10[k]), 0.0)
        idx0 = hp.Alm.getidx(lmax_fit, l, 0)
        E_full[idx0] = np.random.normal(0.0, np.sqrt(cl_E))
        B_full[idx0] = np.random.normal(0.0, np.sqrt(cl_B))
        for m in range(1, l + 1):
            idx = hp.Alm.getidx(lmax_fit, l, m)
            E_full[idx] = (
                np.random.normal(0.0, np.sqrt(cl_E / 2.0))
                + 1j * np.random.normal(0.0, np.sqrt(cl_E / 2.0))
            )
            B_full[idx] = (
                np.random.normal(0.0, np.sqrt(cl_B / 2.0))
                + 1j * np.random.normal(0.0, np.sqrt(cl_B / 2.0))
            )

    return E_full, B_full


def build_det_alm_complex(theta_l3, lmax_sig, lmax_fit):
    """Build alm for lmax_fit using only deterministic l <= lmax_sig modes."""
    return build_full_alm_complex(
        theta_l3,
        np.zeros(lmax_fit - lmax_sig),
        np.zeros(lmax_fit - lmax_sig),
        lmax_sig,
        lmax_fit,
    )


def precompute_noise_cholesky(sig_pmra, sig_pmdec, pm_corr, mask, npix):
    """Pre-compute per-pixel Cholesky factors for the 2x2 noise covariance.

    Returns (L11, L21, L22, valid_pix) for use with generate_pixel_noise.
    """
    valid_pix = (
        mask
        & np.isfinite(sig_pmra) & (sig_pmra > 0)
        & np.isfinite(sig_pmdec) & (sig_pmdec > 0)
    )
    sig_ra = sig_pmra[valid_pix]
    sig_dec = sig_pmdec[valid_pix]
    rho = np.clip(pm_corr[valid_pix], -0.9999, 0.9999)

    L11 = sig_ra
    L21 = rho * sig_dec
    L22 = sig_dec * np.sqrt(np.clip(1.0 - rho**2, 0.0, 1.0))
    return L11, L21, L22, valid_pix


def build_zbinned_solvers(pix_bins, nside, lmax):
    """Build and return list of WeightedSHT instances, one per z-bin.

    Each solver has LU factorised for `lmax` and is ready for solve_fast.

    Parameters
    ----------
    pix_bins : list of dicts from compute_zbinned_pixel_stats
    nside    : int
    lmax     : int — lmax to solve (typically lmax_sig=3)

    Returns
    -------
    list of WeightedSHT, length N_z
    """
    from .forward_model import WeightedSHT
    solvers = []
    for pix in pix_bins:
        sig_ra  = np.where(np.isfinite(pix['sig_pmra_wm']),  pix['sig_pmra_wm'],  0.0)
        sig_dec = np.where(np.isfinite(pix['sig_pmdec_wm']), pix['sig_pmdec_wm'], 0.0)
        corr    = np.nan_to_num(pix['pm_corr_wm'], nan=0.0)
        s = WeightedSHT(nside, pix['mask'],
                        pix['pmra_wm'], pix['pmdec_wm'],
                        sig_ra, sig_dec, corr)
        s.build_lu(lmax)
        solvers.append(s)
    return solvers


def precompute_zbinned_noise(pix_bins, npix):
    """Pre-compute Cholesky noise factors for each z-bin.

    Parameters
    ----------
    pix_bins : list of dicts from compute_zbinned_pixel_stats
    npix     : int — total number of HEALPix pixels

    Returns
    -------
    list of dicts, each with keys: L11, L21, L22, valid_pix
    """
    noise_bins = []
    for pix in pix_bins:
        sig_ra  = np.where(np.isfinite(pix['sig_pmra_wm']),  pix['sig_pmra_wm'],  0.0)
        sig_dec = np.where(np.isfinite(pix['sig_pmdec_wm']), pix['sig_pmdec_wm'], 0.0)
        corr    = np.nan_to_num(pix['pm_corr_wm'], nan=0.0)
        L11, L21, L22, valid_pix = precompute_noise_cholesky(
            sig_ra, sig_dec, corr, pix['mask'], npix
        )
        noise_bins.append({'L11': L11, 'L21': L21, 'L22': L22, 'valid_pix': valid_pix})
    return noise_bins


def generate_pixel_noise(L11, L21, L22, valid_pix, npix):
    """Draw one correlated per-pixel noise realisation."""
    n_valid = valid_pix.sum()
    z = np.random.randn(2, n_valid)
    noise_ra = L11 * z[0]
    noise_dec = L21 * z[0] + L22 * z[1]
    out_ra = np.zeros(npix)
    out_dec = np.zeros(npix)
    out_ra[valid_pix] = noise_ra
    out_dec[valid_pix] = noise_dec
    return out_ra, out_dec


def create_prior_full(n_alm_l3, n_Cl, alm_bound, Cl_bound):
    """BoxUniform prior for full model: theta = [alm_l<=3, Cl_E_4..10, Cl_B_4..10]."""
    low = torch.cat([-alm_bound * torch.ones(n_alm_l3), torch.zeros(n_Cl)])
    high = torch.cat([alm_bound * torch.ones(n_alm_l3), Cl_bound * torch.ones(n_Cl)])
    return sbi_utils.BoxUniform(low=low, high=high)


def create_prior_det(n_alm_l3, alm_bound):
    """BoxUniform prior for det. model: theta = [alm_l<=3]."""
    return sbi_utils.BoxUniform(
        low=-alm_bound * torch.ones(n_alm_l3),
        high=alm_bound * torch.ones(n_alm_l3),
    )


def simulator_full(theta_tensor, solver, L11, L21, L22, valid_pix, npix,
                   lmax_sig, lmax_fit, n_alm_l3, n_Cl):
    """Full forward model: theta (n_alm_l3 + n_Cl)-dim -> x (n_real_params(lmax_fit))-dim."""
    theta = theta_tensor.numpy().astype(np.float64)
    theta_l3 = theta[:n_alm_l3]
    Cl_E = theta[n_alm_l3 : n_alm_l3 + n_Cl // 2]
    Cl_B = theta[n_alm_l3 + n_Cl // 2 :]

    E_full, B_full = build_full_alm_complex(theta_l3, Cl_E, Cl_B, lmax_sig, lmax_fit)
    m_theta, m_phi = hp.alm2map_spin([E_full, B_full], solver.nside, spin=1, lmax=lmax_fit)
    pmra_sim, pmdec_sim = m_phi, -m_theta

    noise_ra, noise_dec = generate_pixel_noise(L11, L21, L22, valid_pix, npix)
    pmra_obs = pmra_sim + noise_ra
    pmdec_obs = pmdec_sim + noise_dec

    x = solver.solve_fast(pmra_obs, pmdec_obs, lmax_fit)
    return torch.tensor(x, dtype=torch.float32), pmra_obs, pmdec_obs


def simulator_det(theta_tensor, solver, L11, L21, L22, valid_pix, npix,
                  lmax_sig, lmax_fit, n_alm_l3):
    """Deterministic forward model: theta (n_alm_l3)-dim -> x (n_real_params(lmax_sig))-dim."""
    theta = theta_tensor.numpy().astype(np.float64)
    theta_l3 = theta[:n_alm_l3]

    E_full, B_full = build_det_alm_complex(theta_l3, lmax_sig, lmax_fit)
    m_theta, m_phi = hp.alm2map_spin([E_full, B_full], solver.nside, spin=1, lmax=lmax_fit)
    pmra_sim, pmdec_sim = m_phi, -m_theta

    noise_ra, noise_dec = generate_pixel_noise(L11, L21, L22, valid_pix, npix)
    pmra_obs = pmra_sim + noise_ra
    pmdec_obs = pmdec_sim + noise_dec

    x = solver.solve_fast(pmra_obs, pmdec_obs, lmax_sig)
    return torch.tensor(x, dtype=torch.float32)
