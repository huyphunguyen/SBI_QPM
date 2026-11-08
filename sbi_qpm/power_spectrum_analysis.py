import numpy as np
import healpy as hp
import pymaster as nmt


def bandpower_to_full_cl(ell_b, cl_b, lmax):
    """Approximate full-ell Cl from band-powers by constant interpolation.

    Propagates band-power value forward to all ells in that band.
    Leading zeros (ells before first band) are filled with the first non-zero value.

    Parameters
    ----------
    ell_b : ndarray -- effective ell per band-power bin
    cl_b  : ndarray -- band-power Cl values
    lmax  : int

    Returns
    -------
    cl_full : ndarray, length lmax+1
    """
    cl_full = np.zeros(lmax + 1)
    for k in range(len(ell_b) - 1):
        lo = int(round(ell_b[k]))
        hi = int(round(ell_b[k + 1]))
        cl_full[lo:hi] = cl_b[k]
    cl_full[int(round(ell_b[-1])):] = cl_b[-1]
    # Fill leading zeros with first non-zero value
    first_nonzero = next((i for i, v in enumerate(cl_full) if v != 0), None)
    if first_nonzero and first_nonzero > 0:
        cl_full[:first_nonzero] = cl_full[first_nonzero]
    return cl_full


def make_bins(nside, nbin=1, lmin=1):
    """Create NmtBin starting from lmin=1 (NaMaster default skips ell=1).

    Parameters
    ----------
    nside : int
    nbin  : int -- multipoles per band-power bin
    lmin  : int -- first ell to include (set to 1 to include dipole)

    Returns
    -------
    NmtBin
    """
    lmax = 3 * nside - 1
    ells = np.arange(lmin, lmax + 1)
    n_ells = len(ells)
    n_bins = n_ells // nbin
    ells_ini = ells[:n_bins * nbin].reshape(n_bins, nbin)[:, 0]
    ells_fin = ells[:n_bins * nbin].reshape(n_bins, nbin)[:, -1]
    bpws = np.repeat(np.arange(n_bins), nbin)
    weights = np.ones(n_bins * nbin)
    return nmt.NmtBin(bpws=bpws, ells=ells[:n_bins * nbin],
                      weights=weights, lmax=lmax)


def make_spin1_field(mask, pmra_map, pmdec_map):
    """Create NmtField for spin-1 proper-motion maps.

    NaMaster spin-1 convention: maps = [-pmdec, pmra] (theta, phi components).
    """
    return nmt.NmtField(mask.astype(float), [-pmdec_map, pmra_map], spin=1)


def make_spin0_field(mask, scalar_map):
    """Create NmtField for a spin-0 scalar map."""
    return nmt.NmtField(mask.astype(float), [scalar_map])


def apodize_mask(mask, apod_scale_deg, apod_type="C1"):
    """Apodize a binary mask for NaMaster."""
    return nmt.mask_apodization(mask.astype(float), apod_scale_deg, apod_type)


def compute_power_spectrum(field1, field2, bins):
    """Run full NaMaster Cl pipeline: couple -> decouple.

    Returns
    -------
    ells       : ndarray -- effective ell per bin
    cl_decoupled : ndarray, shape (n_correlations, n_bins)
      For spin-1 x spin-1: [EE, EB, BE, BB]
      For spin-1 x spin-0: [E×T, B×T]
      For spin-0 x spin-0: [TT]
    """
    ws = nmt.NmtWorkspace()
    ws.compute_coupling_matrix(field1, field2, bins)
    cl_coupled = nmt.compute_coupled_cell(field1, field2)
    cl_decoupled = ws.decouple_cell(cl_coupled)
    ells = bins.get_effective_ells()
    return ells, cl_decoupled


def estimate_noise_covariance(field, bins, n_sims, fiducial_cl):
    """Estimate noise covariance via Monte Carlo of Gaussian random fields.

    Parameters
    ----------
    field       : NmtField (auto-spectrum only)
    bins        : NmtBin
    n_sims      : int -- number of MC realisations
    fiducial_cl : ndarray -- input Cl for Gaussian realisations

    Returns
    -------
    cov : ndarray, shape (n_bins, n_bins)
    """
    nside = hp.npix2nside(field.get_maps().shape[-1])
    cls_sims = []
    for _ in range(n_sims):
        sim_map = hp.synfast(fiducial_cl, nside, verbose=False)
        f_sim = nmt.NmtField(field.get_mask(), [sim_map])
        _, cl_sim = compute_power_spectrum(f_sim, f_sim, bins)
        cls_sims.append(cl_sim[0])
    cls_sims = np.array(cls_sims)
    return np.cov(cls_sims.T)


def evaluate_cross_spectrum(cl_obs, cl_null, cov):
    """Chi-squared significance of observed Cl vs null hypothesis.

    Parameters
    ----------
    cl_obs  : ndarray, shape (n_bins,) -- observed cross-spectrum
    cl_null : ndarray, shape (n_bins,) -- expected null (usually zeros)
    cov     : ndarray, shape (n_bins, n_bins) -- noise covariance

    Returns
    -------
    chi2    : float
    p_value : float (from chi-squared distribution)
    """
    from scipy import stats
    delta = cl_obs - cl_null
    cov_inv = np.linalg.inv(cov)
    chi2 = float(delta @ cov_inv @ delta)
    p_value = 1.0 - stats.chi2.cdf(chi2, df=len(cl_obs))
    return chi2, p_value
