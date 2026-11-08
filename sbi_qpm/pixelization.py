import numpy as np
import healpy as hp


class Pixelize:
    def __init__(self, nside):
        self.nside = nside

    def sum_in_pixels(self, theta, phi, weights=None):
        npix = hp.nside2npix(self.nside)
        indices = hp.ang2pix(self.nside, theta, phi)
        return np.bincount(indices, weights=weights, minlength=npix).astype(float)

    def calculate_weighted_means(self, theta, phi, data, data_errors):
        """Diagonal inverse-variance weighted mean. Visualization use only.
        For science (feeding WeightedSHT), use pixelize_full_covariance()."""
        w = 1.0 / data_errors**2
        sum_w = self.sum_in_pixels(theta, phi, weights=w)
        sum_wd = self.sum_in_pixels(theta, phi, weights=data * w)
        wm = np.zeros_like(sum_wd)
        np.divide(sum_wd, sum_w, out=wm, where=sum_w != 0)
        return wm


def pixelize_full_covariance(theta, phi, pmra, pmdec,
                              sig_pmra, sig_pmdec, pm_corr, nside):
    """Pixelize proper motions using the full per-source 2x2 noise covariance.

    Accumulates per-source C^-1 into pixels, inverts analytically, returns
    jointly optimal pixel-level weighted means and uncertainties.

    Returns
    -------
    pmra_pix, pmdec_pix      : ndarray (npix,) -- covariance-weighted means
    sig_pmra_pix, sig_pmdec_pix : ndarray (npix,) -- effective pixel uncertainties
    pm_corr_pix              : ndarray (npix,) -- effective pixel correlation
    """
    npix = hp.nside2npix(nside)
    indices = hp.ang2pix(nside, theta, phi)

    factor = 1.0 / (1.0 - pm_corr**2)
    Cinv_rr = factor / sig_pmra**2
    Cinv_rd = -factor * pm_corr / (sig_pmra * sig_pmdec)
    Cinv_dd = factor / sig_pmdec**2

    Cinv_rr_pix = np.bincount(indices, weights=Cinv_rr, minlength=npix).astype(float)
    Cinv_rd_pix = np.bincount(indices, weights=Cinv_rd, minlength=npix).astype(float)
    Cinv_dd_pix = np.bincount(indices, weights=Cinv_dd, minlength=npix).astype(float)

    sum_wr = np.bincount(indices, weights=Cinv_rr * pmra + Cinv_rd * pmdec, minlength=npix).astype(float)
    sum_wd = np.bincount(indices, weights=Cinv_rd * pmra + Cinv_dd * pmdec, minlength=npix).astype(float)

    det = Cinv_rr_pix * Cinv_dd_pix - Cinv_rd_pix**2
    valid = det > 0
    safe_det = np.where(valid, det, 1.0)

    Sigma_rr = np.where(valid,  Cinv_dd_pix / safe_det, 0.0)
    Sigma_rd = np.where(valid, -Cinv_rd_pix / safe_det, 0.0)
    Sigma_dd = np.where(valid,  Cinv_rr_pix / safe_det, 0.0)

    pmra_pix  = Sigma_rr * sum_wr + Sigma_rd * sum_wd
    pmdec_pix = Sigma_rd * sum_wr + Sigma_dd * sum_wd

    sig_pmra_pix  = np.where(valid, np.sqrt(np.maximum(Sigma_rr, 0.0)), 0.0)
    sig_pmdec_pix = np.where(valid, np.sqrt(np.maximum(Sigma_dd, 0.0)), 0.0)
    denom = sig_pmra_pix * sig_pmdec_pix
    safe_denom = np.where(denom > 0, denom, 1.0)
    pm_corr_pix = np.where(denom > 0, Sigma_rd / safe_denom, 0.0)

    return pmra_pix, pmdec_pix, sig_pmra_pix, sig_pmdec_pix, pm_corr_pix
