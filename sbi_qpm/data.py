import numpy as np
import pandas as pd
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u

from .pixelization import Pixelize, pixelize_full_covariance


def load_gaia_qso(path):
    """Load Gaia QSO catalog and apply basic selection cuts.

    Returns DataFrame with columns: ra, dec, pmra, pmdec,
    pmra_error, pmdec_error, pmra_pmdec_corr.
    """
    gaia = pd.read_csv(path)
    sel = (
        (np.abs(gaia["ra"]) > 0.0) &
        (np.sqrt(gaia["pmra"]**2 + gaia["pmdec"]**2) < 100)
    )
    return gaia[sel].reset_index(drop=True)


def compute_pixel_stats(ra, dec, pmra, pmdec, sig_pmra, sig_pmdec, pm_corr, nside, galcut=0, z=None):
    """Bin catalog sources into HEALPix pixels using full 2x2 covariance weighting.

    Uses pixelize_full_covariance for jointly optimal pixel means.
    Returns pixel maps suitable for feeding directly into WeightedSHT.

    Parameters
    ----------
    ra, dec, pmra, pmdec, sig_pmra, sig_pmdec, pm_corr : ndarray (N,)
    nside : int
    galcut : float, optional
    z : ndarray (N,), optional
        Per-source redshift. If provided, z_wm (unweighted mean redshift per pixel)
        is added to the returned dict. Zero for empty pixels.

    Returns
    -------
    dict with keys:
        pmra_wm, pmdec_wm         : covariance-weighted proper motion maps
        sig_pmra_wm, sig_pmdec_wm : effective pixel uncertainty maps
        pm_corr_wm                : effective pixel correlation map
        n_per_pixel               : source count map
        mask                      : boolean validity mask
        z_wm                      : unweighted mean redshift per pixel (zero if z=None)
    """
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    mask_galactic = make_galactic_mask(nside, b_cut_deg=galcut)

    pix_obj = Pixelize(nside)
    n_pix = pix_obj.sum_in_pixels(theta, phi)

    pmra_wm, pmdec_wm, sig_pmra_wm, sig_pmdec_wm, pm_corr_wm = \
        pixelize_full_covariance(theta, phi, pmra, pmdec,
                                 sig_pmra, sig_pmdec, pm_corr, nside)

    mask = (
        (pmra_wm != 0) & (pmdec_wm != 0) &
        (sig_pmra_wm != 0) & (sig_pmdec_wm != 0) & mask_galactic
    )

    npix_total = hp.nside2npix(nside)
    if z is not None:
        _idx = hp.ang2pix(nside, theta, phi)
        _z_sum = np.bincount(_idx, weights=np.asarray(z, dtype=float), minlength=npix_total).astype(float)
        z_wm = np.zeros(npix_total)
        populated = n_pix > 0
        z_wm[populated] = _z_sum[populated] / n_pix[populated]
    else:
        z_wm = np.zeros(npix_total)

    return dict(
        pmra_wm=pmra_wm,
        pmdec_wm=pmdec_wm,
        sig_pmra_wm=sig_pmra_wm,
        sig_pmdec_wm=sig_pmdec_wm,
        pm_corr_wm=pm_corr_wm,
        n_per_pixel=n_pix,
        mask=mask,
        z_wm=z_wm,
    )


def compute_zbinned_pixel_stats(ra, dec, pmra, pmdec, sig_pmra, sig_pmdec, pm_corr,
                                 nside, z_edges, galcut=0, z=None):
    """Bin catalog sources into HEALPix pixels within redshift slices.

    Parameters
    ----------
    ra, dec, pmra, pmdec, sig_pmra, sig_pmdec, pm_corr : ndarray (N,)
    nside     : int
    z_edges   : array-like, shape (N_z+1,)
    galcut    : float, galactic latitude cut in degrees
    z         : ndarray (N,), per-source redshift. Must be provided; raises ValueError if None.

    Returns
    -------
    list of N_z dicts, each with same keys as compute_pixel_stats output.
    """
    if z is None:
        raise ValueError("z is required for z-binned pixel stats")

    z_edges = np.asarray(z_edges, dtype=float)
    n_bins = len(z_edges) - 1
    z = np.asarray(z, dtype=float)

    bins = []
    for i in range(n_bins):
        z_lo, z_hi = z_edges[i], z_edges[i + 1]
        sel = (z >= z_lo) & (z < z_hi if i < n_bins - 1 else z <= z_hi)
        if sel.sum() == 0:
            npix = hp.nside2npix(nside)
            bins.append(dict(
                pmra_wm=np.zeros(npix),
                pmdec_wm=np.zeros(npix),
                sig_pmra_wm=np.zeros(npix),
                sig_pmdec_wm=np.zeros(npix),
                pm_corr_wm=np.zeros(npix),
                n_per_pixel=np.zeros(npix),
                mask=np.zeros(npix, dtype=bool),
                z_wm=np.zeros(npix),
            ))
            continue

        pix_i = compute_pixel_stats(
            ra[sel], dec[sel], pmra[sel], pmdec[sel],
            sig_pmra[sel], sig_pmdec[sel], pm_corr[sel],
            nside=nside, galcut=galcut, z=z[sel],
        )
        bins.append(pix_i)

    _make_masks_disjoint(bins)
    return bins


def _make_masks_disjoint(bins):
    """Enforce that each pixel appears in at most one bin's mask.

    For pixels masked in multiple bins, the bin with the highest source count
    wins; the mask is cleared in all other bins. Modifies bins[i]['mask'] in place.
    """
    if len(bins) <= 1:
        return
    counts = np.stack([b['n_per_pixel'] for b in bins], axis=0)  # (n_bins, npix)
    winner = np.argmax(counts, axis=0)                            # (npix,)
    combined_mask = np.stack([b['mask'] for b in bins], axis=0)  # (n_bins, npix)
    contested = combined_mask.sum(axis=0) > 1                     # pixels in >1 bin
    for i, b in enumerate(bins):
        b['mask'][contested & (winner != i)] = False


def make_galactic_mask(nside, b_cut_deg=20.0):
    """Return boolean array: True for pixels with |b| > b_cut_deg."""
    npix = hp.nside2npix(nside)
    pix_idx = np.arange(npix)
    theta_g, phi_g = hp.pix2ang(nside, pix_idx, lonlat=False)
    ra_g = np.degrees(phi_g)
    dec_g = 90.0 - np.degrees(theta_g)
    coords = SkyCoord(ra_g * u.deg, dec_g * u.deg, frame="icrs")
    return np.abs(coords.galactic.b.deg) > b_cut_deg
