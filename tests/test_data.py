# tests/test_data.py
import numpy as np
import healpy as hp
import pytest
from sbi_qpm.data import compute_pixel_stats


def _make_catalog(nside, N=500, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.pi, N)
    phi   = rng.uniform(0, 2 * np.pi, N)
    ra    = np.degrees(phi)
    dec   = 90.0 - np.degrees(theta)
    pmra  = np.zeros(N)
    pmdec = np.zeros(N)
    sig   = np.ones(N) * 0.1
    corr  = np.zeros(N)
    z     = rng.uniform(0.5, 2.0, N)
    return ra, dec, pmra, pmdec, sig, corr, z


def test_z_wm_present_when_z_provided():
    ra, dec, pmra, pmdec, sig, corr, z = _make_catalog(nside=8)
    result = compute_pixel_stats(ra, dec, pmra, pmdec, sig, sig, corr, nside=8, z=z)
    assert 'z_wm' in result


def test_z_wm_shape():
    nside = 8
    ra, dec, pmra, pmdec, sig, corr, z = _make_catalog(nside)
    result = compute_pixel_stats(ra, dec, pmra, pmdec, sig, sig, corr, nside=nside, z=z)
    assert result['z_wm'].shape == (hp.nside2npix(nside),)


def test_z_wm_zero_when_z_none():
    nside = 8
    ra, dec, pmra, pmdec, sig, corr, _ = _make_catalog(nside)
    result = compute_pixel_stats(ra, dec, pmra, pmdec, sig, sig, corr, nside=nside)
    assert 'z_wm' in result
    assert np.all(result['z_wm'] == 0.0)


def test_z_wm_in_data_range():
    nside = 8
    ra, dec, pmra, pmdec, sig, corr, z = _make_catalog(nside)
    result = compute_pixel_stats(ra, dec, pmra, pmdec, sig, sig, corr, nside=nside, z=z)
    populated = result['n_per_pixel'] > 0
    assert np.all(result['z_wm'][populated] >= 0.5)
    assert np.all(result['z_wm'][populated] <= 2.0)


def test_z_wm_empty_pixels_are_zero():
    nside = 64  # many empty pixels guaranteed
    ra, dec, pmra, pmdec, sig, corr, z = _make_catalog(nside, N=100)
    result = compute_pixel_stats(ra, dec, pmra, pmdec, sig, sig, corr, nside=nside, z=z)
    empty = result['n_per_pixel'] == 0
    assert np.all(result['z_wm'][empty] == 0.0)


# ---------------------------------------------------------------------------
# Tests for compute_zbinned_pixel_stats
# ---------------------------------------------------------------------------
from sbi_qpm.data import compute_zbinned_pixel_stats


def test_compute_zbinned_pixel_stats_returns_list():
    rng = np.random.default_rng(0)
    N = 500
    ra  = rng.uniform(0, 360, N)
    dec = rng.uniform(-90, 90, N)
    pmra = rng.normal(0, 0.1, N)
    pmdec = rng.normal(0, 0.1, N)
    sig = np.full(N, 0.1)
    corr = np.zeros(N)
    z = rng.uniform(0.1, 3.0, N)
    z_edges = np.array([0.0, 0.5, 1.0, 1.5, 3.5])

    bins = compute_zbinned_pixel_stats(
        ra, dec, pmra, pmdec, sig, sig, corr,
        nside=16, z_edges=z_edges, z=z,
    )
    assert len(bins) == 4
    keys = {'pmra_wm', 'pmdec_wm', 'sig_pmra_wm', 'sig_pmdec_wm',
            'pm_corr_wm', 'n_per_pixel', 'mask', 'z_wm'}
    for b in bins:
        assert set(b.keys()) == keys


def test_compute_zbinned_pixel_stats_z_wm_in_range():
    rng = np.random.default_rng(1)
    N = 2000
    ra  = rng.uniform(0, 360, N)
    dec = rng.uniform(-90, 90, N)
    pmra = rng.normal(0, 0.1, N)
    pmdec = rng.normal(0, 0.1, N)
    sig = np.full(N, 0.1)
    corr = np.zeros(N)
    z = rng.uniform(0.1, 3.0, N)
    z_edges = np.array([0.0, 1.0, 2.0, 3.5])

    bins = compute_zbinned_pixel_stats(
        ra, dec, pmra, pmdec, sig, sig, corr,
        nside=16, z_edges=z_edges, z=z,
    )
    for i, b in enumerate(bins):
        z_lo, z_hi = z_edges[i], z_edges[i + 1]
        valid_z = b['z_wm'][b['mask']]
        assert np.all(valid_z >= z_lo - 1e-9), f"bin {i}: z_wm below lower edge"
        assert np.all(valid_z <= z_hi + 1e-9), f"bin {i}: z_wm above upper edge"


def test_compute_zbinned_pixel_stats_masks_are_disjoint():
    """Each pixel can only be in one bin's mask."""
    rng = np.random.default_rng(2)
    N = 2000
    ra  = rng.uniform(0, 360, N)
    dec = rng.uniform(-90, 90, N)
    pmra = rng.normal(0, 0.1, N)
    pmdec = rng.normal(0, 0.1, N)
    sig = np.full(N, 0.1)
    corr = np.zeros(N)
    z = rng.uniform(0.1, 3.0, N)
    z_edges = np.array([0.0, 1.0, 2.0, 3.5])

    bins = compute_zbinned_pixel_stats(
        ra, dec, pmra, pmdec, sig, sig, corr,
        nside=16, z_edges=z_edges, z=z,
    )
    mask_sum = sum(b['mask'].astype(int) for b in bins)
    assert np.all(mask_sum <= 1), "Pixel appears in multiple bins"
