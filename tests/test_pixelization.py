# tests/test_pixelization.py
import numpy as np
import healpy as hp
import pytest
from sbi_qpm.pixelization import Pixelize

nside = 16
pix = Pixelize(nside)
npix = hp.nside2npix(nside)

def test_sum_in_pixels_total_count():
    # All sources in pixel 0
    theta = np.array([np.pi / 2])
    phi = np.array([0.0])
    hmap = pix.sum_in_pixels(theta, phi)
    assert hmap.sum() == 1.0
    assert hmap.shape == (npix,)

def test_sum_in_pixels_weighted():
    theta = np.array([np.pi / 2])
    phi = np.array([0.0])
    hmap = pix.sum_in_pixels(theta, phi, weights=np.array([3.5]))
    assert abs(hmap.sum() - 3.5) < 1e-12

def test_calculate_weighted_means_uniform():
    n = 100
    theta = np.full(n, np.pi / 2)
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    data = np.ones(n) * 5.0
    errors = np.ones(n) * 0.1
    wm = pix.calculate_weighted_means(theta, phi, data, errors)
    nonzero = wm[wm != 0]
    np.testing.assert_allclose(nonzero, 5.0, atol=1e-10)

def test_pixelize_full_covariance_shape():
    from sbi_qpm.pixelization import pixelize_full_covariance
    n = 200
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    pmra = rng.normal(0, 0.01, n)
    pmdec = rng.normal(0, 0.01, n)
    sig = np.ones(n) * 0.01
    corr = np.zeros(n)
    out = pixelize_full_covariance(theta, phi, pmra, pmdec, sig, sig, corr, nside)
    assert len(out) == 5
    for arr in out:
        assert arr.shape == (npix,)

def test_pixelize_full_covariance_zero_corr_matches_diagonal():
    """With zero correlation, full-cov means should match diagonal weighted means."""
    from sbi_qpm.pixelization import pixelize_full_covariance
    n = 500
    rng = np.random.default_rng(1)
    theta = rng.uniform(0.1, np.pi - 0.1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    pmra = rng.normal(0, 0.01, n)
    pmdec = rng.normal(0, 0.01, n)
    sig = np.ones(n) * 0.01
    corr = np.zeros(n)
    pmra_pix, pmdec_pix, _, _, _ = pixelize_full_covariance(
        theta, phi, pmra, pmdec, sig, sig, corr, nside
    )
    pmra_diag = pix.calculate_weighted_means(theta, phi, pmra, sig)
    pmdec_diag = pix.calculate_weighted_means(theta, phi, pmdec, sig)
    np.testing.assert_allclose(pmra_pix, pmra_diag, atol=1e-10)
    np.testing.assert_allclose(pmdec_pix, pmdec_diag, atol=1e-10)
