# tests/test_power_spectrum_analysis.py
import numpy as np

from sbi_qpm.power_spectrum_analysis import bandpower_to_full_cl, make_bins


def test_bandpower_to_full_cl_piecewise_constant():
    cl = bandpower_to_full_cl(np.array([2.0, 5.0]), np.array([1.0, 3.0]), lmax=8)
    assert cl.shape == (9,)
    np.testing.assert_array_equal(cl[2:5], 1.0)
    np.testing.assert_array_equal(cl[5:], 3.0)
    np.testing.assert_array_equal(cl[:2], 1.0)   # leading zeros back-filled


def test_make_bins_includes_dipole():
    nside = 16
    bins = make_bins(nside, nbin=1, lmin=1)
    ells = bins.get_effective_ells()
    assert ells[0] == 1
    assert ells[-1] == 3 * nside - 1
    assert len(ells) == 3 * nside - 1


def test_make_bins_nbin_groups_multipoles():
    bins = make_bins(16, nbin=4, lmin=1)
    assert bins.get_n_bands() == (3 * 16 - 1) // 4
