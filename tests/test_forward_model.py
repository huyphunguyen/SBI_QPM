# tests/test_forward_model.py
import numpy as np
import healpy as hp
import math
import pytest
from sbi_qpm.forward_model import WeightedSHT

NSIDE = 8
NPIX = hp.nside2npix(NSIDE)
LMAX = 3
RNG = np.random.default_rng(42)

def _make_solver():
    mask = np.ones(NPIX, dtype=bool)
    sig = np.ones(NPIX) * 0.01
    corr = np.zeros(NPIX)
    pm_ra = RNG.normal(0, 0.01, NPIX)
    pm_dec = RNG.normal(0, 0.01, NPIX)
    return WeightedSHT(NSIDE, mask, pm_ra, pm_dec, sig, sig, corr)

def _n_params(lmax):
    return 2 * ((lmax + 1) + (math.comb(lmax + 2, 2) - (lmax + 1)) * 2) - 2

def test_solve_direct_shape():
    solver = _make_solver()
    x = solver.solve_direct(LMAX)
    assert x.shape == (_n_params(LMAX),)

def test_solve_fast_matches_direct():
    solver = _make_solver()
    solver.build_lu(LMAX)
    x_direct = solver.solve_direct(LMAX)
    x_fast = solver.solve_fast(
        solver.pm_ra_map, solver.pm_dec_map, LMAX
    )
    np.testing.assert_allclose(x_fast, x_direct, atol=1e-8)

def test_generate_map_shape():
    solver = _make_solver()
    n = _n_params(LMAX)
    coeff = np.zeros(n)
    ra_map, dec_map, E_alm, B_alm = solver.generate_map(coeff, LMAX)
    assert ra_map.shape == (NPIX,)
    assert dec_map.shape == (NPIX,)
