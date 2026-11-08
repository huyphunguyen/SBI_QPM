# tests/test_geometry.py
import numpy as np
import pytest
from sbi_qpm.geometry import SkyFunction, direction_summary, icrs_to_galactic

skyfunc = SkyFunction(nside=16)

def test_dir2vec_vec2dir_roundtrip():
    ra, dec = 45.0, 30.0
    vec = skyfunc.dir2vec(ra, dec)
    ra_out, dec_out = skyfunc.vec2dir(vec)
    assert abs(ra_out - ra) < 1e-10
    assert abs(dec_out - dec) < 1e-10

def test_s2g_g2s_roundtrip():
    g_in = (1.5, -2.3, 0.7)
    s = skyfunc.g2s(g_in)
    g_out = skyfunc.s2g(s)
    np.testing.assert_allclose(g_out, g_in, atol=1e-12)

def test_dir2vec_unit_length():
    ra, dec = 120.0, -45.0
    vec = skyfunc.dir2vec(ra, dec)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-12


def test_icrs_to_galactic_galactic_centre():
    # Sgr A* (ICRS) must map to the galactic origin.
    l, b = icrs_to_galactic(266.4050, -28.9362)
    assert abs(l[0]) < 0.1 or abs(l[0] - 360.0) < 0.1
    assert abs(b[0]) < 0.1


def test_direction_summary_recovers_injected_direction():
    rng = np.random.default_rng(0)
    ra_true, dec_true = 269.1, -31.6            # Gaia EDR3 acceleration direction
    n_true = skyfunc.dir2vec(ra_true, dec_true)
    g = 5.0 * n_true + 0.15 * rng.standard_normal((20000, 3))

    s = direction_summary(g)
    assert abs(((s['ra_c'] - ra_true + 180) % 360) - 180) < 0.5
    assert abs(s['dec_c'] - dec_true) < 0.5
    assert s['ra_16'] < s['ra_84'] and s['dec_16'] < s['dec_84']
    assert 0 < s['theta_68'] < s['theta_95']
    assert 0.9 < s['R'] <= 1.0

    l_true, b_true = icrs_to_galactic(ra_true, dec_true)
    assert abs(((s['l_c'] - l_true[0] + 180) % 360) - 180) < 0.5
    assert abs(s['b_c'] - b_true[0]) < 0.5


def test_direction_summary_handles_ra_wraparound():
    # Posterior straddling RA = 0: naive percentiles would land near 180 deg.
    rng = np.random.default_rng(1)
    n_true = skyfunc.dir2vec(0.0, 10.0)
    g = 5.0 * n_true + 0.2 * rng.standard_normal((20000, 3))

    s = direction_summary(g)
    assert min(s['ra_c'], 360.0 - s['ra_c']) < 1.0
    # 16/84 must bracket the centre once unwrapped, not straddle the sky.
    lo = ((s['ra_16'] - s['ra_c'] + 180) % 360) - 180
    hi = ((s['ra_84'] - s['ra_c'] + 180) % 360) - 180
    assert lo < 0 < hi and hi - lo < 20.0


def test_direction_summary_isotropic_has_small_R():
    rng = np.random.default_rng(2)
    g = rng.standard_normal((20000, 3))
    s = direction_summary(g)
    assert s['R'] < 0.05
    assert s['theta_68'] > 100.0        # near-isotropic: 68% radius approaches 120 deg
