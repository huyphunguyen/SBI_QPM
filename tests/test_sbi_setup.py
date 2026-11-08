# tests/test_sbi_setup.py
import numpy as np
import healpy as hp
import pytest
import torch
from sbi_qpm.sbi_setup import (
    n_real_params,
    build_full_alm_complex,
    build_det_alm_complex,
    precompute_noise_cholesky,
    generate_pixel_noise,
    create_prior_full,
    create_prior_det,
    simulator_full,
    simulator_det,
)
from sbi_qpm.forward_model import WeightedSHT

LMAX_SIG = 3
LMAX_FIT = 10
N_ALM_L3 = n_real_params(LMAX_SIG)  # 30
N_CL = 2 * (LMAX_FIT - LMAX_SIG)    # 14
ALM_BOUND = 50e-3                   # mas/yr
CL_BOUND = 130e-6                   # (mas/yr)^2

def test_n_real_params():
    assert n_real_params(1) == 6
    assert n_real_params(3) == 30
    assert n_real_params(10) == 240

def test_build_full_alm_complex_shape():
    theta_l3 = np.zeros(N_ALM_L3)
    Cl_E = np.ones(LMAX_FIT - LMAX_SIG) * 1e-6
    Cl_B = np.ones(LMAX_FIT - LMAX_SIG) * 1e-6
    E, B = build_full_alm_complex(theta_l3, Cl_E, Cl_B, LMAX_SIG, LMAX_FIT)
    expected = hp.Alm.getsize(LMAX_FIT)
    assert E.shape == (expected,)
    assert B.shape == (expected,)

def test_build_det_alm_deterministic():
    # With Cl=0, det and full should give same result
    theta_l3 = np.random.randn(N_ALM_L3) * 1e-3
    E_det, B_det = build_det_alm_complex(theta_l3, LMAX_SIG, LMAX_FIT)
    E_full, B_full = build_full_alm_complex(
        theta_l3, np.zeros(LMAX_FIT - LMAX_SIG), np.zeros(LMAX_FIT - LMAX_SIG),
        LMAX_SIG, LMAX_FIT,
    )
    np.testing.assert_array_equal(E_det, E_full)

def test_generate_pixel_noise_shape():
    nside = 16
    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=bool)
    sig = np.ones(npix) * 0.01
    corr = np.zeros(npix)
    L11, L21, L22, valid_pix = precompute_noise_cholesky(sig, sig, corr, mask, npix)
    noise_ra, noise_dec = generate_pixel_noise(L11, L21, L22, valid_pix, npix)
    assert noise_ra.shape == (npix,)
    assert noise_dec.shape == (npix,)

def _make_solver_and_noise(nside=16, sigma=0.1):
    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=bool)
    sig  = np.ones(npix) * sigma
    corr = np.zeros(npix)
    pmra = np.zeros(npix)
    pmdec = np.zeros(npix)
    solver = WeightedSHT(nside, mask, pmra, pmdec, sig, sig, corr)
    solver.build_lu(LMAX_SIG)
    solver.build_lu(LMAX_FIT)
    L11, L21, L22, valid_pix = precompute_noise_cholesky(sig, sig, corr, mask, npix)
    return solver, L11, L21, L22, valid_pix, npix



def _common(solver, L11, L21, L22, valid_pix, npix):
    return dict(solver=solver, L11=L11, L21=L21, L22=L22, valid_pix=valid_pix,
                npix=npix, lmax_sig=LMAX_SIG, lmax_fit=LMAX_FIT, n_alm_l3=N_ALM_L3)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def test_create_prior_full_dim_and_bounds():
    prior = create_prior_full(N_ALM_L3, N_CL, ALM_BOUND, CL_BOUND)
    s = prior.sample((2000,)).numpy()
    assert s.shape == (2000, N_ALM_L3 + N_CL)   # 44
    assert np.all(np.abs(s[:, :N_ALM_L3]) <= ALM_BOUND)
    assert np.all(s[:, N_ALM_L3:] >= 0.0)
    assert np.all(s[:, N_ALM_L3:] <= CL_BOUND)


def test_create_prior_det_dim_and_bounds():
    prior = create_prior_det(N_ALM_L3, ALM_BOUND)
    s = prior.sample((2000,)).numpy()
    assert s.shape == (2000, N_ALM_L3)          # 30
    assert np.all(np.abs(s) <= ALM_BOUND)


def test_create_prior_full_log_prob_outside_support():
    prior = create_prior_full(N_ALM_L3, N_CL, ALM_BOUND, CL_BOUND)
    theta = prior.sample((1,))
    theta[0, N_ALM_L3] = -1e-6                  # negative Cl
    assert torch.isinf(prior.log_prob(theta)).all()


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

def test_simulator_full_output():
    solver, L11, L21, L22, valid_pix, npix = _make_solver_and_noise()
    theta = create_prior_full(N_ALM_L3, N_CL, ALM_BOUND, CL_BOUND).sample()
    x, pmra, pmdec = simulator_full(theta, n_Cl=N_CL,
                                    **_common(solver, L11, L21, L22, valid_pix, npix))
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert x.shape == (n_real_params(LMAX_FIT),)  # 240
    assert torch.isfinite(x).all()
    assert pmra.shape == (npix,) and pmdec.shape == (npix,)


def test_simulator_det_output():
    solver, L11, L21, L22, valid_pix, npix = _make_solver_and_noise()
    theta = create_prior_det(N_ALM_L3, ALM_BOUND).sample()
    x = simulator_det(theta, **_common(solver, L11, L21, L22, valid_pix, npix))
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert x.shape == (N_ALM_L3,)                 # 30
    assert torch.isfinite(x).all()


def test_simulator_det_recovers_theta_without_noise():
    """Injection-recovery: with negligible noise, the VSH fit returns the input alm."""
    solver, L11, L21, L22, valid_pix, npix = _make_solver_and_noise(sigma=1e-9)
    theta = create_prior_det(N_ALM_L3, ALM_BOUND).sample()
    x = simulator_det(theta, **_common(solver, L11, L21, L22, valid_pix, npix))
    np.testing.assert_allclose(x.numpy(), theta.numpy(), atol=1e-6)


def test_simulator_full_recovers_l3_block_when_cl_zero():
    """With Cl = 0 and negligible noise, the lmax_fit summary equals the input alm."""
    solver, L11, L21, L22, valid_pix, npix = _make_solver_and_noise(sigma=1e-9)
    theta = create_prior_full(N_ALM_L3, N_CL, ALM_BOUND, CL_BOUND).sample()
    theta[N_ALM_L3:] = 0.0
    x, _, _ = simulator_full(theta, n_Cl=N_CL,
                             **_common(solver, L11, L21, L22, valid_pix, npix))
    E, B = build_det_alm_complex(theta[:N_ALM_L3].numpy().astype(np.float64),
                                 LMAX_SIG, LMAX_FIT)
    expected = solver._complex_to_real(E, B, LMAX_FIT)
    np.testing.assert_allclose(x.numpy(), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# z-binned helper tests
# ---------------------------------------------------------------------------
from sbi_qpm.data import compute_zbinned_pixel_stats
from sbi_qpm.sbi_setup import build_zbinned_solvers, precompute_zbinned_noise


def _make_zbinned_pixel_data(nside=16, N=2000, seed=42):
    rng = np.random.default_rng(seed)
    ra   = rng.uniform(0, 360, N)
    dec  = rng.uniform(-90, 90, N)
    pmra_s  = rng.normal(0, 0.1, N)
    pmdec_s = rng.normal(0, 0.1, N)
    sig  = np.full(N, 0.1)
    corr = np.zeros(N)
    z    = rng.uniform(0.1, 3.0, N)
    z_edges = np.array([0.0, 1.0, 2.0, 3.5])
    pix_bins = compute_zbinned_pixel_stats(
        ra, dec, pmra_s, pmdec_s, sig, sig, corr,
        nside=nside, z_edges=z_edges, z=z,
    )
    return pix_bins, nside, hp.nside2npix(nside)


def test_build_zbinned_solvers_returns_list():
    pix_bins, nside, npix = _make_zbinned_pixel_data()
    solvers = build_zbinned_solvers(pix_bins, nside, lmax=LMAX_SIG)
    assert len(solvers) == len(pix_bins)
    from sbi_qpm.forward_model import WeightedSHT
    for s in solvers:
        assert isinstance(s, WeightedSHT)


def test_build_zbinned_solvers_lu_cached():
    pix_bins, nside, npix = _make_zbinned_pixel_data()
    solvers = build_zbinned_solvers(pix_bins, nside, lmax=LMAX_SIG)
    for s in solvers:
        npix_s = hp.nside2npix(nside)
        out = s.solve_fast(np.zeros(npix_s), np.zeros(npix_s), LMAX_SIG)
        assert out.shape == (n_real_params(LMAX_SIG),)


def test_precompute_zbinned_noise_returns_list():
    pix_bins, nside, npix = _make_zbinned_pixel_data()
    noise_bins = precompute_zbinned_noise(pix_bins, npix)
    assert len(noise_bins) == len(pix_bins)
    for nb in noise_bins:
        assert set(nb.keys()) == {'L11', 'L21', 'L22', 'valid_pix'}


