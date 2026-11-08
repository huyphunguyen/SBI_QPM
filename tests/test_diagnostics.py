import numpy as np
import pytest
from sbi_qpm.diagnostics import marginal_coverage


def test_marginal_coverage_perfect_posterior():
    # Posterior always contains the truth → ECP should be ~1 for all alpha
    np.random.seed(0)
    N_test, N_samp = 200, 500
    theta_k = np.random.randn(N_test)
    # Samples centered on truth with tiny spread → always contains truth
    samples_k = theta_k[np.newaxis, :] + np.random.randn(N_samp, N_test) * 1e-10
    alphas, ecp = marginal_coverage(samples_k, theta_k, n_bins=10)
    assert alphas.shape == (11,)
    assert ecp.shape == (11,)
    # All coverage values should be ~1
    np.testing.assert_allclose(ecp[1:], 1.0, atol=0.05)


def test_marginal_coverage_output_range():
    np.random.seed(1)
    N_test, N_samp = 100, 200
    theta_k = np.random.randn(N_test)
    samples_k = np.random.randn(N_samp, N_test)
    alphas, ecp = marginal_coverage(samples_k, theta_k)
    assert np.all(alphas >= 0) and np.all(alphas <= 1)
    assert np.all(ecp >= 0) and np.all(ecp <= 1)
