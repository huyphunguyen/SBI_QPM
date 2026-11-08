import numpy as np
import torch


def marginal_coverage(samples_k, theta_k, n_bins=50):
    """Compute marginal Expected Coverage Probability (ECP).

    Parameters
    ----------
    samples_k : ndarray, shape (N_samples, N_test)
        Posterior samples for one parameter across N_test test points.
    theta_k : ndarray, shape (N_test,)
        True parameter values.
    n_bins : int
        Number of credibility levels.

    Returns
    -------
    alphas : ndarray, shape (n_bins+1,)  -- credibility levels in [0,1]
    ecp    : ndarray, shape (n_bins+1,)  -- fraction of test points covered
    """
    alphas = np.linspace(0, 1, n_bins + 1)
    ecp = np.zeros(n_bins + 1)
    for i, alpha in enumerate(alphas):
        lo = np.percentile(samples_k, 100 * (1 - alpha) / 2, axis=0)
        hi = np.percentile(samples_k, 100 * (1 + alpha) / 2, axis=0)
        ecp[i] = np.mean((theta_k >= lo) & (theta_k <= hi))
    return alphas, ecp


def extract_g_tarp(samples_np, dipole_idx, skyfunc):
    """Convert alm posterior samples -> (gx, gy, gz) for TARP coverage plot.

    Parameters
    ----------
    samples_np : ndarray, shape (N_samples, N_test, n_params)
    dipole_idx : array-like, length 3 -- indices of [a10, Re(a11), Im(a11)]
    skyfunc    : SkyFunction instance

    Returns
    -------
    ndarray, shape (N_samples, N_test, 3)
    """
    N_S, N_T, _ = samples_np.shape
    out = np.zeros((N_S, N_T, 3))
    for i in range(N_S):
        for j in range(N_T):
            out[i, j] = skyfunc.s2g(samples_np[i, j, dipole_idx] * 1e3)
    return out


def extract_g_theta(theta_np, dipole_idx, skyfunc):
    """Convert theta draws -> (gx, gy, gz).

    Parameters
    ----------
    theta_np   : ndarray, shape (N_test, n_params)
    dipole_idx : array-like, length 3
    skyfunc    : SkyFunction instance

    Returns
    -------
    ndarray, shape (N_test, 3)
    """
    return np.array([
        skyfunc.s2g(theta_np[j, dipole_idx] * 1e3)
        for j in range(theta_np.shape[0])
    ])


def run_tarp_test(prior, simulator_fn, posterior, N_TARP, N_TARP_SAMPLES):
    """Run TARP test: sample prior, simulate, draw posterior, return arrays.

    Parameters
    ----------
    prior           : sbi BoxUniform prior
    simulator_fn    : callable(theta_tensor) -> x_tensor
    posterior       : trained sbi posterior
    N_TARP          : number of test simulations
    N_TARP_SAMPLES  : posterior samples per test point

    Returns
    -------
    theta_tarp : Tensor, shape (N_TARP, n_params)
    x_tarp     : Tensor, shape (N_TARP, n_summary)
    samples    : Tensor, shape (N_TARP_SAMPLES, N_TARP, n_params)
    """
    theta_tarp = prior.sample((N_TARP,))
    x_tarp = torch.stack([simulator_fn(theta_tarp[i]) for i in range(N_TARP)])

    with torch.no_grad():
        samples = torch.stack([
            posterior.sample((N_TARP_SAMPLES,), x=x_tarp[i])
            for i in range(N_TARP)
        ], dim=1)

    return theta_tarp, x_tarp, samples
