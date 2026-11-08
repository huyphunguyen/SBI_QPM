# tests/test_imports.py
"""Smoke test: every name the notebooks import from sbi_qpm must exist."""
import importlib

import pytest

NOTEBOOK_IMPORTS = {
    "sbi_qpm.data": ["load_gaia_qso", "compute_pixel_stats", "make_galactic_mask"],
    "sbi_qpm.diagnostics": ["marginal_coverage", "run_tarp_test", "extract_g_tarp", "extract_g_theta"],
    "sbi_qpm.forward_model": ["WeightedSHT"],
    "sbi_qpm.geometry": ["SkyFunction", "icrs_to_galactic", "direction_summary"],
    "sbi_qpm.pixelization": ["Pixelize", "pixelize_full_covariance"],
    "sbi_qpm.plotting": ["plot_power_spectrum", "plot_snr_per_ell"],
    "sbi_qpm.power_spectrum_analysis": [
        "bandpower_to_full_cl", "make_bins", "make_spin1_field", "make_spin0_field",
        "apodize_mask", "compute_power_spectrum", "estimate_noise_covariance",
        "evaluate_cross_spectrum",
    ],
    "sbi_qpm.sbi_setup": [
        "n_real_params", "build_full_alm_complex", "build_det_alm_complex",
        "precompute_noise_cholesky", "generate_pixel_noise",
        "create_prior_full", "create_prior_det", "simulator_full", "simulator_det",
    ],
}


@pytest.mark.parametrize("module", sorted(NOTEBOOK_IMPORTS))
def test_notebook_imports_resolve(module):
    mod = importlib.import_module(module)
    for name in NOTEBOOK_IMPORTS[module]:
        assert hasattr(mod, name), f"{module}.{name} missing"


def test_package_is_sbi_qpm():
    import sbi_qpm
    assert "sbi_qpm" in sbi_qpm.__file__
