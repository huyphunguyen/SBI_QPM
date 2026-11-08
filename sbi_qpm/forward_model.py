import math
import numpy as np
import healpy as hp
from scipy.linalg import lu_factor, lu_solve


class WeightedSHT:
    """Weighted spin-1 spherical harmonic transform solver.

    Solves (AᵀC⁻¹A) â = AᵀC⁻¹d for proper-motion alm coefficients,
    accounting for the full 2×2 per-pixel noise covariance.
    """

    def __init__(self, nside, mask, pm_ra, pm_dec, sigma_pmra, sigma_pmdec, pm_corr):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.mask = mask
        self._lu_cache = {}

        self.pm_ra_map = np.zeros(self.npix)
        self.pm_dec_map = np.zeros(self.npix)
        self.pm_ra_map[mask] = pm_ra[mask]
        self.pm_dec_map[mask] = pm_dec[mask]

        self.Cinv_rr = np.zeros(self.npix)
        self.Cinv_rd = np.zeros(self.npix)
        self.Cinv_dd = np.zeros(self.npix)

        f = 1.0 / (1.0 - pm_corr[mask]**2)
        self.Cinv_rr[mask] = f / sigma_pmra[mask]**2
        self.Cinv_rd[mask] = -f * pm_corr[mask] / (sigma_pmra[mask] * sigma_pmdec[mask])
        self.Cinv_dd[mask] = f / sigma_pmdec[mask]**2

    # ----------------------------------------------------------------
    # Core operations
    # ----------------------------------------------------------------

    def _apply_Cinv(self, ra_map, dec_map):
        return (
            self.Cinv_rr * ra_map + self.Cinv_rd * dec_map,
            self.Cinv_rd * ra_map + self.Cinv_dd * dec_map,
        )

    def _forward(self, E_alm, B_alm, lmax):
        m_theta, m_phi = hp.alm2map_spin(
            [E_alm.astype(complex), B_alm.astype(complex)],
            self.nside, spin=1, lmax=lmax,
        )
        return m_phi, -m_theta

    def _adjoint(self, ra_map, dec_map, lmax):
        E_alm, B_alm = hp.map2alm_spin(
            [(-dec_map).astype(np.float64), ra_map.astype(np.float64)],
            spin=1, lmax=lmax,
        )
        return E_alm, B_alm

    def _matvec(self, x, lmax):
        E, B = self._real_to_complex(x, lmax)
        ra, dec = self._forward(E, B, lmax)
        w_ra, w_dec = self._apply_Cinv(ra, dec)
        E_out, B_out = self._adjoint(w_ra, w_dec, lmax)
        return self._complex_to_real(E_out, B_out, lmax)

    def _compute_rhs(self, ra_map, dec_map, lmax):
        w_ra, w_dec = self._apply_Cinv(ra_map, dec_map)
        E_rhs, B_rhs = self._adjoint(w_ra, w_dec, lmax)
        return self._complex_to_real(E_rhs, B_rhs, lmax)

    # ----------------------------------------------------------------
    # Real <-> Complex alm conversion
    # ----------------------------------------------------------------

    def _real_to_complex(self, coeff, lmax):
        half = len(coeff) // 2
        E_r, B_r = coeff[:half], coeff[half:]
        n_alm = math.comb(lmax + 2, 2)
        E_alm = np.zeros(n_alm, dtype=complex)
        B_alm = np.zeros(n_alm, dtype=complex)
        E_alm[1:lmax + 1] = E_r[0:lmax]
        B_alm[1:lmax + 1] = B_r[0:lmax]
        E_alm[lmax + 1:] = E_r[lmax::2] + 1j * E_r[lmax + 1::2]
        B_alm[lmax + 1:] = B_r[lmax::2] + 1j * B_r[lmax + 1::2]
        return E_alm, B_alm

    def _complex_to_real(self, E_alm, B_alm, lmax):
        E_m0 = E_alm[1:lmax + 1].real
        B_m0 = B_alm[1:lmax + 1].real
        E_mp, B_mp = E_alm[lmax + 1:], B_alm[lmax + 1:]
        E_mp_r = np.empty(2 * len(E_mp))
        E_mp_r[0::2] = E_mp.real
        E_mp_r[1::2] = E_mp.imag
        B_mp_r = np.empty(2 * len(B_mp))
        B_mp_r[0::2] = B_mp.real
        B_mp_r[1::2] = B_mp.imag
        return np.concatenate([E_m0, E_mp_r, B_m0, B_mp_r])

    # ----------------------------------------------------------------
    # Solvers
    # ----------------------------------------------------------------

    @staticmethod
    def _n_params(lmax):
        return 2 * ((lmax + 1) + (math.comb(lmax + 2, 2) - (lmax + 1)) * 2) - 2

    def _build_normal_matrix(self, lmax):
        n = self._n_params(lmax)
        M = np.zeros((n, n))
        e_i = np.zeros(n)
        for i in range(n):
            e_i[i] = 1.0
            M[:, i] = self._matvec(e_i, lmax)
            e_i[i] = 0.0
        return M

    def solve_direct(self, lmax):
        """Solve via np.linalg.solve. One-off use (no caching)."""
        M = self._build_normal_matrix(lmax)
        rhs = self._compute_rhs(self.pm_ra_map, self.pm_dec_map, lmax)
        return np.linalg.solve(M, rhs)

    def build_lu(self, lmax):
        """Build and cache LU factorisation of normal matrix for given lmax.

        Call once before repeated solve_fast calls (e.g. before simulation loop).
        """
        M = self._build_normal_matrix(lmax)
        self._lu_cache[lmax] = lu_factor(M)

    def solve_fast(self, pmra_fullsky, pmdec_fullsky, lmax):
        """Solve for new full-sky maps using cached LU factorisation.

        Requires build_lu(lmax) to have been called first.
        """
        if lmax not in self._lu_cache:
            raise ValueError(f"LU not cached for lmax={lmax}. Call build_lu({lmax}) first.")
        lu_piv = self._lu_cache[lmax]
        rhs = self._compute_rhs(pmra_fullsky, pmdec_fullsky, lmax)
        return lu_solve(lu_piv, rhs)

    # ----------------------------------------------------------------
    # Convenience
    # ----------------------------------------------------------------

    def generate_map(self, coeff, lmax):
        """Convert real alm vector to (pmra_map, pmdec_map, E_alm, B_alm)."""
        E_alm, B_alm = self._real_to_complex(coeff, lmax)
        ra_map, dec_map = self._forward(E_alm, B_alm, lmax)
        return ra_map, dec_map, E_alm, B_alm
