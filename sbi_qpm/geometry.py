import numpy as np


class SkyFunction:
    def __init__(self, nside=None):
        self.nside = nside  # reserved for future HEALPix operations; unused by current methods

    def vec2dir(self, vec):
        gx, gy, gz = vec
        hyp = np.sqrt(gx**2 + gy**2)
        dec = np.arctan2(gz, hyp)
        ra = np.arctan2(gy, gx)
        return np.degrees(ra) % 360, np.degrees(dec)

    def dir2vec(self, ra, dec):
        deg2rad = np.pi / 180.0
        return np.array([
            np.cos(deg2rad * ra) * np.cos(deg2rad * dec),
            np.sin(deg2rad * ra) * np.cos(deg2rad * dec),
            np.sin(deg2rad * dec),
        ])

    def s2g(self, s):
        # Real SH normalization: Y_1^0 ~ sqrt(3/4pi)*cos(theta), so gz gets sqrt(3/8pi)
        # while Y_1^{+/-1} ~ sqrt(3/4pi)*sin(theta), so gx/gy get sqrt(3/4pi).
        s10, s11_r, s11_i = s
        gz =  s10  * np.sqrt(3 / (8 * np.pi))
        gx = -s11_r * np.sqrt(3 / (4 * np.pi))
        gy =  s11_i * np.sqrt(3 / (4 * np.pi))
        return gx, gy, gz

    def g2s(self, g):
        gx, gy, gz = g
        s10  =  np.sqrt((8 * np.pi) / 3) * gz
        s11_r = -np.sqrt((4 * np.pi) / 3) * gx
        s11_i =  np.sqrt((4 * np.pi) / 3) * gy
        return s10, s11_r, s11_i


def icrs_to_galactic(ra_deg, dec_deg):
    """Vectorised ICRS (ra, dec) -> galactic (l, b), all in degrees."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    gal = SkyCoord(np.atleast_1d(ra_deg) * u.deg,
                   np.atleast_1d(dec_deg) * u.deg, frame='icrs').galactic
    return gal.l.deg % 360.0, gal.b.deg


def _wrapped_percentiles(angles_deg, center_deg, q=(16, 50, 84)):
    """Percentiles of a periodic angle, unwrapped onto (-180, 180] about `center_deg`.

    Returned values are absolute angles (mod 360 for longitudes); taking raw
    percentiles of e.g. RA would be meaningless for a posterior straddling 0/360.
    """
    delta = (np.asarray(angles_deg) - center_deg + 180.0) % 360.0 - 180.0
    return center_deg + np.percentile(delta, q)


def direction_summary(g_samples, containment=(68.0, 95.0)):
    """Posterior direction of a 3-vector, in ICRS and galactic coordinates.

    The point estimate is the direction of the mean unit vector (the only
    unbiased centre for a direction; component-wise medians of RA/Dec are not).
    Marginal errors are 16/84 percentiles unwrapped about that centre, and
    `theta_XX` are the angular radii containing XX% of the sampled directions.

    Parameters
    ----------
    g_samples   : (N, 3) array of posterior draws of (gx, gy, gz), ICRS Cartesian.
    containment : percentages for the angular containment radii.

    Returns
    -------
    dict with ra/dec/l/b centres (`*_c`), their 16/50/84 percentiles
    (`*_16`, `*_50`, `*_84`), `theta_68`/`theta_95` [deg], and `R` = length of
    the mean resultant vector (1 = perfectly aligned, 0 = isotropic).
    """
    g = np.atleast_2d(np.asarray(g_samples, dtype=float))
    n = g / np.linalg.norm(g, axis=1, keepdims=True)

    n_bar = n.mean(axis=0)
    R = float(np.linalg.norm(n_bar))
    n_bar = n_bar / R

    sky = SkyFunction(nside=None)
    ra_c, dec_c = sky.vec2dir(n_bar)
    ra, dec = sky.vec2dir(n.T)
    l, b = icrs_to_galactic(ra, dec)
    l_c, b_c = icrs_to_galactic(ra_c, dec_c)
    l_c, b_c = float(l_c[0]), float(b_c[0])

    # Angular separation of each sample from the central direction.
    sep = np.degrees(np.arccos(np.clip(n @ n_bar, -1.0, 1.0)))

    out = {'ra_c': float(ra_c), 'dec_c': float(dec_c),
           'l_c': l_c, 'b_c': b_c, 'R': R}
    for name, arr, centre in (('ra', ra, ra_c), ('l', l, l_c)):
        out[f'{name}_16'], out[f'{name}_50'], out[f'{name}_84'] = \
            _wrapped_percentiles(arr, centre) % 360.0
    for name, arr in (('dec', dec), ('b', b)):
        out[f'{name}_16'], out[f'{name}_50'], out[f'{name}_84'] = \
            np.percentile(arr, [16, 50, 84])
    for c in containment:
        out[f'theta_{int(c)}'] = float(np.percentile(sep, c))
    return out
