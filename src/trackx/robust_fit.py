"""Independent prompt-helix fit using a robust loss on measured coordinates."""

import numpy as np
from scipy.optimize import least_squares

from .core import intersection, wrap


def fit_helix(h, sigma=0.0002):
    """Return (phi0, k, z0, tan_lambda), tangential and z residuals in m."""
    x, y, z = h[:, :3].T
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    if len(r) < 3 or np.ptp(r) < 1e-8 or sigma <= 0:
        raise ValueError("three radially distinct measurements required")
    bound = 1.999 / r.max()

    def residual(par):
        phi0, k, z0, slope = par
        bend, arc = intersection(r, k)
        return np.column_stack((r * wrap(phi - phi0 - bend), z - z0 - slope * arc))

    slope, z0 = np.polyfit(r, z, 1)
    opt = least_squares(
        lambda par: residual(par).ravel() / sigma,
        [phi[0], 0, z0, slope],
        bounds=([-np.inf, -bound, -np.inf, -np.inf], [np.inf, bound, np.inf, np.inf]),
        loss="soft_l1",
        max_nfev=150,
    )
    if not opt.success:
        raise RuntimeError(opt.message)
    return opt.x, residual(opt.x)
