import numpy as np

from trackx.core import RADII, intersection
from trackx.robust_fit import fit_helix


def test_outlying_hit():
    bend, arc = intersection(RADII, 0.8)
    phi = 3.13 + bend
    phi[3] += 0.10
    points = np.column_stack(
        (RADII * np.cos(phi), RADII * np.sin(phi), 0.04 - 0.7 * arc)
    )
    params, residual = fit_helix(points)
    assert abs(params[1] - 0.8) < 0.01
    assert np.argmax(abs(residual[:, 0])) == 3
