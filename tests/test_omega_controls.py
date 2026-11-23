import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trackx.disks import disk_paths, planes
from trackx.extension import Barrel
from trackx.smoothing import smooth_group

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from omega_review import paired


def test_fixed_paired_bootstrap():
    assert paired([1, 1, 1]) == {
        "mean": 1.0,
        "median": 1.0,
        "sd": 0.0,
        "low": 1.0,
        "high": 1.0,
        "positive": 3,
        "negative": 0,
    }
    with pytest.raises(ValueError):
        paired([1.0])
    assert paired([-1, 0, 2]) == paired([-1, 0, 2])


def test_disks_order_and_exclusivity():
    z = np.array([-0.9, -0.7, -0.6, 0.6, 0.7, 0.9])
    h = pd.DataFrame(
        {
            "volume_id": [7] * 3 + [9] * 3,
            "layer_id": [2, 4, 6] * 2,
            "z": z * 1000,
            "x": abs(z) * 40,
            "y": abs(z) * 30,
        }
    )
    m = h.rename(columns={"z": "cz"})
    assert [l for _, l in planes(m, 7)] == [6, 4, 2]
    assert disk_paths(h, m) == [[2, 1, 0], [3, 4, 5]]
    assert disk_paths(h.assign(particle_id=42, weight=0), m) == disk_paths(h, m)


def test_rts_straight_track():
    r = np.array([0.04, 0.08, 0.12, 0.3, 0.55])
    h = pd.DataFrame(
        {
            "x": 1000 * r,
            "y": 0.0,
            "z": 230 * r,
            "volume_id": [8, 8, 8, 13, 17],
            "layer_id": [2, 4, 6, 2, 2],
        }
    )
    b = Barrel(h, h.rename(columns={"x": "cx", "y": "cy"}))
    base, sm, cov = smooth_group(b, [[0, 1, 2]], np.array([[3, 4]]))
    np.testing.assert_allclose(sm, base, atol=1e-12)
    assert (np.linalg.eigvalsh(cov) > 0).all()
    np.testing.assert_allclose(cov, cov.transpose(0, 2, 1), atol=1e-14)
