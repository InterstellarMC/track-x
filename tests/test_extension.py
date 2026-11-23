import numpy as np
import pandas as pd
import pytest

from trackx.extension import Barrel, project, topology, trajectory


@pytest.mark.parametrize("curvature", [0.0, 0.6, -0.6])
def test_helix(curvature):
    r = np.array([0.03, 0.07, 0.11, 0.17])
    params = (curvature, 3.12, 0.8, 0.01)
    phi, z, _ = project(params, r)
    xyz = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))[None]
    got = trajectory(xyz, np.ones((1, 4), bool))
    pred = project(got, 0.7)
    actual = project(params, 0.7)
    assert np.cos(pred[0] - actual[0]) == pytest.approx(1.0, abs=1e-8)
    assert pred[1] == pytest.approx(actual[1], abs=1e-5)


def test_missing_surface_and_exclusivity():
    rs = [30, 70, 110, 170, 260, 360, 500, 660, 820, 1020]
    vs = [8] * 4 + [13] * 4 + [17] * 2
    ls = [2, 4, 6, 8, 2, 4, 6, 8, 2, 4]
    m = pd.DataFrame({"cx": rs, "cy": 0, "volume_id": vs, "layer_id": ls})
    h = (
        m.rename(columns={"cx": "x", "cy": "y"})
        .assign(z=0)
        .drop(index=4)
        .reset_index(drop=True)
    )
    b = Barrel(h, m)
    paths, _, _ = b.extend([[0, 1, 2, 3]])
    assert paths == [list(range(9))]
    assert b.extend([[0, 1, 2, 3]], gaps=0)[0] == [[0, 1, 2, 3]]
    assert [v for v, _, _ in topology(m)] == vs
    with pytest.raises(ValueError):
        topology(m.assign(cx=1))
