from types import SimpleNamespace

import numpy as np
import pytest

from trackx.omega import alternatives, choose, residuals


def fixture():
    r = np.array([0.04, 0.08, 0.12, 0.16, 0.12])
    phi = 3.13 + np.arcsin(0.2 * r)
    xyz = np.column_stack(
        (r * np.cos(phi), r * np.sin(phi), np.arcsin(0.2 * r) / 0.2 * 0.4)
    )
    xyz[2, 2] += 0.003
    return SimpleNamespace(xyz=xyz, r=r), np.array([2, 4, 6, 8, 6]), np.full(5, 8)


def test_correct_replacement_and_retention():
    b, l, v = fixture()
    pool = alternatives(b, l, v, [[0, 1, 2, 3]])
    assert len(pool[0]) <= 3
    assert choose(b, pool, "replace") == [[0, 1, 4, 3]]
    assert choose(b, alternatives(b, l, v, [[0, 1, 4, 3]]), "replace") == [[0, 1, 4, 3]]
    assert residuals(b.xyz, np.array([[0, 1, 4, 3]]))[0] < 0.01


def test_invalid_transitions_and_ownership():
    b, l, v = fixture()
    with pytest.raises(ValueError):
        alternatives(b, l, v, [[0, 2, 1]])
    with pytest.raises(ValueError):
        choose(b, [[(0, (0, 1, 2))], [(0, (2, 3, 4))]], "replace")
    with pytest.raises(ValueError):
        alternatives(b, l, v, [[0, 1, 2]], 5)
