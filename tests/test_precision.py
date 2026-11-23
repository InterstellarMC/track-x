"""Radial EKF numerics, nominal innovation consistency and ownership."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from trackx.precision import innovation, propagate, reconstruct, select_paths, update


@pytest.mark.parametrize("h", [0, 1e-8, 0.45, -1.2])
def test_jacobian_and_covariance(h):
    x = np.array([[0.2, 0.03, h, 0.7]])
    p = np.eye(4)[None] * 0.001
    r0, r1 = np.array([0.11]), np.array([0.75])
    y, cov, f, ok = propagate(x, p, r0, r1)
    assert ok.all()
    for j in range(4):
        delta = np.zeros_like(x)
        delta[0, j] = 1e-6
        diff = (
            propagate(x + delta, p, r0, r1)[0] - propagate(x - delta, p, r0, r1)[0]
        ) / 2e-6
        np.testing.assert_allclose(diff[0], f[0, :, j], atol=1e-7)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    _, res, l, noise = innovation(y, cov, y[:, :2] + 0.0003, r1, 1.0)
    _, post = update(y, cov, res, l, noise)
    np.testing.assert_allclose(post, post.transpose(0, 2, 1), atol=1e-13)
    assert np.all(np.linalg.eigvalsh(post) > 0)
    assert post[0, 0, 0] < cov[0, 0, 0]
    assert not propagate(np.array([[0.0, 0.0, 2.0, 0.0]]), p, r0, r1)[-1].any()


def test_singular_prior_and_nominal_innovations():
    rng = np.random.default_rng(1729)
    n = 5000
    y = np.zeros((n, 4))
    prior = np.zeros((n, 4, 4))
    radius = np.full(n, 0.5)
    meas = rng.normal(size=(n, 2)) * 0.003
    chi, res, l, noise = innovation(y, prior, meas, radius, 1)
    assert abs(chi.mean() - 2) < 0.08
    assert 0.980 < (chi < 9).mean() < 0.997
    _, post = update(y, prior, res, l, noise)
    assert np.isfinite(post).all()


def test_whole_path_conflicts():
    seeds = [[0, 1], [2, 3]]
    paths = select_paths(
        seeds,
        [
            ([[0, 1, 4, 5], [2, 3, 6, 5]], [1.8, 1.5]),
            ([[0, 1, 4], [2, 3, 6]], [0.8, 0.9]),
        ],
    )
    assert paths == [[0, 1, 4, 5], [2, 3, 6]]


@pytest.mark.parametrize("mode", ["M2", "M3", "M4", "M5"])
def test_bounded_multilayer_filter(mode):
    r = np.array([0.04, 0.08, 0.12, 0.3, 0.3, 0.55, 0.8])
    phi = 0.1 + np.arcsin(r * 0.2)
    xyz = np.column_stack(
        (r * np.cos(phi), r * np.sin(phi), np.arcsin(r * 0.2) / 0.2 * 0.4)
    )
    xyz[4, 2] += 0.02
    b = SimpleNamespace(r=r, xyz=xyz, phi=phi, layers=[])
    for layer, ix in enumerate(([3, 4], [5], [6])):
        ix = np.array(ix)
        b.layers.append((13, layer, r[ix[0]], ix, cKDTree(xyz[ix])))
    expected = [[0, 1, 2, 3, 5, 6]]
    for _ in range(2):
        assert reconstruct(b, [[0, 1, 2]], mode)[0] == expected
    b.layers[1] = (13, 1, 0.55, np.array([], int), None)
    assert reconstruct(b, [[0, 1, 2]], mode)[0] == [[0, 1, 2, 3]]


def test_hit_labels_invariant_and_phi_seam():
    import pandas as pd

    from trackx.extension import Barrel

    radius = np.array([0.04, 0.08, 0.12, 0.3, 0.55])
    phi = np.pi - 0.04 + np.arcsin(radius * 0.25)
    hits = pd.DataFrame(
        {
            "x": 1000 * radius * np.cos(phi),
            "y": 1000 * radius * np.sin(phi),
            "z": 1000 * np.arcsin(radius * 0.25) / 0.25 * 0.1,
            "volume_id": [8, 8, 8, 13, 17],
            "layer_id": [2, 4, 6, 2, 2],
        }
    )
    modules = hits.rename(columns={"x": "cx", "y": "cy"})
    rng = np.random.default_rng(72)
    base = Barrel(hits, modules)
    shuffled = Barrel(
        hits.assign(
            particle_id=rng.integers(0, 10, len(hits)), weight=rng.random(len(hits))
        ),
        modules,
    )
    for mode in ("M2", "M3", "M4", "M5"):
        expect = reconstruct(base, [[0, 1, 2]], mode)[0]
        assert expect == [[0, 1, 2, 3, 4]]
        assert reconstruct(shuffled, [[0, 1, 2]], mode)[0] == expect
