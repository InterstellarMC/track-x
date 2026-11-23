"""Sparse layer-pair neighbors in prompt direction coordinates."""

import numpy as np
import torch
from scipy.spatial import cKDTree

from .core import wrap


def directional_graph(h, curvature_scale=1.5):
    if len(h) > 4096 or not np.isfinite(h).all():
        raise ValueError("invalid sector")
    x, y, z, layer = h.T
    r = np.hypot(x, y)
    if np.any(r <= 0):
        raise ValueError("invalid radius")
    phi = np.arctan2(y, x)
    nodes = np.column_stack((r / 0.32, np.cos(phi), np.sin(phi), z / 0.5, layer / 7))
    edges = []
    attrs = []
    for outer in np.unique(layer):
        target = np.flatnonzero(layer == outer)
        for gap in (1, 2):
            src = np.flatnonzero(layer == outer - gap)
            if not len(src):
                continue
            ri, ro = np.median(r[src]), np.median(r[target])
            # Angular tolerance grows with radial flight. A vertex-z tolerance
            # maps to a slope difference proportional to 1/ri - 1/ro.
            az = 0.025 + abs(ro - ri) * curvature_scale
            longitudinal = 0.05 + 0.065 * abs(1 / ri - 1 / ro)
            vec = np.column_stack(
                (np.cos(phi) / az, np.sin(phi) / az, z / r / longitudinal)
            )
            tree = cKDTree(vec[target])
            dist, j = tree.query(
                vec[src],
                k=list(range(1, min(8, len(target)) + 1)),
                distance_upper_bound=2,
                workers=1,
            )
            a, b = np.nonzero(np.isfinite(dist))
            i = src[a]
            dest = target[j[a, b]]
            dr = r[dest] - r[i]
            ok = dr > 0
            i, dest, dr = i[ok], dest[ok], dr[ok]
            slope = (z[dest] - z[i]) / dr
            edges.append(np.column_stack((i, dest)))
            attrs.append(
                np.column_stack(
                    (
                        wrap(phi[dest] - phi[i]) / 0.12,
                        slope / 2,
                        (z[i] - r[i] * slope) / 0.05,
                        dr / 0.08,
                    )
                )
            )

    def array(parts, n, dtype):
        return torch.as_tensor(
            np.concatenate(parts) if parts else np.empty((0, n)), dtype=dtype
        )

    return (
        torch.as_tensor(nodes, dtype=torch.float32),
        array(edges, 2, torch.long).T.contiguous(),
        array(attrs, 4, torch.float32),
    )
