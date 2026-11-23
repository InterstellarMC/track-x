"""Ideal barrel simulation and sparse outward association."""

import numpy as np
import torch
from scipy.spatial import cKDTree

RADII = np.arange(1, 9) * 0.04


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def intersection(r, k):
    bend = np.arcsin(k * r / 2)
    return bend, r / np.sinc(bend / np.pi)


def event(c, number):
    rng = np.random.default_rng([c["seed"], number])
    if c["tracks"] < 0 or c["noise"] < 0 or 8 * c["tracks"] + c["noise"] > 4096:
        raise ValueError("invalid hit budget")
    if not 0 <= c["missing"] <= 1:
        raise ValueError("missing probability outside [0,1]")
    if not 0 < c.get("pt_min", 0.3) <= 2.5 or abs(0.299792458 * c["field"]) * RADII[
        -1
    ] >= 2 * c.get("pt_min", 0.3):
        raise ValueError("configured momentum must reach the outer cylinder")
    vertex_count = c.get("vertices", 0)
    if not 0 <= vertex_count <= 4096:
        raise ValueError("vertex budget exceeded")
    vertices = rng.normal(0, c.get("vertex_sigma", 0.012), vertex_count)
    rows = []
    ids = []
    pars = {}

    def add(layer, phi, z, pid):
        rows.append([RADII[layer] * np.cos(phi), RADII[layer] * np.sin(phi), z, layer])
        ids.append(pid)

    for pid in range(1, c["tracks"] + 1):
        pt = rng.uniform(c.get("pt_min", 0.3), 2.5)
        q = rng.choice([-1, 1])
        pars[pid] = (float(pt), int(q))
        bend, arc = intersection(RADII, -0.299792458 * q * c["field"] / pt)
        phi = rng.uniform(-np.pi, np.pi) + bend
        z0 = (
            rng.choice(vertices)
            if vertex_count
            else rng.normal(0, c.get("vertex_sigma", 0.012))
        )
        z = z0 + arc * rng.uniform(-1.6, 1.6)
        for i in range(len(RADII)):
            if rng.random() >= c["missing"]:
                add(
                    i,
                    phi[i] + rng.normal(0, 0.0002 / RADII[i]),
                    z[i] + rng.normal(0, 0.0002),
                    pid,
                )
    for _ in range(c["noise"]):
        add(rng.integers(8), rng.uniform(-np.pi, np.pi), rng.uniform(-0.5, 0.5), 0)
    order = rng.permutation(len(rows))
    return np.array(rows).reshape(-1, 4)[order], np.array(ids, dtype=int)[order], pars


def graph(h, settings=None):
    settings = settings or {}
    if (
        not 1 <= settings.get("max_gap", 2) <= 2
        or not 1 <= settings.get("neighbors", 8) <= 8
    ):
        raise ValueError("allowed layer gap 1–2 and neighbor count 1–8")
    if (
        not 0 < settings.get("phi_chord", 0.23) <= 2
        or settings.get("max_slope", 2) <= 0
    ):
        raise ValueError("invalid graph window")
    if len(h) > 4096 or not np.isfinite(h).all():
        raise ValueError("invalid or oversized event")
    x, y, z, l = h.T
    r = np.hypot(x, y)
    if np.any(r <= 0) or np.any(l != l.astype(int)):
        raise ValueError("invalid detector coordinates")
    p = np.arctan2(y, x)
    nodes = np.column_stack((r / 0.32, np.cos(p), np.sin(p), z / 0.5, l / 7))
    edges = []
    attrs = []
    for layer in np.unique(l):
        target = np.flatnonzero(l == layer)
        tree = cKDTree(np.column_stack((np.cos(p[target]), np.sin(p[target]))))
        for i in np.flatnonzero(
            (l < layer) & (l >= layer - settings.get("max_gap", 2))
        ):
            near = target[
                tree.query_ball_point(
                    [np.cos(p[i]), np.sin(p[i])], settings.get("phi_chord", 0.23)
                )
            ]
            dr = r[near] - r[i]
            good = (dr > 0) & (
                abs(z[near] - z[i]) < settings.get("max_slope", 2) * dr + 0.003
            )
            j = near[good]
            dp = wrap(p[j] - p[i])
            dz = z[j] - z[i]
            slope = dz / dr[good]
            intercept = z[i] - r[i] * slope
            cost = abs(dp) / 0.12 + abs(intercept) / 0.05
            for a in np.argsort(cost, kind="stable")[
                : min(settings.get("neighbors", 8), 8)
            ]:
                edges.append((i, j[a]))
                attrs.append(
                    (
                        dp[a] / 0.12,
                        slope[a] / 2,
                        intercept[a] / 0.05,
                        dr[good][a] / 0.08,
                    )
                )
    return (
        torch.tensor(nodes, dtype=torch.float32),
        torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
        torch.tensor(attrs, dtype=torch.float32).reshape(-1, 4),
    )


def labels(ids, edge):
    i, j = edge.numpy()
    return torch.tensor((ids[i] == ids[j]) & (ids[i] > 0), dtype=torch.float32)
