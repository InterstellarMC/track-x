"""Outward propagation of disjoint prompt barrel seeds. All distances in metres."""

from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .core import wrap


def topology(modules):
    barrel = modules[modules.volume_id.isin([8, 13, 17])].copy()
    barrel["r"] = np.hypot(barrel.cx, barrel.cy) / 1000
    layers = (
        barrel.groupby(["volume_id", "layer_id"])
        .r.agg(["min", "median", "max"])
        .sort_values("median")
    )
    if list(layers.index.get_level_values(0).unique()) != [8, 13, 17] or np.any(
        layers["min"].values[1:] <= layers["max"].values[:-1]
    ):
        raise ValueError("unsupported or overlapping nominal radial layers")
    return [(int(v), int(l), float(r)) for (v, l), r in layers["median"].items()]


def regression(x, y, w):
    n = w.sum(1)
    mx = (x * w).sum(1) / n
    my = (y * w).sum(1) / n
    dx = x - mx[:, None]
    slope = (dx * (y - my[:, None]) * w).sum(1) / np.maximum((dx**2 * w).sum(1), 1e-12)
    return slope, my - slope * mx


def trajectory(points, mask, refine=True):
    r = np.hypot(points[:, :, 0], points[:, :, 1])
    p = np.arctan2(points[:, :, 1], points[:, :, 0])
    p = p[:, :1] + wrap(p - p[:, :1])
    w = mask.astype(float)
    half, phi = regression(r, p, w)
    limit = 0.995 / np.maximum(r.max(1), 1e-8)
    half = np.clip(half, -limit, limit)
    if refine:
        for _ in range(3):
            u = np.clip(half[:, None] * r, -0.995, 0.995)
            jac = r / np.sqrt(1 - u * u)
            res = p - phi[:, None] - np.arcsin(u)
            weights = w / np.maximum(1, abs(res) * r / 0.001)
            step, intercept = regression(jac, res, weights)
            half = np.clip(half + step, -limit, limit)
            phi += intercept
    bend = np.arcsin(np.clip(half[:, None] * r, -0.999, 0.999))
    arc = r / np.sinc(bend / np.pi)
    slope, z0 = regression(arc, points[:, :, 2], w)
    return half, phi, slope, z0


def project(params, r):
    half, phi, slope, z0 = params
    u = half * r
    angle = np.arcsin(np.clip(u, -0.999999, 0.999999))
    return phi + angle, z0 + slope * r / np.sinc(angle / np.pi), abs(u) < 0.999


class Barrel:
    def __init__(self, hits, modules):
        self.xyz = hits[["x", "y", "z"]].to_numpy() / 1000
        self.r = np.hypot(*self.xyz[:, :2].T)
        self.phi = np.arctan2(self.xyz[:, 1], self.xyz[:, 0])
        self.layers = []
        for vol, layer, r in topology(modules):
            if vol == 8:
                continue
            ix = np.flatnonzero((hits.volume_id == vol) & (hits.layer_id == layer))
            p = self.phi[ix]
            coords = np.column_stack((r * np.cos(p), r * np.sin(p), self.xyz[ix, 2]))
            self.layers.append((vol, layer, r, ix, cKDTree(coords)))

    def extend(
        self,
        seeds,
        *,
        refine=True,
        arbitrate=True,
        gaps=2,
        outer=True,
        straight=False,
        bridge=True,
    ):
        paths = [list(p) for p in seeds]
        if not paths:
            return paths, [], {}
        n = len(paths)
        mat = np.full((n, 10), -1, int)
        for i, p in enumerate(paths):
            mat[i, : len(p)] = p
        size = (mat >= 0).sum(1)
        misses = np.zeros(n, int)
        traces = []
        times = {"fit": 0.0, "search": 0.0, "ownership": 0.0}
        for vol, layer, r, ix, tree in self.layers:
            if vol == 17 and not outer:
                break
            t = perf_counter()
            mask = mat >= 0
            points = np.where(mask[:, :, None], self.xyz[np.maximum(mat, 0)], 0.0)
            params = trajectory(points, mask, refine)
            if straight:
                params = (
                    np.zeros(n),
                    self.phi[mat[np.arange(n), size - 1]],
                    params[2],
                    params[3],
                )
            last = self.r[mat[np.arange(n), size - 1]]
            pp, zz, valid = project(params, r)
            times["fit"] += perf_counter() - t
            t = perf_counter()
            active = valid & (misses <= gaps)
            if not bridge and vol == 17:
                active &= last > 0.5
            rows = np.flatnonzero(active)
            if len(ix) and len(rows):
                pos = np.column_stack(
                    (r * np.cos(pp[rows]), r * np.sin(pp[rows]), zz[rows])
                )
                _, near = tree.query(
                    pos, k=list(range(1, min(8, len(ix)) + 1)), workers=1
                )
                a = np.repeat(rows, near.shape[1])
                b = ix[near.ravel()]
                phi, z, ok = project(tuple(v[a] for v in params), self.r[b])
                flight = np.maximum(self.r[b] - last[a], 0)
                # Empirical residual scales grow with extrapolation distance.
                cost = np.hypot(
                    wrap(phi - self.phi[b]) * self.r[b] / (0.0018 + 0.025 * flight),
                    (z - self.xyz[b, 2]) / (0.003 + 0.04 * flight),
                )
                keep = ok & (cost < 3)
                a, b, cost = a[keep], b[keep], cost[keep]
            else:
                a = b = np.array([], int)
                cost = np.array([])
            times["search"] += perf_counter() - t
            t = perf_counter()
            order = np.lexsort((b, a, cost)) if arbitrate else np.lexsort((b, cost, a))
            used = set()
            claimed = set()
            misses += 1
            for k in order:
                i, j = int(a[k]), int(b[k])
                if i in claimed or j in used:
                    continue
                used.add(j)
                claimed.add(i)
                mat[i, size[i]] = j
                size[i] += 1
                misses[i] = 0
                paths[i].append(j)
            times["ownership"] += perf_counter() - t
            traces.append((vol, layer, a, b))
        return paths, traces, times
