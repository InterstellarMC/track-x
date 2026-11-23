"""Exploratory forward-only pixel disk follower with exclusive hit ownership."""

import numpy as np
from scipy.spatial import cKDTree


def planes(modules, volume):
    m = modules[modules.volume_id == volume]
    rows = sorted(
        (abs(float(g.cz.median())) / 1000, int(layer))
        for layer, g in m.groupby("layer_id")
    )
    if (
        len(rows) < 3
        or len({r for r, _ in rows}) != len(rows)
        or (m.cz.min() * m.cz.max() <= 0)
    ):
        raise ValueError("invalid disk inventory")
    return rows


def disk_paths(hits, modules):
    xyz = hits[["x", "y", "z"]].to_numpy() / 1000
    depth = abs(xyz[:, 2])
    vol = hits.volume_id.to_numpy()
    layer = hits.layer_id.to_numpy()
    paths = []
    for v in (7, 9):
        levels = planes(modules, v)
        tables = [np.flatnonzero((vol == v) & (layer == l)) for _, l in levels]
        if any(len(i) > 14000 for i in tables):
            raise ValueError("disk occupancy exceeds configured budget")
        if any(not len(i) for i in tables[:3]):
            continue
        a, b, c = tables[:3]
        # Vertex-constrained search on the second plane. Test finite-difference
        # direction on the third. Exact hit depths refine propagation subsequently.
        pos = xyz[a, :2] * levels[1][0] / depth[a, None]
        _, near = cKDTree(xyz[b, :2]).query(
            pos, k=np.arange(1, min(len(b), 4) + 1), distance_upper_bound=0.012
        )
        i, k = np.nonzero(near < len(b))
        x = a[i]
        y = b[near[i, k]]
        slope = (xyz[y, :2] - xyz[x, :2]) / (depth[y] - depth[x])[:, None]
        pred = xyz[y, :2] + slope * (levels[2][0] - depth[y])[:, None]
        dist, j = cKDTree(xyz[c, :2]).query(pred, distance_upper_bound=0.002)
        used = set()
        part = []
        for k in np.argsort(dist, kind="stable"):
            if not np.isfinite(dist[k]):
                break
            s = [int(x[k]), int(y[k]), int(c[j[k]])]
            if used.isdisjoint(s):
                used.update(s)
                part.append(s)
        active = np.arange(len(part))
        for (z, _), idx in zip(levels[3:], tables[3:]):
            if not len(idx) or not len(active):
                break
            a, b = np.array([part[i][-2:] for i in active]).T
            pos = (
                xyz[b, :2]
                + (xyz[b, :2] - xyz[a, :2])
                * ((z - depth[b]) / (depth[b] - depth[a]))[:, None]
            )
            dist, j = cKDTree(xyz[idx, :2]).query(pos, distance_upper_bound=0.003)
            nxt = []
            for k in np.argsort(dist, kind="stable"):
                if not np.isfinite(dist[k]):
                    break
                h = int(idx[j[k]])
                if h not in used:
                    used.add(h)
                    i = active[k]
                    part[i].append(h)
                    nxt.append(i)
            active = np.array(nxt, int)
        paths += part
    return paths
