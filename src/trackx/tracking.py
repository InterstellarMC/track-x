import numpy as np
import torch
from torch import nn


class EdgeNet(nn.Module):
    def __init__(self, depth, width=32, node_features=None, edge_features=None):
        super().__init__()

        def block(n):
            return nn.Sequential(nn.Linear(n, width), nn.LayerNorm(width), nn.SiLU())

        self.register_buffer("mean", torch.zeros(9))
        self.register_buffer("scale", torch.ones(9))
        self.node_features = list(range(5)) if node_features is None else node_features
        self.edge_features = list(range(4)) if edge_features is None else edge_features
        for indices, size in [(self.node_features, 5), (self.edge_features, 4)]:
            if (
                not indices
                or len(set(indices)) != len(indices)
                or min(indices) < 0
                or max(indices) >= size
            ):
                raise ValueError("invalid feature indices")
        self.nodes = block(len(self.node_features))
        self.edges = block(len(self.edge_features) + 2 * width)
        self.updates = nn.ModuleList(block(2 * width) for _ in range(depth))
        self.messages = nn.ModuleList(block(3 * width) for _ in range(depth))
        self.out = nn.Linear(width, 1)

    def forward(self, x, edge, attr):
        x = (x - self.mean[:5]) / self.scale[:5]
        attr = (attr - self.mean[5:]) / self.scale[5:]
        i, j = edge
        h = self.nodes(x[:, self.node_features])
        e = self.edges(torch.cat([h[i], h[j], attr[:, self.edge_features]], 1))
        for update, msg in zip(self.updates, self.messages):
            agg = torch.zeros_like(h).index_add(0, j, e)
            count = h.new_zeros(len(h)).index_add(0, j, h.new_ones(len(j)))
            h = h + update(torch.cat([h, agg / count.clamp_min(1)[:, None]], 1))
            e = e + msg(torch.cat([h[i], h[j], e], 1))
        return self.out(e).flatten()


def assemble(h, edge, scores, gate):
    """Greedy edge linking with one predecessor and one successor."""
    nxt = {}
    prev = {}

    def plausible(a, b, c):
        u = h[b, :3] - h[a, :3]
        v = h[c, :3] - h[b, :3]
        return np.dot(u, v) > 0.985 * np.linalg.norm(u) * np.linalg.norm(v)

    for idx in np.argsort(-scores, kind="stable"):
        if scores[idx] < 0.5:
            break
        a, b = edge[:, idx]
        if a in nxt or b in prev:
            continue
        if gate and (
            (a in prev and not plausible(prev[a], a, b))
            or (b in nxt and not plausible(a, b, nxt[b]))
        ):
            continue
        nxt[a] = b
        prev[b] = a
    paths = []
    for start in sorted(nxt.keys() - prev.keys()):
        path = [start]
        while path[-1] in nxt:
            path.append(nxt[path[-1]])
        if len(path) >= 3:
            paths.append(path)
    return paths


def fit(points, field):
    """Beam-axis constrained circle, linear z versus transverse arc."""
    x, y, z = points[:, :3].T
    u, s, vt = np.linalg.svd(np.column_stack((x, y)), full_matrices=False)
    r = np.hypot(x, y)
    cross = x[0] * y[-1] - y[0] * x[-1]
    k = 0.0
    if s[-1] > 1e-10 * s[0]:
        center = vt.T @ ((u.T @ (r * r / 2)) / s)
        k = np.sign(cross) / np.linalg.norm(center)
    bend = np.arcsin(np.clip(k * r / 2, -1, 1))
    arc = r / np.sinc(bend / np.pi)
    dz = z - np.polyval(np.polyfit(arc, z, 1), arc)
    return k, abs(0.299792458 * field / k) if k else np.inf, dz
