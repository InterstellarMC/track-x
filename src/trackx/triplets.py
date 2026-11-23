"""Rank complete three/four-layer hypotheses before assigning pixel hits."""

import numpy as np

from .core import wrap


def hypotheses(h, edge, score=None):
    """Linearize prompt azimuth and z locally in radius; rank compatible continuations."""
    n = len(h)
    i, j = edge
    if not len(i):
        return []
    degree = np.bincount(i, minlength=n)
    if degree.max() > 16:
        raise ValueError("out-degree exceeds graph contract")
    # Each row maps a hit to its outgoing edge IDs. No hit-by-hit dense matrix.
    links = np.full((n, int(degree.max())), -1, dtype=int)
    order = np.argsort(i, kind="stable")
    slot = np.arange(len(i)) - np.repeat(np.cumsum(degree) - degree, degree)
    links[i[order], slot] = order
    first, k = np.nonzero(links[j] >= 0)
    second = links[j[first], k]
    a, b, c = i[first], j[first], j[second]
    r = np.hypot(h[:, 0], h[:, 1])
    phi = np.arctan2(h[:, 1], h[:, 0])
    scale = (r[c] - r[b]) / (r[b] - r[a])
    dp = wrap(phi[c] - phi[b] - wrap(phi[b] - phi[a]) * scale)
    dz = h[c, 2] - h[b, 2] - (h[b, 2] - h[a, 2]) * scale
    residual = np.hypot(dp / 0.010, dz / 0.003)
    ok = residual < 3
    cost = residual.copy()
    if score is not None:
        cost -= 0.2 * np.log(np.maximum(score[first] * score[second], 1e-8))
    ranked = np.flatnonzero(ok)
    ranked = ranked[np.argsort(cost[ranked], kind="stable")]
    best = np.full(len(i), -1, dtype=int)
    # Reversed assignment leaves the lowest-cost continuation at each edge.
    best[first[ranked[::-1]]] = ranked[::-1]
    next_triplet = best[second[ranked]]
    extended = next_triplet >= 0
    quads = ranked[extended]
    nxt = next_triplet[extended]
    sort4 = np.argsort(cost[quads] + cost[nxt], kind="stable")
    taken = np.zeros(n, dtype=bool)
    result = []
    for seq in [
        np.column_stack((a[quads], b[quads], c[quads], c[nxt]))[sort4],
        np.column_stack((a[ranked], b[ranked], c[ranked])),
    ]:
        for path in seq:
            if not taken[path].any():
                result.append(path.tolist())
                taken[path] = True
    return result
