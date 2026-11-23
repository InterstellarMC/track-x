"""Vectorized evaluation of the frozen triplet extrapolation ranking."""

import numpy as np

from .core import wrap


def resolve(candidates, xyz):
    unique = sorted({tuple(s) for s in candidates})
    ranked = []
    for n in (4, 3):
        group = np.asarray([s for s in unique if len(s) == n], int)
        if not len(group):
            continue
        pts = xyz[group]
        dr = np.diff(np.hypot(pts[:, :, 0], pts[:, :, 1]), axis=1)
        k = dr[:, 1:] / dr[:, :-1]
        dp = wrap(np.diff(np.arctan2(pts[:, :, 1], pts[:, :, 0]), axis=1))
        dz = np.diff(pts[:, :, 2], axis=1)
        cost = np.hypot(
            wrap(dp[:, 1:] - k * dp[:, :-1]) / 0.010,
            (dz[:, 1:] - k * dz[:, :-1]) / 0.003,
        ).mean(1)
        ranked.extend(group[np.argsort(cost, kind="stable")].tolist())
    used = set()
    out = []
    for s in ranked:
        if used.isdisjoint(s):
            out.append(s)
            used.update(s)
    return out
