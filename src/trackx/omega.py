"""Bounded pixel-seed alternatives. Coordinates in metres; no truth inputs."""

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .core import wrap
from .extension import trajectory
from .precision import initialize, innovation, propagate


def residuals(xyz, paths):
    """Largest 2D residual using a prompt-circle fit across the whole seed."""
    q = xyz[paths]
    r = np.hypot(q[:, :, 0], q[:, :, 1])
    h, phi, t, z0 = trajectory(q, np.ones(paths.shape, bool))
    angle = np.arcsin(np.clip(h[:, None] * r, -0.999, 0.999))
    tang = wrap(np.arctan2(q[:, :, 1], q[:, :, 0]) - phi[:, None] - angle) * r
    dz = q[:, :, 2] - z0[:, None] - t[:, None] * r / np.sinc(angle / np.pi)
    return np.hypot(tang / 0.00035, dz / 0.0007).max(1), h


def alternatives(barrel, layers, volumes, seeds, neighbors=4):
    """Offer single same-layer substitutions, retain original plus two alternatives."""
    if not 1 <= neighbors <= 4:
        raise ValueError("neighbor cap is four")
    xyz = barrel.xyz
    pix = np.flatnonzero(volumes == 8)
    nearby = np.full((len(xyz), neighbors), -1, int)
    for layer in np.unique(layers[pix]):
        ix = pix[layers[pix] == layer]
        _, near = cKDTree(xyz[ix]).query(
            xyz[ix],
            k=np.arange(1, min(len(ix), neighbors) + 1),
            distance_upper_bound=0.008,
        )
        valid = near < len(ix)
        i, j = np.nonzero(valid)
        nearby[ix[i], j] = ix[near[i, j]]
    pool = defaultdict(list)
    for length in (3, 4):
        owners = np.array([i for i, s in enumerate(seeds) if len(s) == length], int)
        if not len(owners):
            continue
        for start in range(0, len(owners), 256):
            group = owners[start : start + 256]
            orig = np.array([seeds[i] for i in group])
            if np.any(volumes[orig] != 8) or np.any(np.diff(layers[orig], axis=1) <= 0):
                raise ValueError("invalid pixel seed ordering")
            options = [orig]
            origin = [group]
            for col in range(length):
                n = nearby[orig[:, col]]
                a, k = np.nonzero((n >= 0) & (n != orig[:, col, None]))
                alt = orig[a].copy()
                alt[:, col] = n[a, k]
                options.append(alt)
                origin.append(group[a])
            mat = np.concatenate(options)
            source = np.concatenate(origin)
            cost, _ = residuals(xyz, mat)
            cost[np.any(np.diff(barrel.r[mat], axis=1) <= 0, axis=1)] = np.inf
            for i, s, c in zip(source, mat, cost):
                if np.isfinite(c):
                    pool[int(i)].append((float(c), tuple(s.tolist())))
    result = []
    for i, s in enumerate(seeds):
        original = tuple(s)
        rows = sorted(pool[i])
        if not any(p == original for _, p in rows):
            raise ValueError("non-outward original seed")
        other = [v for v in rows if v[1] != original][:2]
        result.append([next(v for v in rows if v[1] == original), *other])
    return result


def corroboration(b, paths):
    """Best first outer-layer innovation, capped at the nominal gate."""
    if not paths:
        return np.empty(0)
    x, p, r = initialize(b, paths)
    _, _, radius, ix, tree = b.layers[0]
    if not len(ix):
        return np.zeros(len(paths))
    y, _, _, ok = propagate(x, p, r, np.full(len(r), radius))
    coords = np.column_stack(
        (radius * np.cos(y[:, 0]), radius * np.sin(y[:, 0]), y[:, 1])
    )
    _, j = tree.query(coords, k=np.arange(1, min(4, len(ix)) + 1))
    a = np.repeat(np.arange(len(paths)), j.shape[1])
    h = ix[j.ravel()]
    y, c, _, valid = propagate(x[a], p[a], r[a], b.r[h])
    d, *_ = innovation(y, c, np.column_stack((b.phi[h], b.xyz[h, 2])), b.r[h], 1.0)
    return np.where(
        ok,
        np.where(valid, np.minimum(d, 9.0), 9.0).reshape(len(r), -1).min(1) / 9.0,
        0.0,
    )


def choose(b, pool, mode):
    originals = [p[0][1] for p in pool]
    if mode == "strict":
        return [list(p[0][1]) for p in pool if p[0][0] <= 2.0]
    if mode not in ("replace", "beam", "joint"):
        raise ValueError("unknown seed strategy")
    owner = {j: i for i, p in enumerate(originals) for j in p}
    if len(owner) != sum(map(len, originals)):
        raise ValueError("shared original seed")
    entries = [
        (i, c, p)
        for i, opts in enumerate(pool)
        for c, p in opts
        if mode == "joint" or all(owner.get(j, i) == i for j in p)
    ]
    ranks = np.array([c for _, c, _ in entries])
    if mode in ("beam", "joint"):
        for start in range(0, len(entries), 512):
            paths = [p for _, _, p in entries[start : start + 512]]
            ranks[start : start + len(paths)] += corroboration(b, paths)
    # Score a whole measurement set. Prefer lower worst normalized seed residual
    # plus optional capped extension cost. Deterministic fallback for conflicts.
    used = set()
    accepted = {}
    for k in sorted(
        range(len(entries)), key=lambda k: (ranks[k], entries[k][0], entries[k][2])
    ):
        i, _, p = entries[k]
        if i not in accepted and used.isdisjoint(p):
            accepted[i] = list(p)
            used.update(p)
    return [p for _, p in sorted(accepted.items())]
