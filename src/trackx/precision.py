"""Prompt outward radial EKF and bounded whole-path ambiguity resolution (metres)."""

from time import perf_counter

import numpy as np

from .core import wrap
from .extension import project, trajectory


def arc(radius, half):
    u = np.clip(radius * half, -0.999999, 0.999999)
    angle = np.arcsin(u)
    length = radius / np.sinc(angle / np.pi)
    small = abs(u) < 1e-3
    derivative = np.where(
        small,
        radius**2 * (u / 3 + 3 * u**3 / 10),
        (u / np.sqrt(1 - u * u) - angle) / np.where(small, 1, half**2),
    )
    return angle, length, derivative, radius / np.sqrt(1 - u * u)


def propagate(x, cov, r0, r1):
    half, slope = x[:, 2:].T
    a0, l0, d0, b0 = arc(r0, half)
    a1, l1, d1, b1 = arc(r1, half)
    step = l1 - l0
    y = x.copy()
    y[:, 0] = wrap(y[:, 0] + a1 - a0)
    y[:, 1] += step * slope
    jac = np.tile(np.eye(4), (len(x), 1, 1))
    jac[:, 0, 2] = b1 - b0
    jac[:, 1, 2] = slope * (d1 - d0)
    jac[:, 1, 3] = step
    # Effective independent diffusion per metre of radial travel.
    q = np.abs(r1 - r0)[:, None] * np.array([0.004**2, 0.003**2, 0.025**2, 0.012**2])
    pred = jac @ cov @ jac.transpose(0, 2, 1)
    pred[:, np.arange(4), np.arange(4)] += q
    return y, pred, jac, (abs(r1 * half) < 0.999) & (r1 >= r0)


def initialize(b, paths):
    mat = np.full((len(paths), max(map(len, paths))), -1, int)
    for i, p in enumerate(paths):
        mat[i, : len(p)] = p
    mask = mat >= 0
    pts = np.where(mask[:, :, None], b.xyz[np.maximum(mat, 0)], 0.0)
    params = trajectory(pts, mask)
    end = np.array([p[-1] for p in paths])
    r = b.r[end]
    phi, z, _ = project(params, r)
    lever = np.maximum(r - b.r[mat[:, 0]], 0.015)
    std = np.column_stack(
        (0.00065 / r, np.full(len(r), 0.0012), 0.002 / (r * lever), 0.0025 / lever)
    )
    cov = np.zeros((len(r), 4, 4))
    cov[:, np.arange(4), np.arange(4)] = std**2
    return np.column_stack((phi, z, params[0], params[2])), cov, r


def innovation(y, p, meas, r, scale):
    noise = np.zeros((len(y), 2, 2))
    noise[:, 0, 0] = (0.0015 * scale / r) ** 2
    noise[:, 1, 1] = (0.003 * scale) ** 2
    residual = meas - y[:, :2]
    residual[:, 0] = wrap(residual[:, 0])
    chol = np.linalg.cholesky(p[:, :2, :2] + noise)
    white = np.linalg.solve(chol, residual[:, :, None])[:, :, 0]
    return (white**2).sum(1), residual, chol, noise


def update(y, p, residual, chol, noise):
    k = np.linalg.solve(
        chol.transpose(0, 2, 1), np.linalg.solve(chol, p[:, :2, :])
    ).transpose(0, 2, 1)
    y = y + (k @ residual[:, :, None])[:, :, 0]
    y[:, 0] = wrap(y[:, 0])
    a = np.tile(np.eye(4), (len(y), 1, 1))
    a[:, :, :2] -= k
    cov = a @ p @ a.transpose(0, 2, 1) + k @ noise @ k.transpose(0, 2, 1)
    return y, (cov + cov.transpose(0, 2, 1)) / 2


def select_paths(seeds, families):
    """Greedy whole extensions by positive score with deterministic conflict checks."""
    choices = []
    for paths, values in families:
        for i, (p, v) in enumerate(zip(paths, values, strict=True)):
            tail = tuple(p[len(seeds[i]) :])
            if tail and v > 0:
                choices.append((-float(v), i, tail))
    result = [list(p) for p in seeds]
    used = {j for p in seeds for j in p}
    taken = set()
    for _, i, tail in sorted(choices):
        if i not in taken and used.isdisjoint(tail):
            result[i].extend(tail)
            used.update(tail)
            taken.add(i)
    return result


def follow(b, seeds, mode, independent=False, threshold=9.0, scale=1.0):
    paths = [list(p) for p in seeds]
    gains = np.zeros((len(paths), len(b.layers) + 1))
    value = np.zeros(len(paths))
    x, p, r = initialize(b, paths)
    active = np.ones(len(paths), bool)
    trace = []
    times = {"search": 0.0, "update": 0.0, "ownership": 0.0}
    samples = []
    accepted = []
    for vol, layer, radius, ix, tree in b.layers:
        tic = perf_counter()
        refit = mode in ("M2", "fixed", "M4")
        if refit:
            x, p, r = initialize(b, paths)
        y, _, _, valid = propagate(x, p, r, np.full(len(r), radius))
        rows = np.flatnonzero(active & valid)
        if not len(rows) or not len(ix):
            active[:] = False
            trace.append((vol, layer, np.array([], int), np.array([], int)))
            continue
        phi, z = y[rows, :2].T
        pos = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
        _, near = tree.query(pos, k=np.arange(1, min(8, len(ix)) + 1), workers=1)
        a = np.repeat(rows, near.shape[1])
        h = ix[near.ravel()]
        # Use each module hit's observed radius for the measurement surface.
        y, prior, _, ok = propagate(x[a], p[a], r[a], b.r[h])
        meas = np.column_stack((b.phi[h], b.xyz[h, 2]))
        chi, res, chol, noise = innovation(y, prior, meas, b.r[h], scale)
        tangent = res[:, 0] * b.r[h]
        hard = (abs(tangent) < 0.035) & (abs(res[:, 1]) < 0.055)
        cost = chi
        if mode in ("fixed", "M4"):
            flight = b.r[h] - r[a]
            cost = (tangent / (0.0018 + 0.025 * flight)) ** 2 + (
                res[:, 1] / (0.003 + 0.04 * flight)
            ) ** 2
        good = np.flatnonzero(ok & hard & (cost < threshold))
        trace.append((vol, layer, a[good], h[good]))
        times["search"] += perf_counter() - tic
        tic = perf_counter()
        active[:] = False
        claimed = set()
        picks = []
        for k in good[np.lexsort((h[good], a[good], cost[good]))]:
            i, j = int(a[k]), int(h[k])
            if active[i] or (not independent and j in claimed):
                continue
            active[i] = True
            claimed.add(j)
            paths[i].append(j)
            value[i] += 1 - cost[k] / threshold
            gains[i, len(paths[i]) - len(seeds[i])] = value[i]
            picks.append(k)
        times["ownership"] += perf_counter() - tic
        tic = perf_counter()
        k = np.asarray(picks, int)
        if len(k):
            accepted.extend(chi[k].tolist())
            if not refit:
                x[a[k]], p[a[k]] = update(y[k], prior[k], res[k], chol[k], noise[k])
                r[a[k]] = b.r[h[k]]
        # Keep a single deterministic seed's eight competing proposals per surface.
        inspect = rows[0] if not samples else samples[0]["seed"]
        for j in np.flatnonzero(a == inspect):
            samples.append(
                {
                    "seed": int(inspect),
                    "volume": vol,
                    "layer": layer,
                    "hit": int(h[j]),
                    "radius": float(b.r[h[j]]),
                    "phi": float(y[j, 0]),
                    "z": float(y[j, 1]),
                    "residual": res[j].tolist(),
                    "innovation_cov": (chol[j] @ chol[j].T).tolist(),
                    "chi2": float(chi[j]),
                    "accepted": int(h[j]) in paths[inspect],
                }
            )
        times["update"] += perf_counter() - tic
    return (
        paths,
        value,
        trace,
        times,
        {
            "accepted_chi2_quantiles": np.quantile(accepted, [0.1, 0.5, 0.9]).tolist()
            if accepted
            else [],
            "sample": samples,
            "gains": gains,
        },
    )


def reconstruct(b, seeds, mode):
    if not seeds:
        return [], [], {}, {}
    global_mode = mode in ("M4", "M5")
    threshold = 4.0 if mode == "tight" else 9.0
    scale = 2.0 if mode == "wide_R" else 1.0
    paths, val, trace, times, diagnostics = follow(
        b, seeds, mode, global_mode, threshold, scale
    )
    gains = diagnostics.pop("gains")
    if global_mode:
        # Prefix alternatives allow partial extensions if full candidates conflict.
        start = perf_counter()
        families = [(paths, val)]
        for cut in range(1, 6):
            pref = [
                p[:-cut] if len(p) - cut > len(s) else list(s)
                for s, p in zip(seeds, paths)
            ]
            families.append(
                (
                    pref,
                    [
                        gains[i, len(p) - len(s)]
                        for i, (p, s) in enumerate(zip(pref, seeds))
                    ],
                )
            )
        paths = select_paths(seeds, families)
        diagnostics["altered_by_conflict"] = sum(
            p != f for p, f in zip(paths, families[0][0])
        )
        for sample in diagnostics["sample"]:
            sample["final_accepted"] = sample["hit"] in paths[sample["seed"]]
        times["global"] = perf_counter() - start
    return paths, trace, times, diagnostics
