"""Truth-assisted diagnostics after M3 reconstruction on the validation event."""

from pathlib import Path

import numpy as np
import torch
from phase3 import seeds
from phase4 import inputs

from trackx.core import wrap
from trackx.extension import Barrel
from trackx.official import geometry, load
from trackx.phase2 import save
from trackx.precision import initialize, innovation, propagate, reconstruct, update


def main():
    torch.set_num_threads(1)
    root, doc, [eid] = inputs("validation")
    ev = load(root, doc, eid)
    b = Barrel(ev.hits, geometry(root / "detectors.csv"))
    seed = seeds(ev.hits, b.xyz)[1][0]
    paths = reconstruct(b, seed, "M3")[0]
    # All inference finishes before attribution. Evaluate seed-majority truth hits
    # on the next reachable layer, including missed continuations.
    truth = ev.truth
    ids = truth.particle_id.to_numpy()
    major = []
    for s in seed:
        pid, n = np.unique(ids[s], return_counts=True)
        major.append(pid[n.argmax()] if n.max() > len(s) / 2 else 0)
    x, p, r = initialize(b, seed)
    active = np.ones(len(seed), bool)
    chis = []
    records = []
    for step, (v, l, _, ix, _) in enumerate(b.layers):
        lookup = {}
        for hit in ix:
            if ids[hit] > 0:
                lookup.setdefault(ids[hit], []).append(hit)
        # For module overlaps use the first outward truth intersection, chosen by
        # radius and hit index, independently of prediction error or gate result.
        pairs = [
            (i, min(lookup[major[i]], key=lambda j: (b.r[j], j)))
            for i in np.flatnonzero(active)
            if major[i] in lookup
        ]
        a, h = np.array(pairs).T
        y, c, _, ok = propagate(x[a], p[a], r[a], b.r[h])
        chi, *_ = innovation(y, c, np.column_stack((b.phi[h], b.xyz[h, 2])), b.r[h], 1)
        chis.extend(chi[ok].tolist())
        records.append([v, l, int(ok.sum()), float(np.mean(chi[ok] < 9))])
        a = np.array(
            [i for i in np.flatnonzero(active) if len(paths[i]) > len(seed[i]) + step],
            int,
        )
        h = np.array([paths[i][len(seed[i]) + step] for i in a])
        active[:] = False
        active[a] = True
        y, c, _, _ = propagate(x[a], p[a], r[a], b.r[h])
        _, res, chol, noise = innovation(
            y, c, np.column_stack((b.phi[h], b.xyz[h, 2])), b.r[h], 1
        )
        x[a], p[a] = update(y, c, res, chol, noise)
        r[a] = b.r[h]
    a = np.array(
        [
            i
            for i in range(len(seed))
            if len(paths[i]) > len(seed[i])
            and len(set(ids[paths[i]])) == 1
            and ids[paths[i][0]] > 0
        ]
    )
    first = np.array([paths[i][0] for i in a])
    last = np.array([paths[i][-1] for i in a])
    point = truth[["tx", "ty"]].to_numpy() / 1000
    tangent = np.arctan2(truth.tpy, truth.tpx).to_numpy()
    chord = np.linalg.norm(point[last] - point[first], axis=1)
    ok = chord > 1e-6
    effective = 2 * np.sin(wrap(tangent[last] - tangent[first]) / 2) / chord
    errors = 2 * x[a[ok], 2] - effective[ok]
    pulls = errors / (2 * np.sqrt(p[a[ok], 2, 2]))

    def stats(z, limits):
        z = np.asarray(z)
        counts, edges = np.histogram(z, bins=50, range=limits)
        return {
            "n": len(z),
            "quantiles": np.quantile(z, [0.1, 0.5, 0.9]).tolist(),
            "histogram": {
                "edges": edges.tolist(),
                "counts": counts.tolist(),
                "underflow": int((z < edges[0]).sum()),
                "overflow": int((z > edges[-1]).sum()),
            },
        }

    save(
        Path("reports/phase4/calibration.json"),
        {
            "event": eid,
            "by_surface": records,
            "true_innovations": stats(chis, (0, 12)),
            "gate_coverage": float(np.mean(np.array(chis) < 9)),
            "curvature_error_per_m": stats(errors, (-0.05, 0.05)),
            "curvature_pull": stats(pulls, (-3, 3)),
            "within_one_sigma": float(np.mean(abs(pulls) < 1)),
        },
    )


if __name__ == "__main__":
    main()
