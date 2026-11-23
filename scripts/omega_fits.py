"""Post-assignment smoothing check on historical validation only."""

from time import perf_counter

import numpy as np
import torch
from omega import OUT
from phase3 import seeds
from phase4 import inputs

from trackx.core import wrap
from trackx.extension import Barrel
from trackx.official import geometry, load
from trackx.phase2 import save
from trackx.precision import reconstruct
from trackx.smoothing import smooth_group


def main():
    torch.set_num_threads(1)
    root, doc, [eid] = inputs("validation")
    ev = load(root, doc, eid)
    b = Barrel(ev.hits, geometry(root / "detectors.csv"))
    seed = seeds(ev.hits, b.xyz)[1][0]
    paths = reconstruct(b, seed, "M3")[0]
    errors = []
    ids = ev.truth.particle_id.to_numpy()
    xy = ev.truth[["tx", "ty"]].to_numpy() / 1000
    tangent = np.arctan2(ev.truth.tpy, ev.truth.tpx).to_numpy()
    elapsed = 0.0
    for depth in range(1, 7):
        ix = [i for i, p in enumerate(paths) if len(p) - len(seed[i]) == depth]
        if not ix:
            continue
        s = [seed[i] for i in ix]
        tail = np.array([paths[i][len(seed[i]) :] for i in ix])
        t = perf_counter()
        x, y, c = smooth_group(b, s, tail)
        elapsed += perf_counter() - t
        a, z = np.array([[p[0], p[-1]] for p in s]).T
        chord = np.linalg.norm(xy[z] - xy[a], axis=1)
        keep = np.array(
            [
                ids[paths[i][0]] > 0 and np.all(ids[paths[i]] == ids[paths[i][0]])
                for i in ix
            ]
        ) & (chord > 1e-7)
        ref = 2 * np.sin(wrap(tangent[z[keep]] - tangent[a[keep]]) / 2) / chord[keep]
        errors.extend(
            np.column_stack(
                (2 * x[keep, 2] - ref, 2 * y[keep, 2] - ref, 2 * np.sqrt(c[keep, 2, 2]))
            )
        )
    err = np.array(errors)
    save(
        OUT / "smoothing.json",
        {
            "event": eid,
            "count": len(err),
            "filter_median_abs_error": float(np.median(abs(err[:, 0]))),
            "smooth_median_abs_error": float(np.median(abs(err[:, 1]))),
            "smooth_one_sigma": float(np.mean(abs(err[:, 1]) < err[:, 2])),
            "fit_s": elapsed,
        },
    )


if __name__ == "__main__":
    main()
