"""Historical development-only measurement on disjoint forward pixel surfaces."""

from time import perf_counter

import numpy as np
import torch
from omega import OUT
from phase3 import seeds
from phase4 import inputs

from trackx.disks import disk_paths, planes
from trackx.extension import Barrel
from trackx.official import geometry, load, official_score
from trackx.phase2 import save
from trackx.precision import reconstruct


def main():
    torch.set_num_threads(1)
    root, doc, events = inputs("development")
    m = geometry(root / "detectors.csv")
    data = []
    for eid in events:
        ev = load(root, doc, eid)
        b = Barrel(ev.hits, m)
        s = seeds(ev.hits, b.xyz)[1][0]
        p = reconstruct(b, s, "M3")[0]
        t = perf_counter()
        d = disk_paths(ev.hits, m)
        dt = perf_counter() - t
        ids = ev.truth.particle_id.to_numpy()
        v = ev.hits.volume_id.to_numpy()
        region = {}
        for vol in (7, 9):
            paths = [q for q in d if v[q[0]] == vol]
            region[str(vol)] = {
                "paths": len(paths),
                "mixed": np.mean([len(set(ids[q]) - {0}) > 1 for q in paths]),
                "assigned_hits": sum(map(len, paths)),
                "available_hits": int((v == vol).sum()),
            }
        data.append(
            {
                "event": eid,
                "barrel_score": official_score(ev, p),
                "combined_score": official_score(ev, p + d),
                "disk_only_score": official_score(ev, d),
                "disk_s": dt,
                "regions": region,
            }
        )
    save(
        OUT / "disks.json",
        {"planes": {str(v): planes(m, v) for v in (7, 9)}, "development": data},
    )


if __name__ == "__main__":
    main()
