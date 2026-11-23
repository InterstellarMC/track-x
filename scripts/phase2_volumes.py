"""Controlled single-volume transfer and post-fit diagnostics on exploratory events."""

import json
from pathlib import Path

import numpy as np
import torch

from trackx.barrel import directional_graph
from trackx.core import wrap
from trackx.experiment import matching
from trackx.official import load, official_score, sector_indices
from trackx.phase2 import fraction, save
from trackx.tracking import fit
from trackx.triplets import hypotheses


def main():
    torch.set_num_threads(1)
    out = Path("reports/phase2")
    doc = json.loads((out / "manifest.json").read_text())
    rows = []
    for eid in doc["split"]["test"]:
        ev = load(Path("data/trackml"), doc, eid)
        xyz = ev.hits[["x", "y", "z"]].to_numpy() / 1000
        truth = ev.truth
        true_xy = truth[["tx", "ty"]].to_numpy() / 1000
        tangent = np.arctan2(truth.tpy.to_numpy(), truth.tpx.to_numpy())
        ids = truth.particle_id.to_numpy()
        for volume in (8, 13, 17):
            paths = []
            for ix, h in sector_indices(ev.hits, 32, volume=volume):
                g = directional_graph(h)
                paths.extend(ix[p].tolist() for p in hypotheses(h, g[1].numpy()))
            mask = ev.hits.volume_id == volume
            scoped = np.where(mask, ids, 0)
            eligible, q, dups = matching(paths, scoped)
            curvature_error = []
            for pid, path, *_ in q:
                a, b = path[0], path[-1]
                chord = np.linalg.norm(true_xy[b] - true_xy[a])
                if ids[a] != pid or ids[b] != pid or chord <= 1e-8:
                    continue
                ref = 2 * np.sin(0.5 * wrap(tangent[b] - tangent[a])) / chord
                k, _, _ = fit(xyz[path], 2)
                curvature_error.append(float(k - ref))
            rows.append(
                {
                    "event": eid,
                    "volume": volume,
                    "hits": int(mask.sum()),
                    "eligible": len(eligible),
                    "paths": len(paths),
                    "efficiency": fraction(len(q), len(eligible)),
                    "fake": fraction(len(paths) - len(q) - dups, len(paths)),
                    "official_score": official_score(ev, paths),
                    "curvature_comparisons": len(curvature_error),
                    "median_abs_curvature_delta_per_m": float(
                        np.median(np.abs(curvature_error))
                    )
                    if curvature_error
                    else None,
                }
            )
    save(out / "volumes.json", rows)


if __name__ == "__main__":
    main()
