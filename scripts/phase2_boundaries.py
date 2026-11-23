"""Test a second, half-sector-shifted pass and resolve candidate ownership."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from trackx.barrel import directional_graph
from trackx.core import wrap
from trackx.experiment import matching
from trackx.official import load, official_score, sector_indices
from trackx.phase2 import save
from trackx.triplets import hypotheses


def resolve(candidates, xyz):
    """Prefer longer paths, then smaller local extrapolation residual."""
    unique = {tuple(p) for p in candidates}

    def rank(path):
        pts = xyz[list(path)]
        r = np.hypot(pts[:, 0], pts[:, 1])
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        k = np.diff(r)[1:] / np.diff(r)[:-1]
        dp = wrap(np.diff(phi))
        dz = np.diff(pts[:, 2])
        cost = np.hypot(
            wrap(dp[1:] - k * dp[:-1]) / 0.010, (dz[1:] - k * dz[:-1]) / 0.003
        )
        return -len(path), np.mean(cost), path

    used = set()
    paths = []
    for path in sorted(unique, key=rank):
        if used.isdisjoint(path):
            used.update(path)
            paths.append(list(path))
    return paths


def main():
    torch.set_num_threads(1)
    out = Path("reports/phase2")
    doc = json.loads((out / "manifest.json").read_text())
    rows = []
    for eid in doc["split"]["validation"] + doc["split"]["test"]:
        ev = load(Path("data/trackml"), doc, eid)
        xyz = ev.hits[["x", "y", "z"]].to_numpy() / 1000
        ids = ev.truth.particle_id.to_numpy()
        scoped = np.where(ev.hits.volume_id == 8, ids, 0)
        candidates = []
        start = time.perf_counter()
        for passes, offset in enumerate((0, np.pi / 32), 1):
            for ix, h in sector_indices(ev.hits, 32, offset):
                g = directional_graph(h)
                candidates.extend(ix[p].tolist() for p in hypotheses(h, g[1].numpy()))
            paths = resolve(candidates, xyz)
            eligible, q, dups = matching(paths, scoped)
            rows.append(
                {
                    "event": eid,
                    "passes": passes,
                    "tracks": len(paths),
                    "efficiency": len(q) / len(eligible),
                    "fake": (len(paths) - len(q) - dups) / len(paths),
                    "duplicates": dups / len(paths),
                    "purity": float(np.mean([a for _, _, a, _ in q])),
                    "completeness": float(np.mean([b for _, _, _, b in q])),
                    "official_score": official_score(ev, paths),
                    "elapsed_s": time.perf_counter() - start,
                }
            )
    save(out / "boundaries.json", rows)


if __name__ == "__main__":
    main()
