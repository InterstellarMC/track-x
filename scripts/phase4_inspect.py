"""Bounded validation-only path atlas and origin of mixed-signal assignments."""

from collections import Counter
from pathlib import Path

import torch
from phase3 import seeds
from phase4 import inputs

from trackx.extension import Barrel
from trackx.official import geometry, load
from trackx.phase2 import save
from trackx.precision import follow, reconstruct


def main():
    torch.set_num_threads(1)
    root, doc, [eid] = inputs("validation")
    ev = load(root, doc, eid)
    b = Barrel(ev.hits, geometry(root / "detectors.csv"))
    seed = seeds(ev.hits, b.xyz)[1][0]
    local = reconstruct(b, seed, "M3")[0]
    independent = follow(b, seed, "M5", independent=True)[0]
    resolved = reconstruct(b, seed, "M5")[0]
    truth = ev.truth.particle_id.to_numpy()
    counts = Counter()
    cases = {}
    ownership = Counter(j for p in independent for j in p)
    for i, (s, p, u, v) in enumerate(
        zip(seed, local, independent, resolved, strict=True)
    ):

        def signals(path):
            return Counter(int(x) for x in truth[path] if x)

        labels = signals(p)
        flags = {
            "seed already mixed": len(signals(s)) > 1,
            "new mixing in extension": len(labels) > 1 and len(signals(s)) <= 1,
            "severe merge": sum(n >= 2 for n in labels.values()) >= 2,
            "no signal mixing": len(labels) <= 1,
            "conflicting proposals": any(ownership[j] > 1 for j in u),
            "changed after global selection": u != v,
        }
        for key, match in flags.items():
            if not match:
                continue
            counts[key] += 1
            group = cases.setdefault(key, [])
            if len(group) >= 8:
                continue
            seen = sorted(set(p + u + v))
            group.append(
                {
                    "seed_index": i,
                    "seed": s,
                    "local": p,
                    "independent": u,
                    "resolved": v,
                    "hits": {
                        j: {
                            "xyz": b.xyz[j].tolist(),
                            "particle_id": str(truth[j]),
                            "volume": int(ev.hits.volume_id.iloc[j]),
                            "layer": int(ev.hits.layer_id.iloc[j]),
                        }
                        for j in seen
                    },
                }
            )
    save(
        Path("reports/phase4/validation-atlas.json"),
        {
            "event": eid,
            "total_paths": len(seed),
            "counts": dict(counts),
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
