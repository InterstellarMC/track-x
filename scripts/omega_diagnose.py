"""Post-inference discrimination, repair/damage counts and isolated scorer timing."""

from collections import Counter
from time import perf_counter

import numpy as np
import torch
from omega import OUT
from phase3 import seeds
from phase4 import inputs
from sklearn.metrics import average_precision_score, roc_auc_score

from trackx.extension import Barrel
from trackx.official import geometry, load, official_score
from trackx.omega import alternatives, choose
from trackx.phase2 import save
from trackx.precision import reconstruct


def main():
    torch.set_num_threads(1)
    root, doc, [eid] = inputs("validation")
    ev = load(root, doc, eid)
    b = Barrel(ev.hits, geometry(root / "detectors.csv"))
    orig = seeds(ev.hits, b.xyz)[1][0]
    pool = alternatives(
        b, ev.hits.layer_id.to_numpy(), ev.hits.volume_id.to_numpy(), orig
    )
    revised = choose(b, pool, "beam")
    paths = reconstruct(b, revised, "M3")[0]
    t = perf_counter()
    score = official_score(ev, paths)
    scorer = perf_counter() - t
    ids = ev.truth.particle_id.to_numpy()
    mixed = lambda p: len(set(ids[p]) - {0}) > 1
    labels = [mixed(p) for p in orig]
    cost = np.array([opts[0][0] for opts in pool])
    changes = Counter()
    for src, dst, opts in zip(orig, revised, pool, strict=True):
        if src == dst:
            changes["unchanged"] += 1
        else:
            changes[f"changed_mixed_{int(mixed(src))}_to_{int(mixed(dst))}"] += 1
        if len(opts) > 1:
            changes["alternative_available"] += 1
            changes["mixed_with_alternative"] += int(mixed(src))
    save(
        OUT / "seed-detection.json",
        {
            "event": eid,
            "n": len(orig),
            "mixed_prevalence": np.mean(labels),
            "residual_auc": roc_auc_score(labels, cost),
            "residual_ap": average_precision_score(labels, cost),
            "changes": dict(changes),
            "official_score": score,
            "scoring_s": scorer,
        },
    )


if __name__ == "__main__":
    main()
