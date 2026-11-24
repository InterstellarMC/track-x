"""Run the frozen seed-replacement tracker with batched final ranking on historical events."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from phase3 import seeds
from phase4 import inputs

from trackx.barrel import directional_graph
from trackx.extension import Barrel
from trackx.official import geometry, load, official_score, sector_indices
from trackx.omega import alternatives, choose
from trackx.phase2 import save
from trackx.precision import reconstruct
from trackx.seed_order import resolve
from trackx.triplets import hypotheses


def fast_seeds(hits, xyz):
    pool = []
    for angle in (0.0, np.pi / 32):
        for ix, h in sector_indices(hits, 32, angle):
            pool.extend(
                ix[p].tolist() for p in hypotheses(h, directional_graph(h)[1].numpy())
            )
    return resolve(pool, xyz)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=["development", "validation"])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    root, doc, events = inputs(args.stage)
    torch.set_num_threads(1)
    modules = geometry(root / "detectors.csv")
    results = []
    for number, eid in enumerate(events):
        ev = load(root, doc, eid)
        timing = {"frozen": [], "batched": []}
        expected = None
        for rep in range(3):
            order = (
                ["frozen", "batched"]
                if (rep + number) % 2 == 0
                else ["batched", "frozen"]
            )
            for method in order:
                t = perf_counter()
                b = Barrel(ev.hits, modules)
                raw = (
                    fast_seeds(ev.hits, b.xyz)
                    if method == "batched"
                    else seeds(ev.hits, b.xyz)[1][0]
                )
                pool = alternatives(
                    b, ev.hits.layer_id.to_numpy(), ev.hits.volume_id.to_numpy(), raw
                )
                seed = choose(b, pool, "beam")
                paths = reconstruct(b, seed, "M3")[0]
                timing[method].append(perf_counter() - t)
                if expected is None:
                    expected = (raw, seed, paths)
                elif (raw, seed, paths) != expected:
                    raise ValueError(
                        "batched reconstruction changed an ordered assignment"
                    )
        score = official_score(ev, expected[2])
        reference = json.loads(
            (Path("reports/omega") / args.stage / (eid + ".json")).read_text()
        )["rows"]["beam"]["score"]
        if score != reference:
            raise ValueError("scientific result differs from frozen benchmark")
        results.append(
            {"event": eid, "score": score, "wall_s": timing, "paths": len(expected[2])}
        )
    save(args.output, results)


if __name__ == "__main__":
    main()
