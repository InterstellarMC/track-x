"""Matched seed ranking microbenchmark, with exact whole-event output checks."""

from time import perf_counter

import numpy as np
from omega import OUT
from phase2_boundaries import resolve
from phase4 import inputs

from trackx.barrel import directional_graph
from trackx.official import load, sector_indices
from trackx.phase2 import save
from trackx.seed_order import resolve as vector
from trackx.triplets import hypotheses


def main():
    root, doc, events = inputs("development")
    report = []
    for eid in events:
        ev = load(root, doc, eid)
        xyz = ev.hits[["x", "y", "z"]].to_numpy() / 1000
        pool = []
        for angle in (0.0, np.pi / 32):
            for ix, h in sector_indices(ev.hits, 32, angle):
                pool.extend(
                    ix[p].tolist()
                    for p in hypotheses(h, directional_graph(h)[1].numpy())
                )
        expected = resolve(pool, xyz)
        assert expected == vector(pool, xyz)
        times = [[], []]
        for rep in range(4):
            for i in [0, 1] if rep % 2 == 0 else [1, 0]:
                t = perf_counter()
                result = (resolve, vector)[i](pool, xyz)
                times[i].append(perf_counter() - t)
                assert result == expected
        report.append(
            {
                "event": eid,
                "candidates": len(pool),
                "selected": len(expected),
                "reference_s": times[0],
                "vector_s": times[1],
            }
        )
    save(OUT / "ranking-profile.json", report)


if __name__ == "__main__":
    main()
