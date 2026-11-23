"""Profile identical fits on the first training event before choosing a data layout."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from trackx.barrel import directional_graph
from trackx.core import graph
from trackx.experiment import predict
from trackx.official import sector_indices
from trackx.phase2 import save
from trackx.tracking import assemble, fit


def main():
    root = Path("data/trackml")
    hits = pd.read_csv(root / "event000001000-hits.csv")
    truth = (
        pd.read_csv(root / "event000001000-truth.csv")
        .set_index("hit_id")
        .loc[hits.hit_id]
    )
    pilot = {}
    for builder in [graph, directional_graph]:
        t = time.perf_counter()
        paths = []
        candidates = positive = 0
        for ix, h in sector_indices(hits):
            g = builder(h)
            ids = truth.particle_id.to_numpy()[ix]
            i, j = g[1].numpy()
            candidates += len(i)
            positive += int(((ids[i] > 0) & (ids[i] == ids[j])).sum())
            paths.extend(
                ix[p]
                for p in assemble(
                    h, g[1].numpy(), predict("classical", {}, g, "cpu"), True
                )
            )
        pilot[builder.__name__] = {
            "seconds": time.perf_counter() - t,
            "edges": candidates,
            "positive": positive,
        }
    paths = paths[:500]
    t = time.perf_counter()
    original = [
        fit(hits.iloc[p][["x", "y", "z"]].to_numpy() / 1000, 2)[2] for p in paths
    ]
    old = time.perf_counter() - t
    t = time.perf_counter()
    xyz = hits[["x", "y", "z"]].to_numpy() / 1000
    revised = [fit(xyz[p], 2)[2] for p in paths]
    new = time.perf_counter() - t
    for a, b in zip(original, revised, strict=True):
        np.testing.assert_array_equal(a, b)
    save(
        Path("reports/phase2/profile.json"),
        {
            "training_event": 1000,
            "pilot": pilot,
            "fits": len(paths),
            "pandas_s": old,
            "array_s": new,
            "identical": True,
        },
    )


if __name__ == "__main__":
    import torch

    torch.set_num_threads(1)
    main()
