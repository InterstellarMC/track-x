"""Stress fixed weights and whole-path assembly on validation, then held-out events."""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from trackx.barrel import directional_graph
from trackx.experiment import matching, predict
from trackx.official import load, official_score, sector_indices
from trackx.phase2 import fraction, save
from trackx.tracking import EdgeNet, assemble
from trackx.triplets import hypotheses


def summarize(ev, paths, scoped):
    eligible, q, dup = matching(paths, scoped)
    return {
        "eligible": len(eligible),
        "matched": len(q),
        "tracks": len(paths),
        "efficiency": fraction(len(q), len(eligible)),
        "fake": fraction(len(paths) - len(q) - dup, len(paths)),
        "official_score": official_score(ev, paths),
    }


def run(ev, eid, model, sectors, drop, curvature, mode):
    ids = ev.truth.particle_id.to_numpy()
    scoped = np.where(ev.hits.volume_id == 8, ids, 0)
    paths = []
    stats = []
    rng = np.random.default_rng([1729, int(eid[5:])])
    # Fixed keep mask indexed by event hit, independent of truth and sector count.
    keep = rng.uniform(size=len(ids)) >= drop
    close = set()
    hit_tracks = Counter()
    fragments = Counter()
    crossing = Counter()
    graph_seconds = assembly_seconds = 0.0
    for ix, h in sector_indices(ev.hits, sectors):
        h = h[keep[ix]]
        ix = ix[keep[ix]]
        start = time.perf_counter()
        g = directional_graph(h, curvature_scale=curvature)
        graph_seconds += time.perf_counter() - start
        s = predict("gnn", {"gnn": model}, g, "cpu")
        start = time.perf_counter()
        if mode == "edge":
            seq = assemble(h, g[1].numpy(), s, True)
        else:
            seq = hypotheses(h, g[1].numpy(), s if mode == "triplet_ml" else None)
        assembly_seconds += time.perf_counter() - start
        paths.extend([ix[p].tolist() for p in seq])
        pid = ids[ix]
        for layer in np.unique(h[:, 3]):
            ind = np.flatnonzero(h[:, 3] == layer)
            if len(ind) > 1:
                dist, j = cKDTree(h[ind, :3]).query(h[ind, :3], k=2)
                crowded = (dist[:, 1] < 0.002) & (pid[ind] != pid[ind[j[:, 1]]])
                close.update(pid[ind[crowded]].tolist())
        for p in seq:
            counts = Counter(pid[p].tolist())
            fragments.update(
                [particle for particle, n in counts.items() if particle and n >= 2]
            )
        eligible, q, dups = matching(seq, pid)
        stats.append(
            {
                "nodes": len(ix),
                "edges": g[1].shape[1],
                "noise_fraction": float(np.mean(pid == 0)),
                "tracks": len(seq),
                "sector_fakes": len(seq) - len(q) - dups,
            }
        )
        for p in np.unique(pid[pid > 0]):
            crossing[p] += 1
        hit_tracks.update(pid[pid > 0].tolist())
    eligible, q, dups = matching(paths, scoped)
    good = {p for p, *_ in q}

    def group(condition):
        group = {p for p in eligible if condition(p)}
        return {"eligible": len(group), "matched": len(group & good)}

    res = summarize(ev, paths, scoped)
    res.update(
        {
            "event": eid,
            "sectors": sectors,
            "drop": drop,
            "curvature": curvature,
            "method": mode,
            "graph_s": graph_seconds,
            "assembly_s": assembly_seconds,
            "occupancy": stats,
            "fragmented": sum(fragments[p] >= 2 for p in eligible),
            "spans_sector": group(lambda p: crossing[p] > 1),
            "crowded": group(lambda p: p in close),
            "sparse": group(lambda p: hit_tracks[p] == 3),
        }
    )
    return res


def main():
    torch.set_num_threads(1)
    out = Path("reports/phase2")
    doc = json.loads((out / "manifest.json").read_text())
    train = json.loads((out / "training.json").read_text())
    model = EdgeNet(2, 32)
    model.load_state_dict(
        torch.load(
            Path("experiments") / ("official-" + train["config_id"]) / "local_gnn.pt"
        )
    )
    model.eval()
    records = []
    for eid in doc["split"]["validation"] + doc["split"]["test"]:
        ev = load(Path("data/trackml"), doc, eid)
        for sectors, drop, bend, mode in [
            (32, 0, 1.5, "edge"),
            (32, 0, 1.5, "triplet"),
            (32, 0, 1.5, "triplet_ml"),
            (32, 0.1, 1.5, "triplet_ml"),
            (16, 0, 1.5, "triplet_ml"),
            (64, 0, 1.5, "triplet_ml"),
            (32, 0, 0.0, "triplet_ml"),
        ]:
            records.append(run(ev, eid, model, sectors, drop, bend, mode))
        save(out / "stress.json", records)
        print("processed", eid, flush=True)


if __name__ == "__main__":
    main()
