"""Run paired Phase III methods on event-disjoint official data."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from phase2_boundaries import resolve

from trackx.barrel import directional_graph
from trackx.experiment import matching, predict
from trackx.extension import Barrel, topology
from trackx.official import (
    digest,
    discover,
    geometry,
    load,
    official_score,
    sector_indices,
)
from trackx.phase2 import save
from trackx.tracking import EdgeNet
from trackx.triplets import hypotheses

OUT = Path("reports/phase3")
VARIANTS = {
    "extended": {},
    "simple_fit": {"refine": False},
    "seed_order": {"arbitrate": False},
    "middle_only": {"outer": False},
    "no_gaps": {"gaps": 0},
    "strict_bridge": {"bridge": False},
    "straight": {"straight": True},
}


def seeds(hits, xyz, net=None):
    pool = []
    results = []
    graphs = other = 0.0
    for angle in (0.0, np.pi / 32):
        for ix, h in sector_indices(hits, 32, angle):
            t = time.perf_counter()
            g = directional_graph(h)
            graphs += time.perf_counter() - t
            t = time.perf_counter()
            score = predict("gnn", {"gnn": net}, g, "cpu") if net else None
            pool.extend(ix[p].tolist() for p in hypotheses(h, g[1].numpy(), score))
            other += time.perf_counter() - t
        t = time.perf_counter()
        paths = resolve(pool, xyz)
        results.append((paths, graphs, other + time.perf_counter() - t))
    return results


def evaluate(ev, paths, seeds, traces):
    ids = ev.truth.particle_id.to_numpy()
    vol = ev.hits.volume_id.to_numpy()
    lay = ev.hits.layer_id.to_numpy()
    eligible, q, dup = matching(paths, ids)
    seed_ids = []
    for p in seeds:
        pid, n = np.unique(ids[p], return_counts=True)
        k = n.argmax()
        seed_ids.append(pid[k] if n[k] > len(p) / 2 else 0)
    seed_ids = np.array(seed_ids)
    opportunities = accepted = 0
    transitions = []
    for v, l, a, b in traces:
        target = np.unique(ids[(vol == v) & (lay == l)])
        possible = int(((seed_ids > 0) & np.isin(seed_ids, target)).sum())
        good = np.unique(a[(seed_ids[a] > 0) & (seed_ids[a] == ids[b])])
        opportunities += possible
        accepted += len(good)
        transitions.append(
            {"volume": v, "layer": l, "eligible": possible, "offered": len(good)}
        )
    mixed = wrong = added = wrong_particle = 0
    for i, p in enumerate(paths):
        signal = ids[p]
        mixed += len(np.unique(signal[signal > 0])) > 1
        tail = p[len(seeds[i]) :]
        added += len(tail)
        if seed_ids[i] > 0:
            wrong += int((ids[tail] != seed_ids[i]).sum())
            wrong_particle += len(tail)
    region = {}
    for v in (8, 13, 17):
        regional = [a for p in paths if len(a := [j for j in p if vol[j] == v]) >= 2]
        # Hit-level coverage conditional on membership in a full-event match.
        sel = (vol == v) & (ids > 0)
        correct = sum(int(((vol[p] == v) & (ids[p] == pid)).sum()) for pid, p, *_ in q)
        region[str(v)] = {
            "matched_hit_efficiency": correct / int(sel.sum()),
            "paths_with_two_hits": len(regional),
        }
    return {
        "score": official_score(ev, paths),
        "efficiency": len(q) / len(eligible),
        "fake": (len(paths) - len(q) - dup) / len(paths),
        "duplicate": dup / len(paths),
        "purity": float(np.mean([a for _, _, a, _ in q])),
        "completeness": float(np.mean([a for _, _, _, a in q])),
        "mixed": float(mixed / len(paths)),
        "coverage": sum(map(len, paths)) / len(ids),
        "added": added,
        "wrong_outer": wrong,
        "assessable_outer": wrong_particle,
        "candidate_recall": accepted / opportunities if opportunities else None,
        "transition_counts": transitions,
        "regions": region,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["development", "confirmation"])
    args = p.parse_args()
    torch.set_num_threads(1)
    root = Path("data/trackml")
    doc = json.loads(Path("reports/phase2/manifest.json").read_text())
    events = doc["split"]["validation"] + doc["split"]["test"]
    if args.stage == "confirmation":
        root = Path("data/trackml-phase3")
        groups = discover(root)
        events = [f"event{i:09}" for i in range(1009, 1012)]
        if (
            sorted(groups) != events
            or digest(root / "detectors.csv") != doc["geometry"]["sha256"]
        ):
            raise ValueError("unexpected holdout")
        doc = dict(
            doc,
            split={"confirmation": events},
            events={
                e: {
                    k: {"path": v.name, "bytes": v.stat().st_size, "sha256": digest(v)}
                    for k, v in parts.items()
                }
                for e, parts in groups.items()
            },
        )
        manifest = OUT / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != doc:
            raise ValueError("holdout hash mismatch")
        save(manifest, doc)
    modules = geometry(root / "detectors.csv")
    save(OUT / "layers.json", topology(modules))
    meta = json.loads(Path("reports/phase2/training.json").read_text())
    net = EdgeNet(2, 32)
    net.load_state_dict(
        torch.load(
            Path("experiments") / ("official-" + meta["config_id"]) / "local_gnn.pt"
        )
    )
    net.eval()
    for eid in events:
        dest = OUT / (eid + ".json")
        if dest.exists() and (Path("experiments/phase3") / (eid + ".json")).exists():
            continue
        t = time.perf_counter()
        ev = load(root, doc, eid)
        loading = time.perf_counter() - t
        t = time.perf_counter()
        barrel = Barrel(ev.hits, modules)
        indexing = time.perf_counter() - t
        mask = ev.hits.volume_id.isin([8, 13, 17])
        result = {
            "event": eid,
            "stage": args.stage,
            "loading_s": loading,
            "index_s": indexing,
            "supported_weight": float(
                ev.truth.weight[mask].sum() / ev.truth.weight.sum()
            ),
            "methods": {},
        }
        for tag, network in [("geometry", None), ("gnn", net)]:
            passes = seeds(ev.hits, barrel.xyz, network)
            for name, settings in (
                list(VARIANTS.items())
                + [("one_pass", {}), ("baseline", {}), ("original_one", {})]
                if network is None
                else [("gnn_extended", {})]
            ):
                seed, graph_s, assembly_s = passes[
                    0 if name in ("one_pass", "original_one") else 1
                ]
                t = time.perf_counter()
                if name in ("baseline", "original_one"):
                    paths, traces, timing = seed, [], {}
                else:
                    paths, traces, timing = barrel.extend(seed, **settings)
                elapsed = time.perf_counter() - t
                result["methods"][name] = dict(
                    **evaluate(ev, paths, seed, traces),
                    graph_s=graph_s,
                    seed_s=assembly_s,
                    extension_s=elapsed,
                    total_s=graph_s
                    + assembly_s
                    + elapsed
                    + (indexing if traces else 0),
                    timing=timing,
                )
                if name == "extended":
                    directory = Path("experiments/phase3")
                    directory.mkdir(exist_ok=True, parents=True)
                    save(directory / (eid + ".json"), {"seed": seed, "paths": paths})
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(dest, result)
        print(
            eid,
            result["methods"]["baseline"]["score"],
            result["methods"]["extended"]["score"],
            flush=True,
        )


if __name__ == "__main__":
    main()
