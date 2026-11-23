"""Run frozen, paired seed interventions on bounded official TrackML events."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from phase3 import evaluate, seeds
from phase4 import inputs

from trackx.experiment import matching
from trackx.extension import Barrel
from trackx.official import digest, geometry, load
from trackx.omega import alternatives, choose, residuals
from trackx.phase2 import save
from trackx.precision import reconstruct

OUT = Path("reports/omega")
CANDIDATES = ("strict", "replace", "beam", "joint")
ALL = ("geometry", "local", "global", "tight", *CANDIDATES)
CODE = (
    "scripts/omega.py",
    "src/trackx/omega.py",
    "src/trackx/precision.py",
    "src/trackx/extension.py",
    "src/trackx/triplets.py",
    "scripts/phase3.py",
    "scripts/phase2_boundaries.py",
    "src/trackx/barrel.py",
    "src/trackx/official.py",
    "src/trackx/experiment.py",
    "uv.lock",
    "reports/omega/EXPERIMENT_PROTOCOL.md",
)


def diagnostics(ev, b, original, seed, paths):
    ids = ev.truth.particle_id.to_numpy()
    vol = ev.hits.volume_id.to_numpy()
    layer = ev.hits.layer_id.to_numpy()
    eligible, q, _ = matching(seed, np.where(vol == 8, ids, 0))
    good = {tuple(p) for p in original if ids[p[0]] > 0 and (ids[p] == ids[p[0]]).all()}
    pt = ev.particles.set_index("particle_id")
    pt = np.hypot(pt.px, pt.py).to_dict()
    costs = {}
    for n in (3, 4):
        group = np.array([p for p in seed if len(p) == n])
        if len(group):
            cost, h = residuals(b.xyz, group)
            costs.update((tuple(p), (c, k)) for p, c, k in zip(group, cost, h))
    strata = defaultdict(Counter)
    first = Counter()
    mixed = severe = upstream = 0
    exact = {tuple(p) for p in seed}
    pix_counts = Counter(ids[vol == 8])
    purities = []
    completeness = []
    for s, p in zip(seed, paths, strict=True):
        si = ids[s]
        sign = Counter(int(k) for k in si if k)
        pid = max(sign, key=sign.get, default=0)
        width = len(sign)
        mixed += width > 1
        full = Counter(int(k) for k in ids[p] if k)
        upstream += len(full) > 1
        severe += sum(v >= 2 for v in full.values()) >= 2
        purities.append(sign.get(pid, 0) / len(s))
        completeness.append(sign.get(pid, 0) / max(1, pix_counts[pid]))
        if width > 1:
            lead = next(k for k in si if k)
            first[int(next(layer[j] for j in s if ids[j] and ids[j] != lead))] += 1
        c, k = costs[tuple(s)]
        seam = (
            len(set(np.floor((b.phi[s] + np.pi) * 32 / (2 * np.pi)).astype(int) % 32))
            > 1
        )
        for key, val in (
            ("length", len(s)),
            ("abs_half_curvature", np.digitize(abs(k), [0.25, 1, 3])),
            ("residual", np.digitize(c, [0.5, 1, 2])),
            ("majority_pt", np.digitize(pt[pid], [0.5, 1, 2]) if pid else "noise"),
            ("cross_sector", int(seam)),
            ("skipped_pixel_layer", int(np.any(np.diff(layer[s]) > 2))),
        ):
            strata[f"{key}/{val}"]["total"] += 1
            strata[f"{key}/{val}"]["mixed"] += int(width > 1)
    return {
        "seed_count": len(seed),
        "seed_mixed": mixed / len(seed),
        "seed_purity": np.mean(purities),
        "seed_completeness": np.mean(completeness),
        "seed_efficiency": len(q) / len(eligible),
        "retention": len(good & exact) / max(len(good), 1),
        "correct_baseline": len(good),
        "upstream_count": int(mixed),
        "mixed_paths": int(upstream),
        "severe": severe / len(paths),
        "first_change_layer": first,
        "strata": strata,
    }


def run(stage, out, config=None):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    if stage == "confirmation":
        manifest = OUT / "manifest.json"
        doc = json.loads(manifest.read_text())
        root = Path("data/omega")
        events = sorted(doc["events"])
        selected = json.loads((OUT / "selection.json").read_text())
        modes = ["local", selected["challenger"]]
    else:
        root, doc, events = inputs("development" if stage == "smoke" else stage)
        if stage == "smoke":
            events = events[:1]
        modes = list(ALL)
    config_id = hashlib.sha256(
        json.dumps({k: digest(k) for k in CODE}, sort_keys=True).encode()
    ).hexdigest()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    modules = geometry(root / "detectors.csv")
    for eid in events:
        f = out / (eid + ".json")
        meta = {
            "event": eid,
            "stage": stage,
            "configuration_hash": config_id,
            "input_hashes": {k: m["sha256"] for k, m in doc["events"][eid].items()},
            "geometry_sha256": doc["geometry"]["sha256"],
            "modes": modes,
        }
        if f.exists():
            cached = json.loads(f.read_text())
            if cached["provenance"] != meta or list(cached["rows"]) != modes:
                raise ValueError(
                    f"checkpoint mismatch at {f}; use a fresh output directory"
                )
            continue
        t = perf_counter()
        ev = load(root, doc, eid)
        load_s = perf_counter() - t
        t = perf_counter()
        b = Barrel(ev.hits, modules)
        indexing = perf_counter() - t
        t = perf_counter()
        orig, gs, ss = seeds(ev.hits, b.xyz)[1]
        common = perf_counter() - t + indexing
        layers = ev.hits.layer_id.to_numpy()
        vol = ev.hits.volume_id.to_numpy()
        t = perf_counter()
        pool = alternatives(b, layers, vol, orig)
        pool_s = perf_counter() - t
        rows = {}
        atlas = []
        for mode in modes:
            t = perf_counter()
            seed = choose(b, pool, mode) if mode in CANDIDATES else orig
            choose_s = perf_counter() - t
            t = perf_counter()
            if mode == "geometry":
                paths, trace, times = b.extend(seed)
            else:
                paths, trace, times, _ = reconstruct(
                    b, seed, {"global": "M5", "tight": "tight"}.get(mode, "M3")
                )
            ext = perf_counter() - t
            t = perf_counter()
            row = evaluate(ev, paths, seed, trace)
            evaluation = perf_counter() - t
            row.update(
                diagnostics(ev, b, orig, seed, paths),
                runtime=common + ext + choose_s + (pool_s if mode in CANDIDATES else 0),
                extension_s=ext,
                select_s=choose_s,
                evaluate_s=evaluation,
                times=times,
            )
            rows[mode] = row
            if stage == "validation" and mode in CANDIDATES:
                included = {tuple(p) for p in seed}
                for opt in pool:
                    if len(atlas) >= 20 * (1 + CANDIDATES.index(mode)):
                        break
                    if opt[0][1] not in included and len(opt) > 1:
                        atlas.append(
                            {
                                "mode": mode,
                                "options": [
                                    {
                                        "hit": list(p),
                                        "xyz": b.xyz[list(p)].tolist(),
                                        "truth": [
                                            str(t)
                                            for t in ev.truth.particle_id.to_numpy()[
                                                list(p)
                                            ]
                                        ],
                                        "residual": c,
                                        "selected": p in included,
                                    }
                                    for c, p in opt
                                ],
                            }
                        )
        weights = ev.truth.weight.to_numpy()
        save(
            f,
            {
                "provenance": meta,
                "revision": revision,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "rng_seed": 1729,
                "model_checkpoint": None,
                "hits": len(ev.hits),
                "pixel_hits": int((vol == 8).sum()),
                "load_s": load_s,
                "graph_s": gs,
                "seed_assembly_s": ss,
                "seed_common_s": common,
                "pool_s": pool_s,
                "peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (2**20 if sys.platform == "darwin" else 1024),
                "region_weight": {
                    str(k): float(weights[vol == k].sum() / weights.sum())
                    for k in np.unique(vol)
                },
                "rows": rows,
                "atlas": atlas,
            },
        )
        print(eid, {m: round(r["score"], 6) for m, r in rows.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["smoke", "development", "validation", "confirmation"]
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    run(args.stage, args.output or OUT / args.stage)
