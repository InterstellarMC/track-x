"""Paired, event-checkpointed Phase IV experiments on official TrackML simulation."""

import argparse
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from phase3 import evaluate, seeds

from trackx.core import wrap
from trackx.extension import Barrel, trajectory
from trackx.official import digest, discover, geometry, load
from trackx.phase2 import save
from trackx.tracking import EdgeNet

OUT = Path("reports/phase4")
SPLITS = {
    "gap-a": range(1012, 1017),
    "gap-b": range(1017, 1022),
    "development": range(1003, 1006),
    "validation": [1002],
    "final": range(1022, 1027),
}


def inputs(stage):
    expected = [f"event{i:09}" for i in SPLITS[stage]]
    original = json.loads(Path("reports/phase2/manifest.json").read_text())
    if stage in ("development", "validation"):
        return Path("data/trackml"), original, expected
    root = Path("data") / ("trackml4-" + stage)
    parts = discover(root)
    if (
        sorted(parts) != expected
        or digest(root / "detectors.csv") != original["geometry"]["sha256"]
    ):
        raise ValueError("input sample differs from protocol")
    doc = dict(
        original,
        split={stage: expected},
        events={
            e: {
                k: {"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
                for k, p in files.items()
            }
            for e, files in parts.items()
        },
    )
    dest = OUT / (stage + "-manifest.json")
    if dest.exists() and json.loads(dest.read_text()) != doc:
        raise ValueError("input hashes changed")
    released = Path("reports/phase4") / dest.name
    if released.is_file() and json.loads(released.read_text()) != doc:
        raise ValueError("input hashes differ from released cohort")
    save(dest, doc)
    return root, doc, expected


def quality(ev, b, paths, seed):
    ids = ev.truth.particle_id.to_numpy()
    weights = ev.truth.weight.to_numpy()
    theta = np.arctan2(ev.truth.tpy, ev.truth.tpx).to_numpy()
    tx = ev.truth[["tx", "ty"]].to_numpy() / 1000
    counts = Counter()
    fragments = Counter()
    length = {}
    errors = []
    residuals = []
    assigned = np.concatenate(paths)
    if len(assigned) != len(np.unique(assigned)):
        raise ValueError("conflicting output ownership")
    for p, s in zip(paths, seed, strict=True):
        y = ids[p]
        signals, n = np.unique(y[y > 0], return_counts=True)
        major = signals[np.argmax(n)] if len(n) else -1
        counts["severe"] += int((n >= 2).sum() >= 2)
        counts["single_background"] += int(len(n) == 1 and (y == 0).sum() == 1)
        counts["multiple_wrong"] += int((y != major).sum() >= 2)
        fragments.update(signals[n >= 2].tolist())
        tail = y[len(s) :]
        if len(tail):
            counts["transition_switch"] += int(
                y[len(s) - 1] != tail[0] and y[len(s) - 1] > 0 and tail[0] > 0
            )
        bucket = length.setdefault(len(p), [0, 0.0])
        bucket[0] += 1
        bucket[1] += max(n, default=0) / len(p)
    for n in sorted(length):
        ix = np.array([p for p in paths if len(p) == n])
        xyz = b.xyz[ix]
        mask = np.ones(ix.shape, bool)
        half, phi, slope, z0 = trajectory(xyz, mask)
        radii = b.r[ix]
        bend = np.arcsin(np.clip(half[:, None] * radii, -0.999, 0.999))
        tang = wrap(b.phi[ix] - phi[:, None] - bend) * radii
        dz = xyz[:, :, 2] - z0[:, None] - slope[:, None] * radii / np.sinc(bend / np.pi)
        residuals.extend(np.sqrt(np.mean(tang**2 + dz**2, axis=1)).tolist())
        a, c = ix[:, 0], ix[:, -1]
        chord = np.linalg.norm(tx[c] - tx[a], axis=1)
        ok = (ids[a] == ids[c]) & (ids[a] > 0) & (chord > 1e-6)
        errors.extend(
            (
                2 * half[ok]
                - 2 * np.sin(wrap(theta[c[ok]] - theta[a[ok]]) / 2) / chord[ok]
            ).tolist()
        )
    from trackx.experiment import matching

    eligible, q, _ = matching(paths, ids)
    particle = ev.particles.set_index("particle_id")
    idx = np.array(sorted(eligible))
    pt = np.hypot(particle.px.loc[idx], particle.py.loc[idx]).to_numpy()
    matched = np.isin(idx, [p for p, *_ in q])
    bins = np.digitize(pt, [0.5, 1, 2, 5])
    return dict(
        **{k: v / len(paths) for k, v in counts.items()},
        assigned_weight=weights[assigned].sum() / weights.sum(),
        fragmented=sum(fragments[p] >= 2 for p in eligible) / len(eligible),
        lengths=length,
        pt_bins=[
            [int((bins == i).sum()), int(matched[bins == i].sum())] for i in range(5)
        ],
        residual_m=np.quantile(residuals, [0.1, 0.5, 0.9]).tolist(),
        curvature_error_per_m=np.quantile(errors, [0.1, 0.5, 0.9]).tolist(),
    )


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=list(SPLITS))
    p.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help="Use an empty directory for an independent replay",
    )
    args = p.parse_args()
    OUT = args.output
    OUT.mkdir(exist_ok=True, parents=True)
    stage = args.stage
    torch.set_num_threads(1)
    root, doc, events = inputs(stage)
    modules = geometry(root / "detectors.csv")
    gap = stage.startswith("gap")
    modes = ["M0", "M1"] if gap else ["M0", "M1", "M2", "M3", "M4", "M5"]
    if stage == "development":
        modes += ["tight", "wide_R", "fixed", "standard", "middle"]
    if stage == "final":
        modes += ["M6"]
    for eid in events:
        dest = OUT / (eid + ".json")
        if dest.exists():
            continue
        tic = time.perf_counter()
        ev = load(root, doc, eid)
        loading = time.perf_counter() - tic
        tic = time.perf_counter()
        b = Barrel(ev.hits, modules)
        index_s = time.perf_counter() - tic
        cpu = time.process_time()
        seed, gs, ss = seeds(ev.hits, b.xyz)[1]
        seed_cpu = time.process_time() - cpu
        rows = {}
        for mode in modes:
            chosen = mode
            if mode == "M6":
                chosen = json.loads(
                    (Path("reports/phase4/selection.json")).read_text()
                )["backbone"]
                meta = json.loads(Path("reports/phase2/training.json").read_text())
                net = EdgeNet(2, 32)
                net.load_state_dict(
                    torch.load(
                        Path("experiments")
                        / ("official-" + meta["config_id"])
                        / "local_gnn.pt"
                    )
                )
                weight_hash = hashlib.sha256(
                    b"".join(t.numpy().tobytes() for t in net.state_dict().values())
                ).hexdigest()
                if weight_hash != meta["model_sha256"]["local_gnn"]:
                    raise ValueError(
                        "GNN checkpoint differs from frozen Phase II weights"
                    )
                cpu = time.process_time()
                seed, gs, ss = seeds(ev.hits, b.xyz, net.eval())[1]
                seed_cpu = time.process_time() - cpu
            t = time.perf_counter()
            cpu = time.process_time()
            diagnostics = {}
            if chosen in ("M0", "M1", "standard", "middle"):
                paths, traces, times = b.extend(
                    seed,
                    gaps=0 if chosen == "M1" else 2,
                    refine=chosen != "standard",
                    outer=chosen != "middle",
                )
            else:
                from trackx.precision import reconstruct

                paths, traces, times, diagnostics = reconstruct(b, seed, chosen)
            wall = time.perf_counter() - t + index_s + gs + ss
            rec_cpu = time.process_time() - cpu + seed_cpu
            t = time.perf_counter()
            row = evaluate(ev, paths, seed, traces)
            row.update(
                quality(ev, b, paths, seed),
                wall_s=wall,
                cpu_seed_extension_s=rec_cpu,
                graph_s=gs,
                seed_s=ss,
                times=times,
                diagnostics=diagnostics,
            )
            row["evaluation_s"] = time.perf_counter() - t
            rows[mode] = row
        save(
            dest,
            {
                "event": eid,
                "stage": stage,
                "hits": len(ev.hits),
                "loading_s": loading,
                "index_s": index_s,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                "rows": rows,
            },
        )
        print(eid, {k: round(v["score"], 5) for k, v in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
