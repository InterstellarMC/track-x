"""Bounded official TrackML reconstruction and paired evaluation."""

import argparse
import hashlib
import json
import platform
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve

from .barrel import directional_graph
from .core import graph, labels
from .experiment import atomic_save, matching, models, predict
from .official import digest, freeze, load, official_score, sector_indices
from .tracking import EdgeNet, assemble, fit

BUILDERS = {"original": graph, "directional": directional_graph}


def save(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def train(root, doc, c, work):
    """Train per-sector, keeping the last epoch and best validation checkpoint."""
    history = {}
    for name, depth, width, embedding in [
        ("local_mlp", 0, 32, False),
        ("local_small", 2, 16, False),
        ("local_gnn", c["depth"], c["width"], False),
        ("local_layer", c["depth"], c["width"], True),
    ]:
        torch.manual_seed(c["seed"])
        net = LayerNet(depth, width) if embedding else EdgeNet(depth, width)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
        last = work / (name + "-last.pt")
        losses = []
        if last.exists():
            state = torch.load(last)
            losses = state["losses"]
            net.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
        else:
            # Fixed physical input scales inherited from the graph, standardized
            # with train-only sufficient statistics.
            totals = [[], []]
            for eid in doc["split"]["train"]:
                ev = load(root, doc, eid)
                for _, h in sector_indices(ev.hits, c["sectors"]):
                    x, _, a = directional_graph(h)
                    for stats, t in zip(totals, (x, a)):
                        stats.append(
                            (len(t), t.double().sum(0), t.double().square().sum(0))
                        )
            for offset, stats in zip((0, 5), totals):
                n, s, ss = map(sum, zip(*stats))
                mean = s / n
                sel = slice(offset, offset + len(mean))
                net.mean[sel] = mean
                net.scale[sel] = (ss / n - mean.square()).clamp_min(1e-6).sqrt()
        for epoch in range(len(losses), c["epochs"]):
            for stage in ("train", "validation"):
                values = []
                net.train(stage == "train")
                for eid in doc["split"][stage]:
                    ev = load(root, doc, eid)
                    for ix, h in sector_indices(ev.hits, c["sectors"]):
                        g = directional_graph(h)
                        y = labels(ev.truth.particle_id.to_numpy()[ix], g[1])
                        if not len(y):
                            continue
                        with torch.set_grad_enabled(stage == "train"):
                            out = net(*g)
                            weight = (
                                ((1 - y).sum() / y.sum().clamp_min(1)).clamp(0.25, 20)
                                if stage == "train"
                                else None
                            )
                            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                out, y, pos_weight=weight
                            )
                            if stage == "train":
                                optimizer.zero_grad()
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                                optimizer.step()
                        values.append(loss.item())
            val = float(np.mean(values))
            if not losses or val < min(losses):
                atomic_save(net.state_dict(), work / (name + ".pt"))
            losses.append(val)
            atomic_save(
                {
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "losses": losses,
                },
                last,
            )
        history[name] = losses
    return history


class LayerNet(EdgeNet):
    """A four-category pixel-layer embedding added to the node encoder."""

    def __init__(self, depth, width):
        super().__init__(depth, width)
        self.layer_embedding = torch.nn.Embedding(4, 5)
        torch.nn.init.zeros_(self.layer_embedding.weight)

    def forward(self, x, edge, attr):
        layer = (7 * x[:, 4]).round().long()
        return super().forward(x + self.layer_embedding(layer), edge, attr)


def fraction(a, b):
    return float(a / b) if b else None


def truth_pairs(ev):
    """All same-particle outward pairs across one or two pixel barrel layers."""
    df = ev.hits.loc[ev.hits.volume_id == 8, ["layer_id"]].copy()
    df["pid"] = ev.truth.particle_id
    counts = (
        df[df.pid > 0]
        .groupby(["pid", "layer_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[2, 4, 6, 8], fill_value=0)
    )
    arr = counts.to_numpy()
    return counts.index.to_numpy(), (arr[:, :-1] * arr[:, 1:]).sum(1) + (
        arr[:, :-2] * arr[:, 2:]
    ).sum(1)


def bench(ev, c, nets):
    rows = []
    diagnostics = []
    particle_ids = ev.truth.particle_id.to_numpy()
    pts = ev.particles.set_index("particle_id")
    pt = np.hypot(pts.px, pts.py)
    eta = np.arcsinh(pts.pz / pt.clip(lower=1e-10))
    bins = c["pt_bins"]
    truth_pid, nt = truth_pairs(ev)
    xyz = ev.hits[["x", "y", "z"]].to_numpy() / 1000
    barrel = ev.hits.volume_id.to_numpy() == 8
    scoped = np.where(barrel, particle_ids, 0)
    for builder, build in BUILDERS.items():
        modes = [
            ("classical", "classical", True, 0.5),
            ("mlp", "mlp", False, 0.5),
            ("gnn", "gnn", False, 0.5),
            ("hybrid", "gnn", True, 0.5),
        ]
        if builder == "directional":
            modes += [
                (m, m, False, 0.5)
                for m in ("local_mlp", "local_small", "local_gnn", "local_layer")
            ]
            modes += [
                ("local_hybrid", "local_gnn", True, 0.5),
                ("local_hybrid_high", "local_gnn", True, 0.7),
            ]
        modes = [
            (mode, model, gate, c["threshold"] if threshold == 0.5 else threshold)
            for mode, model, gate, threshold in modes
        ]
        paths = defaultdict(list)
        timing = defaultdict(Counter)
        score_lists = defaultdict(list)
        positives = []
        edge_count = 0
        kept = Counter()
        transitions = Counter()
        sector_stats = []
        for ix, h in sector_indices(ev.hits, c["sectors"]):
            start = time.perf_counter()
            g = build(h)
            build_sec = time.perf_counter() - start
            edge = g[1].numpy()
            y = labels(particle_ids[ix], g[1]).numpy().astype(bool)
            positives.append(y)
            kept.update(particle_ids[ix][edge[0, y]].tolist())
            sector_stats.append(
                {
                    "hits": len(h),
                    "edges": len(y),
                    "true_edges": int(y.sum()),
                    "seconds": build_sec,
                }
            )
            edge_count += len(y)
            for a, b in zip(h[edge[0], 3], h[edge[1], 3]):
                transitions[f"{int(a)}-{int(b)}"] += 1
            for mode, model, gate, threshold in modes:
                start = time.perf_counter()
                score = predict(model, nets, g, "cpu")
                infer = time.perf_counter() - start
                start = time.perf_counter()
                selected = assemble(
                    h, edge, np.where(score >= threshold, score, 0.0), gate
                )
                assembly = time.perf_counter() - start
                paths[mode].extend(ix[p].tolist() for p in selected)
                score_lists[mode].append(score)
                timing[mode].update(
                    {"graph": build_sec, "inference": infer, "assembly": assembly}
                )
        y = np.concatenate(positives)
        retained = np.array([kept[p] for p in truth_pid])
        bucket = np.digitize(pt.loc[truth_pid], bins) - 1
        diagnostics.append(
            {
                "builder": builder,
                "edges": edge_count,
                "false_candidates": int(edge_count - y.sum()),
                "edges_per_hit": edge_count / int(barrel.sum()),
                "directed_density": edge_count
                / (int(barrel.sum()) * (int(barrel.sum()) - 1)),
                "truth_pairs": int(nt.sum()),
                "recall": fraction(retained.sum(), nt.sum()),
                "by_pt": [
                    {
                        "lower": b,
                        "retained": int(retained[bucket == k].sum()),
                        "total": int(nt[bucket == k].sum()),
                    }
                    for k, b in enumerate(bins)
                ],
                "transitions": dict(transitions),
                "sectors": sector_stats,
            }
        )
        for mode, model, gate, threshold in modes:
            seq = paths[mode]
            start = time.perf_counter()
            rms = [float(np.sqrt(np.mean(fit(xyz[p], 2)[2] ** 2))) for p in seq]
            timing[mode]["fit"] = time.perf_counter() - start
            eligible, q, dups = matching(seq, scoped)
            full, full_q, _ = matching(seq, particle_ids)
            scores = np.concatenate(score_lists[mode])
            pred = scores >= threshold
            tp = int((y & pred).sum())
            precision, recall, _ = precision_recall_curve(y, scores)
            samples = np.linspace(
                0, len(precision) - 1, min(80, len(precision)), dtype=int
            )
            matched = {p for p, *_ in q}

            def strata(prop, edges, eligible=eligible, matched=matched):
                a = np.array(sorted(eligible), dtype=np.int64)
                k = np.digitize(prop.loc[a], edges) - 1
                found = np.isin(a, list(matched))
                return [
                    {
                        "lower": float(b),
                        "eligible": int((k == i).sum()),
                        "matched": int(found[k == i].sum()),
                    }
                    for i, b in enumerate(edges)
                ]

            purity = [p for _, _, p, _ in q]
            completeness = [p for _, _, _, p in q]
            official = official_score(ev, seq)
            rows.append(
                {
                    "builder": builder,
                    "method": mode,
                    "official_full_event": official,
                    "barrel_efficiency": fraction(len(q), len(eligible)),
                    "full_efficiency": fraction(len(full_q), len(full)),
                    "fake_rate": fraction(len(seq) - len(q) - dups, len(seq)),
                    "duplicates": fraction(dups, len(seq)),
                    "matched": len(q),
                    "eligible": len(eligible),
                    "tracks": len(seq),
                    "purity": float(np.mean(purity)) if q else None,
                    "completeness": float(np.mean(completeness)) if q else None,
                    "edge_ap": float(average_precision_score(y, scores)),
                    "edge_precision": fraction(tp, pred.sum()),
                    "edge_recall": fraction(tp, y.sum()),
                    "edge_f1": fraction(2 * tp, pred.sum() + y.sum()),
                    "time_s": dict(timing[mode]),
                    "pt": strata(pt, bins),
                    "abs_eta": strata(abs(eta), [0, 1, 2, 3]),
                    "z_rms_quantiles_m": np.quantile(rms, [0.1, 0.5, 0.9]).tolist()
                    if rms
                    else [],
                    "pr": np.column_stack(
                        (recall[samples], precision[samples])
                    ).tolist(),
                }
            )
    return rows, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/trackml"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase2"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase2.yaml"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    doc = freeze(args.root, args.output / "manifest.json")
    reference = Path(__file__).resolve().parents[2] / "reports/phase2/manifest.json"
    if not reference.is_file() or doc != json.loads(reference.read_text()):
        raise ValueError("official benchmark requires the released sample hashes")
    c = yaml.safe_load(args.config.read_text())
    token = hashlib.sha256(
        (digest(args.config) + digest(args.output / "manifest.json")).encode()
    ).hexdigest()[:16]
    work = Path("experiments") / ("official-" + token)
    work.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    history = train(args.root, doc, c, work)
    nets = models(yaml.safe_load(Path(c["source_model"]).read_text()), "cpu")
    for name in history:
        net = (
            LayerNet(c["depth"], c["width"])
            if name == "local_layer"
            else EdgeNet(
                0
                if name == "local_mlp"
                else (2 if name == "local_small" else c["depth"]),
                16
                if name == "local_small"
                else (32 if name == "local_mlp" else c["width"]),
            )
        )
        net.load_state_dict(torch.load(work / (name + ".pt")))
        nets[name] = net.eval()
    hashes = {
        n: hashlib.sha256(
            b"".join(t.cpu().numpy().tobytes() for t in net.state_dict().values())
        ).hexdigest()
        for n, net in nets.items()
    }
    save(
        args.output / "training.json",
        {
            "validation_bce": history,
            "elapsed_s": time.perf_counter() - start,
            "model_sha256": hashes,
            "config": c,
            "config_id": token,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
        },
    )
    for eid in doc["split"]["test"]:
        start = time.perf_counter()
        ev = load(args.root, doc, eid)
        loading = time.perf_counter() - start
        rows, diagnostics = bench(ev, c, nets)
        save(
            args.output / (eid + ".json"),
            {
                "event": eid,
                "config_id": token,
                "official_dataset": True,
                "full_hits": len(ev.hits),
                "cells": ev.cells,
                "particles": len(ev.particles),
                "barrel_hits": int((ev.hits.volume_id == 8).sum()),
                "loading_s": loading,
                "experiment_wall_s": time.perf_counter() - start,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024),
                "rows": rows,
                "graphs": diagnostics,
            },
        )
        print("evaluated", eid, flush=True)


if __name__ == "__main__":
    main()
