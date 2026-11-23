import hashlib
import json
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn.functional import binary_cross_entropy_with_logits as bce

from .cache import cached
from .core import event, graph, labels
from .tracking import EdgeNet, assemble, fit


def splits(c):
    sizes = [c[k] for k in ("train", "validation", "test")]
    if (
        min(sizes) < 1
        or sum(sizes) > min(c["max_events"], 10000)
        or 8 * c["tracks"] + c["noise"] > 4096
    ):
        raise ValueError("configured event budget exceeded")
    return dict(
        zip(
            ("train", "validation", "test"),
            np.split(np.arange(sum(sizes)), np.cumsum(sizes)[:2]),
        )
    )


def device(c):
    if c["device"] not in ("cpu", "auto", "mps"):
        raise ValueError("unknown accelerator")
    if c["device"] != "cpu" and torch.backends.mps.is_available():
        try:
            net = EdgeNet(2).to("mps")
            g = graph(event(c, 0)[0])
            net(*(t.to("mps") for t in g)).sum().backward()
            torch.mps.synchronize()
            return "mps"
        except (RuntimeError, NotImplementedError) as exc:
            print("MPS probe failed, selecting CPU", exc)
    return "cpu"


def folder(c):
    digest = hashlib.sha256(b"v2" + json.dumps(c, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    p = Path("experiments") / digest
    p.mkdir(parents=True, exist_ok=True)
    return p


def training(c, dev):
    ids = splits(c)
    path = folder(c)
    for name, depth in [("mlp", 0), ("gnn", c["depth"])]:
        torch.manual_seed(c["seed"])
        net = EdgeNet(
            depth, c["width"], c.get("node_features"), c.get("edge_features")
        ).to(dev)

        def features(h):
            return cached(
                h,
                Path("experiments/.cache"),
                c.get("cache_mb", 32) * 1024**2,
                c.get("graph"),
            )

        moments = [[], []]
        for number in ids["train"]:
            x, _, attr = features(event(c, number)[0])
            for stats, t in zip(moments, (x, attr)):
                stats.append((len(t), t.double().sum(0), t.double().square().sum(0)))
        for offset, stats in zip((0, 5), moments):
            n, total, squares = map(sum, zip(*stats))
            if not n:
                raise ValueError("empty training features")
            mean = total / n
            selection = slice(offset, offset + len(mean))
            net.mean[selection] = mean
            net.scale[selection] = (squares / n - mean.square()).clamp_min(1e-6).sqrt()
        opt = torch.optim.AdamW(net.parameters(), lr=0.002)
        best = float("inf")
        stale = 0
        last = path / (name + "-last.pt")
        start = 0
        if last.exists():
            state = torch.load(last, map_location=dev)
            net.load_state_dict(state["net"])
            opt.load_state_dict(state["opt"])
            start, best, stale = state["epoch"], state["best"], state["stale"]
        for epoch in range(start, c["epochs"]):
            if stale >= 4:
                break
            for phase in ("train", "validation"):
                train = phase == "train"
                net.train(train)
                losses = []
                for number in ids[phase]:
                    h, y, _ = event(c, number)
                    g = features(h)
                    target = labels(y, g[1]).to(dev)
                    if not len(target):
                        continue
                    with torch.set_grad_enabled(train):
                        logits = net(*(t.to(dev) for t in g))
                        weight = (
                            ((1 - target).sum() / target.sum().clamp_min(1)).clamp(
                                0.25, 20
                            )
                            if train
                            else None
                        )
                        loss = bce(logits, target, pos_weight=weight)
                        if train:
                            opt.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                            opt.step()
                    losses.append(loss.item())
                if not losses:
                    raise ValueError("no candidate edges in split")
            val = float(np.mean(losses))
            if val < best:
                best, stale = val, 0
                atomic_save(net.state_dict(), path / (name + ".pt"))
            else:
                stale += 1
            atomic_save(
                {
                    "net": net.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch + 1,
                    "best": best,
                    "stale": stale,
                },
                last,
            )
        print(name, "best validation BCE", round(best, 5))


def atomic_save(state, path):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def models(c, dev):
    result = {}
    for name, depth in [("mlp", 0), ("gnn", c["depth"])]:
        net = EdgeNet(
            depth, c["width"], c.get("node_features"), c.get("edge_features")
        ).to(dev)
        net.load_state_dict(torch.load(folder(c) / (name + ".pt"), map_location=dev))
        result[name] = net.eval()
    return result


@torch.inference_mode()
def predict(name, nets, g, dev):
    if name == "classical":
        cost = g[2][:, 0].abs() + g[2][:, 2].abs() + 0.3 * g[2][:, 3]
        return torch.exp(-0.5 * cost).numpy()
    net = nets["gnn" if name == "hybrid" else name]
    return net(*(t.to(dev) for t in g)).sigmoid().cpu().numpy()


def matching(paths, ids):
    counts = dict(zip(*np.unique(ids[ids > 0], return_counts=True)))
    eligible = {p for p, n in counts.items() if n >= 3}
    found = set()
    quality = []
    duplicate = 0
    for path in paths:
        pid, n = np.unique(ids[path], return_counts=True)
        for p, m in zip(pid, n):
            if p in eligible and m / len(path) > 0.5 and m / counts[p] >= 0.5:
                if p in found:
                    duplicate += 1
                else:
                    quality.append((p, path, m / len(path), m / counts[p]))
                found.add(p)
                break
    return eligible, quality, duplicate


def evaluate(c, dev, training_config=None):
    nets = models(training_config or c, dev)
    records = []
    ids = splits(c)
    digest = hashlib.sha256()
    for number in ids["test"]:
        h, y, pars = event(c, number)
        digest.update(h.tobytes() + y.tobytes())
        start = time.perf_counter()
        g = graph(h, c.get("graph"))
        prep = time.perf_counter() - start
        target = labels(y, g[1]).numpy().astype(bool)
        for name in ("classical", "mlp", "gnn", "hybrid"):

            def run(name=name, g=g, h=h):
                score = predict(name, nets, g, dev)
                paths = assemble(
                    h, g[1].numpy(), score, name in ("classical", "hybrid")
                )
                fits = [fit(h[p], c["field"]) for p in paths]
                return score, paths, fits

            run()
            start = time.perf_counter()
            score, paths, fits = run()
            ms = 1000 * (time.perf_counter() - start)
            eligible, q, dups = matching(paths, y)
            pred = score >= 0.5
            tp = (target & pred).sum()

            def ratio(a, b):
                return float(a / b) if b else None

            fit_map = {tuple(p): f for p, f in zip(paths, fits)}
            errors = [fit_map[tuple(path)][1] - pars[p][0] for p, path, _, _ in q]
            errors = [abs(e) for e in errors if np.isfinite(e)]
            records.append(
                {
                    "event": int(number),
                    "method": name,
                    "nodes": len(h),
                    "edges": len(score),
                    "preprocess_ms": prep * 1000,
                    "latency_ms": ms,
                    "events_per_s": 1000 / (ms + 1000 * prep),
                    "efficiency": ratio(len(q), len(eligible)),
                    "fake": ratio(len(paths) - len(q) - dups, len(paths)),
                    "duplicate": ratio(dups, len(paths)),
                    "precision": ratio(tp, pred.sum()),
                    "recall": ratio(tp, target.sum()),
                    "f1": ratio(2 * tp, pred.sum() + target.sum()),
                    "ap": float(average_precision_score(target, score))
                    if target.any()
                    else None,
                    "purity": np.mean([a for _, _, a, _ in q]) if q else None,
                    "completeness": np.mean([a for _, _, _, a in q]) if q else None,
                    "pt_mae_GeV": float(np.mean(errors)) if errors else None,
                    "z_rms_m": np.sqrt(
                        np.mean(np.concatenate([f[2] for f in fits]) ** 2)
                    )
                    if fits
                    else None,
                }
            )
    result = {
        "dataset": "synthetic",
        "device": dev,
        "config": c,
        "training_config": training_config or c,
        "split": {k: v.tolist() for k, v in ids.items()},
        "test_sha256": digest.hexdigest(),
        "test_array_bytes": sum(
            r["nodes"] for r in records if r["method"] == "classical"
        )
        * 40,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            k: version(k)
            for k in ("torch", "numpy", "scipy", "scikit-learn", "plotly", "pyyaml")
        },
        "records": records,
    }
    output = folder(c) / "metrics.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(output, "peak RSS bytes", result["peak_rss_bytes"])
    for name in nets.keys() | {"classical", "hybrid"}:
        print(
            name,
            "mean event efficiency",
            round(
                np.mean(
                    [
                        r["efficiency"]
                        for r in records
                        if r["method"] == name and r["efficiency"] is not None
                    ]
                ),
                4,
            ),
        )
    return result
