"""Apply the prespecified sector and ranking comparison on untouched events."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from phase2_boundaries import resolve

from trackx.barrel import directional_graph
from trackx.experiment import matching, predict
from trackx.official import digest, discover, load, official_score, sector_indices
from trackx.phase2 import save
from trackx.tracking import EdgeNet, fit
from trackx.triplets import hypotheses

OUT = Path("reports/phase2")
ROOT = Path("data/trackml-confirmation")


def manifest():
    expected = [f"event{n:09d}" for n in (1006, 1007, 1008)]
    groups = discover(ROOT)
    if sorted(groups) != expected:
        raise ValueError("confirmation requires exactly frozen event IDs")
    first = json.loads((OUT / "manifest.json").read_text())
    assert not set(expected) & set(first["events"])

    def entry(path):
        return {"path": path.name, "sha256": digest(path), "bytes": path.stat().st_size}

    doc = {
        "source": first["source"],
        "archive": first["archive"],
        "version": 1,
        "geometry": entry(ROOT / "detectors.csv"),
        "events": {
            e: {k: entry(p) for k, p in parts.items()} for e, parts in groups.items()
        },
        "split": {"confirmation": expected},
        "protocol_commit": "aeb2d15",
    }
    if doc["geometry"] != first["geometry"]:
        raise ValueError("different geometry release")
    path = OUT / "confirmation-manifest.json"
    if path.exists() and json.loads(path.read_text()) != doc:
        raise ValueError("confirmation manifest mismatch")
    save(path, doc)
    return doc


def summaries(ev, paths):
    ids = ev.truth.particle_id.to_numpy()
    scoped = np.where(ev.hits.volume_id == 8, ids, 0)
    eligible, q, dups = matching(paths, scoped)
    matched = {p for p, *_ in q}
    pars = ev.particles.set_index("particle_id")
    pt = np.hypot(pars.px, pars.py)
    eta = np.arcsinh(pars.pz / pt.clip(lower=1e-10))

    def bins(values, bounds):
        ids = np.array(sorted(eligible))
        strata = np.digitize(values.loc[ids], bounds) - 1
        good = np.isin(ids, list(matched))
        return [
            {
                "lower": b,
                "eligible": int((strata == i).sum()),
                "matched": int(good[strata == i].sum()),
            }
            for i, b in enumerate(bounds)
        ]

    xyz = ev.hits[["x", "y", "z"]].to_numpy() / 1000
    residuals = [np.sqrt(np.mean(fit(xyz[path], 2)[2] ** 2)) for path in paths]
    return {
        "eligible": len(eligible),
        "matched": len(q),
        "tracks": len(paths),
        "efficiency": len(q) / len(eligible),
        "fake": (len(paths) - len(q) - dups) / len(paths),
        "duplicate": dups / len(paths),
        "purity": float(np.mean([a for _, _, a, _ in q])),
        "completeness": float(np.mean([a for _, _, _, a in q])),
        "official_score": official_score(ev, paths),
        "pt": bins(pt, [0, 0.5, 1, 2, 5]),
        "abs_eta": bins(abs(eta), [0, 1, 2, 3]),
        "z_rms_m": np.quantile(residuals, [0.1, 0.5, 0.9]).tolist(),
    }


def main():
    torch.set_num_threads(1)
    doc = manifest()
    metadata = json.loads((OUT / "training.json").read_text())
    net = EdgeNet(2, 32)
    net.load_state_dict(
        torch.load(
            Path("experiments") / ("official-" + metadata["config_id"]) / "local_gnn.pt"
        )
    )
    net.eval()
    import hashlib

    model_sha = hashlib.sha256(
        b"".join(v.numpy().tobytes() for v in net.state_dict().values())
    ).hexdigest()
    if model_sha != metadata["model_sha256"]["local_gnn"]:
        raise ValueError("confirmation checkpoint mismatch")
    rows = []
    for eid in doc["split"]["confirmation"]:
        start = time.perf_counter()
        ev = load(ROOT, doc, eid)
        load_s = time.perf_counter() - start
        xyz = ev.hits[["x", "y", "z"]].to_numpy() / 1000
        for model in ("geometry", "gnn_weighted"):
            candidates = []
            stage = 0.0
            for passes, offset in enumerate((0.0, np.pi / 32), 1):
                t = time.perf_counter()
                for ix, h in sector_indices(ev.hits, 32, offset):
                    g = directional_graph(h)
                    scores = (
                        predict("gnn", {"gnn": net}, g, "cpu")
                        if model == "gnn_weighted"
                        else None
                    )
                    candidates.extend(
                        ix[p].tolist() for p in hypotheses(h, g[1].numpy(), scores)
                    )
                stage += time.perf_counter() - t
                t = time.perf_counter()
                paths = resolve(candidates, xyz)
                elapsed = stage + time.perf_counter() - t
                rows.append(
                    dict(
                        event=eid,
                        method=model,
                        passes=passes,
                        load_s=load_s,
                        reconstruction_s=elapsed,
                        full_hits=len(xyz),
                        config_id=metadata["config_id"],
                        model_sha256=model_sha if model == "gnn_weighted" else None,
                        **summaries(ev, paths),
                    )
                )
        print("confirmation", eid, flush=True)
    save(OUT / "confirmation.json", rows)


if __name__ == "__main__":
    main()
