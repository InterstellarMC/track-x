"""Post-reconstruction diagnoses; overlapping categories are counted independently."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from phase3 import OUT

from trackx.official import load
from trackx.phase2 import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["development", "confirmation"])
    stage = parser.parse_args().stage
    confirmation = stage == "confirmation"
    root = Path("data/trackml-phase3" if confirmation else "data/trackml")
    manifest = (
        OUT / "manifest.json" if confirmation else Path("reports/phase2/manifest.json")
    )
    doc = json.loads(manifest.read_text())
    report = []
    for path in sorted(OUT.glob("event*.json")):
        result = json.loads(path.read_text())
        if result["stage"] != stage:
            continue
        eid = result["event"]
        ev = load(root, doc, eid)
        rec = json.loads((Path("experiments/phase3") / (eid + ".json")).read_text())
        pid = ev.truth.particle_id.to_numpy()
        h = ev.hits
        xyz = h[["x", "y", "z"]].to_numpy() / 1000
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        volume = h.volume_id.to_numpy()
        keys = h.layer_id.to_numpy() + volume * 100
        particles = ev.particles.set_index("particle_id")
        pt = np.hypot(particles.px, particles.py)
        groups = h[volume != 8].groupby(pid[volume != 8]).indices
        outer = np.flatnonzero(volume != 8)
        count = Counter()
        pairs = Counter()
        for seed, p in zip(rec["seed"], rec["paths"], strict=True):
            ids, n = np.unique(pid[seed], return_counts=True)
            who = ids[n.argmax()]
            if who == 0 or n.max() <= len(seed) / 2:
                count["no_signal_seed_majority"] += 1
                continue
            related = outer[groups.get(who, [])]
            target = related[np.isin(volume[related], [13, 17])]
            if not len(target):
                continue
            tail = np.array(p[len(seed) :], int)
            correct = tail[pid[tail] == who]
            count["extendible_majority_seeds"] += 1
            missed = len(correct) == 0
            count["no_correct_extension"] += missed
            count["incorrect_assignment_seed"] += bool(np.any(pid[tail] != who))
            count["missing_nominal_surface"] += len(np.unique(keys[target])) < 6
            count["missed_and_missing_surface"] += (
                missed and len(np.unique(keys[target])) < 6
            )
            count["seeds_pt_below_0.5"] += pt.loc[who] < 0.5
            count["missed_pt_below_0.5"] += missed and pt.loc[who] < 0.5
            count["pixel_seed_spans_original_seam"] += (
                len(
                    np.unique(((phi[seed] + np.pi) * 32 / (2 * np.pi)).astype(int) % 32)
                )
                > 1
            )
            for v in (13, 17):
                if np.any(volume[target] == v):
                    pairs[f"available_{v}"] += 1
                    pairs[f"missed_{v}"] += not np.any(volume[correct] == v)
        report.append(
            {
                "event": eid,
                "seed_counts": {k: int(v) for k, v in count.items()},
                "volume_opportunities": {k: int(v) for k, v in pairs.items()},
            }
        )
    save(OUT / (stage + "-failures.json"), report)


if __name__ == "__main__":
    main()
