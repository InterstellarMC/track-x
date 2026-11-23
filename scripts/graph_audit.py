"""Audit acceptance before classification using held-out labels only."""

import json
from pathlib import Path

import numpy as np

from trackx.core import event, graph, labels

out = Path("reports")
records = []
for scenario in ("nominal", "stress", "pileup"):
    for seed in (1729, 1730, 1731):
        data = json.loads((out / f"{scenario}-{seed}.json").read_text())
        total = retained = 0
        for number in data["split"]["test"]:
            h, ids, _ = event(data["config"], number)
            _, e, _ = graph(h, data["config"].get("graph"))
            retained += int(labels(ids, e).sum())
            for pid in np.unique(ids[ids > 0]):
                layers = np.sort(h[ids == pid, 3])
                total += sum(
                    np.count_nonzero((layers > layer) & (layers <= layer + 2))
                    for layer in layers
                )
        records.append(
            {
                "scenario": scenario,
                "seed": seed,
                "allowed_true_pairs": int(total),
                "retained_true_pairs": retained,
                "acceptance": retained / total,
            }
        )
(out / "graph-acceptance.json").write_text(json.dumps(records, indent=2))
for scenario in ("nominal", "stress", "pileup"):
    print(
        scenario,
        np.mean([r["acceptance"] for r in records if r["scenario"] == scenario]),
    )
