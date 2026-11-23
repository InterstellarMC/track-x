"""Create print-quality vector and 300-dpi figures from released numerical rows."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("reports/omega")
mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.hashsalt": "track-x-omega",
    }
)
m = json.loads((OUT / "selection.json").read_text())["challenger"]
docs = [
    json.loads(p.read_text())
    for p in sorted((OUT / "confirmation").glob("event*.json"))
]
assert [d["provenance"]["event"] for d in docs] == json.loads(
    (OUT / "cohort.json").read_text()
)["events"]
fig, axs = plt.subplots(1, 2, figsize=(8, 4.8), layout="constrained")
for ax, k, label, factor in zip(
    axs,
    ["score", "mixed"],
    ["Official full-event score change", "Mixed paths, change (percentage points)"],
    [1, 100],
):
    diffs = [factor * (d["rows"][m][k] - d["rows"]["local"][k]) for d in docs]
    y = np.arange(len(docs))
    ax.axvline(0, color=".55", lw=1)
    ax.hlines(y, 0, diffs, color="#96b3c2", lw=2)
    ax.scatter(diffs, y, color="#005477", s=24)
    ax.set_yticks(y, [d["provenance"]["event"][-4:] for d in docs])
    ax.invert_yaxis()
    ax.set_xlabel(label)
axs[0].set_ylabel("TrackML simulation event")
fig.suptitle("Seed replacement with outer-layer corroboration versus frozen local EKF")
for suffix in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"confirmation-pairs.{suffix}", dpi=300)
