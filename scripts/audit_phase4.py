"""Check all deterministic Phase III confirmation results against an isolated replay."""

import json
import sys
from pathlib import Path
from statistics import mean

root = Path(sys.argv[1])
stored = Path("reports/phase3")
timings = {"graph_s", "seed_s", "extension_s", "total_s", "timing"}
rows = []
for event in range(1009, 1012):
    filename = f"event{event:09}.json"
    old, new = [json.loads((p / filename).read_text()) for p in (stored, root)]
    assert old["supported_weight"] == new["supported_weight"]
    assert old["methods"].keys() == new["methods"].keys()
    for key, a in old["methods"].items():
        b = new["methods"][key]
        assert {k: v for k, v in a.items() if k not in timings} == {
            k: v for k, v in b.items() if k not in timings
        }, (event, key)
    rows.append(new)
for key in ("baseline", "extended", "gnn_extended", "no_gaps"):
    print(
        key,
        {
            k: mean(r["methods"][key][k] for r in rows)
            for k in ("score", "efficiency", "mixed", "total_s")
        },
    )
