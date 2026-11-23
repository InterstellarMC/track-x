"""Measure full-event truth weight in the supported barrel after confirmation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from phase4_report import OUT, table


def main():
    events = {}
    for p in sorted(OUT.glob("event*.json")):
        doc = json.loads(p.read_text())
        stage = doc["stage"]
        root = (
            Path("data/trackml")
            if stage in ("development", "validation")
            else Path("data") / ("trackml4-" + stage)
        )
        eid = doc["event"]
        hits = pd.read_csv(root / (eid + "-hits.csv"), usecols=["hit_id", "volume_id"])
        truth = pd.read_csv(root / (eid + "-truth.csv"), usecols=["hit_id", "weight"])
        t = hits.merge(truth, on="hit_id", validate="one_to_one")
        assert len(t) == len(hits) == len(truth)
        events[eid] = {
            str(v): float(t.weight[t.volume_id == v].sum() / t.weight.sum())
            for v in (8, 13, 17)
        }
    (OUT / "region-weights.json").write_text(json.dumps(events, indent=2) + "\n")
    final = [
        json.loads((OUT / f"event{i:09}.json").read_text()) for i in range(1022, 1027)
    ]
    ids = [d["event"] for d in final]
    w = [[e, *events[e].values(), sum(events[e].values())] for e in ids]
    perf = [
        [
            m,
            *[
                np.mean(
                    [
                        d["rows"][m]["regions"][str(v)]["matched_hit_efficiency"]
                        for d in final
                    ]
                )
                for v in (8, 13, 17)
            ],
        ]
        for m in final[0]["rows"]
    ]
    (OUT / "DETECTOR_COVERAGE.md").write_text(
        "# Detector coverage on final events\n\nFull-event truth weight within each supported barrel volume, divided by complete-event truth weight. This gives context for the score attainable by barrel-only assignments.\n\n"
        + table(["event", "volume 8", "volume 13", "volume 17", "supported total"], w)
        + "\n\nFraction of regional signal hits in correct associations on uniquely matched paths, averaged across events.\n\n"
        + table(["method", "volume 8", "volume 13", "volume 17"], perf)
        + "\n"
    )


if __name__ == "__main__":
    main()
