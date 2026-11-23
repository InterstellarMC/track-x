"""Frozen scientific result verification and paired event inference."""

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("reports/omega")


def read_stage(stage, directory=None):
    folder = directory or OUT / stage
    docs = [json.loads(p.read_text()) for p in sorted(folder.glob("event*.json"))]
    expected = (
        json.loads((OUT / "cohort.json").read_text())["events"]
        if stage == "confirmation"
        else [
            f"event{i:09}"
            for i in {
                "development": [1003, 1004, 1005],
                "validation": [1002],
                "smoke": [1003],
            }[stage]
        ]
    )
    if [d["provenance"]["event"] for d in docs] != expected or any(
        d["provenance"]["stage"] != stage for d in docs
    ):
        raise ValueError("incomplete cohort or split mismatch")
    if len({d["provenance"]["configuration_hash"] for d in docs}) != 1:
        raise ValueError("mixed code/configuration revisions")
    return docs


def paired(differences):
    x = np.asarray(differences, float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(
            "paired inference requires at least two finite event differences"
        )
    boot = np.random.default_rng(1729).choice(x, (10000, len(x))).mean(1)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "sd": float(x.std(ddof=1)),
        "low": float(np.quantile(boot, 0.025)),
        "high": float(np.quantile(boot, 0.975)),
        "positive": int((x > 0).sum()),
        "negative": int((x < 0).sum()),
    }


def verify(stage, replay):
    fields = {"runtime", "extension_s", "select_s", "evaluate_s", "times"}
    for a, b in zip(read_stage(stage), read_stage(stage, replay), strict=True):
        assert a["provenance"] == b["provenance"]
        assert a["atlas"] == b["atlas"] and a["region_weight"] == b["region_weight"]
        for m in a["rows"]:
            assert {k: v for k, v in a["rows"][m].items() if k not in fields} == {
                k: v for k, v in b["rows"][m].items() if k not in fields
            }, (a["provenance"]["event"], m)
    print(f"Exact scientific replay matched for {stage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()
    verify(args.stage, args.replay)
