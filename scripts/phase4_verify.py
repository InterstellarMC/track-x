"""Verify split integrity, input provenance and deterministic replay rows."""

import argparse
import json
from pathlib import Path

from phase4 import OUT, SPLITS

from trackx.official import digest


def verify(directory, replay=None, stages=None, receipt=None):
    seen = set()
    results = {}
    for stage, events in SPLITS.items():
        expected = {f"event{i:09}" for i in events}
        assert seen.isdisjoint(expected), stage
        seen.update(expected)
        manifest = directory / (stage + "-manifest.json")
        if stage.startswith("gap") or stage == "final":
            data = json.loads(manifest.read_text())
            assert set(data["events"]) == expected
            for parts in data["events"].values():
                for v in parts.values():
                    assert (
                        digest(Path("data") / ("trackml4-" + stage) / v["path"])
                        == v["sha256"]
                    )
        for eid in expected:
            obj = json.loads((directory / (eid + ".json")).read_text())
            assert (obj["event"], obj["stage"]) == (eid, stage)
            if replay and (not stages or stage in stages):
                other = json.loads((replay / (eid + ".json")).read_text())
                assert (other["event"], other["stage"], other["hits"]) == (
                    eid,
                    stage,
                    obj["hits"],
                )
                results[eid] = {
                    "stage": stage,
                    "peak_rss_bytes": other["peak_rss_bytes"],
                    "wall_s": {m: r["wall_s"] for m, r in other["rows"].items()},
                }
                assert other["rows"].keys() == obj["rows"].keys()
                for m, row in obj["rows"].items():
                    assert {
                        k: v
                        for k, v in row.items()
                        if k != "times" and not k.endswith("_s")
                    } == {
                        k: v
                        for k, v in other["rows"][m].items()
                        if k != "times" and not k.endswith("_s")
                    }, (eid, m)
    if receipt and results:
        receipt.write_text(
            json.dumps(
                {
                    "comparison": "Exact equality for every released method field except timing",
                    "replays": dict(sorted(results.items())),
                },
                indent=2,
            )
            + "\n"
        )
    if replay and not list(replay.glob("event*.json")):
        raise ValueError("empty replay")
    print(
        f"{len(seen)} distinct event records verified; {len(results)} event replays matched all scientific fields"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path)
    parser.add_argument(
        "--stage",
        action="append",
        choices=SPLITS,
        help="Verify a partial replay for this complete stage",
    )
    parser.add_argument(
        "--receipt", type=Path, help="Save measured replay times and matched event IDs"
    )
    args = parser.parse_args()
    verify(OUT, args.replay, args.stage, args.receipt)
