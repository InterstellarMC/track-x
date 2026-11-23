"""Compare scientific fields and model hashes from an independent output directory."""

import argparse
import json
from pathlib import Path


def scientific(value):
    if isinstance(value, list):
        return [scientific(x) for x in value]
    if isinstance(value, dict):
        timings = {
            "time_s",
            "seconds",
            "loading_s",
            "experiment_wall_s",
            "peak_rss_bytes",
            "elapsed_s",
        }
        return {k: scientific(v) for k, v in value.items() if k not in timings}
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", type=Path)
    args = p.parse_args()
    released = Path("reports/phase2")
    doc = json.loads((released / "manifest.json").read_text())
    checked = []
    for filename in [
        "manifest.json",
        "training.json",
        *[e + ".json" for e in doc["split"]["test"]],
    ]:
        a = json.loads((released / filename).read_text())
        b = json.loads((args.replay / filename).read_text())
        if scientific(a) != scientific(b):
            raise ValueError(f"scientific replay differs in {filename}")
        checked.append(filename)
    print(
        json.dumps(
            {"matched": checked, "scope": "frozen weights and event-level evaluation"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
