"""Apply the preregistered validation and trade-off rule."""

import json

from omega import CANDIDATES, OUT

from trackx.phase2 import save


def main():
    table = json.loads((OUT / "validation/event000001002.json").read_text())["rows"]
    base = table["local"]
    eligible = [
        m
        for m in CANDIDATES
        if table[m]["score"] >= base["score"]
        and table[m]["mixed"] <= base["mixed"]
        and table[m]["efficiency"] >= base["efficiency"] - 0.002
    ]
    challenger = max(eligible or CANDIDATES, key=lambda m: table[m]["score"])
    result = {
        "baseline": "local",
        "challenger": challenger,
        "validation_choice": challenger if eligible else "local",
        "eligible": eligible,
        "event": 1002,
    }
    target = OUT / "selection.json"
    if target.exists() and json.loads(target.read_text()) != result:
        raise ValueError("frozen selection differs")
    save(target, result)
    print(result)


if __name__ == "__main__":
    main()
