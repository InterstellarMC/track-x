"""Re-evaluate frozen checkpoints without writing Phase I benchmark artifacts."""

import json
from pathlib import Path

import torch

from trackx.experiment import evaluate


def main():
    torch.set_num_threads(1)
    report = []
    for path in sorted(Path("reports").glob("*-17??.json")):
        old = json.loads(path.read_text())
        new = evaluate(old["config"], "cpu", old["training_config"])
        keys = set(old["records"][0]) - {"preprocess_ms", "latency_ms", "events_per_s"}
        equal = (
            old["test_sha256"] == new["test_sha256"]
            and len(old["records"]) == len(new["records"])
            and all(
                a[k] == b[k]
                for a, b in zip(old["records"], new["records"])
                for k in keys
            )
        )
        report.append(
            {"sample": path.stem, "rows": len(new["records"]), "identical": equal}
        )
    out = Path("reports/phase2")
    out.mkdir(exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(report, indent=2))
    if len(report) != 9 or not all(r["identical"] for r in report):
        raise RuntimeError("baseline mismatch")


if __name__ == "__main__":
    main()
