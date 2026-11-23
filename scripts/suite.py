"""Reproduce multi-seed and stress experiments plus the report table."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import yaml

from trackx.experiment import evaluate, training


def main():
    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path("configs/mac_m1.yaml").read_text())
    out = Path("reports")
    for seed in [1729, 1730, 1731]:
        c = dict(cfg, seed=seed)
        training(c, "cpu")
        for tag, changes in [
            ("nominal", {}),
            ("pileup", {"vertices": 5, "tracks": 54, "vertex_sigma": 0.045}),
            (
                "stress",
                {
                    "noise": 220,
                    "tracks": 54,
                    "vertex_sigma": 0.045,
                    "missing": 0.24,
                    "pt_min": 0.15,
                },
            ),
        ]:
            test = dict(c, **changes)
            data = evaluate(test, "cpu", c)
            data["training_config"] = c
            (out / f"{tag}-{seed}.json").write_text(json.dumps(data, indent=2))
    summarize(out)


def summarize(out):
    summaries = []
    for tag in ("nominal", "stress", "pileup"):
        for seed in (1729, 1730, 1731):
            data = json.loads((out / f"{tag}-{seed}.json").read_text())
            for method in ("classical", "mlp", "gnn", "hybrid"):
                rows = [r for r in data["records"] if r["method"] == method]
                summaries.append(
                    dict(
                        scenario=tag,
                        method=method,
                        seed=seed,
                        **{
                            k: np.mean([r[k] for r in rows if r[k] is not None])
                            for k in (
                                "efficiency",
                                "fake",
                                "duplicate",
                                "ap",
                                "f1",
                                "latency_ms",
                                "pt_mae_GeV",
                            )
                        },
                    )
                )
    lines = [
        "| Scenario | Method | Efficiency ± SD | Fake | Duplicate | Edge AP | Edge F1 | Latency ms | pT MAE GeV/c |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fig = go.Figure()
    for tag in ("nominal", "stress", "pileup"):
        vals = []
        for method in ("classical", "mlp", "gnn", "hybrid"):
            rows = [
                r for r in summaries if r["scenario"] == tag and r["method"] == method
            ]
            vals.append(np.mean([r["efficiency"] for r in rows]))
            spread = np.std([r["efficiency"] for r in rows], ddof=1)
            fields = [
                f"{np.mean([r[k] for r in rows]):.3f}"
                for k in ("fake", "duplicate", "ap", "f1", "latency_ms", "pt_mae_GeV")
            ]
            lines.append(
                "| "
                + " | ".join([tag, method, f"{vals[-1]:.3f} ± {spread:.3f}", *fields])
                + " |"
            )
        fig.add_bar(name=tag, x=["classical", "mlp", "gnn", "hybrid"], y=vals)
    fig.update_layout(
        title="Synthetic barrel reconstruction",
        yaxis_title="Mean track efficiency",
        yaxis_range=[0, 1],
    )
    fig.write_html(out / "efficiency.html", include_plotlyjs="directory")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    if parser.parse_args().report_only:
        summarize(Path("reports"))
    else:
        main()
