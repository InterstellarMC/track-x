"""Generate paired event-level tables, bootstrap estimates and research figures."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT = Path("reports/phase4")
METRICS = (
    "score",
    "efficiency",
    "mixed",
    "severe",
    "fake",
    "duplicate",
    "coverage",
    "assigned_weight",
    "purity",
    "completeness",
    "fragmented",
    "wall_s",
    "cpu_seed_extension_s",
)


def summarize(a):
    a = np.asarray(a)
    draws = np.random.default_rng(1729).choice(a, (10000, len(a))).mean(axis=1)
    return [
        float(x)
        for x in (
            a.mean(),
            np.median(a),
            a.std(ddof=1),
            *np.quantile(draws, [0.025, 0.975]),
        )
    ]


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "---|" * len(headers),
            *[
                "| "
                + " | ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in r)
                + " |"
                for r in rows
            ],
        ]
    )


def main():
    docs = [json.loads(p.read_text()) for p in sorted(OUT.glob("event*.json"))]
    sets = {
        k: [d for d in docs if d["stage"].startswith(k)]
        for k in ("gap", "development", "validation", "final")
    }
    summaries = {}
    for stage, name in [
        ("gap", "NO_GAP_CONFIRMATION"),
        ("development", "ABLATION_STUDY"),
        ("final", "CONFIRMATION_RESULTS"),
    ]:
        ds = sets[stage]
        if not ds:
            continue
        expected = {
            "gap": range(1012, 1022),
            "development": range(1003, 1006),
            "final": range(1022, 1027),
        }[stage]
        if [d["event"] for d in ds] != [f"event{i:09}" for i in expected]:
            raise ValueError(
                f"Incomplete {stage} cohort; finish the event runs before reporting"
            )

        lines = [
            f"# {stage.capitalize()} event results",
            f"{len(ds)} complete official TrackML simulated events. Fractions are unitless. Runtime is reconstructed wall time per event including geometry indexing and both seed passes.",
        ]
        modes = list(ds[0]["rows"])
        lines.append(
            table(
                ["event", *modes],
                [[d["event"], *[d["rows"][m]["score"] for m in modes]] for d in ds],
            )
        )
        lines.append(
            table(
                ["method", *METRICS],
                [
                    [m, *[np.mean([d["rows"][m][k] for d in ds]) for k in METRICS]]
                    for m in modes
                ],
            )
        )
        comps = [("M1", "M0")] if stage == "gap" else [(m, "M0") for m in modes[1:]]
        if stage == "final":
            comps += [
                ("M3", "M1"),
                ("M5", "M3"),
                ("M6", json.loads((OUT / "selection.json").read_text())["backbone"]),
            ]
        summary = {}
        for m, base in comps:
            key = f"{m} minus {base}"
            summary[key] = {
                k: summarize([d["rows"][m][k] - d["rows"][base][k] for d in ds])
                for k in METRICS
            }
            lines.append(f"## {key}")
            lines.append(
                table(
                    [
                        "quantity",
                        "paired mean",
                        "median",
                        "sample SD",
                        "95% low",
                        "95% high",
                    ],
                    [[k, *v] for k, v in summary[key].items()],
                )
            )
        summaries[stage] = summary
        (OUT / (name + ".md")).write_text("\n\n".join(lines) + "\n")
    (OUT / "paired.json").write_text(json.dumps(summaries, indent=2) + "\n")
    ds = sets["final"]
    if not ds:
        return

    def avg(m, k):
        return np.mean([d["rows"][m][k] for d in ds])

    modes = list(ds[0]["rows"])
    lines = [
        "# Full-event computational measurements",
        "Single-process CPU evaluation. RSS is the process lifetime high-water mark including all evaluated methods, libraries and scoring. Timing components can vary with thermal state and other processes. CPU covers seeding and extension. Wall excludes CSV loading, scoring and diagnostic reductions.",
        table(
            ["event", "hits", "CSV validation/loading s", "outer index s", "peak MiB"],
            [
                [
                    d["event"],
                    d["hits"],
                    d["loading_s"],
                    d["index_s"],
                    d["peak_rss_bytes"] / 2**20,
                ]
                for d in ds
            ],
        ),
    ]
    lines.append(
        table(
            [
                "method",
                "wall s",
                "CPU s",
                "graph s",
                "seed assembly s",
                "evaluation s",
                "extension stage means (s)",
            ],
            [
                [
                    m,
                    *[
                        avg(m, k)
                        for k in (
                            "wall_s",
                            "cpu_seed_extension_s",
                            "graph_s",
                            "seed_s",
                            "evaluation_s",
                        )
                    ],
                    ", ".join(
                        f"{k} {np.mean([d['rows'][m]['times'][k] for d in ds]):.4f}"
                        for k in ds[0]["rows"][m]["times"]
                    ),
                ]
                for m in modes
            ],
        )
    )
    receipt = OUT / "replay.json"
    if receipt.exists():
        repeated = [
            r
            for r in json.loads(receipt.read_text())["replays"].values()
            if r["stage"] == "final"
        ]
        lines.append("## Fresh-directory replay")
        lines.append(
            table(
                ["method", "confirmation wall s/event", "replay wall s/event"],
                [
                    [m, avg(m, "wall_s"), np.mean([r["wall_s"][m] for r in repeated])]
                    for m in modes
                ],
            )
        )
        lines.append(
            f"Replay peak process RSS was {max(r['peak_rss_bytes'] for r in repeated) / 2**20:.1f} MiB. Session load varied and no exclusive-core or thermal controls were used. These two measurements show the variation in absolute timing."
        )
    (OUT / "PERFORMANCE.md").write_text("\n\n".join(lines) + "\n")
    captions = [
        "Full-event score",
        "Efficiency by truth pT",
        "Assigned signal purity by length",
        "Mixed versus occupancy",
        "Fake versus occupancy",
        "Accepted innovation quantiles",
        "Endpoint curvature error quantiles (1/m)",
        "Wall time versus hits (s)",
        "Process peak versus hits (MiB)",
    ]
    fig = make_subplots(rows=3, cols=3, subplot_titles=captions)
    for m in modes:
        rows = [d["rows"][m] for d in ds]
        bins = np.sum([r["pt_bins"] for r in rows], axis=0)
        lens = sorted({int(k) for r in rows for k in r["lengths"]})
        sums = np.array(
            [
                np.sum([r["lengths"].get(str(k), [0, 0]) for r in rows], axis=0)
                for k in lens
            ]
        )
        hits = [d["hits"] for d in ds]
        xy = [
            ([d["event"][-4:] for d in ds], [r["score"] for r in rows]),
            (["<0.5", "0.5–1", "1–2", "2–5", "≥5"], bins[:, 1] / bins[:, 0]),
            (lens, sums[:, 1] / sums[:, 0]),
            (hits, [r["mixed"] for r in rows]),
            (hits, [r["fake"] for r in rows]),
            ([], []),
            (
                [0.1, 0.5, 0.9],
                np.mean([r["curvature_error_per_m"] for r in rows], axis=0),
            ),
            (hits, [r["wall_s"] for r in rows]),
            ([], []),
        ]
        if "accepted_chi2_quantiles" in rows[0]["diagnostics"]:
            xy[5] = (
                [0.1, 0.5, 0.9],
                np.mean(
                    [r["diagnostics"]["accepted_chi2_quantiles"] for r in rows], axis=0
                ),
            )
        if m == "M0":
            xy[8] = (hits, [d["peak_rss_bytes"] / 2**20 for d in ds])
        for i, (x, y) in enumerate(xy):
            if len(x):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=m,
                        legendgroup=m,
                        showlegend=i == 0,
                        mode="markers" if i in (3, 4, 7, 8) else "lines+markers",
                    ),
                    row=i // 3 + 1,
                    col=i % 3 + 1,
                )
    fig.update_layout(
        height=950, title="TRACK-X · five held-out TrackML simulated events"
    )
    fig.write_html(
        OUT / "measurements.html", include_plotlyjs="../plotly.min.js", div_id="phase4"
    )
    cal = json.loads((OUT / "calibration.json").read_text())
    panels = [
        ("true_innovations", "True-hit innovation χ²"),
        ("curvature_error_per_m", "Signed curvature error (1/m)"),
        ("curvature_pull", "Curvature pull"),
    ]
    hist = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in panels])
    for i, (key, label) in enumerate(panels, 1):
        counts = cal[key]["histogram"]
        edges = np.array(counts["edges"])
        hist.add_trace(
            go.Bar(
                x=(edges[1:] + edges[:-1]) / 2,
                y=counts["counts"],
                width=np.diff(edges),
                name=label,
            ),
            row=1,
            col=i,
        )
        hist.update_xaxes(
            title_text=f"Below / above range: {counts['underflow']} / {counts['overflow']}",
            row=1,
            col=i,
        )
    hist.update_layout(showlegend=False, title="Validation event 1002 · counts per bin")
    hist.write_html(
        OUT / "calibration.html",
        include_plotlyjs="../plotly.min.js",
        div_id="phase4-calibration",
    )
    keys = [
        "mixed",
        "severe",
        "single_background",
        "multiple_wrong",
        "transition_switch",
        "fragmented",
        "duplicate",
    ]
    (OUT / "FALSE_MERGE_ANALYSIS.md").write_text(
        "# Contamination on final events\n\n"
        + table(["method", *keys], [[m, *[avg(m, k) for k in keys]] for m in modes])
        + "\n\nMixed requires two signal IDs. Severe requires two hits apiece from two signal IDs. Single background requires exactly one noise hit and only one signal ID. Multiple wrong counts paths with two or more hits outside their largest signal contribution. Transition switch compares the last seed and first extension hit and requires distinct nonzero IDs. These path fractions overlap. Fragmentation is the fraction of eligible particles contributing at least two hits to two or more paths. Duplicate counts repeat qualifying matches divided by all paths. Final assignments are exclusive. The measurements do not establish that splitting a mixed seed recovers either trajectory. No track-splitting operation was selected or tested.\n"
    )


if __name__ == "__main__":
    main()
