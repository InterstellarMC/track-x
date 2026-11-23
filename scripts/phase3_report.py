"""Generate Phase III reports from event-level records."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

OUT = Path("reports/phase3")


def read(name):
    return json.loads((OUT / name).read_text())


def table(df):
    def fmt(x):
        return f"{x:.5f}" if isinstance(x, (float, np.floating)) else str(x)

    return "\n".join(
        [
            "| " + " | ".join(df.columns) + " |",
            "|" + "---|" * len(df.columns),
            *[
                "| " + " | ".join(map(fmt, r)) + " |"
                for r in df.itertuples(index=False, name=None)
            ],
        ]
    )


def doc(name, title, *items):
    (OUT / (name + ".md")).write_text("# " + title + "\n\n" + "\n\n".join(items) + "\n")


def main():
    events = [read(p.name) for p in sorted(OUT.glob("event*.json"))]
    df = pd.DataFrame(
        [
            dict(event=e["event"], stage=e["stage"], method=m, **r)
            for e in events
            for m, r in e["methods"].items()
        ]
    )
    test = df[df.stage == "confirmation"]
    if len(test) != 33:
        raise ValueError("incomplete confirmation")
    primary = test[test.method.isin(["baseline", "extended"])]
    comp = primary.pivot(index="event", columns="method", values="score")
    comp["paired_delta"] = comp.extended - comp.baseline
    means = (
        test.groupby("method")[
            [
                "score",
                "efficiency",
                "fake",
                "duplicate",
                "purity",
                "completeness",
                "mixed",
                "coverage",
                "total_s",
            ]
        ]
        .mean()
        .reset_index()
    )
    dev = (
        df[(df.stage == "development") & (df.event != "event000001002")]
        .groupby("method")[["score", "mixed", "total_s"]]
        .mean()
        .reset_index()
    )
    doc(
        "MULTILAYER_GEOMETRY",
        "Measured barrel surfaces",
        table(
            pd.DataFrame(
                read("layers.json"), columns=["volume", "layer", "median_radius_m"]
            )
        ),
        "Group the official module centres by volume and layer, then order by median radius. Require non-overlapping ranges of centre radii. The map covers outward first crossings; endcaps and transitions between modules on one nominal layer are outside the model.",
    )
    doc(
        "DETECTOR_EXTENSION",
        "Cross-volume navigation",
        "Seed hits occupy volume 8. Query volume 13 layers 2, 4, 6 and 8, then volume 17 layers 2 and 4. A missing observation on one surface can be followed by an association on a later surface. strict_bridge requires a hit beyond 0.5 m before proceeding into volume 17. Geometry is loaded once per run and each event builds one spatial index per target layer. [Geometry](MULTILAYER_GEOMETRY.md)",
    )
    doc(
        "TRACK_EXTENSION",
        "Trajectory propagation and ownership",
        "For a prompt outward helix, φ(r) = φ₀ + asin(hr) and z(r) = z₀ + tanλ r/sinc(asin(hr)/π), where h is half the signed transverse curvature in inverse metres. Regress unwrapped azimuth on radius, then use three bounded Gauss–Newton updates with tangential residual downweighting above 1 mm. Fit longitudinal slope and offset by least squares on arc length. Reject unreachable projections. The straight-track limit uses normalized sinc. simple_fit retains curvature but skips nonlinear azimuth refinement.",
        "Query eight nearest positions in (R cosφ, R sinφ, z) at nominal module radius R. Recompute residuals at each actual measured radius. Tangential and longitudinal scales are 1.8 mm + 0.025Δr and 3 mm + 0.04Δr in consistent metre units. Keep normalized distance below three. These tolerances approximate increasing extrapolation uncertainty and lack a calibrated probability interpretation. Sort competing proposals globally by residual, then break ties by seed and hit index. Claim at most one hit per seed and one seed per hit on each surface. Retain seeds with unsuccessful extensions. Two missed surfaces may precede recovery.",
    )
    doc(
        "ABLATION_RESULTS",
        "Matched algorithm comparisons",
        "Development test event means (1003–1005), excluding validation 1002.",
        table(dev),
        "Frozen confirmation event means (1009–1011). Each row reuses the same hits. gnn_extended uses learned pixel-seed ranking and the same geometric outer extension. The score difference measures only that learned intervention. No trained outer-edge model was evaluated.",
        table(means),
    )
    doc(
        "OFFICIAL_SCORE_RESULTS",
        "Full-event scoring",
        table(comp.reset_index()),
        "The pinned organizers’ score_event receives every hit, with a single shared track ID zero for unassigned hits. The score uses strict majority in both directions and correctly assigned normalized truth weight. Custom track matching uses purity >0.5 and completeness ≥0.5 on all observed event hits, including disks. Truth particles require at least three observed hits. Fake and duplicate rates divide by output tracks. Purity and completeness average first qualifying matches. Mixed fraction counts paths containing hits of two or more nonzero particle IDs; it also counts one-hit contaminants. Coverage is owned hits / all event hits.",
        table(
            pd.DataFrame(
                [
                    {"event": e["event"], "supported_weight": e["supported_weight"]}
                    for e in events
                ]
            )
        ),
        "Supported weight sums original truth weights in volumes 8, 13 and 17 and divides by full-event truth weight. It measures available weight before reconstruction. Regional matched-hit efficiency divides correctly assigned hits from full-event matched paths by all signal hits in that region.",
        table(
            pd.DataFrame(
                [
                    dict(
                        event=r.event,
                        **{
                            v: d["matched_hit_efficiency"] for v, d in r.regions.items()
                        },
                    )
                    for r in test[test.method == "extended"].itertuples()
                ]
            )
        ),
    )
    doc(
        "CONFIRMATION_RESULTS",
        "Prospective three-event confirmation",
        table(comp.reset_index()),
        f"Mean paired official score change {comp.paired_delta.mean():.5f}; between-event sample SD {comp.paired_delta.std():.5f}. The three consecutive archive events provide limited information on population uncertainty. All deterministic methods were specified before acquisition. The fixed seed-ranking checkpoint was trained in Phase II.",
    )
    performance = pd.DataFrame(
        [
            dict(
                event=e["event"],
                load_validate_s=e["loading_s"],
                spatial_index_s=e["index_s"],
                peak_MB=e["peak_rss_bytes"] / 1e6,
                graph_s=e["methods"]["extended"]["graph_s"],
                seed_assembly_s=e["methods"]["extended"]["seed_s"],
                extension_s=e["methods"]["extended"]["extension_s"],
                **e["methods"]["extended"]["timing"],
            )
            for e in events
        ]
    )
    doc(
        "PERFORMANCE",
        "Apple M1 CPU resource measurements",
        table(performance),
        "Process high-water RSS includes earlier event work in the process. Reconstruction time includes indexing, pixel graph, pixel assembly and extension. Loading, official scoring and diagnostic work are timed separately or excluded. Timings are single-run measurements. All seed construction uses one torch CPU thread; layer queries use one worker. At most eight cross-surface proposals per active seed enter ranking on each layer.",
    )
    fails = read("confirmation-failures.json")
    doc(
        "FAILURE_ANALYSIS",
        "Tracking failure diagnostics",
        table(
            pd.DataFrame(
                [
                    dict(
                        event=r["event"],
                        **r["seed_counts"],
                        **r["volume_opportunities"],
                    )
                    for r in fails
                ]
            )
        ),
        table(
            test[test.method == "extended"][
                [
                    "event",
                    "candidate_recall",
                    "added",
                    "wrong_outer",
                    "assessable_outer",
                    "mixed",
                    "duplicate",
                ]
            ]
        ),
        "Candidate recall divides the number of seed/surface opportunities with at least one correct proposed hit by all opportunities for majority-signal seeds with a truth hit on that surface, including terminated seeds. Accepted wrong-outer counts include noise. Missed transition counts refer to extendible seeds without a correct hit in the given volume. Sparse trajectories and low-pT groups can overlap. Seam counts identify seeds crossing an original sector boundary; the one-pass comparison measures the net effect. The straight and simple_fit controls diagnose sensitivity to propagation errors. These overlapping indicators do not establish exclusive causal fractions.",
    )
    fig = go.Figure()
    for m in ("baseline", "middle_only", "extended", "gnn_extended"):
        x = test[test.method == m]
        fig.add_bar(x=x.event, y=x.score, name=m)
    fig.update_layout(
        title="TrackML complete-event score · frozen confirmation",
        yaxis_title="Official score",
        barmode="group",
    )
    fig.write_html(OUT / "confirmation.html", include_plotlyjs="../plotly.min.js")
    doc(
        "FINAL_REPORT",
        "TRACK-X Phase III | Extending pixel seeds across the barrel",
        "## Abstract",
        f"On three prospectively reserved TrackML simulated events, geometric propagation changed mean official full-event score from {comp.baseline.mean():.5f} to {comp.extended.mean():.5f}. All methods scored complete events. The extension preserves the original pixel seed and adds measurements from six outer surfaces. [Event-level comparison](CONFIRMATION_RESULTS.md) · [Figure](confirmation.html)",
        "## Reconstruction quality and cost",
        table(means[means.method.isin(["baseline", "extended", "gnn_extended"])]),
        "Hit coverage and full-event majority matching benefit from longer paths. The mixed fraction tracks contamination independently of the fake rate. A path can contain contamination and still pass majority matching. The rates above describe different failure modes.",
        "## Methods and limitations",
        "[Protocol](EXPERIMENT_PROTOCOL.md) · [Trajectory model](TRACK_EXTENSION.md) · [Ablations](ABLATION_RESULTS.md) · [Failure analysis](FAILURE_ANALYSIS.md) · [Reproducibility](REPRODUCIBILITY.md). Seeds still require ≥3 pixel layers. Forward particles, displaced vertices, curling returns, material transport and module-level covariances are incompletely modeled. A next experiment should compare endcap propagation with a frozen barrel reference, measuring mixed-hit contamination and gain in covered truth weight on fresh events. ACTS motivates surface navigation and compatible-measurement selection. No matched ACTS run was performed. [ACTS tracking](https://acts.readthedocs.io/en/latest/tracking.html) · [TrackML accuracy challenge](https://arxiv.org/abs/1904.06778)",
    )


if __name__ == "__main__":
    main()
