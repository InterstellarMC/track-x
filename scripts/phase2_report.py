"""Generate Phase II tables, figures and scientific summaries from saved records."""

import functools
import json
import operator
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

OUT = Path("reports/phase2")


def read(name):
    return json.loads((OUT / name).read_text())


def table(rows):
    df = pd.DataFrame(rows)

    def cell(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.5f}" if np.isfinite(x) else "undefined"
        return str(x)

    return "\n".join(
        [
            "| " + " | ".join(df.columns) + " |",
            "|" + "---|" * len(df.columns),
            *[
                "| " + " | ".join(map(cell, row)) + " |"
                for row in df.itertuples(index=False, name=None)
            ],
        ]
    )


def write(filename, title, sections):
    (OUT / (filename + ".md")).write_text(
        "# " + title + "\n\n" + "\n\n".join(sections) + "\n"
    )


def main():
    manifest = read("manifest.json")
    test = manifest["split"]["test"]
    events = [read(e + ".json") for e in test]
    df = pd.DataFrame([dict(event=e["event"], **r) for e in events for r in e["rows"]])
    metrics = [
        "official_full_event",
        "barrel_efficiency",
        "full_efficiency",
        "fake_rate",
        "edge_ap",
    ]
    mean = df.groupby(["builder", "method"], sort=False)[metrics].mean().reset_index()
    reasons = (
        pd.DataFrame(
            [
                dict(builder=r["builder"], **r["counts"])
                for r in read("candidate-losses.json")
            ]
        )
        .groupby("builder")
        .sum()
        .reset_index()
    )
    both = pd.DataFrame(read("boundaries.json"))
    boundary = (
        both[both.event.isin(test)]
        .groupby("passes")[
            [
                "official_score",
                "efficiency",
                "fake",
                "duplicates",
                "purity",
                "completeness",
            ]
        ]
        .mean()
        .reset_index()
    )
    graph = pd.DataFrame(
        [dict(event=e["event"], **g) for e in events for g in e["graphs"]]
    )
    gs = (
        graph.groupby("builder")[
            [
                "truth_pairs",
                "edges",
                "false_candidates",
                "recall",
                "edges_per_hit",
                "directed_density",
            ]
        ]
        .mean()
        .reset_index()
    )
    stress = pd.DataFrame(read("stress.json"))
    stress = stress[stress.event.isin(test)]
    ab = (
        stress.groupby(["method", "sectors", "drop", "curvature"], sort=False)[
            ["official_score", "efficiency", "fake", "graph_s", "assembly_s"]
        ]
        .mean()
        .reset_index()
    )
    std = (
        stress.groupby(["method", "sectors", "drop", "curvature"], sort=False)
        .efficiency.std()
        .to_numpy()
    )
    ab["efficiency_event_SD"] = std
    geom = read("geometry.json")
    validation = pd.DataFrame(geom["events"])
    write(
        "TRACKML_DATA_VALIDATION",
        "Official sample validation",
        [
            "The authenticated acquisition uses range reads of six complete event prefixes in sorted order. CRC checks on extracted members and SHA-256 identities freeze inputs. Every event passed finite numeric, unique key, hit/truth alignment, particle reference, detector module and cell association checks. Particle IDs retain integer precision. The official source describes simulated silicon tracking measurements. [Organizers](https://sites.google.com/site/trackmlparticle/dataset) · [Frozen manifest](manifest.json)",
            table(validation),
            "Input sizes, full split membership and hashes are in manifest.json. The cells table is streamed in chunks; it is validated but supplies no inference features.",
        ],
    )
    write(
        "DETECTOR_GEOMETRY",
        "Detector geometry and scope",
        [
            f"The official geometry table contains {geom['modules']} modules. The reconstruction covers all azimuths in volume 8, divided into disjoint sectors. Module centres describe sensor placement; the minimum and maximum centre radii below do not describe complete sensor outlines.",
            table(pd.DataFrame(geom["layers"]).query("volume_id==8")),
            "TrackML has overlapping modules, unequal radial steps and disks at different z. A true particle may produce multiple hits on the same nominal layer. The Phase I generator permits at most one hit per layer. Radial ordering alone is unsuitable for endcaps. Geometry and occupancy for every volume are in [the detector figure](geometry.html) and [machine-readable inventory](geometry.json).",
        ],
    )
    write(
        "CANDIDATE_GRAPH_STUDY",
        "Paired candidate study",
        [
            "Denominator includes every same, nonzero particle pair on volume-8 layers one or two transitions apart across the entire barrel, including pairs separated by sector boundaries. Multiple module hits contribute multiple possible pairs. This is an available association graph, not a unique chronological trajectory. Pairs on a curling return can be counted even when outward tracking is inappropriate. No momentum, truth or labels enter either builder.",
            "The directional search takes at most eight target hits per layer in scaled (cos φ, sin φ, z/r). Its azimuth scale increases with measured layer separation; its longitudinal tolerance reflects beamline vertex spread. Both builders share partitions and graph features. Directed density divides by N(N−1) across barrel hits, including cross-sector pairs.",
            table(gs),
            table(reasons),
            "Reason counts pool all held-out eligible pairs. A boundary failure takes precedence over geometric rejection. neighbor_cap counts in-window, in-sector outward pairs excluded by candidate ranking. More detailed occupancy bins are in candidate-losses.json.",
            "Averaging is per held-out event. Transition counts and sector occupancy are stored per event. Remaining losses include sector seams, finite neighbor budget, projective and curvature windows, and radial-order failures. The separate stress study varies the curvature allowance and sector count.",
        ],
    )
    write(
        "REAL_DATA_BENCHMARKS",
        "Official TrackML simulation benchmarks",
        [
            table(mean),
            "All rows are equal-weight means across three complete held-out events. official_full_event calls the pinned organizers’ score_event on every hit. Unused hits share track ID zero, making one explicit background candidate. Only a strict majority in both candidate and full truth particle earns its correctly assigned weight, normalized by event weight. Full-event efficiency uses every truth particle with at least three hits. The barrel overlap metrics use eligible volume-8 particles with at least three barrel hits. Their rule requires >50% purity and ≥50% completeness. Each truth particle can be claimed once. Extra qualifying candidates are duplicates; remaining unmatched candidates are fakes. Their denominators are all reconstructed candidates. Undefined event ratios are omitted from means.",
            "Edge AP, precision, recall and F1 refer to candidate links only. PR uses sigmoid scores or geometric ranking. Scores are uncalibrated; conditioning and matching rules differ across endpoints. See the event JSON for individual measurements, PR samples, fit residual quantiles and denominator counts. Whole-path reconstruction results follow.",
            table(ab.query("sectors==32 and drop==0 and curvature==1.5")),
        ],
    )
    write(
        "MODEL_ABLATIONS",
        "Matched ablations",
        [
            table(mean.query('builder=="directional"')),
            "Within the directional graph, local_mlp removes message updates, local_small uses 16 hidden units, local_gnn uses 32, and local_layer adds a learned four-layer embedding. All train on identical events with the same seed, optimizer and epoch budget. Minimum unweighted validation BCE selects each checkpoint. Weighted training scores at threshold 0.5 can produce poor path efficiency. local_hybrid_high raises the gate to 0.7; a missing fake rate means there were no candidates. The embedding and width comparisons also change parameter count. A detector-volume embedding cannot vary within this single-volume reconstruction.",
            table(ab),
            "triplet uses whole-path geometric ranking. triplet_ml breaks geometric ambiguities with an added −0.2 log product of edge scores. Both prefer compatible four-hit hypotheses to triples before exclusive assignment. The edge method shares the same local GNN weights, but uses thresholded greedy links. Curvature is the coefficient of the radial-gap term in the search window, not fitted truth curvature. The missing-hit study removes observations before reconstruction and scores against the original unthinned truth.",
        ],
    )
    times = pd.DataFrame(
        [
            dict(event=r.event, builder=r.builder, method=r.method, **r.time_s)
            for r in df.itertuples()
        ]
    )
    selected = (
        times.query('method in ["classical","local_gnn","hybrid"]')
        .groupby(["builder", "method"])[["graph", "inference", "assembly", "fit"]]
        .mean()
        .reset_index()
    )
    prof = read("profile.json")
    write(
        "PERFORMANCE",
        "M1 performance",
        [
            "Stage times are seconds per event aggregated across sectors on one PyTorch CPU thread. The benchmark executes alternative methods sequentially. Its complete wall time includes validation, model comparisons, scoring and diagnostics. Graph time is shared across methods and must not be summed over rows. Phase II run timings describe local load; device microbenchmarks warm up separately.",
            table(selected),
            table(
                [
                    {
                        "event": e["event"],
                        "hits": e["full_hits"],
                        "loading_and_validation_s": e["loading_s"],
                        "all_method_wall_s": e["experiment_wall_s"],
                        "process_peak_MB": e["peak_rss_bytes"] / 1e6,
                    }
                    for e in events
                ]
            ),
            f"On {prof['fits']} matched training-event hit sequences, moving coordinate extraction outside the fit loop changed elapsed time from {prof['pandas_s']:.4f} to {prof['array_s']:.4f} seconds. Every longitudinal residual was identical. The training pilot measured {prof['pilot']['graph']['seconds']:.3f} s with the original builder plus classical assembly and {prof['pilot']['directional_graph']['seconds']:.3f} s with directional candidates. That comparison changes the graph and therefore the work performed.",
            table(read("devices.json")),
            "Independent device runs include the event loader in process peak RSS. Median score latency includes transfers and synchronized execution for a fixed validation sector. [Full run wall time and OS maximum RSS](process-time.txt)",
        ],
    )
    base = stress.query(
        'method=="triplet_ml" and sectors==32 and drop==0 and curvature==1.5'
    )
    fail = []
    for r in base.itertuples():
        for k in ("spans_sector", "sparse", "crowded"):
            v = getattr(r, k)
            fail.append(
                dict(
                    event=r.event,
                    group=k,
                    **v,
                    efficiency=v["matched"] / max(1, v["eligible"]),
                )
            )
    write(
        "FAILURE_ANALYSIS",
        "Failure modes",
        [
            table(fail),
            "Sparse denotes exactly three retained barrel hits. Crowded means at least one hit whose nearest same-layer sector neighbor is from a different truth association within 2 mm. Boundary particles occupy more than one sector. Groups overlap. No truth is used to create candidates or resolve ownership. Figures and per-sector fake counts expose occupancy dependence. Zero denominator is undefined.",
            "Four layers limit completeness under the official full-detector score. Fixed angular sectors split viable tracks. The 16/32/64 sector sweep measures the cost of boundaries together with changes in candidate competition. Exclusivity can suppress duplicates while assigning a hit to the wrong candidate. Fragment counts in stress.json count truth particles represented by two or more hits on multiple output paths. They differ from duplicate counts under the majority matching rule.",
        ],
    )
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[
            "Efficiency by true pT",
            "Candidate recall by pT",
            "Candidate precision–recall",
            "Fake rate by sector occupancy",
            "Score and event size",
            "Graph construction by sector size",
            "Peak RSS by graph size",
            "Longitudinal fit residuals",
            "False edges and candidate retention",
        ],
    )
    for method in ["classical", "gnn", "local_gnn", "local_hybrid"]:
        sub = df.query('builder=="directional" and method==@method')
        b = (
            pd.DataFrame(functools.reduce(operator.iadd, sub.pt.tolist(), []))
            .groupby("lower")
            .sum()
        )
        fig.add_scatter(
            x=b.index,
            y=b.matched / b.eligible,
            mode="lines+markers",
            name=method,
            row=1,
            col=1,
        )
        r = sub.iloc[0]
        pr = np.array(r.pr)
        fig.add_scatter(
            x=pr[:, 0], y=pr[:, 1], name=method, showlegend=False, row=1, col=3
        )
        fig.add_scatter(
            x=[e["full_hits"] for e in events],
            y=sub.official_full_event,
            mode="markers",
            name=method,
            showlegend=False,
            row=2,
            col=2,
        )
        fig.add_scatter(
            x=[0.1, 0.5, 0.9],
            y=np.mean(np.array(sub.z_rms_quantiles_m.tolist()), axis=0) * 1000,
            showlegend=False,
            name=method,
            row=3,
            col=2,
        )
    for name, g in graph.groupby("builder"):
        pts = (
            pd.DataFrame(functools.reduce(operator.iadd, g.by_pt.tolist(), []))
            .groupby("lower")
            .sum()
        )
        fig.add_scatter(
            x=pts.index, y=pts.retained / pts.total, name=name, row=1, col=2
        )
        ss = pd.DataFrame(functools.reduce(operator.iadd, g.sectors.tolist(), []))
        fig.add_scatter(
            x=ss.hits,
            y=ss.seconds * 1000,
            mode="markers",
            name=name,
            showlegend=False,
            row=2,
            col=3,
        )
        fig.add_scatter(
            x=g.false_candidates,
            y=g.recall,
            mode="markers",
            name=name,
            showlegend=False,
            row=3,
            col=3,
        )
    ss = pd.DataFrame(functools.reduce(operator.iadd, base.occupancy.tolist(), []))
    fig.add_scatter(
        x=ss.nodes,
        y=ss.sector_fakes / ss.tracks,
        mode="markers",
        name="Triplet + GNN",
        row=2,
        col=1,
    )
    for dev, g in pd.DataFrame(read("devices.json")).groupby("device"):
        fig.add_scatter(
            x=g.edges, y=g.peak_rss_bytes / 1e6, mode="markers", name=dev, row=3, col=1
        )
    for k, (x, y) in enumerate(
        [
            ("pT lower edge (GeV/c)", "Efficiency"),
            ("pT lower edge (GeV/c)", "Recall"),
            ("Recall, first test event", "Precision"),
            ("Sector hits", "Sector fake fraction"),
            ("Full event hits", "Official score"),
            ("Sector hits", "Graph time (ms)"),
            ("Sector edges", "Process peak MB"),
            ("Quantile", "z RMS (mm)"),
            ("False candidate edges", "Recall"),
        ]
    ):
        fig.update_xaxes(title_text=x, row=1 + k // 3, col=1 + k % 3)
        fig.update_yaxes(title_text=y, row=1 + k // 3, col=1 + k % 3)
    fig.update_layout(
        height=1000, title="Official TrackML simulation · pixel barrel reconstruction"
    )
    fig.write_html(OUT / "measurements.html", include_plotlyjs="../plotly.min.js")
    main_ab = ab.query("sectors==32 and drop==0 and curvature==1.5")
    write(
        "FINAL_RESEARCH_REPORT",
        "TRACK-X: Physics-Constrained Graph Neural Networks for Charged-Particle Track Reconstruction",
        [
            "## Abstract\n\nWe tested graph construction, edge classification and ambiguity resolution on six official TrackML simulated events. Two events trained the compact networks, one selected checkpoints, and three supplied the paired evaluation. All tracked hits belong to the four-layer pixel barrel. A geometry-only whole-path ranker performed similarly to its GNN-weighted counterpart. The experiment gives stronger evidence for a change in candidate generation and assembly than for a learned advantage.",
            "## Physics, data and methods\n\nPrompt charged trajectories in a uniform axial field bend transversely while advancing longitudinally with arc length. Locally, azimuth and z can be extrapolated in radius. We score triplets by extrapolation residuals scaled by 0.010 rad and 3 mm. A norm below three retains a hypothesis. These are engineering tolerances; they are not a detector covariance or a calibrated chi-square. Ranking before hit ownership resolves conflicts that greedy edge selection makes irreversible. Indexed sparse neighborhoods and bounded out-degree limit hypothesis enumeration. [Detector geometry](DETECTOR_GEOMETRY.md) · [Input provenance](TRACKML_DATA_VALIDATION.md)",
            "## Results on official simulation\n\n"
            + table(gs[["builder", "recall", "edges", "false_candidates"]]),
            table(
                main_ab[
                    [
                        "method",
                        "official_score",
                        "efficiency",
                        "efficiency_event_SD",
                        "fake",
                    ]
                ]
            ),
            "## Sector boundary follow-up\n\nA second tiling is shifted by half a sector. Geometry-only paths from both passes compete globally by length, then triplet residual, then deterministic hit index. Both rows use the same final ownership resolver. This follow-up used the original test events after diagnosing boundary losses; the result is exploratory.",
            table(boundary),
            "The official score is small because full truth and weights include outer layers and disks that the reconstruction leaves unassigned. Custom barrel efficiency also remains far below one. These measures answer different questions. The table gives event means and between-event sample SD for efficiency. Three events from the start of one archive and one model seed cannot establish general-population precision. The path-assembly follow-up was motivated by initial transfer results and is exploratory on this test split. An independent untouched sample is needed for confirmation.",
            "## Synthetic comparison\n\nThe original synthetic results remain separate. The audit reproduced every stored non-timing metric for nine Phase I experiments. Detector overlap, multiple hits per layer, high occupancy and abrupt sector boundaries expose failure modes absent from the ideal eight-cylinder model. [Phase I audit](BASELINE_AUDIT.md) · [Synthetic results](../RESULTS.md)",
            "## Related work and comparison limits\n\nThe TrackML accuracy challenge supplies weighted truth associations and an official scoring implementation. Graph-based seeding and labeling motivated the learned comparison. ACTS uses geometry-aware propagation, measurement selection, seeding, combinatorial Kalman branching, fitting and ambiguity resolution. TRACK-X tests local compatibility and shared-hit resolution with fixed tolerances. A matched ACTS execution would require the same digitization, source-link geometry, seeding, field, uncertainties and selection denominators. No such setup was executed. A useful first upstream discussion would concern a small seam/overlap regression fixture with known geometry and licensing. [TrackML paper](https://arxiv.org/abs/1904.06778) · [Graph seeding](https://arxiv.org/abs/2007.00149) · [ACTS tracking](https://acts.readthedocs.io/en/latest/tracking.html) · [Official scorer](https://github.com/LAL/trackml-library/blob/53a165e15a2c885f54c2bef1bd1ed53db6ed9648/trackml/score.py)",
            "## Validation, limitations and next experiment\n\n[Matched baselines](REAL_DATA_BENCHMARKS.md) · [Model and physics ablations](MODEL_ABLATIONS.md) · [Failure strata](FAILURE_ANALYSIS.md) · [Measured performance](PERFORMANCE.md) · [Figures](measurements.html) · [Reproduction](REPRODUCIBILITY.md). Freeze the two-pass geometry-only baseline and the GNN-weighted alternative, then test them on new complete event IDs. Compare outer-barrel extension under the same official full-event score. Endcap and outer-barrel reconstruction, independent seed uncertainty, material transport and a matched ACTS performance comparison remain open.",
        ],
    )


if __name__ == "__main__":
    main()
