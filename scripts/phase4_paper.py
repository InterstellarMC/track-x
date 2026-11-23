"""Render the Phase IV scientific summary only from a complete confirmation run."""

import json
from pathlib import Path

import numpy as np
from phase4_report import OUT, table


def main():
    models = {
        "M0": "Phase III, two gaps",
        "M1": "Phase III, no gaps",
        "M2": "Robust refit, statistical gate",
        "M3": "Local EKF",
        "M4": "Geometry, global prefixes",
        "M5": "EKF, global prefixes",
        "M6": "M3 with GNN seed rank",
    }
    docs = [
        json.loads((OUT / f"event{i:09}.json").read_text()) for i in range(1022, 1027)
    ]
    pairs = json.loads((OUT / "paired.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["backbone"] == "M3"

    def difference(stage, comp, key):
        mean, _, _, low, high = pairs[stage][comp][key]
        return f"{mean:+.6f} (paired 95% interval {low:+.6f} to {high:+.6f})"

    def avg(m, k):
        return float(np.mean([d["rows"][m][k] for d in docs]))

    atlas = json.loads((OUT / "validation-atlas.json").read_text())
    counts = atlas["counts"]
    peak = max(d["peak_rss_bytes"] for d in docs) / 2**20
    comparison = table(
        [
            "method",
            "official score",
            "efficiency",
            "mixed",
            "severe",
            "fake",
            "wall s/event",
        ],
        [
            [
                models[m],
                *[
                    avg(m, k)
                    for k in [
                        "score",
                        "efficiency",
                        "mixed",
                        "severe",
                        "fake",
                        "wall_s",
                    ]
                ],
            ]
            for m in models
        ],
    )
    results = f"""# TRACK-X Phase IV · precision tracking on official TrackML simulation

## Abstract

A no-gap rule reduced contamination in a ten-event prospective comparison. A prompt radial extended Kalman filter (M3) then improved full-event score over the frozen Phase III geometry tracker (M0) on a separate five-event cohort. Their mean paired score difference was {difference("final", "M3 minus M0", "score")}. The study concerns simulated events and ten outward barrel surfaces. Endcaps, displaced seeding and detector-calibrated momentum remain outside its scope.

## Methods

Two half-sector-shifted passes produce pixel seeds. M0 allows two missed outer layers and M1 terminates after the first miss. M2 refits prompt geometry and propagates an approximate covariance to select hits. M3 carries a four-parameter EKF posterior across surfaces. M4 and M5 rank competing complete extensions and prefixes under exclusive ownership. M6 uses the frozen Phase II GNN to rank pixel seeds for the validation-selected M3. Seed formation, spatial indexing, complete-event scoring and matching conventions follow Phase III. [State model](KALMAN_MODEL.md) · [Association](HIT_ASSOCIATION.md) · [Ambiguity objective](GLOBAL_AMBIGUITY.md)

## Independent comparisons

Ten fresh events (1012–1021) tested M1 against M0. The mean change in mixed fraction was {difference("gap", "M1 minus M0", "mixed")}. The change in official score was {difference("gap", "M1 minus M0", "score")}. The latter interval includes zero. The experiment set no noninferiority margin. [No-gap event data](NO_GAP_CONFIRMATION.md)

Three historical events supplied development ablations. On validation event 1002, M3 scored {selection["validation_scores"]["M3"]:.6f}, the highest of M0–M5. After freezing the methods, we acquired simulated events 1022–1026. Each row below averages those same five events.

{comparison}

M3 minus M1 gave a score change of {difference("final", "M3 minus M1", "score")}. M3 minus M0 changed broad mixing by {difference("final", "M3 minus M0", "mixed")}, severe mixing by {difference("final", "M3 minus M0", "severe")}, efficiency by {difference("final", "M3 minus M0", "efficiency")} and wall seconds by {difference("final", "M3 minus M0", "wall_s")}. M5 minus M3 changed score by {difference("final", "M5 minus M3", "score")}. GNN seed weighting changed score by {difference("final", "M6 minus M3", "score")}. Full tables also give event-level distributions, fake and duplicate fractions, purity, completeness, timing and coverage. [Frozen event comparisons](CONFIRMATION_RESULTS.md) · [Development ablations](ABLATION_STUDY.md) · [Figures](measurements.html)

## Failure analysis and computation

In validation event 1002, {counts["seed already mixed"]} M3 candidates already contained two signal identities in their seed, and {counts["new mixing in extension"]} first acquired a second identity during extension. Only {counts["severe merge"]} of {atlas["total_paths"]} paths contained two hits apiece from two signal particles. Diagnosis uses truth only after inference. The validation atlas contains deterministically selected examples, and frequency estimates use every path. Narrow innovations on correctly attributed hits and curvature pulls with heavy concentration near zero limit uncertainty interpretation. [Covariance study](COVARIANCE_VALIDATION.md) · [Contamination categories](FALSE_MERGE_ANALYSIS.md)

Peak process RSS across the final cohort was {peak:.1f} MiB on the Apple M1. Timings include real spatial indexing and both seeding passes. Whole-path proposal ranking adds work and the GNN adds pixel inference. The event benchmark reuses its common seed computation across matched methods. [Stage measurements](PERFORMANCE.md)

## Interpretation and next experiment

M3 scored higher than M0 in all five events. The local EKF is the prespecified validation choice. The paired score and purity results should be weighed against runtime, fewer assigned hits and the behavior of the severe-merge endpoint. Five consecutive events from a single simulation archive cannot establish generalization across pileup settings or detector conditions. The bootstrap treats events as exchangeable and gives no protection against dataset shift. The alternative arms are descriptive multiple comparisons. The selected truth curvature diagnostic is an endpoint proxy on pure paths; the measurement model has no material map. Whole-path selection tests a bounded greedy prefix pool. Splitting, backward smoothing, exact set packing and a matched ACTS comparison were not evaluated.

The next experiment should target mixed pixel seeds before extension. Prespecify a competitive seed replacement or outlier-rejection rule, preserve the local EKF and compare against it on a disjoint simulation shard with sufficient independent events for rare severe merges. Evaluate truth-free seed quality, coverage, official score and time alongside contamination. [Protocol](EXPERIMENT_PROTOCOL.md) · [Reproduction](REPRODUCIBILITY.md)
"""
    (OUT / "FINAL_RESEARCH_REPORT.md").write_text(results)
    paper = Path("reports/TRACK_X.md")
    marker = "## Phase IV · independent precision-tracking experiments"
    paper.write_text(
        paper.read_text().split(marker)[0].rstrip()
        + "\n\n"
        + marker
        + "\n\n"
        + "At barrel radius r, the filter carries (φ,z,h,t), with signed half-curvature h and longitudinal slope t. An outward prompt circle advances azimuth by Δasin(hr) and z by tΔ[asin(hr)/h]. Its Jacobian propagates covariance. A wrapped two-coordinate innovation ranks at most eight hits per surface. A Joseph update incorporates the accepted hit. The global alternative selects disjoint complete extensions or prefixes by their summed hit compatibility. The covariance scales are effective tracking tolerances. Validation found conservative curvature pulls on pure tracks. [Model and Jacobian](phase4/KALMAN_MODEL.md) · [Uncertainty validation](phase4/COVARIANCE_VALIDATION.md)\n\n"
        + results.split("## Independent comparisons\n\n")[1]
        .split("## Failure analysis")[0]
        .replace("](NO_GAP", "](phase4/NO_GAP")
        .replace("](CONFIRMATION", "](phase4/CONFIRMATION")
        .replace("](ABLATION", "](phase4/ABLATION")
        .replace("](measurements", "](phase4/measurements")
        .rstrip()
        + "\n\n[Phase IV model, failure analysis and limitations](phase4/FINAL_RESEARCH_REPORT.md) · [Reproduction instructions](phase4/REPRODUCIBILITY.md)\n"
    )


if __name__ == "__main__":
    main()
