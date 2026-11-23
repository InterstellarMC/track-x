"""Build the Omega manuscript and supplementary tables from complete event records."""

import json

import numpy as np
from omega_review import OUT, paired, read_stage

KEYS = [
    "score",
    "efficiency",
    "fake",
    "duplicate",
    "mixed",
    "severe",
    "purity",
    "coverage",
    "seed_mixed",
    "seed_purity",
    "seed_completeness",
    "seed_efficiency",
    "retention",
    "runtime",
]


def table(headers, rows):
    def fmt(x):
        return f"{x:.6f}" if isinstance(x, float) else str(x)

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "---|" * len(headers),
            *["| " + " | ".join(map(fmt, r)) + " |" for r in rows],
        ]
    )


def write(name, title, *parts):
    (OUT / (name + ".md")).write_text("\n\n".join(["# " + title, *parts]) + "\n")


def mean(ds, m, k):
    return float(np.mean([d["rows"][m][k] for d in ds]))


def summary(ds, keys):
    return table(
        ["method", *keys], [[m, *[mean(ds, m, k) for k in keys]] for m in ds[0]["rows"]]
    )


def main():
    dev = read_stage("development")
    val = read_stage("validation")
    selection = json.loads((OUT / "selection.json").read_text())
    m = selection["challenger"]
    write(
        "MODEL_ABLATIONS",
        "Matched seed and extension controls",
        "Equal-event development means (three historical events).",
        summary(dev, KEYS),
        "Validation event 1002.",
        summary(val, KEYS),
        f"Frozen challenger: {m}.",
    )
    write(
        "SEED_CONTAMINATION",
        "Where signal identity changes",
        "The first change is the first outward hit with a nonzero ID different from the first signal ID. It identifies a discrepancy, not which particle owned the intended trajectory. Noise hits alone do not count as a second signal. Exact retention divides unchanged all-one-signal baseline seeds by all such originals. Seed purity is the largest signal contribution divided by seed length, averaged over all seeds. Seed completeness divides that contribution by all volume-8 hits of that particle. All-noise seeds contribute zero. Seed efficiency counts uniquely matched pixel particles with ≥3 hits, strict majority purity and ≥50% pixel completeness.",
        table(
            [
                "event",
                "pixel occupancy",
                "baseline seeds",
                "mixed at seed",
                "mixed final paths",
                "fraction already mixed",
                "first divergent layer counts",
            ],
            [
                [
                    d["provenance"]["event"],
                    d["pixel_hits"],
                    r["seed_count"],
                    r["upstream_count"],
                    r["mixed_paths"],
                    r["upstream_count"] / r["mixed_paths"],
                    r["first_change_layer"],
                ]
                for d in [*dev, *val]
                for r in [d["rows"]["local"]]
            ],
        ),
        table(
            ["baseline stratum", "seeds (development)", "mixed fraction"],
            [
                [
                    k,
                    sum(
                        d["rows"]["local"]["strata"].get(k, {}).get("total", 0)
                        for d in dev
                    ),
                    sum(
                        d["rows"]["local"]["strata"].get(k, {}).get("mixed", 0)
                        for d in dev
                    )
                    / sum(
                        d["rows"]["local"]["strata"].get(k, {}).get("total", 0)
                        for d in dev
                    ),
                ]
                for k in sorted({k for d in dev for k in d["rows"]["local"]["strata"]})
            ],
        ),
        "Bins use |half-curvature| boundaries 0.25, 1 and 3 per metre; residual boundaries 0.5, 1 and 2; dominant particle pT boundaries 0.5, 1 and 2 GeV. The sector flag describes crossing an unshifted seam. The baseline reconciles complete paths from two tilings without splicing paths. A skipped pixel layer can arise from missing detections or pattern recognition. These overlapping diagnostics do not identify unique causes. Nearby measurements and ownership conflicts are directly tested by replacement and joint selection.",
    )
    write(
        "SEED_PURIFICATION",
        "Seed residual rejection and same-layer replacement",
        "Given a frozen three/four-hit seed, compare its measured positions to a robust transverse prompt circle and an arc-length z regression. Use the largest normalized 2D residual, with effective scales 0.35 mm tangential and 0.7 mm longitudinal. The strict arm discards seeds above two units. Replacement considers one same-layer substitution at a time among four spatial neighbors inside 8 mm, including the hit itself. Each original plus two lowest-cost alternatives reaches assignment. All variants preserve radial and layer order.",
        summary(dev, ["score", "seed_purity", "retention", "efficiency", "mixed"]),
        "Retention is intentionally exact. A genuine same-particle replacement also changes the seed identity under this test.",
    )
    write(
        "MULTI_HYPOTHESIS",
        "Bounded competing seed hypotheses",
        "The beam arm reranks the original and two alternatives by maximum pixel residual plus first outer-layer corroboration. For four nearest volume-13 hits, compute the minimum innovation χ² at the actual radius, cap at 9 and divide by 9. Unreachable extrapolations receive neutral corroboration. Sorting all candidates by cost reconciles collisions on unowned alternatives. Reserved original hits guarantee a fallback for each seed. Every candidate has the same length as its parent. This is a one-step beam, without recursive combinatorial branching.",
        summary(dev, ["score", "mixed", "severe", "retention", "runtime"]),
    )
    detection = json.loads((OUT / "seed-detection.json").read_text())
    with (OUT / "SEED_CONTAMINATION.md").open("a") as out:
        print(
            f"\nOn validation event 1002, the truth-free maximum seed residual had ROC AUC {detection['residual_auc']:.4f} and average precision {detection['residual_ap']:.4f} for detecting mixed seeds at prevalence {detection['mixed_prevalence']:.4f}. These are ranking statistics for an untrained scalar predictor. A good detector of bad seeds can still make a poor rejection policy. Replacement converted {detection['changes']['changed_mixed_1_to_0']} mixed seeds to single-signal or noise-only paths and introduced mixing in {detection['changes']['changed_mixed_0_to_1']} cases. Other changed seeds retained their mixing status. The reference scorer alone took {detection['scoring_s']:.3f} s on this validation run. [Diagnostic record](seed-detection.json)",
            file=out,
        )
    smooth = json.loads((OUT / "smoothing.json").read_text())
    write(
        "ROBUST_KALMAN",
        "Fixed-path backward state estimation",
        "The forward filter is frozen Phase IV M3. Each assigned outer hit supplies a predicted and updated state. The extended RTS gain is Pᵢ Fᵢᵀ(Pᵢ₊₁⁻)⁻¹ evaluated by Cholesky solves. Apply the gain to the wrapped smoothed-minus-predicted state and the covariance difference. No measurement or assignment changes. Compare seed-exit curvature estimates with the endpoint tangent-direction change across the pixel chord on pure extended validation tracks.",
        table(list(smooth), [list(smooth.values())]),
        "The smoother worsened median absolute curvature error on this selected validation sample. It was excluded from reconstruction selection. The tighter forward gate is a separate development control. Material variation, displaced trajectories and the seed-to-endpoint proxy limit interpretation. Splitting was not evaluated.",
    )
    write(
        "GLOBAL_AMBIGUITY",
        "Exclusive pixel and outer hit ownership",
        "The joint arm relaxes the original pixel ownership reservation. Seeds compete by the best cost among all offered alternatives, and can lose their original hits. Greedy assignment keeps maximum one candidate per original and one owner per hit. By comparison, global in the benchmark uses the frozen Phase IV outer-path prefix packer and original pixel seeds. Neither procedure solves exact weighted set packing.",
        summary(dev, ["score", "mixed", "severe", "fake", "retention"]),
    )
    disk = json.loads((OUT / "disks.json").read_text())
    write(
        "DETECTOR_COVERAGE",
        "Forward pixel disk feasibility",
        "Inspect absolute median module z by volume and layer. Volumes 7 and 9 have seven signed-z surfaces each. For the first two planes use a prompt-origin radial scaling and at most four neighbors within 12 mm in xy. A linear xy extrapolation to the third median plane accepts the nearest candidate within 2 mm. Claim disjoint triplets by residual, then extend with a 3 mm gate on subsequent disks. Terminate on misses. This exploratory follower omits curvature transport, barrel-to-disk linking and the outer strip disks.",
        table(
            ["volume", "ordered (absolute z in metres, layer)"], disk["planes"].items()
        ),
        table(
            [
                "development event",
                "barrel score",
                "disk only",
                "disjoint union",
                "disk seconds",
                "vol 7 mixed",
                "vol 9 mixed",
            ],
            [
                [
                    d["event"],
                    d["barrel_score"],
                    d["disk_only_score"],
                    d["combined_score"],
                    d["disk_s"],
                    d["regions"]["7"]["mixed"],
                    d["regions"]["9"]["mixed"],
                ]
                for d in disk["development"]
            ],
        ),
        "All three scores use complete-event truth. These exploratory arms were excluded from Omega confirmation. Per-event truth weights for all nine volumes accompany the main benchmark.",
    )
    ds = read_stage("confirmation")
    stats = {
        k: paired([d["rows"][m][k] - d["rows"]["local"][k] for d in ds]) for k in KEYS
    }
    (OUT / "paired.json").write_text(json.dumps(stats, indent=2) + "\n")
    result = table(
        [
            "method",
            "score",
            "efficiency",
            "fake",
            "mixed",
            "severe",
            "retention",
            "s/event",
        ],
        [
            [
                mode,
                *[
                    mean(ds, mode, k)
                    for k in [
                        "score",
                        "efficiency",
                        "fake",
                        "mixed",
                        "severe",
                        "retention",
                        "runtime",
                    ]
                ],
            ]
            for mode in ("local", m)
        ],
    )
    delta = table(
        [
            "candidate minus local",
            "mean",
            "median",
            "SD",
            "95% low",
            "95% high",
            "positive events",
            "negative events",
        ],
        [[k, *s.values()] for k, s in stats.items()],
    )
    event = table(
        [
            "event",
            "hits",
            "baseline score",
            "candidate score",
            "delta",
            "baseline mixed",
            "candidate mixed",
            "peak MiB",
        ],
        [
            [
                d["provenance"]["event"],
                d["hits"],
                d["rows"]["local"]["score"],
                d["rows"][m]["score"],
                d["rows"][m]["score"] - d["rows"]["local"]["score"],
                d["rows"]["local"]["mixed"],
                d["rows"][m]["mixed"],
                d["peak_mib"],
            ]
            for d in ds
        ],
    )
    accept = (
        stats["score"]["mean"] > 0
        and stats["mixed"]["mean"] < 0
        and stats["efficiency"]["mean"] >= -0.002
    )
    write(
        "CONFIRMATION_RESULTS",
        "Prospective independent confirmation",
        result,
        event,
        delta,
        "The intervals are percentile intervals from 10,000 equal-event paired resamples, seed 1729. Occupancy and detector simulation conditions are not independently randomized. Secondary endpoints are descriptive.",
    )
    write(
        "BENCHMARK_RESULTS",
        "Complete official-event benchmark",
        summary(ds, KEYS),
        f"The frozen decision rule {'promotes ' + m if accept else 'retains the local EKF baseline'}. Paired estimates and individual regressions are in [confirmation results](CONFIRMATION_RESULTS.md).",
    )
    prof = json.loads((OUT / "ranking-profile.json").read_text())
    write(
        "SCALABILITY",
        "Single-process reconstruction costs",
        table(
            [
                "event",
                "load s",
                "graph s",
                "seed assembly s",
                "alternative pool s",
                "candidate choice s",
                "EKF extension s",
                "metric evaluation s",
                "peak MiB",
            ],
            [
                [
                    d["provenance"]["event"],
                    d["load_s"],
                    d["graph_s"],
                    d["seed_assembly_s"],
                    d["pool_s"],
                    d["rows"][m]["select_s"],
                    d["rows"][m]["extension_s"],
                    d["rows"][m]["evaluate_s"],
                    d["peak_mib"],
                ]
                for d in ds
            ],
        ),
        "Runtime in comparisons includes common two-pass seeding, geometry indexing, method-specific selection and extension. Validation, reference scoring and truth diagnostics run outside reconstruction. Evaluation timing combines matching and scoring. RSS is the process-lifetime high-water mark across both arms.",
        table(
            [
                "rank microbenchmark",
                "unique selected",
                "scalar median s",
                "vector median s",
            ],
            [
                [
                    d["event"],
                    d["selected"],
                    float(np.median(d["reference_s"])),
                    float(np.median(d["vector_s"])),
                ]
                for d in prof
            ],
        ),
        "The batched same-length residual kernel exactly reproduced the ordered seed list in four alternating timed repetitions per event. Only the resolver was timed in that comparison. The confirmation used the frozen scalar baseline. End-to-end speedup for the vectorized variant remains unmeasured.",
    )
    archive = json.loads((OUT / "cohort.json").read_text())["archive"]
    s = stats["score"]
    write(
        "FINAL_RESEARCH_REPORT",
        "TRACK-X: Pixel seed replacement and uncertainty-aware barrel tracking on simulated TrackML events",
        "## Abstract\n\nWe tested whether correcting pixel seeds improves a local extended Kalman barrel tracker. On twelve prospectively selected complete events, the validation-selected "
        + m
        + f" method changed mean official score by {s['mean']:+.6f} (paired event-bootstrap 95% interval {s['low']:+.6f} to {s['high']:+.6f}). "
        + (
            "The prespecified trade-off rule selected the challenger."
            if accept
            else "The prespecified rule retained the baseline."
        ),
        "## Problem and related work\n\nPattern recognition must distinguish charged trajectories that pass close together, with finite resolution, inefficiency and high occupancy. TrackML supplies simulated measurements, particle identities and weighted event scoring. ACTS provides experiment-independent propagation and track fitting. The original TRACK-X synthetic GNN study used an idealized cylinder geometry. Historical official-data controls found little score benefit from learned seed ranking. This study tests whether correcting a measurement before extension addresses an earlier source of error. [TrackML challenge](https://arxiv.org/abs/1904.06778) · [ACTS framework](https://arxiv.org/abs/2106.13593) · [Phase IV](../phase4/FINAL_RESEARCH_REPORT.md)",
        "## Detector, algorithms and measurement\n\nSeeds use three or four measurements in the pixel barrel, arranged in overlapping azimuthal tilings. The baseline ranks sparse triplets and quadruplets, resolves ownership and extrapolates outward into six barrel layers. In an axial field a prompt track advances by Δasin(hr) in azimuth and tΔ[asin(hr)/h] in z, with signed half-curvature h and longitudinal slope t. The EKF propagates their covariance and gates innovations at each actual hit radius. Omega retains original seeds and considers bounded same-layer substitutions ranked by prompt residuals and an outer-layer innovation. Candidate generation uses measured hits and geometry only. Truth enters after assignment. No network was trained in Omega. [Equations](../phase4/KALMAN_MODEL.md) · [Seed methods](SEED_PURIFICATION.md) · [Competing hypotheses](MULTI_HYPOTHESIS.md)",
        "## Experimental design\n\nEvents 1003–1005 supplied development controls and 1002 selected the challenger. We excluded all historical IDs through 1026. Twelve evenly spaced eligible positions in "
        + archive
        + " supplied final confirmation after selection was committed. All methods use the organizers’ complete-event score. Custom efficiency requires a unique majority match recovering at least half of a particle’s hits among particles with at least three hits. Fake and duplicate fractions divide unmatched and repeat qualifying candidates by all paths. Mixed paths contain two signal identities; severe merges contain two hits apiece from at least two signal particles. Runtime uses one CPU process. [Protocol](EXPERIMENT_PROTOCOL.md) · [Input hashes](manifest.json)",
        "## Independent confirmation",
        result,
        event,
        "## Seed contamination and ablations\n\n"
        + f"The baseline had {mean(ds, 'local', 'seed_mixed'):.2%} mixed seeds on the new events. The candidate had {mean(ds, m, 'seed_mixed'):.2%} and retained {mean(ds, m, 'retention'):.2%} of exactly pure original seeds. The mean paired change in final mixed fraction was {stats['mixed']['mean']:+.6f} and in severe fraction {stats['severe']['mean']:+.6f}. Strict rejection cost score on development data. The alternative-hit and global competition arms isolate retention and ownership costs. [Ablation tables](MODEL_ABLATIONS.md) · [Seed failure strata](SEED_CONTAMINATION.md)",
        "## Fit, coverage and computing\n\nBackward smoothing worsened the median curvature discrepancy on pure validation paths. A separate forward-disk prototype added full-event score in three historical development events. Its prospective generalization remains untested. Seed ordering admitted an exactly matching vectorized kernel in three event-level comparisons. The highest confirmation process RSS was "
        + f"{max(d['peak_mib'] for d in ds):.1f} MiB. [Fit study](ROBUST_KALMAN.md) · [Disk study](DETECTOR_COVERAGE.md) · [Performance](SCALABILITY.md)",
        "## Limitations and conclusions\n\nThe study tests one frozen intervention on official simulation. Selection used one validation event. Approximate covariance, prompt constraints, greedy ownership, fixed small neighbor pools and incomplete detector coverage limit transportability. Consecutive detector-production conditions can correlate events even in an evenly spaced archive sample. The bootstrap does not model simulation or calibration systematics. No matched ACTS run or external replication has been performed. The most informative further check would independently reproduce seed-level and full-event results across different simulation conditions and test barrel/disk ambiguity on a newly frozen cohort. [Full paired uncertainty](CONFIRMATION_RESULTS.md) · [Internal review](INDEPENDENT_REVIEW.md) · [Reproduction](REPRODUCIBILITY.md)",
    )
    print(result)


if __name__ == "__main__":
    main()
