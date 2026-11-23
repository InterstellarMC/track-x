# TRACK-X Phase IV · precision tracking on official TrackML simulation

## Abstract

A no-gap rule reduced contamination in a ten-event prospective comparison. A prompt radial extended Kalman filter (M3) then improved full-event score over the frozen Phase III geometry tracker (M0) on a separate five-event cohort. Their mean paired score difference was +0.003272 (paired 95% interval +0.002429 to +0.004387). The study concerns simulated events and ten outward barrel surfaces. Endcaps, displaced seeding and detector-calibrated momentum remain outside its scope.

## Methods

Two half-sector-shifted passes produce pixel seeds. M0 allows two missed outer layers and M1 terminates after the first miss. M2 refits prompt geometry and propagates an approximate covariance to select hits. M3 carries a four-parameter EKF posterior across surfaces. M4 and M5 rank competing complete extensions and prefixes under exclusive ownership. M6 uses the frozen Phase II GNN to rank pixel seeds for the validation-selected M3. Seed formation, spatial indexing, complete-event scoring and matching conventions follow Phase III. [State model](KALMAN_MODEL.md) · [Association](HIT_ASSOCIATION.md) · [Ambiguity objective](GLOBAL_AMBIGUITY.md)

## Independent comparisons

Ten fresh events (1012–1021) tested M1 against M0. The mean change in mixed fraction was -0.018860 (paired 95% interval -0.020233 to -0.017538). The change in official score was +0.000346 (paired 95% interval -0.000143 to +0.000830). The latter interval includes zero. The experiment set no noninferiority margin. [No-gap event data](NO_GAP_CONFIRMATION.md)

Three historical events supplied development ablations. On validation event 1002, M3 scored 0.233050, the highest of M0–M5. After freezing the methods, we acquired simulated events 1022–1026. Each row below averages those same five events.

| method | official score | efficiency | mixed | severe | fake | wall s/event |
|---|---|---|---|---|---|---|
| Phase III, two gaps | 0.221185 | 0.260268 | 0.432438 | 0.003284 | 0.507881 | 1.878690 |
| Phase III, no gaps | 0.221889 | 0.261491 | 0.413543 | 0.003038 | 0.505704 | 1.746463 |
| Robust refit, statistical gate | 0.218389 | 0.258244 | 0.392268 | 0.001945 | 0.511860 | 1.949355 |
| Local EKF | 0.224457 | 0.264673 | 0.394608 | 0.002302 | 0.499711 | 1.897135 |
| Geometry, global prefixes | 0.220542 | 0.260014 | 0.397348 | 0.002046 | 0.508477 | 2.011809 |
| EKF, global prefixes | 0.223658 | 0.263398 | 0.385570 | 0.001728 | 0.502122 | 2.103614 |
| M3 with GNN seed rank | 0.224455 | 0.264660 | 0.395057 | 0.002264 | 0.499993 | 2.642818 |

M3 minus M1 gave a score change of +0.002567 (paired 95% interval +0.001766 to +0.003328). M3 minus M0 changed broad mixing by -0.037829 (paired 95% interval -0.039987 to -0.036559), severe mixing by -0.000982 (paired 95% interval -0.001254 to -0.000709), efficiency by +0.004405 (paired 95% interval +0.002870 to +0.005940) and wall seconds by +0.018444 (paired 95% interval -0.179795 to +0.142909). M5 minus M3 changed score by -0.000799 (paired 95% interval -0.000996 to -0.000550). GNN seed weighting changed score by -0.000002 (paired 95% interval -0.000233 to +0.000259). Full tables also give event-level distributions, fake and duplicate fractions, purity, completeness, timing and coverage. [Frozen event comparisons](CONFIRMATION_RESULTS.md) · [Development ablations](ABLATION_STUDY.md) · [Figures](measurements.html)

## Failure analysis and computation

In validation event 1002, 1886 M3 candidates already contained two signal identities in their seed, and 260 first acquired a second identity during extension. Only 10 of 5261 paths contained two hits apiece from two signal particles. Diagnosis uses truth only after inference. The validation atlas contains deterministically selected examples, and frequency estimates use every path. Narrow innovations on correctly attributed hits and curvature pulls with heavy concentration near zero limit uncertainty interpretation. [Covariance study](COVARIANCE_VALIDATION.md) · [Contamination categories](FALSE_MERGE_ANALYSIS.md)

Peak process RSS across the final cohort was 350.8 MiB on the Apple M1. Timings include real spatial indexing and both seeding passes. Whole-path proposal ranking adds work and the GNN adds pixel inference. The event benchmark reuses its common seed computation across matched methods. [Stage measurements](PERFORMANCE.md)

## Interpretation and next experiment

M3 scored higher than M0 in all five events. The local EKF is the prespecified validation choice. The paired score and purity results should be weighed against runtime, fewer assigned hits and the behavior of the severe-merge endpoint. Five consecutive events from a single simulation archive cannot establish generalization across pileup settings or detector conditions. The bootstrap treats events as exchangeable and gives no protection against dataset shift. The alternative arms are descriptive multiple comparisons. The selected truth curvature diagnostic is an endpoint proxy on pure paths; the measurement model has no material map. Whole-path selection tests a bounded greedy prefix pool. Splitting, backward smoothing, exact set packing and a matched ACTS comparison were not evaluated.

The next experiment should target mixed pixel seeds before extension. Prespecify a competitive seed replacement or outlier-rejection rule, preserve the local EKF and compare against it on a disjoint simulation shard with sufficient independent events for rare severe merges. Evaluate truth-free seed quality, coverage, official score and time alongside contamination. [Protocol](EXPERIMENT_PROTOCOL.md) · [Reproduction](REPRODUCIBILITY.md)
