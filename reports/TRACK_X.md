# TRACK-X: Physics-Constrained Graph Neural Networks for Charged-Particle Track Reconstruction

## Abstract

We studied which parts of a laptop-scale reconstruction pipeline improve complete-event TrackML score. A synthetic study compared graph classification with geometry gates. An official-event study tested whole-track ranking and sector boundaries. A prospective outward-extension study on three further events raised mean official score from 0.01858 to 0.22320. The increase coincided with more mixed-particle candidates.

## From simulated cylinders to official detector geometry

The Phase I generator produced independent truth associations for supervised learning and post-reconstruction evaluation. The original three-seed stress study found a trade-off between fake suppression and reconstruction efficiency under a triplet gate. Phase II introduced the official detector metadata, full-event scoring and a frozen pixel-barrel benchmark. Three prespecified confirmation events supported a second shifted tiling, while learned seed ranking gave little additional benefit. [Synthetic study](TECHNICAL_REPORT.md) · [Phase II study](phase2/FINAL_RESEARCH_REPORT.md)

## Multi-layer experiment

The third study fixed the two-pass pixel tracker and propagated its paths into volumes 13 and 17. Fits use wrapped transverse azimuth, prompt-circle radial intersections and longitudinal arc length. Geometry proposes hits independently of truth. Competing paths claim hits under a deterministic residual ranking. The comparison uses a fixed set of methods on the same complete events and calls the TrackML reference scorer.

| method | score | efficiency | mixed | total_s |
|---|---|---|---|---|
| baseline | 0.01858 | 0.03071 | 0.33512 | 1.02682 |
| extended | 0.22320 | 0.26644 | 0.41696 | 1.18896 |
| gnn_extended | 0.22308 | 0.26644 | 0.41706 | 1.88845 |
| no_gaps | 0.22393 | 0.26835 | 0.39926 | 1.17456 |

The supported barrel region held 57.4% of event truth weight on average. Curvature-aware propagation mattered: the zero-bend control scored 0.04709. Stopping after the first missed surface scored 0.22393, slightly above the prespecified primary extension; both remained in the report. Allowing gaps also increased the mixed fraction relative to stopping. A GNN trained for pixel edge scoring was applied only at seed ranking. Its measured end-to-end score provides no evidence of an advantage here.

## Limits and next test

The confirmation consists of three consecutive simulated events, a single frozen set of classifier weights, and fixed field and beam-line assumptions. Track finding still depends on successful pixel seeds. Extrapolation windows lack material-aware covariance calibration. Before extending this method to the endcaps, compare an ambiguity or rejection rule calibrated on development events against the no-gap baseline. Freeze it and measure both correctly recovered hit weight and new contaminating hits on fresh events. ACTS provided conceptual guidance on propagation and compatible measurement selection. A matched ACTS reconstruction benchmark remains unrun.

[Phase III findings and figures](phase3/FINAL_REPORT.md) · [Event-level results](phase3/CONFIRMATION_RESULTS.md) · [Run instructions](phase3/REPRODUCIBILITY.md) · [Protocol](phase3/EXPERIMENT_PROTOCOL.md) · [TrackML accuracy phase](https://arxiv.org/abs/1904.06778) · [ACTS tracking](https://acts.readthedocs.io/en/latest/tracking.html)

## Phase IV · independent precision-tracking experiments

At barrel radius r, the filter carries (φ,z,h,t), with signed half-curvature h and longitudinal slope t. An outward prompt circle advances azimuth by Δasin(hr) and z by tΔ[asin(hr)/h]. Its Jacobian propagates covariance. A wrapped two-coordinate innovation ranks at most eight hits per surface. A Joseph update incorporates the accepted hit. The global alternative selects disjoint complete extensions or prefixes by their summed hit compatibility. The covariance scales are effective tracking tolerances. Validation found conservative curvature pulls on pure tracks. [Model and Jacobian](phase4/KALMAN_MODEL.md) · [Uncertainty validation](phase4/COVARIANCE_VALIDATION.md)

Ten fresh events (1012–1021) tested M1 against M0. The mean change in mixed fraction was -0.018860 (paired 95% interval -0.020233 to -0.017538). The change in official score was +0.000346 (paired 95% interval -0.000143 to +0.000830). The latter interval includes zero. The experiment set no noninferiority margin. [No-gap event data](phase4/NO_GAP_CONFIRMATION.md)

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

M3 minus M1 gave a score change of +0.002567 (paired 95% interval +0.001766 to +0.003328). M3 minus M0 changed broad mixing by -0.037829 (paired 95% interval -0.039987 to -0.036559), severe mixing by -0.000982 (paired 95% interval -0.001254 to -0.000709), efficiency by +0.004405 (paired 95% interval +0.002870 to +0.005940) and wall seconds by +0.018444 (paired 95% interval -0.179795 to +0.142909). M5 minus M3 changed score by -0.000799 (paired 95% interval -0.000996 to -0.000550). GNN seed weighting changed score by -0.000002 (paired 95% interval -0.000233 to +0.000259). Full tables also give event-level distributions, fake and duplicate fractions, purity, completeness, timing and coverage. [Frozen event comparisons](phase4/CONFIRMATION_RESULTS.md) · [Development ablations](phase4/ABLATION_STUDY.md) · [Figures](phase4/measurements.html)

[Phase IV model, failure analysis and limitations](phase4/FINAL_RESEARCH_REPORT.md) · [Reproduction instructions](phase4/REPRODUCIBILITY.md)
