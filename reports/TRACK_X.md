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
