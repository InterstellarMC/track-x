# TRACK-X: Physics-Constrained Graph Neural Networks for Charged-Particle Track Reconstruction

## Abstract

We studied nine official TrackML simulated events. Two events trained the compact networks, one selected checkpoints, three supplied the initial paired evaluation, and three newly acquired events tested a prespecified sector-boundary change. The primary reconstruction uses the four-layer pixel barrel. A separate transfer check examines other barrel volumes. A geometry-only whole-path ranker performed similarly to its GNN-weighted counterpart. The experiment gives stronger evidence for a change in candidate generation and assembly than for a learned advantage.

## Physics, data and methods

Prompt charged trajectories in a uniform axial field bend transversely while advancing longitudinally with arc length. Locally, azimuth and z can be extrapolated in radius. We score triplets by extrapolation residuals scaled by 0.010 rad and 3 mm. A norm below three retains a hypothesis. These are engineering tolerances; they are not a detector covariance or a calibrated chi-square. Ranking before hit ownership resolves conflicts that greedy edge selection makes irreversible. Indexed sparse neighborhoods and bounded out-degree limit hypothesis enumeration. [Detector geometry](DETECTOR_GEOMETRY.md) · [Input provenance](TRACKML_DATA_VALIDATION.md)

## Results on official simulation

| builder | recall | edges | false_candidates |
|---|---|---|---|
| directional | 0.53725 | 259688.66667 | 241401.00000 |
| original | 0.39961 | 306759.00000 | 293163.66667 |

| method | official_score | efficiency | efficiency_event_SD | fake |
|---|---|---|---|---|
| edge | 0.00021 | 0.02600 | 0.00483 | 0.80489 |
| triplet | 0.01552 | 0.48630 | 0.02003 | 0.48550 |
| triplet_ml | 0.01546 | 0.48664 | 0.01961 | 0.48537 |

## Sector boundary follow-up

A second tiling is shifted by half a sector. Geometry-only paths from both passes compete globally by length, then triplet residual, then deterministic hit index. Both rows use the same final ownership resolver. This follow-up used the original test events after diagnosing boundary losses; the result is exploratory.

| passes | official_score | efficiency | fake | duplicates | purity | completeness |
|---|---|---|---|---|---|---|
| 1 | 0.01552 | 0.48630 | 0.48550 | 0.00024 | 0.97491 | 0.80660 |
| 2 | 0.01930 | 0.62153 | 0.34265 | 0.00129 | 0.98371 | 0.80939 |

The official score is small because full truth and weights include outer layers and disks that the reconstruction leaves unassigned. Custom barrel efficiency also remains far below one. These measures answer different questions. The table gives event means and between-event sample SD for efficiency. Three events from the start of one archive and one model seed cannot establish general-population precision. The path-assembly follow-up was motivated by initial transfer results and is exploratory on this test split. An independent untouched sample is needed for confirmation.

## Independent confirmation

After freezing the boundary protocol in Git, three further official events were acquired. Two passes improved full-event score and barrel efficiency in every confirmation event. Adding a learned edge weight to triplet ranking had little effect. [Paired event results](CONFIRMATION.md)

| method | passes | official_score | efficiency | fake |
|---|---|---|---|---|
| geometry | 1 | 0.01633 | 0.50267 | 0.47141 |
| geometry | 2 | 0.01959 | 0.63415 | 0.32905 |
| gnn_weighted | 1 | 0.01627 | 0.50303 | 0.47050 |
| gnn_weighted | 2 | 0.01956 | 0.63416 | 0.32890 |

## Synthetic comparison

The original synthetic results remain separate. The audit reproduced every stored non-timing metric for nine Phase I experiments. Detector overlap, multiple hits per layer, high occupancy and abrupt sector boundaries expose failure modes absent from the ideal eight-cylinder model. [Phase I audit](BASELINE_AUDIT.md) · [Synthetic results](../RESULTS.md)

## Related work and comparison limits

The TrackML accuracy challenge supplies weighted truth associations and an official scoring implementation. Graph-based seeding and labeling motivated the learned comparison. ACTS uses geometry-aware propagation, measurement selection, seeding, combinatorial Kalman branching, fitting and ambiguity resolution. TRACK-X tests local compatibility and shared-hit resolution with fixed tolerances. A matched ACTS execution would require the same digitization, source-link geometry, seeding, field, uncertainties and selection denominators. No such setup was executed. A useful first upstream discussion would concern a small seam/overlap regression fixture with known geometry and licensing. [TrackML paper](https://arxiv.org/abs/1904.06778) · [Graph seeding](https://arxiv.org/abs/2007.00149) · [ACTS tracking](https://acts.readthedocs.io/en/latest/tracking.html) · [Official scorer](https://github.com/LAL/trackml-library/blob/53a165e15a2c885f54c2bef1bd1ed53db6ed9648/trackml/score.py)

## Validation, limitations and next experiment

[Matched baselines](REAL_DATA_BENCHMARKS.md) · [Model and physics ablations](MODEL_ABLATIONS.md) · [Failure strata](FAILURE_ANALYSIS.md) · [Measured performance](PERFORMANCE.md) · [Figures](measurements.html) · [Reproduction](REPRODUCIBILITY.md). Use the confirmed two-pass geometry-only baseline to test extension across barrel volumes under the same full-event score. Compare added detector coverage, false merges and CPU cost on new events. Endcap and outer-barrel reconstruction, independent seed uncertainty, material transport and a matched ACTS performance comparison remain open.
