# TRACK-X Phase III | Extending pixel seeds across the barrel

## Abstract

On three prospectively reserved TrackML simulated events, geometric propagation changed mean official full-event score from 0.01858 to 0.22320. All methods scored complete events. The extension preserves the original pixel seed and adds measurements from six outer surfaces. [Event-level comparison](CONFIRMATION_RESULTS.md) · [Figure](confirmation.html)

## Reconstruction quality and cost

| method | score | efficiency | fake | duplicate | purity | completeness | mixed | coverage | total_s |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.01858 | 0.03071 | 0.94266 | 0.00000 | 0.98233 | 0.70561 | 0.33512 | 0.14769 | 1.02682 |
| extended | 0.22320 | 0.26644 | 0.50241 | 0.00022 | 0.96029 | 0.73504 | 0.41696 | 0.28194 | 1.18896 |
| gnn_extended | 0.22308 | 0.26644 | 0.50279 | 0.00036 | 0.96036 | 0.73520 | 0.41706 | 0.28210 | 1.88845 |

Hit coverage and full-event majority matching benefit from longer paths. The mixed fraction tracks contamination independently of the fake rate. A path can contain contamination and still pass majority matching. The rates above describe different failure modes.

## Methods and limitations

[Protocol](EXPERIMENT_PROTOCOL.md) · [Trajectory model](TRACK_EXTENSION.md) · [Ablations](ABLATION_RESULTS.md) · [Runtime and score](tradeoffs.html) · [Candidate recall by layer](surface-recall.html) · [Failure analysis](FAILURE_ANALYSIS.md) · [Reproducibility](REPRODUCIBILITY.md). Seeds still require ≥3 pixel layers. Forward particles, displaced vertices, curling returns, material transport and module-level covariances are incompletely modeled. A next experiment should compare endcap propagation with a frozen barrel reference, measuring mixed-hit contamination and gain in covered truth weight on fresh events. ACTS motivates surface navigation and compatible-measurement selection. No matched ACTS run was performed. [ACTS tracking](https://acts.readthedocs.io/en/latest/tracking.html) · [TrackML accuracy challenge](https://arxiv.org/abs/1904.06778)
