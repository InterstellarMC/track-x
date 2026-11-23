# Baseline replay

At 8b5e985 the clean checkout passed all 23 existing tests. An isolated working directory reused the Phase II checkpoints and replayed both the original method comparison and the confirmation script. All 12 confirmation rows matched exactly on every field except timing. The original evaluation also passed the scientific replay verifier. See baseline-replay.json for measured latencies and phase2-replay.json for checked outputs. Phase I was covered by the existing smoke and regression tests; the complete three-seed synthetic suite was not rerun.

| Phase II replay | Barrel efficiency | Barrel fake rate | Full score | Time s/event |
|---|---:|---:|---:|---:|
| geometry, 1 pass | 0.50267 | 0.47141 | 0.01633 | 1.01093 |
| geometry, 2 pass | 0.63415 | 0.32905 | 0.01959 | 2.11356 |
| gnn_weighted, 2 pass | 0.63416 | 0.32890 | 0.01956 | 2.08401 |

The Phase III event tables measure efficiency and fakes against full-particle hit counts. Phase II barrel efficiency and barrel fake fractions above use the pixel-barrel denominator.
