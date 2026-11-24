# Experiment registry

The frozen Omega configuration hash is `e5b114d73118f877c70dd2ce1e7d1e77ba84e5dca5b87d4baa689589b6a42a43`. The exact digest and source revision of each run appear in its event JSON. Seed ranking uses no model weights. The Phase IV replay uses the previously archived GNN weights for its learned arm.

| Study | Dataset and status | Result record |
|---|---|---|
| Baseline audit | Phase IV full simulation, 1022–1026, historical replay | [Audit](reports/omega/BASELINE_AUDIT.md) |
| Omega development | 1003–1005, eight methods, historical | [Ablations](reports/omega/MODEL_ABLATIONS.md) |
| Validation | 1002, one selection, beam selected | [Selection](reports/omega/selection.json) |
| Prospective comparison | Twelve events from train_1.zip, consumed holdout, never reuse as untouched | [IDs](reports/omega/cohort.json) · [Hashes](reports/omega/manifest.json) · [Event differences](reports/omega/CONFIRMATION_RESULTS.md) |
| Seed diagnosis | 1002, residual discrimination and changed-path inspection | [Repair/damage counts](reports/omega/seed-detection.json) |
| Forward pixel disks | 1003–1005, exploratory development, independent validation pending | [Coverage](reports/omega/DETECTOR_COVERAGE.md) |
| Backward smoothing | 1002, worsened median absolute seed-exit curvature error on pure extended paths | [Fit](reports/omega/ROBUST_KALMAN.md) |
| Seed ordering optimization | 1003–1005, equality of complete seed lists, kernel timing only | [Timing](reports/omega/ranking-profile.json) |
| ACTS | Matched benchmark NOT RUN; missing geometry/source-link equivalence | [Prerequisites](reports/omega/ACTS_COMPARISON.md) |

Strict rejection and the tight EKF gate were negative development controls. Global pixel competition sometimes gave a larger development score, but validation selected reserved ownership with corroboration. A material-calibrated fit, severe-merge reduction and barrel/disk joining remain unresolved.

The clean checkout reproduced all 16 Omega event records. A separate end-to-end batched-ranker replay produced exactly the frozen assignments and scores on all three development events. [Clean replay](reports/omega/clean-replay.json) · [Paired timing trials](reports/omega/fast-replay.json)
