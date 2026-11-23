# Research question and checkpoints

Does message passing improve hit association over a local edge model on identical candidate graphs, and what is the efficiency cost of enforcing triplet consistency?

- [x] Verify native tooling, study ACTS tracking and the TrackML table schema.
- [x] Generate repeatable prompt helices with missing hits, clutter and independent truth.
- [x] Run a classical follower, learned edge scoring, exclusive assembly and matching on one complete small split.
- [x] Add event-level normalization, checkpoints, cache eviction, device probes and local visualization.
- [x] Benchmark matched methods, a same-weights triplet ablation, fit residuals and three seeded conditions.
- [x] Verify clean installation, CPU/MPS execution, tests and lint.
- [ ] Validate official-file transfer on user-supplied TrackML events.
- [ ] Compare detector-aware propagation and fitting with an ACTS Examples reference.

The minimal deliverable is the smoke command. The research extension is the reproducible multi-seed stress study and robust-fit comparison. Dataset-scale claims require the two open validation steps.
