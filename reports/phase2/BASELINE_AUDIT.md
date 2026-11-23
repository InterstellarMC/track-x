# Baseline audit

Started at 13b9764 on a clean main branch on the native Apple M1. The locked Python 3.12 installation succeeded. All 14 original tests passed, including MPS. The CPU smoke command reused the existing checkpoints, recovered validation BCE 0.32890 (MLP) and 0.15421 (GNN), and produced efficiency 1.0 for each method on the four smoke events. Its in-command elapsed time was 1.534 s and evaluation-stage maximum RSS was 343064576 bytes. The earlier 39.07 s measurement covered a cold process and fresh environment.

Re-evaluation with the original checkpoints reproduced the test digests and every non-timing event-method field for all nine Phase I sample/seed combinations, 540 rows. This checks frozen-weight inference. The original clean smoke regression also trains new synthetic weights in a temporary directory. The earlier cold mac_m1 training time was not independently remeasured in this audit. [Machine-readable audit](baseline.json)

The implementation reuses EdgeNet, trained Phase I checkpoints, graph inputs, exclusive path assembly, circle fits and the original majority-overlap evaluation. Phase II artifacts reside in this directory. The reference Phase I JSON and reports remain in their original paths.
