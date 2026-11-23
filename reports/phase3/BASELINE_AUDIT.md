# Baseline replay

At 8b5e985 the clean checkout passed all 23 existing tests. An isolated working directory reused the Phase II checkpoints and replayed both the original method comparison and the confirmation script. All 12 confirmation rows matched exactly on every field except timing. The original evaluation also passed the scientific replay verifier. See baseline-replay.json for measured latencies and phase2-replay.json for checked outputs. Phase I was covered by the existing smoke and regression tests; the complete three-seed synthetic suite was not rerun.
