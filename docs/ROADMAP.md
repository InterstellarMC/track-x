# Path toward an upstream contribution

1. Acquire a permitted TrackML sample, record filenames, checksums, detector geometry and field conventions. Freeze a held-out event list. Measure candidate-edge acceptance and efficiency versus pT, radial layer and occupancy before interpreting classifier scores.
2. Generate ACTS Examples events with recorded geometry, field and digitization settings. Compare shared measurements against a documented ACTS seeding and combinatorial Kalman-filter reference, including displaced and low-pT failures. Use a common matching definition.
3. Profile the candidate builder. If spatial search dominates, port that kernel to native C++ and verify identical candidate sets against frozen fixtures, including wraparound and ties.
4. Discuss a narrow test fixture or benchmark with ACTS maintainers before proposing an API. Establish build, licensing and CI compatibility on the supported ACTS revision. Upstream integration remains future work.
