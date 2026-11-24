# TRACK-X | Research brief

**Research question.** Can correcting one ambiguous pixel measurement improve downstream charged-particle reconstruction with a fixed barrel EKF? The starting observation was that 1,886 of 2,146 mixed validation paths already contained two signal identities in their seeds. Across twelve later confirmation events, the corresponding baseline fraction ranged from 86.3% to 89.0%.

**Method.** The tracker forms three- and four-hit pixel seeds in overlapping azimuthal sectors. For each seed, a spatial index offers single-hit substitutions on the same layer. A robust prompt-circle residual retains two alternatives. An outer-layer EKF innovation ranks them against the original seed before resolving shared hits. The procedure uses measured positions and detector geometry. Truth identifies failures after reconstruction.

**Evidence.** On twelve untouched TrackML simulated events from an official training archive, mean full-event score increased from 0.227243 to 0.228094. The paired difference was +0.000851 (95% event-bootstrap interval +0.000663 to +0.001054). Mixed paths fell from 37.70% to 37.04%; severe merges changed from 0.1623% to 0.1694%. Efficiency improved on every event. Exact clean-seed retention was 89.00%. Mean CPU reconstruction wall time increased from 1.742 to 1.953 seconds/event. The complete benchmark process peaked at 369.5 MiB on an Apple M1.

**Interpretation.** The change passed the prespecified score, efficiency and contamination rule. A rarer severe-merge benefit was unresolved. Strict seed rejection lost official score in development. The backward smoother worsened median curvature discrepancy on selected pure tracks. The exploratory pixel-disk follower increased development scores; its performance across detector transitions remains unknown.

**Collaboration directions.** Review module-overlap alternatives, replicate the paired experiment on a different simulation production, or validate a common source-link geometry and particle-matching contract for an ACTS comparison. [Paired results](CONFIRMATION_RESULTS.md) · [Manuscript](manuscript.pdf) · [Reproduction](REPRODUCIBILITY.md)
