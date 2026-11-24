# Internal critical review

| Severity | Check | Finding and disposition |
|---|---|---|
| High | Confirmation reuse | Cohort and selection rule committed separately before acquisition. Earlier studies end at 1026. All twelve new IDs exceed that bound. Distinct archive names alone do not imply different simulation physics. |
| High | Inference leakage | Seed interventions consume coordinates, radial/layer ordering and volume. The EKF look-ahead uses measured outer hits. Particle identity and truth momentum are read by post-inference diagnostics. The validation atlas carries truth for visualization only. |
| High | Score equivalence | Phase IV replay verified seven matched arms using full submissions and the pinned reference scorer. Omega uses the same scorer and unassigned-hit convention. Disk results also score complete events. |
| High | Unfair tuning | Strict, replacement, beam and joint alternatives were compared on historical data. One validation event chose beam. The disk study and smoother did not enter the final search. |
| Medium | Purity selection bias | Matched-track purity excludes fakes. Report all-path mixing, severe merges, fake fraction and exact clean-seed retention alongside official score. Exact seed retention counts an equivalent pure substitution as a change. |
| Medium | Timing attribution | Shared baseline work is charged to each arm. Peak RSS covers both and scoring. Strict rejection is charged for construction of the common alternatives pool; a specialized rejection-only implementation could be faster. Component microbenchmarks do not establish whole-pipeline acceleration. |
| Medium | Failure attribution | Identity changes identify disagreement with the first signal hit. They do not locate a unique causal wrong hit. Low pT, missing layers, seam crossing and local occupancy can overlap. |
| Medium | Limited independence | One production archive, deterministic evenly spaced IDs, twelve events, one validation event and one beam design. Bootstrap intervals cover between-event sampling only. No external peer review. |
| Medium | Model assumptions | Prompt radial EKF and coarse effective covariance, no material transport. A capped beam can omit the correct alternate or assign neutral corroboration to an unreachable next plane. The disk model does not join barrel and forward segments. |
| Medium | Provenance | Development code was run before its source commit. Event records include the exact source/config hash and then-current HEAD. dbf9ba0 archives the executed algorithm. A clean replay checks the hash and scientific outcomes. |
| Low | Upstream scorer maintenance | The pinned TrackML scorer emits pandas deprecation warnings. Regression tests currently agree with the published definition; any dependency upgrade requires another score check. |

A separate investigator should reproduce the work, review seed replacement semantics on module overlaps and evaluate a different simulation production. The internal checks support a reproducible research prototype.
