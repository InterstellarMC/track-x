# TRACK-X Omega state

The scientific release is anchored at d3cc150. Selection was frozen at 0548c0f before the twelve-event acquisition. Resolve the current checkout with `git rev-parse HEAD`; subsequent commits carry review materials and verification receipts.

* **Validated barrel method:** one same-layer pixel replacement, original plus two alternatives, outer-layer innovation corroboration, exclusive ownership, frozen M3 extension. The independent decision rule selected this configuration.
* **Scientific result:** 0.227243 to 0.228094 mean official full-event score. Broad mixing fell from 37.70% to 37.04%. Severe merges changed from 0.1623% to 0.1694%; the paired interval includes zero. Mean reconstruction wall time was 1.742 versus 1.953 s/event and process peak was 369.5 MiB.
* **Completed studies:** baseline replay; contamination diagnosis; rejection, replacement, beam and ownership controls; forward/backward fitting; exploratory pixel disks; CPU ranking optimization; twelve-event confirmation; publication manuscript, figures and local event viewer.
* **Review limits:** no independent external review, matched ACTS setup, calibrated material model, tested barrel-to-disk joining or confirmation under another simulation production. Rare severe merges and runtime differences remain uncertain.
* **Release artifacts:** reports/omega is the primary record. Use EXPERIMENT_REGISTRY.md for study provenance, NEXT_SESSION.md for reproducible commands and RESEARCH_BACKLOG.md for optional independent work. The current action is local technical review; no publication is scheduled.

**Verified tests and replay.** The provisioned M1 suite passed 53 tests. The clean scientific checkout installed its own locked environment offline and reproduced every recorded scientific field in three development events, one validation event and twelve confirmation events. The data-free test initially passed 48 tests and skipped two licensed-input tests before three additional release tests were added. [Replay receipt](reports/omega/clean-replay.json)
