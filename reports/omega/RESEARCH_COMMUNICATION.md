# Research communication materials

## Abstract

A sparse pixel-seed replacement procedure improved the official full-event TrackML score in twelve prospectively selected simulated events. A prompt-trajectory fit and outer-layer innovation rank the original seed alongside two same-layer measurement alternatives. Mean paired score gain over a fixed local EKF was 0.000851, with lower broad contamination and higher full-event efficiency. The severe-merge endpoint remained unresolved. A resource-bounded Python implementation, SHA-256 event manifests and event-level replays make the comparison inspectable on a laptop.

## Engineering portfolio

The research pipeline streams and validates one complete event, indexes spatial neighborhoods on actual detector volumes, runs bounded seed alternatives and propagates small EKF covariance matrices. A separate evaluation layer runs the organizers’ scorer and records per-event timing, memory, seed retention and truth-stratified failure counts. Checkpoints are atomic. The benchmark suite preserves frozen baselines and writes separate outputs for independent replay. A scalar-to-vectorized ranking experiment checked equality of the entire ordered seed list before reporting kernel timings. A local Streamlit viewer inspects where alternatives acquired or lost measurements.

## Five-slide talk outline

1. Show an ambiguous pixel seed and explain why one early hit affects six subsequent surfaces.
2. Draw the beam of three measurements sets, the residual gate and the first outer-layer prediction.
3. Show the twelve paired score and mixed-fraction differences with uncertainty.
4. Show a repair and a damaged seed from the validation atlas. Explain why stricter rejection can improve purity and still lose score.
5. Compare CPU cost, remaining detector coverage and the requirements for a matched external benchmark.

## Poster content

**Question:** Which pixel-hit substitutions survive a downstream geometric check? **Method panel:** Two sector tilings, one same-layer substitution, three retained hypotheses and local EKF extension. **Central figure:** Use confirmation-pairs.pdf at native vector resolution. **Failure panel:** Present single-hit contamination and severe mixing separately. **Reproduction footer:** Include the frozen algorithm revision, archive and manifest, bootstrap seed, full-event scorer, license and local verification command.

## 75-second demonstration

* 0–15 s: Show the official-simulation indicator and identify the ten barrel surfaces.
* 15–35 s: Select a validation example in the Omega lab. Toggle original and alternative trajectory traces. Read the layer, hit index and truth identity for diagnosis.
* 35–50 s: Open the matched event records and identify the complete-event score, both contamination measures and exact clean-seed retention.
* 50–65 s: Show the twelve-event paired figure and point out that lower broad contamination did not establish a severe-merge improvement.
* 65–75 s: Show the manifest, the reproduction command and the separate development-only disk result.

## Draft reviewer invitation

I have prepared a reproducible study of pixel-seed replacement on official TrackML simulation. The method uses an outer-layer measurement to rank bounded seed alternatives before EKF extension. I would value technical review of the module-overlap assumptions, clean-seed retention measure and paired event protocol. The local package contains the manuscript, data hashes, benchmark commands and event-level outputs.
