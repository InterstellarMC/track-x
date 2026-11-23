# Reproducing the official-event experiment

Use the repository root and native CPython 3.12. The tested system is an Apple M1 with 8 GiB unified memory. `uv.lock` pins the native dependencies, including pandas and the organizers’ TrackML library at revision 53a165e15a2c885f54c2bef1bd1ed53db6ed9648. On this installation the upstream scorer emits pandas deprecation warnings. The numeric reference tests pass.

## Install and acquire

```sh
uv sync --locked --python 3.12
uv run python -m trackx train --config configs/mac_m1.yaml
uv run python scripts/fetch_trackml.py
```

First sign in to Kaggle and accept the competition rules with your own account. The fetch script uses your local legacy Kaggle API credentials from kaggle.json in KAGGLE_CONFIG_DIR or ~/.kaggle. Keep credentials outside the repository. Alternatively supply the exact files listed in manifest.json under data/trackml. The official competition supplies detectors.zip and train_sample.zip. The latter was listed at 903185165 bytes. This run read 57311261 bytes of sample ZIP ranges and 179575 bytes of geometry ZIP ranges, plus the single-byte range probes. It never downloaded the complete event archive. The fetch script verifies each member against the archive CRC and resumes by skipping correct local files. The experiment checks all individual SHA-256 hashes against the frozen manifest. [Competition data](https://www.kaggle.com/competitions/trackml-particle-identification/data) · [Rules](https://www.kaggle.com/competitions/trackml-particle-identification/rules)

## Run and inspect

```sh
uv run python -m trackx.phase2
uv run python scripts/profile_phase2.py
uv run python scripts/phase2_geometry.py
uv run python scripts/phase2_stress.py
uv run python scripts/phase2_diagnostics.py
uv run python scripts/phase2_boundaries.py
uv run python scripts/phase2_devices.py
uv run python scripts/phase2_volumes.py
uv run python scripts/phase2_surfaces.py
uv run python scripts/phase2_display.py
uv run python scripts/fetch_trackml.py --confirmation
uv run python scripts/phase2_confirm.py
uv run python scripts/phase2_report.py
uv run pytest -q
uv run ruff check .
uv run --extra demo streamlit run scripts/pages/Detector_lab.py --server.address 127.0.0.1
```

The main runner uses configs/phase2.yaml and writes checkpoint state under experiments/official-<config ID>. Each checkpoint identifies the last completed epoch and best validation model. Repeating the experiment reuses these weights and evaluates the frozen events again. For a cold independent reproduction, use a fresh checkout. The Git-tracked results preserve the release measurements; new runs can change timing files. The Phase I stress-suite replay independently checks every scientific output field against its original report and requires all three Phase I seeds' weights. To recreate missing Phase I weights, run scripts/suite.py in a fresh checkout before scripts/replay_phase1.py. The official experiment needs only the default mac_m1 transfer weights.

## Provenance and measurement scope

- manifest.json records train/validation/test IDs, source archive, file sizes and SHA-256 hashes. Training and test partitions share no event IDs or sectors.
- training.json records validation history, model weight digests, seed, config ID and PyTorch platform. Raw hits, labels and checkpoints are excluded from Git.
- event*.json contains per-event comparisons on identical measurements. stress.json includes the validation event and all three test events; report generation filters to test IDs. candidate-losses.json attributes every eligible true pair to its first failed gate or acceptance.
- devices.json measures warmed forward passes on identical observed sectors with independent processes. process-time.txt contains the cold main experiment's wall time and peak process RSS; secondary studies have their own timing scopes.
- Run the standalone Phase I smoke to generate its interactive synthetic HTML. The separate Phase II Streamlit page loads a complete official event and visualizes a bounded sector. geometry.html and measurements.html use the checked-in Plotly runtime in the parent reports directory.
- CI is configured for public code and fixtures without credentials. The official-file smoke test skips when its licensed input is absent. Hosted CI execution was not performed. No raw data or reconstructed per-hit submission is published.

The confirmation acquisition fetched an additional 28216223 sample ZIP range bytes plus a second detector table. These events live in data/trackml-confirmation. The protocol commit precedes acquisition. To replay the initial evaluation in a separate directory run `uv run python -m trackx.phase2 --output /tmp/trackx-independent` followed by `uv run python scripts/verify_phase2_replay.py /tmp/trackx-independent`. The replay uses existing checkpoints.

## Release verification

The local suite finished with 23 tests passed, zero skipped and six pandas deprecation warnings from the pinned upstream scorer. Ruff lint and formatting passed. The Streamlit application test loaded an official exploratory event, switched to confirmation event 1006 and rendered the RZ projection. A local failure-case HTML display was generated in experiments/phase2-failure/.

Two independent output directories reproduced the frozen manifest, validation histories, model hashes, all non-timing event metrics and graph diagnostics. The first reused the checkpoints. The second started with no Phase II optimizer or model state and retrained all four Phase II models, retaining only the fixed Phase I transfer weights. Cold replay took 115.20 s wall time and reached 426016768 bytes maximum RSS. [Replay comparison](replay.json) · [Cold process measurements](cold-replay-process-time.txt)
