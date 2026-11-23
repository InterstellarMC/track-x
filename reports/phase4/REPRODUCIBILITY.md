# Reproduction on Apple silicon

The experiment uses official TrackML simulated events, SHA-256 per table, the same detector geometry and the reference scorer pinned in uv.lock. The data source, competition acceptance and credentials are documented in data/README.md. Geometry-only methods use no trained parameters. M6 uses the Phase II pixel ranking checkpoint and CPU. Event IDs 1000–1001 trained that model. Event 1002 supplies validation and 1003–1005 are historical development. Events 1006–1011 were previously examined. Commit 993b8e1 fixed the two Phase IV cohorts. Commit 4cdccf3 fixed the estimator, global selection, validation winner and thresholds before fetching 1022–1026.

Run from the repository root. Each data command is bounded and validates existing files. To obtain the ignored Phase II weights on a clean checkout, first run the Phase II training/evaluation command below with the original six-event sample. The release contains the original checkpoint hashes in the Phase II training record.

```sh
uv sync --locked --python 3.12
uv run python scripts/fetch_trackml.py
uv run python -m trackx train --config configs/mac_m1.yaml
uv run python -m trackx.phase2 --output /tmp/trackx-phase2-replay
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
for group in gap-a gap-b final; do
  uv run python scripts/fetch_trackml.py --phase4 "$group"
done
replay=$(mktemp -d)
for stage in gap-a gap-b development validation final; do
  uv run python scripts/phase4.py "$stage" --output "$replay"
done
uv run python scripts/phase4_verify.py --replay "$replay" --receipt "$replay/verification.json"
uv run python scripts/phase4_calibration.py
uv run python scripts/phase4_inspect.py
uv run python scripts/phase4_report.py
uv run python scripts/phase4_regions.py
uv run python scripts/phase4_paper.py
uv run --extra demo streamlit run scripts/demo.py
```

With the default output directory, completed event files are skipped. Use an empty replay directory to force fresh computation. Reports read the released event records in reports/phase4. To regenerate reports for different timing measurements, use a separate checkout with the new event records. Raw simulation, access tokens and virtual environments are excluded from Git.

Paired bootstrap uses 10,000 equal-event resamples with replacement and NumPy seed 1729. Inference uses deterministic sorting and one Torch CPU thread. The model and detector are fixed across comparisons. Inference wall estimates share the same measured seed/index cost across local geometry methods within an event. CPU time covers both seed passes and extension. Per-event high-water RSS includes all loaded libraries, data validation and scoring. Tests for statistical consistency use independently generated draws under the declared measurement model. Process times and RSS depend on local load.

## Release verification

All 19 event records were recomputed in an empty output directory. Every released scientific field matched exactly in all rows, including innovation samples and contamination diagnostics. Timing and RSS were independently measured. The five final-event replay means were 1.225 s/event for M0 and 1.267 s/event for M3. The comparison receipt lists each event. The full suite passed 45 tests with 12 upstream pandas deprecation warnings, Ruff lint and formatting passed, and the Streamlit test harness rendered the gate and selected resolved path. [Replay receipt](replay.json) · [Environment and checkpoint digests](environment.json)
