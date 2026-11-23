# Local reproduction

From the repository root, use the locked Python 3.12 environment. The acquisition commands require access to the official Kaggle competition. Phase II transfer weights can be rebuilt using its published reproduction guide. Keep official CSVs and per-hit derived paths under the ignored data and experiments directories.

```sh
uv sync --locked --python 3.12
uv run pytest -q
uv run python scripts/phase3.py development
uv run python scripts/fetch_trackml.py --phase3
uv run python scripts/phase3.py confirmation
uv run python scripts/phase3_failures.py development
uv run python scripts/phase3_failures.py confirmation
uv run python scripts/phase3_quality.py
uv run python scripts/phase3_report.py
uv run --extra demo streamlit run scripts/pages/Multilayer_lab.py --server.address 127.0.0.1
```

Each event is an atomic checkpoint. The runner skips completed JSON rows. To force an independent replay or regenerate per-hit viewer data on a fresh clone, use a separate working directory containing links to the repository's configs, data and experiments directories, a reports/phase2 link to its released reports, and an empty reports/phase3 directory. Invoke the scripts by absolute path with the installed interpreter. Run development first to establish the layer inventory. For a clean release clone which already contains the published event summaries but lacks local paths, the runner should regenerate the missing path artifact.

The development loader verifies the Phase II manifest and the confirmation loader records event SHA-256 hashes and checks the official detector hash. All production timing was on ARM64 macOS, single-threaded PyTorch CPU with no accelerator. Per-method total excludes CSV loading and truth scoring. See the Phase II software manifest and uv.lock for versions. New confirmation events were preassigned by ID, independent of scores; these are simulated collider events.

## Verified release

The method and exploratory results were committed at 238f624 before the three new event files were requested. The fetcher read 27278669 bytes of the official sample ZIP and 179575 bytes of the detector ZIP, plus single-byte size probes. Raw files are local. [Frozen source and package versions](provenance.json) · [File hashes](manifest.json) · [Phase II prerequisite weights and data](../phase2/REPRODUCIBILITY.md).

A separate empty output directory reconstructed all three prospective events again. All 33 method/event records agreed exactly on every scientific field, including candidate opportunity counts and per-region quantities. The replay of eleven configurations over three events took 30.15 s wall time and peaked at 382124032 bytes RSS including evaluation. [Operating-system measurements](confirmation-replay-time.txt)

The complete local test suite passed 31 tests with zero skips. Twelve warnings came from the pinned official scorer using deprecated pandas inplace operations. Ruff lint and format checks passed. A headless Streamlit test exercised the mixed-particle filter and switched the visible volume list to both outer barrels. Interactive desktop inspection of every control was not performed.
