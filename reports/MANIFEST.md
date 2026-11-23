# Reproducibility manifest

Verified locally on 2026-11-23. Apple M1, eight physical CPU cores, 8,589,934,592 bytes installed RAM. Native ARM64 macOS 27.0.1 build 26A434. CPython 3.12.14, uv 0.12.15. PyTorch 2.15.0, NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, Plotly 7.1.0, PyYAML 6.0.4. Optional Streamlit 1.67.0. Full transitive versions and package hashes are in [uv.lock](../uv.lock).

The nine nominal-, stress- and pileup- JSON files in this directory record exact configuration, training configuration, seeds, split IDs, test-array SHA-256, raw data footprint, platform and event-by-method measurements. For mac_m1, event IDs 0–59 train, 60–69 validate and 70–84 test. For smoke, 0–11 train, 12–14 validate and 15–18 test. Track identity is event-local. Seed 1729 is the default; the suite also uses 1730 and 1731. Detector data are generated locally without external models or datasets.

## Executed reproduction commands

From a local clone of the implementation commit df9e6cf, with empty experiments and a fresh project environment:

```sh
uv sync --locked --python 3.12
/usr/bin/time -l .venv/bin/python -m trackx smoke
/usr/bin/time -l .venv/bin/python -m trackx train --config configs/mac_m1.yaml
.venv/bin/python -m trackx evaluate --config configs/mac_m1.yaml
```

| Command | Wall seconds (macOS time) | Peak RSS bytes (macOS time) | CLI elapsed seconds |
|---|---:|---:|---:|
| smoke, CPU, fresh environment | 39.07 | 390987776 | 5.319 |
| train, mac_m1, CPU, fresh checkpoints | 12.14 | 374538240 | 9.392 |

CLI timing starts after Python imports, config loading and device selection. Smoke wall time includes first imports in a newly installed environment. The fresh nominal evaluation reproduced all 60 non-timing metric rows and the test digest from nominal-1729.json. On this configuration the MLP and GNN best validation BCE values were 0.12610 and 0.10175.

From the working repository:

```sh
uv run python -m trackx generate --config configs/smoke.yaml
uv run python -m trackx visualize --config configs/smoke.yaml
uv run python scripts/suite.py
uv run python scripts/suite.py --report-only
uv run python scripts/fit_study.py
uv run python scripts/graph_audit.py
uv run python scripts/devices.py
uv run python -m trackx smoke --config configs/smoke_mps.yaml
uv run pytest -q
uv run ruff check .
uv run --extra demo python -c "from streamlit.testing.v1 import AppTest; a=AppTest.from_file('scripts/demo.py').run(timeout=60); assert not a.exception"
```

CPU smoke generated the NPZ, trained both models, evaluated 16 event-method rows and wrote the interactive HTML. MPS smoke completed both training loops and evaluation; evaluation-stage peak process RSS was 620478464 bytes. The model-scoring microbenchmark includes device transfers and synchronization. [Device measurements](devices.json)

Tests passed on CPU and MPS, 14 passed with zero skipped. Ruff passed. Streamlit's initial synthetic inspector passed its headless application check. TrackML schema fixtures passed. Official TrackML file verification, extended-scale timing, non-Apple GPU runs and browser interaction testing of every viewer control: **NOT RUN**.

Timing reflects local machine load and filesystem caching. Exact seed replay on this installed stack is verified for the repeated nominal run; cross-platform floating-point equivalence remains untested.
