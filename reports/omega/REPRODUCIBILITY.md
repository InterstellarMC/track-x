# Reproduce Omega

Use native Python 3.12 on Apple silicon from the repository root. uv.lock pins the numerical stack and the organizers’ scorer. The verification system used the same Apple M1 and 8 GiB machine as Phase IV. All reconstruction runs used CPU, one Torch thread, sparse bounded neighborhoods and event-level atomic checkpoints. A GPU and model checkpoint are unnecessary for the Omega comparison. Phase IV GNN replay uses the archived Phase II weights. Exact software versions are in [the environment record](environment.json).

```sh
uv sync --locked --python 3.12
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/fetch_trackml.py
uv run python scripts/omega_fetch.py
replay=$(mktemp -d)
for stage in development validation confirmation; do
    uv run python scripts/omega.py "$stage" --output "$replay/$stage"
    uv run python scripts/omega_review.py "$stage" "$replay/$stage"
done
uv run python scripts/omega_select.py
uv run python scripts/omega_fits.py
uv run python scripts/omega_disks.py
uv run python scripts/omega_profile.py
uv run python scripts/omega_diagnose.py
uv run python scripts/omega_report.py
uv run --no-project --with matplotlib==3.10.8 --python 3.12 python scripts/omega_figures.py
uv run python scripts/omega_manuscript.py
uv run --extra demo streamlit run scripts/pages/Omega_lab.py --server.address 127.0.0.1
```

Accept the official competition rules before attempting acquisition. Use a local Kaggle credential as described in data/README.md. The twelve-event fetcher uses authenticated ZIP ranges, per-member bounds, archive CRC and SHA-256, and checks disk headroom. On interrupted transfers rerun it. Existing CSVs are validated before reuse. Do not edit the frozen cohort or selection. Data files remain outside version control.

A one-event historical smoke run is `uv run python scripts/omega.py smoke --output /tmp/trackx-omega-smoke`. The local tier uses the four historical events and a bounded confirmation sample. The existing extended configuration remains available for synthetic scaling. An expanded real-data run needs its own prespecified event cohort and output directory. Named phases are benchmark workloads, with no implicit downloads. The default output resumes existing files after comparing provenance, dimensions of the method roster and input identifiers. Use an empty directory for scientific replication. Timing fields vary. The verifier compares the entire recorded scientific payload, including diagnostic strata and inspection examples.

Reports are regenerated from the released per-event JSON. Summary medians and bootstrap intervals operate on paired per-event differences. Fresh validation/confirmation labels should never be reused to propose threshold changes. Each input file and geometry table has a SHA-256 in its manifest. Per-event records include source digest, HEAD at execution, device, installed versions, random seed, method IDs, timings and peak process RSS.
