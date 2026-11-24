# Resume TRACK-X Omega

Read PROJECT_STATE.md, EXPERIMENT_REGISTRY.md and RESEARCH_BACKLOG.md before changing reconstruction. Start at the repository root. The last experimental task was the prospective twelve-event comparison and a fresh-checkout replay. Confirmation labels are now spent. The current algorithm freeze is 0548c0f; the research package checkpoint is d3cc150. Use git log -1 for the final documentation revision.

```sh
cd /Users/manojkumarchaturvedi/track-x
git status --short --branch
git log -5 --oneline
uv sync --locked --python 3.12
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
replay=$(mktemp -d)
uv run python scripts/omega.py confirmation --output "$replay"
uv run python scripts/omega_review.py confirmation "$replay"
```

The existing laptop already holds the required CSVs in data/omega. A new machine must follow reports/omega/REPRODUCIBILITY.md to obtain the licensed inputs. Use an empty directory to replay; repeating the same command there resumes completed records. Twelve pandas deprecation warnings from the pinned scorer are expected in the data-provisioned suite. Use the complete twelve events for comparison.

The next independent task is external reproduction and review of module-overlap substitutions. If new research is desired, design a separate barrel-to-disk experiment, freeze the algorithm and a new cohort before inspecting its labels, and explicitly compare complete-event fake and duplicate fractions. No further run or external contact is scheduled.

The optional fast ranker was subsequently verified on three development events. The data-provisioned suite now contains 53 passing tests. The local release tip is identified by `git log -1 --oneline`.
