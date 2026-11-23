# Baseline audit

The checkout began on clean main at a67e798 on the 8 GiB Apple M1 with 6.4 GiB free. Locked native Python 3.12 and all 45 baseline tests passed (12 reference-scorer pandas deprecation warnings). The installed scorer is called by trackx.official.official_score with a complete hit submission. It uses the pinned TrackML library. The archived geometry hash is e01684d65c088794618b916655f989dae6f099de4ecc7a462987e97b4037639a. The official Phase II checkpoint group is experiments/official-3bcc0e660f314bd6. Exact package, file and tensor digests are in [the environment manifest](../phase4/environment.json).

Replayed seven methods on 1022–1026 into an empty directory. All scientific fields matched. Mean official scores were 0.221185 for geometry, 0.224457 for local EKF, 0.223658 for global EKF and 0.224455 for GNN-ranked local EKF. The original results include 39.46% mixed local EKF paths and 0.230% severe merges. Their denominators differ from matched-track purity. Phase I concerns a synthetic detector. Phase II covers four pixel layers. Phase III adds six barrel layers. Phase IV tests uncertainty-aware extension. Measurements in this audit concern Phase IV; earlier phase replays are archived. [Input hashes](../phase4/final-manifest.json) · [Original measures](../phase4/CONFIRMATION_RESULTS.md)

```sh
uv sync --locked --python 3.12
uv run pytest -q
d=$(mktemp -d)
uv run python scripts/phase4.py final --output "$d"
uv run python scripts/phase4_verify.py --stage final --replay "$d"
```

