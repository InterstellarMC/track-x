"""Post-fit residuals and purity on frozen seed/extension pairs."""

import json
from pathlib import Path

import numpy as np

from trackx.core import wrap
from trackx.extension import project, trajectory
from trackx.official import load
from trackx.phase2 import save


def summarize(xyz, pids, paths):
    # Length-grouped arrays avoid padded observations in residual histograms.
    records = []
    for length in sorted(set(map(len, paths))):
        ix = np.array([p for p in paths if len(p) == length])
        pts = xyz[ix]
        mask = np.ones(ix.shape, bool)
        params = trajectory(pts, mask)
        radius = np.hypot(pts[:, :, 0], pts[:, :, 1])
        phi, z, valid = project(tuple(v[:, None] for v in params), radius)
        dphi = wrap(np.arctan2(pts[:, :, 1], pts[:, :, 0]) - phi)
        rms = 1000 * np.sqrt(
            np.mean((radius * dphi) ** 2 + (pts[:, :, 2] - z) ** 2, axis=1)
        )
        for seq, error, ok in zip(ix, rms, valid.all(1), strict=True):
            signal = pids[seq]
            _, count = np.unique(signal[signal > 0], return_counts=True)
            records.append(
                (np.count_nonzero(count >= 2) >= 2, len(count) > 1, error, bool(ok))
            )
    major, mixed, rms, valid = np.asarray(records).T
    return {
        "paths": len(paths),
        "two_particle_merge_fraction": float(major.mean()),
        "any_contamination_fraction": float(mixed.mean()),
        "rms_mm_quantiles": np.quantile(rms, [0.5, 0.9, 0.99]).tolist(),
        "fraction_rms_over_5mm": float(np.mean(rms > 5)),
        "reachable_fit_fraction": float(valid.mean()),
    }


def main():
    out = Path("reports/phase3")
    rows = []
    for file in sorted(out.glob("event*.json")):
        bench = json.loads(file.read_text())
        final = bench["stage"] == "confirmation"
        manifest = (
            out / "manifest.json" if final else Path("reports/phase2/manifest.json")
        )
        root = Path("data/trackml-phase3" if final else "data/trackml")
        ev = load(root, json.loads(manifest.read_text()), bench["event"])
        rec = json.loads((Path("experiments/phase3") / file.name).read_text())
        xyz = ev.hits[["x", "y", "z"]].to_numpy() / 1000
        for key in ("seed", "paths"):
            rows.append(
                dict(
                    event=bench["event"],
                    stage=bench["stage"],
                    method=key,
                    **summarize(xyz, ev.truth.particle_id.to_numpy(), rec[key]),
                )
            )
    save(out / "fit-quality.json", rows)


if __name__ == "__main__":
    main()
