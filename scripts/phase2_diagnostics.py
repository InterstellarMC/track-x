"""Attribute true-pair losses with exhaustive small per-particle truth groups."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from trackx.core import wrap
from trackx.official import load, sector_indices
from trackx.phase2 import BUILDERS, save


def main():
    out = Path("reports/phase2")
    doc = json.loads((out / "manifest.json").read_text())
    rows = []
    for eid in doc["split"]["test"]:
        ev = load(Path("data/trackml"), doc, eid)
        for name, build in BUILDERS.items():
            local = defaultdict(list)
            sectors = {}
            retained = set()
            radii = {}
            totals = Counter()
            occupancy = defaultdict(Counter)
            for s, (ix, h) in enumerate(sector_indices(ev.hits)):
                n = len(h)
                r = np.hypot(h[:, 0], h[:, 1])
                p = np.arctan2(h[:, 1], h[:, 0])
                ys = ev.truth.particle_id.to_numpy()[ix]
                for vi, li in enumerate(h[:, 3].astype(int)):
                    if ys[vi]:
                        local[ys[vi]].append(
                            (int(ix[vi]), li, s, r[vi], p[vi], h[vi, 2])
                        )
                for li in np.unique(h[:, 3]):
                    radii[s, int(li)] = np.median(r[h[:, 3] == li])
                sectors[s] = n
                i, j = build(h)[1].numpy()
                true = (ys[i] > 0) & (ys[i] == ys[j])
                retained.update(zip(ix[i[true]].tolist(), ix[j[true]].tolist()))
            for group in local.values():
                for a, l, s, r, phi, z in group:
                    for b, m, t, ro, p2, z2 in group:
                        if not 1 <= m - l <= 2:
                            continue
                        dr = ro - r
                        delta = abs(wrap(p2 - phi))
                        if (a, b) in retained:
                            reason = "kept"
                        elif s != t:
                            reason = "boundary"
                        elif dr <= 0:
                            reason = "inward"
                        elif name == "original":
                            if 2 * np.sin(delta / 2) > 0.23:
                                reason = "azimuth"
                            elif abs(z2 - z) >= 2 * dr + 0.003:
                                reason = "longitudinal"
                            else:
                                reason = "neighbor_cap"
                        else:
                            ri = radii[s, l]
                            rt = radii[s, m]
                            distance = (
                                2 * np.sin(delta / 2) / (0.025 + 1.5 * abs(rt - ri))
                            ) ** 2 + (
                                (z2 / ro - z / r)
                                / (0.05 + 0.065 * abs(1 / ri - 1 / rt))
                            ) ** 2
                            reason = (
                                "direction_window" if distance >= 4 else "neighbor_cap"
                            )
                        totals[reason] += 1
                        occupancy[sectors[s] // 250 * 250][reason] += 1
            recorded = json.loads((out / (eid + ".json")).read_text())
            g = next(g for g in recorded["graphs"] if g["builder"] == name)
            assert sum(totals.values()) == g["truth_pairs"] and totals["kept"] == len(
                retained
            )
            rows.append(
                {
                    "event": eid,
                    "builder": name,
                    "counts": dict(totals),
                    "by_source_sector_occupancy": {
                        k: dict(v) for k, v in occupancy.items()
                    },
                }
            )
    save(out / "candidate-losses.json", rows)


if __name__ == "__main__":
    main()
