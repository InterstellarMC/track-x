"""Check module orientation and hit placement in the published local frame."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trackx.official import KEY, load
from trackx.phase2 import save


def main():
    out = Path("reports/phase2")
    doc = json.loads((out / "manifest.json").read_text())
    root = Path("data/trackml")
    m = pd.read_csv(root / "detectors.csv")
    rotation = [f"rot_{lab}{local}" for lab in "xyz" for local in "uvw"]
    axes = m[rotation].to_numpy().reshape(-1, 3, 3)
    error = np.max(abs(axes.transpose(0, 2, 1) @ axes - np.eye(3)), axis=(1, 2))
    det = np.linalg.det(axes)
    data = []
    for eid in doc["events"]:
        ev = load(root, doc, eid)
        h = ev.hits.merge(m, on=KEY, validate="many_to_one")
        delta = h[list("xyz")].to_numpy() - h[["cx", "cy", "cz"]].to_numpy()
        u, v, w = np.einsum(
            "ni,nij->nj", delta, h[rotation].to_numpy().reshape(-1, 3, 3)
        ).T
        # Trapezoidal half-width interpolated between the two v endpoints.
        half_u = h.module_minhu + (h.module_maxhu - h.module_minhu) * (
            0.5 + 0.5 * v / h.module_hv
        )
        for volume in sorted(h.volume_id.unique()):
            mask = h.volume_id == volume
            excess = np.maximum(abs(u) - half_u, abs(v) - h.module_hv)
            data.append(
                {
                    "event": eid,
                    "volume": int(volume),
                    "hits": int(mask.sum()),
                    "normal_abs_mm_quantiles": np.quantile(
                        abs(w[mask]), [0.5, 0.99, 1]
                    ).tolist(),
                    "in_plane_excess_over_1mm": int((excess[mask] > 1).sum()),
                }
            )
    save(
        out / "surfaces.json",
        {
            "orthogonality_max": float(error.max()),
            "determinant_range": [float(det.min()), float(det.max())],
            "records": data,
        },
    )


if __name__ == "__main__":
    main()
