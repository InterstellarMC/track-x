"""Compare two fitters on identical matched classical candidates."""

import json
from pathlib import Path

import numpy as np
import yaml

from trackx.core import event, graph
from trackx.experiment import matching, predict, splits
from trackx.robust_fit import fit_helix
from trackx.tracking import assemble, fit

c = yaml.safe_load(Path("configs/mac_m1.yaml").read_text())
records = []
for n in splits(c)["test"]:
    h, y, pars = event(c, n)
    g = graph(h)
    tracks = assemble(h, g[1].numpy(), predict("classical", {}, g, "cpu"), True)
    for pid, path, _, _ in matching(tracks, y)[1]:
        pt, q = pars[pid]
        target = -0.299792458 * q * c["field"] / pt
        algebraic = fit(h[path], c["field"])[0]
        par, residual = fit_helix(h[path])
        records.append(
            {
                "event": int(n),
                "pid": int(pid),
                "algebraic_error": algebraic - target,
                "robust_error": par[1] - target,
                "robust_rms_m": float(np.sqrt(np.mean(residual**2))),
            }
        )
Path("reports/fit-study.json").write_text(json.dumps(records, indent=2))
for name in ("algebraic", "robust"):
    print(
        name,
        len(records),
        "median absolute curvature error /m",
        np.median([abs(r[name + "_error"]) for r in records]),
    )
