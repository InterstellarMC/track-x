"""Inventory all detector modules, layer occupancies and validation counts."""

import json
from pathlib import Path

import numpy as np
from plotly.subplots import make_subplots

from trackx.official import KEY, geometry, load
from trackx.phase2 import save


def main():
    out = Path("reports/phase2")
    root = Path("data/trackml")
    doc = json.loads((out / "manifest.json").read_text())
    modules = geometry(root / "detectors.csv")
    modules["r_mm"] = np.hypot(modules.cx, modules.cy)
    layers = (
        modules.groupby(KEY[:2])
        .agg(
            modules=("module_id", "size"),
            r_min_mm=("r_mm", "min"),
            r_max_mm=("r_mm", "max"),
            z_min_mm=("cz", "min"),
            z_max_mm=("cz", "max"),
        )
        .reset_index()
    )
    events = []
    occupancy = []
    for eid in doc["events"]:
        ev = load(root, doc, eid)
        h = ev.hits
        group = (
            h.groupby(KEY[:2])
            .agg(hits=("hit_id", "size"), active_modules=("module_id", "nunique"))
            .reset_index()
        )
        group["event"] = eid
        occupancy.extend(group.to_dict("records"))
        barrel = h.volume_id == 8
        t = ev.truth.loc[barrel]
        signal = t.particle_id > 0
        l = h.loc[barrel, "layer_id"]
        grouped = l[signal].groupby(t.particle_id[signal])
        unique = grouped.nunique()
        events.append(
            {
                "event": eid,
                "hits": len(h),
                "cells": ev.cells,
                "particles": len(ev.particles),
                "weight_sum": float(ev.truth.weight.sum()),
                "volume8_hits": int(barrel.sum()),
                "noise_hits": int((~signal).sum()),
                "barrel_particles": len(unique),
                "repeated_layer_particles": int((grouped.size() > unique).sum()),
                "particles_lt3_distinct_layers": int((unique < 3).sum()),
            }
        )
    save(
        out / "geometry.json",
        {
            "modules": len(modules),
            "layers": layers.to_dict("records"),
            "occupancy": occupancy,
            "events": events,
        },
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Module centres", "Observed hits by layer (all six events)"],
    )
    for v, g in modules.groupby("volume_id"):
        fig.add_scattergl(
            x=g.cz,
            y=g.r_mm,
            mode="markers",
            marker={"size": 2},
            name=f"Volume {v}",
            row=1,
            col=1,
        )
    import pandas as pd

    occ = pd.DataFrame(occupancy).groupby(KEY[:2]).hits.sum()
    fig.add_bar(
        x=[f"{v}/{l}" for v, l in occ.index],
        y=occ.values,
        showlegend=False,
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="z (mm)", row=1, col=1)
    fig.update_yaxes(title_text="r (mm)", row=1, col=1)
    fig.update_xaxes(title_text="Volume / layer", row=1, col=2)
    fig.update_yaxes(title_text="Hits", row=1, col=2)
    fig.write_html(out / "geometry.html", include_plotlyjs="../plotly.min.js")


if __name__ == "__main__":
    main()
