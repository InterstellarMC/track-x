"""Create a local, deterministic failure display without redistributing event hits."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trackx.barrel import directional_graph
from trackx.experiment import matching
from trackx.official import load, sector_indices
from trackx.triplets import hypotheses


def main():
    out = Path("reports/phase2")
    doc = json.loads((out / "manifest.json").read_text())
    eid = doc["split"]["test"][0]
    ev = load(Path("data/trackml"), doc, eid)
    paths = []
    sector_id = np.full(len(ev.hits), -1)
    for s, (ix, h) in enumerate(sector_indices(ev.hits)):
        sector_id[ix] = s
        g = directional_graph(h)
        paths.extend(ix[p] for p in hypotheses(h, g[1].numpy()))
    ids = np.where(ev.hits.volume_id == 8, ev.truth.particle_id, 0)
    eligible, quality, _ = matching(paths, ids)
    matched = {p for p, *_ in quality}
    # First missed particle in integer ID order crossing a sector boundary.
    pid = next(
        p for p in sorted(eligible - matched) if len(np.unique(sector_id[ids == p])) > 1
    )
    indices = np.flatnonzero(ids == pid)
    indices = indices[np.argsort(ev.hits.layer_id.to_numpy()[indices], kind="stable")]
    xyz = ev.hits[["x", "y", "z"]].to_numpy() / 1000
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {}]],
        subplot_titles=["3D position (m)", "Transverse projection (m)"],
    )

    def draw(ix, name, mode):
        x, y, z = xyz[ix].T
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, name=name, mode=mode), 1, 1)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode=mode, name=name, showlegend=False), 1, 2
        )

    draw(indices, "Missed truth trajectory", "lines+markers")
    overlapping = 0
    for p in paths:
        if np.any(ids[p] == pid):
            draw(p, f"Candidate {overlapping + 1}", "lines+markers")
            overlapping += 1
    fig.update_layout(
        title=f"Official TrackML simulation · {eid} · missed barrel particle {pid}",
        scene={"aspectmode": "data"},
    )
    target = Path("experiments/phase2-failure")
    target.mkdir(exist_ok=True, parents=True)
    fig.write_html(target / "event.html", include_plotlyjs="directory")
    print(
        eid,
        len(indices),
        "truth hits;",
        overlapping,
        "touching candidates;",
        target / "event.html",
    )


if __name__ == "__main__":
    main()
