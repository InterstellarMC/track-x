"""Inspect locally reconstructed multi-volume paths and their truth diagnostics."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from trackx.extension import project, trajectory
from trackx.official import geometry, load

st.set_page_config(page_title="TRACK-X | Barrel extension", layout="wide")
st.title("Barrel track extension")
out = Path("reports/phase3")
events = [
    p.stem
    for p in sorted(out.glob("event*.json"))
    if (Path("experiments/phase3") / p.name).exists()
]
if not events:
    st.info("Run scripts/phase3.py development to generate local tracks.")
    st.stop()
eid = st.selectbox("Official TrackML simulated event", events)
metrics = json.loads((out / (eid + ".json")).read_text())
confirmation = metrics["stage"] == "confirmation"
root = Path("data/trackml-phase3" if confirmation else "data/trackml")
manifest = (
    out / "manifest.json" if confirmation else Path("reports/phase2/manifest.json")
)


@st.cache_resource(max_entries=1)
def event(eid, root, manifest):
    return load(root, json.loads(manifest.read_text()), eid)


ev = event(eid, root, manifest)
rec = json.loads((Path("experiments/phase3") / (eid + ".json")).read_text())
ids = ev.truth.particle_id.to_numpy()
vol = ev.hits.volume_id.to_numpy()
xyz = ev.hits[["x", "y", "z"]].to_numpy() / 1000
from trackx.experiment import matching

eligible, q, _ = matching(rec["paths"], ids)
first = {tuple(p) for _, p, *_ in q}
counts = dict(zip(*np.unique(ids, return_counts=True)))


def duplicate(p):
    particles, n = np.unique(ids[p], return_counts=True)
    k = n.argmax()
    return (
        tuple(p) not in first
        and particles[k] in eligible
        and n[k] > len(p) / 2
        and n[k] >= counts[particles[k]] / 2
    )


filter_by = st.selectbox(
    "Inspect", ["Extended", "Mixed signal particles", "Unextended", "Duplicate"]
)
paths = [
    i
    for i, (s, p) in enumerate(zip(rec["seed"], rec["paths"]))
    if {
        "Extended": len(p) > len(s),
        "Mixed signal particles": len(set(ids[p]) - {0}) > 1,
        "Unextended": len(p) == len(s),
        "Duplicate": duplicate(p),
    }[filter_by]
]
st.caption(f"{len(paths)} candidates · {metrics['stage']} · frozen extended method")
if not paths:
    st.stop()
i = st.selectbox("Candidate index", paths)
s, p = rec["seed"][i], rec["paths"][i]
volumes = st.multiselect("Visible volumes", [8, 13, 17], default=[8, 13, 17])
fig = go.Figure()
mods = geometry(root / "detectors.csv")
for v in volumes:
    df = mods[mods.volume_id == v]
    fig.add_scatter3d(
        x=df.cx / 1000,
        y=df.cy / 1000,
        z=df.cz / 1000,
        mode="markers",
        marker={"size": 1},
        opacity=0.1,
        name=f"Volume {v}",
    )
for label, seq in [("Extended trajectory", p), ("Frozen seed", s)]:
    seq = [j for j in seq if vol[j] in volumes]
    if seq:
        x, y, z = xyz[seq].T
        fig.add_scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines+markers",
            name=label,
            text=[f"hit {ev.hits.hit_id.iloc[j]} · particle {ids[j]}" for j in seq],
        )
fig.update_layout(
    scene={
        "aspectmode": "data",
        "xaxis_title": "x (m)",
        "yaxis_title": "y (m)",
        "zaxis_title": "z (m)",
    }
)
st.plotly_chart(fig)
r = np.hypot(xyz[p, 0], xyz[p, 1])
params = trajectory(xyz[p][None], np.ones((1, len(p)), bool))
_, pred, _ = project(params, r)
st.bar_chart({"z residual (mm)": 1000 * (xyz[p, 2] - pred)})
st.dataframe(metrics["methods"]["extended"]["transition_counts"])
st.caption(
    "Eligible and offered count seed-to-surface opportunities with same-particle hits. The table covers the complete event."
)
st.dataframe(
    [
        {
            "method": k,
            "official_score": v["score"],
            "mixed": v["mixed"],
            "seconds": v["total_s"],
        }
        for k, v in metrics["methods"].items()
    ]
)
