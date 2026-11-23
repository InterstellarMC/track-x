"""Official TrackML detector inspector for the frozen Phase II sample."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from trackx.barrel import directional_graph
from trackx.experiment import matching, predict
from trackx.official import geometry, load, sector_indices
from trackx.tracking import EdgeNet, assemble, fit
from trackx.triplets import hypotheses

st.set_page_config(page_title="TRACK-X | Detector lab", layout="wide")
st.title("Pixel barrel reconstruction")
out = Path("reports/phase2")
root = Path("data/trackml")
if not (root / "detectors.csv").exists() or not (out / "training.json").exists():
    st.info(
        "Prepare the official sample and run the Phase II experiment. See reports/phase2/REPRODUCIBILITY.md."
    )
    st.stop()
doc = json.loads((out / "manifest.json").read_text())
training = json.loads((out / "training.json").read_text())
eid = st.sidebar.selectbox("Held-out TrackML event", doc["split"]["test"])
st.caption(
    f"Official TrackML detector simulation · train_sample.zip · {eid} · run {training['config_id']}"
)
torch.set_num_threads(1)


@st.cache_resource(max_entries=1)
def event_data(event, manifest):
    return load(root, json.loads(manifest), event)


ev = event_data(eid, json.dumps(doc))
parts = list(sector_indices(ev.hits))
sector = st.sidebar.slider("Azimuth sector", 0, len(parts) - 1, 0)
ix, h = parts[sector]
pid = ev.truth.particle_id.to_numpy()[ix]
method = st.sidebar.selectbox(
    "Track assembly",
    ["Triplet + GNN", "Triplet geometry", "GNN edge gate", "Classical edge gate"],
)
net = EdgeNet(2, 32)
path = Path("experiments") / ("official-" + training["config_id"]) / "local_gnn.pt"
if not path.exists():
    st.info("Run uv run python -m trackx.phase2 to build the selected checkpoint.")
    st.stop()
net.load_state_dict(torch.load(path))
net.eval()
t = time.perf_counter()
g = directional_graph(h)
scores = predict(
    "classical" if method.startswith("Classical") else "gnn", {"gnn": net}, g, "cpu"
)
paths = (
    hypotheses(h, g[1].numpy(), scores if method == "Triplet + GNN" else None)
    if method.startswith("Triplet")
    else assemble(h, g[1].numpy(), scores, True)
)
ms = (time.perf_counter() - t) * 1000
eligible, q, dups = matching(paths, pid)
st.caption(
    f"{len(h)} sector hits · {g[1].shape[1]} directed edges · {len(paths)} paths · {ms:.1f} ms graph, score and assembly on CPU"
)
tab, detector, benchmark = st.tabs(
    ["Event inspection", "Detector geometry", "Frozen benchmarks"]
)
with tab:
    view = st.radio("Projection", ["3D", "XY", "RZ"], horizontal=True)
    color = st.selectbox(
        "Hit color", ["Layer", "Volume", "Predicted track", "Truth particle"]
    )
    assignments = np.zeros(len(h), dtype=int)
    for number, p in enumerate(paths, 1):
        assignments[p] = number
    colors = {
        "Layer": h[:, 3],
        "Volume": ev.hits.iloc[ix].volume_id,
        "Predicted track": assignments % 37,
        "Truth particle": np.unique(pid, return_inverse=True)[1] % 37,
    }
    fig = go.Figure()

    def trace(pts, mode, name, **kw):
        x, y, z = pts[:, :3].T
        if view == "3D":
            fig.add_scatter3d(x=x, y=y, z=z, mode=mode, name=name, **kw)
        else:
            fig.add_scatter(
                x=x if view == "XY" else z,
                y=y if view == "XY" else np.hypot(x, y),
                mode=mode,
                name=name,
                **kw,
            )

    trace(
        h,
        "markers",
        "Measured hits",
        marker={"size": 3, "color": colors[color], "colorscale": "Viridis"},
    )
    if paths:
        matched = {tuple(p) for _, p, *_ in q}
        indices = (
            [i for i, p in enumerate(paths) if tuple(p) not in matched]
            if st.checkbox("Inspect unmatched or duplicate paths")
            else list(range(len(paths)))
        )
        if indices:
            selected = st.selectbox("Candidate", indices)
            p = paths[selected]
            trace(
                h[p],
                "lines+markers",
                "Reconstructed",
                line={"color": "#ff8040", "width": 5},
            )
            majority, count = np.unique(pid[p], return_counts=True)
            particle = majority[np.argmax(count)]
            if particle:
                true = np.flatnonzero(pid == particle)
                trace(
                    h[true[np.argsort(h[true, 3])]],
                    "lines+markers",
                    "Majority truth particle",
                )
                fragments = sum(
                    np.count_nonzero(pid[t] == particle) >= 2 for t in paths
                )
                st.caption(
                    f"Majority truth particle {particle} · {fragments} reconstructed paths contain at least two of its hits in this sector"
                )
            _, _, dz = fit(h[p], 2.0)
            st.bar_chart(
                pd.DataFrame(
                    {"z fit residual (mm)": 1000 * dz}, index=h[p, 3].astype(int)
                )
            )
    kind = st.selectbox(
        "Candidate-edge errors", ["Hide", "False positives", "Missed positives"]
    )
    if kind != "Hide":
        i, j = g[1].numpy()
        positive = (pid[i] == pid[j]) & (pid[i] > 0)
        mask = (
            (scores >= 0.5) & ~positive
            if kind == "False positives"
            else (scores < 0.5) & positive
        )
        display = np.flatnonzero(mask)[:150]
        if len(display):
            segments = np.full((len(display), 3, 4), np.nan)
            segments[:, 0] = h[i[display]]
            segments[:, 1] = h[j[display]]
            trace(segments.reshape(-1, 4), "lines", kind, line={"color": "red"})
        st.caption("Overlay limited to 150 candidate edges at score threshold 0.5.")
    fig.update_layout(
        scene={
            "aspectmode": "data",
            "xaxis_title": "x (m)",
            "yaxis_title": "y (m)",
            "zaxis_title": "z (m)",
        },
        xaxis_title="x (m)" if view == "XY" else "z (m)",
        yaxis_title="y (m)" if view == "XY" else "r (m)",
    )
    st.plotly_chart(fig, width="stretch")
with detector:
    m = geometry(root / "detectors.csv")
    fig = go.Figure()
    for vol, df in m.groupby("volume_id"):
        fig.add_scattergl(
            x=df.cz,
            y=np.hypot(df.cx, df.cy),
            mode="markers",
            name=f"Volume {vol}",
            marker={"size": 2},
        )
    st.plotly_chart(
        fig.update_layout(xaxis_title="Module z (mm)", yaxis_title="Module r (mm)"),
        width="stretch",
    )
with benchmark:
    result = json.loads((out / (eid + ".json")).read_text())
    st.dataframe(
        pd.DataFrame(result["rows"])[
            [
                "builder",
                "method",
                "official_full_event",
                "barrel_efficiency",
                "fake_rate",
                "edge_ap",
            ]
        ],
        hide_index=True,
    )
    st.caption(
        "Official scores include every event hit. Reconstruction covers volume 8. The other metrics use the documented candidate or barrel denominator."
    )
