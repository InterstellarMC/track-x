"""Local event inspector. Launch from the project root."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

from trackx.core import event, graph, labels
from trackx.experiment import folder, models, predict, splits
from trackx.tracking import assemble, fit
from trackx.trackml import load

st.set_page_config(page_title="TRACK-X | Event inspector", layout="wide")
st.title("TRACK-X event inspector")
c = yaml.safe_load(Path("configs/smoke.yaml").read_text())
torch.set_num_threads(1)
root = folder(c)
if not (root / "gnn.pt").exists():
    st.info("Run uv run python -m trackx smoke to prepare models.")
    st.stop()
source = st.sidebar.radio("Dataset", ["Synthetic barrel", "TrackML sector"])
if source == "Synthetic barrel":
    number = st.sidebar.selectbox("Held-out event", splits(c)["test"].tolist())
    h, y, _ = event(c, number)
else:
    prefix = st.sidebar.text_input("Local event prefix")
    if not prefix:
        st.stop()
    try:
        h, y = load(prefix)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        st.stop()
st.caption(
    f"{source} · synthetic-trained checkpoint {root.name} · seed {c['seed']} · CPU"
)
method = st.sidebar.selectbox(
    "Reconstruction", ["hybrid", "gnn", "mlp", "classical", "truth"]
)
g = graph(h, c.get("graph"))
i, j = g[1].numpy()
if method == "truth":
    paths = [np.flatnonzero(y == p) for p in np.unique(y[y > 0])]
else:
    scores = predict(method, models(c, "cpu"), g, "cpu")
    paths = assemble(h, g[1].numpy(), scores, method in ("classical", "hybrid"))
    target = labels(y, g[1]).numpy().astype(bool)
    predicted = scores >= 0.5
    st.caption(f"{len(paths)} reconstructed tracks · {len(scores)} candidate edges")
fig = go.Figure(
    go.Scatter3d(
        x=h[:, 0],
        y=h[:, 1],
        z=h[:, 2],
        mode="markers",
        marker={"size": 2, "color": "#8292a4"},
        name="Hits",
    )
)
for idx, path in enumerate(paths):
    pts = h[sorted(path, key=lambda k: h[k, 3])]
    fig.add_scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines", name=f"Track {idx}"
    )
if method != "truth":
    for title, mask, color in [
        ("False positive", predicted & ~target, "#ff6243"),
        ("Missed positive", ~predicted & target, "#4b98ff"),
    ]:
        if st.sidebar.checkbox(title + " edges"):
            xyz = []
            for a, b in zip(i[mask][:500], j[mask][:500]):
                xyz.extend([h[a, :3], h[b, :3], [np.nan] * 3])
            if xyz:
                x, yc, z = np.asarray(xyz).T
                fig.add_scatter3d(
                    x=x,
                    y=yc,
                    z=z,
                    mode="lines",
                    line={"color": color, "width": 4},
                    name=title,
                )
    st.sidebar.caption("Error overlay shows up to 500 candidate edges per class.")
fig.update_layout(
    showlegend=False,
    height=500,
    scene={
        "aspectmode": "data",
        "xaxis_title": "x (m)",
        "yaxis_title": "y (m)",
        "zaxis_title": "z (m)",
    },
)
st.plotly_chart(fig, width="stretch")
left, right = st.columns(2)
eligible = [idx for idx, path in enumerate(paths) if len(path) >= 3]
if eligible:
    selected = left.selectbox("Track fit", eligible)
    pts = h[sorted(paths[selected], key=lambda k: h[k, 3])]
    curvature, pt, dz = fit(pts, c["field"])
    left.plotly_chart(
        go.Figure(go.Scatter(x=pts[:, 3], y=1000 * dz)).update_layout(
            xaxis_title="Layer", yaxis_title="Longitudinal residual (mm)"
        ),
        width="stretch",
    )
    if source == "Synthetic barrel":
        left.caption(f"Curvature {curvature:.3f} /m · pT {pt:.3f} GeV/c")
if (root / "metrics.json").exists():
    records = json.loads((root / "metrics.json").read_text())["records"]
    methods = ["classical", "mlp", "gnn", "hybrid"]
    chart = go.Figure()
    for metric in ("efficiency", "fake"):
        chart.add_bar(
            name=metric,
            x=methods,
            y=[
                np.mean(
                    [
                        r[metric]
                        for r in records
                        if r["method"] == m and r[metric] is not None
                    ]
                )
                for m in methods
            ],
        )
    right.plotly_chart(
        chart.update_layout(title="Synthetic smoke test split", yaxis_range=[0, 1]),
        width="stretch",
    )
