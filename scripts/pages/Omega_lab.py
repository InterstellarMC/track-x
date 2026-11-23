"""Inspect bounded seed alternatives, truth identity and full-event outcomes."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TRACK-X | Pixel seeds", layout="wide")
st.title("Pixel seed reconstruction")
root = Path("reports/omega")
files = sorted(root.glob("*/event*.json"))
if not files:
    st.info("Run the Omega validation benchmark to generate a seed atlas.")
    st.stop()
record = json.loads(
    st.selectbox(
        "Official simulated event",
        files,
        format_func=lambda f: f"{f.parent.name} · {f.stem}",
    ).read_text()
)
st.dataframe(
    [
        {
            "method": m,
            **{
                k: r[k]
                for k in (
                    "score",
                    "efficiency",
                    "seed_mixed",
                    "mixed",
                    "severe",
                    "retention",
                    "runtime",
                )
            },
        }
        for m, r in record["rows"].items()
    ]
)
atlas = json.loads((root / "validation/event000001002.json").read_text())["atlas"]
st.subheader("Alternative pixel measurements")
st.caption(
    "Historical validation event 1002 · first changed examples per method · truth used for inspection"
)
mode = st.selectbox("Seed method", sorted({x["mode"] for x in atlas}))
items = [x for x in atlas if x["mode"] == mode]
case = items[st.selectbox("Example", range(len(items)))]
fig = go.Figure()
for i, opt in enumerate(case["options"]):
    xyz = np.array(opt["xyz"])
    label = ("Original" if i == 0 else f"Alternative {i}") + (
        " · selected" if opt["selected"] else ""
    )
    fig.add_scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers+lines",
        name=label,
        text=[f"row {h} · particle {p}" for h, p in zip(opt["hit"], opt["truth"])],
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
st.dataframe(
    [
        {
            "hit rows": o["hit"],
            "max normalized residual": o["residual"],
            "selected": o["selected"],
            "signal IDs": len(set(o["truth"]) - {"0"}),
        }
        for o in case["options"]
    ]
)
