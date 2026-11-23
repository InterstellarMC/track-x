"""Display archived prospective full-event measurements and real EKF innovations."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TRACK-X | Precision tracking", layout="wide")
st.title("Precision tracking")
root = Path("reports/phase4")
files = sorted(root.glob("event*.json"))
if not files:
    st.info("Run a Phase IV event to inspect its results.")
    st.stop()
data = json.loads(
    st.selectbox(
        "Official TrackML simulated event", files, format_func=lambda p: p.stem
    ).read_text()
)
rows = data["rows"]
st.caption(f"{data['stage']} · {data['hits']:,} hits · complete-event scoring")
st.dataframe(
    [
        {
            "method": m,
            **{
                k: r[k]
                for k in (
                    "score",
                    "efficiency",
                    "mixed",
                    "severe",
                    "fragmented",
                    "duplicate",
                    "wall_s",
                )
            },
        }
        for m, r in rows.items()
    ]
)
available = [m for m, r in rows.items() if r["diagnostics"].get("sample")]
if not available:
    st.info("The no-gap control records event metrics only.")
    st.stop()
mode = st.selectbox(
    "Trajectory estimator",
    available,
    index=available.index("M3") if "M3" in available else 0,
)
d = rows[mode]["diagnostics"]
samples = d["sample"]
surface = st.selectbox(
    "Candidate measurement surface",
    sorted({(s["volume"], s["layer"]) for s in samples}),
)
measurements = [s for s in samples if (s["volume"], s["layer"]) == surface]
s = st.selectbox("Hit row index", measurements, format_func=lambda x: str(x["hit"]))
cov = np.asarray(s["innovation_cov"])
theta = np.linspace(0, 2 * np.pi, 160)
threshold = 4 if mode == "tight" else 9
ellipse = (
    np.sqrt(threshold)
    * np.linalg.cholesky(cov)
    @ np.array([np.cos(theta), np.sin(theta)])
)
fig = go.Figure()
fig.add_scatter(
    x=ellipse[0] * s["radius"] * 1000,
    y=ellipse[1] * 1000,
    name="Innovation gate",
    mode="lines",
)
fig.add_scatter(x=[0], y=[0], name="Prediction", mode="markers")
fig.add_scatter(
    x=[s["residual"][0] * s["radius"] * 1000],
    y=[s["residual"][1] * 1000],
    name="Observed hit",
    mode="markers",
)
fig.update_layout(
    xaxis_title="Tangential innovation (mm)",
    yaxis_title="Longitudinal innovation (mm)",
    yaxis_scaleanchor="x",
)
st.plotly_chart(fig)
st.caption(
    f"Seed {s['seed']} · predicted φ {s['phi']:.5f} rad · z {s['z']:.5f} m · innovation χ² {s['chi2']:.3f} · local assignment {s['accepted']} · final assignment {s.get('final_accepted', s['accepted'])}"
)
if mode in ("M4", "fixed"):
    st.caption(
        "This method uses the fixed geometric gate. The ellipse illustrates the covariance model for comparison."
    )
st.dataframe(measurements)

atlas_file = root / "validation-atlas.json"
if atlas_file.exists():
    atlas = json.loads(atlas_file.read_text())
    st.subheader("Validation path inspection")
    st.caption(
        f"{atlas['event']} · first eight examples per diagnostic · coordinates in metres"
    )
    kind = st.selectbox("Diagnostic", list(atlas["cases"]))
    case = st.selectbox(
        "Seed index", atlas["cases"][kind], format_func=lambda c: str(c["seed_index"])
    )
    view = st.radio(
        "Assignment", ["seed", "local", "independent", "resolved"], horizontal=True
    )
    path = case[view]
    hits = [case["hits"][str(i)] for i in path]
    xyz = np.array([h["xyz"] for h in hits])
    track = go.Figure(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="lines+markers",
            text=[
                f"volume {h['volume']} · layer {h['layer']} · truth {h['particle_id']}"
                for h in hits
            ],
        )
    )
    track.update_layout(scene_aspectmode="data")
    st.plotly_chart(track)
    st.dataframe([{"hit row": i, **h} for i, h in zip(path, hits)])
