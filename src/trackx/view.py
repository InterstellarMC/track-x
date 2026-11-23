import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .core import event, graph
from .experiment import folder, models, predict, splits
from .tracking import assemble, fit


def render(c, dev):
    number = splits(c)["test"][0]
    h, y, _ = event(c, number)
    g = graph(h, c.get("graph"))
    nets = models(c, dev)
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {}], [{}, {}]],
        subplot_titles=[
            "3D event (m)",
            "XY (m)",
            "XZ (m)",
            "Longitudinal fit residual (mm)",
        ],
    )
    for name in ("truth", "classical", "mlp", "gnn", "hybrid"):
        if name == "truth":
            paths = [np.flatnonzero(y == p) for p in np.unique(y[y > 0])]
        else:
            scores = predict(name, nets, g, dev)
            paths = assemble(h, g[1].numpy(), scores, name in ("classical", "hybrid"))

        def add(trace, row, col, name=name):
            trace.update(legendgroup=name, visible=name == "truth", showlegend=False)
            fig.add_trace(trace, row=row, col=col)

        for path in paths:
            pts = h[sorted(path, key=lambda i: h[i, 3])]
            x, yc, z, _ = pts.T
            add(go.Scatter3d(x=x, y=yc, z=z, mode="lines"), 1, 1)
            add(go.Scatter(x=x, y=yc, mode="lines"), 1, 2)
            add(go.Scatter(x=x, y=z, mode="lines"), 2, 1)
            if len(path) >= 3:
                add(
                    go.Scatter(
                        x=pts[:, 3],
                        y=1000 * fit(pts, c["field"])[2],
                        mode="lines+markers",
                    ),
                    2,
                    2,
                )
    fig.add_trace(
        go.Scatter3d(
            x=h[:, 0],
            y=h[:, 1],
            z=h[:, 2],
            mode="markers",
            marker={"size": 2, "color": "gray"},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    buttons = [
        {
            "label": n,
            "method": "update",
            "args": [{"visible": [t.legendgroup in (None, n) for t in fig.data]}],
        }
        for n in ("truth", "classical", "mlp", "gnn", "hybrid")
    ]
    fig.update_layout(
        title=f"Synthetic barrel | event {number} | seed {c['seed']}",
        updatemenus=[{"buttons": buttons, "x": 1.0, "y": 1.16}],
        height=820,
    )
    path = folder(c) / "event.html"
    fig.write_html(path, include_plotlyjs="directory")
    print(path)
