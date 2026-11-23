import numpy as np
import pytest
import torch

from trackx.core import graph
from trackx.tracking import EdgeNet


def test_empty_graph_and_feature_subset():
    g = graph(np.empty((0, 4)))
    net = EdgeNet(3, 32, node_features=[0, 3], edge_features=[0])
    assert net(*g).shape == (0,)


def test_coincident_tracks_compete():
    h = np.array(
        [
            [0.04, 0, 0, 0],
            [0.04, 0, 0, 0],
            [0.08, 0, 0, 1],
            [0.08, 0, 0, 1],
            [0.12, 0, 0, 2],
            [0.12, 0, 0, 2],
        ]
    )
    x, e, a = graph(h, {"max_gap": 1, "neighbors": 1})
    assert e.shape == (2, 4)
    assert torch.equal(x[0], x[1])
    assert torch.isfinite(a).all()
    with pytest.raises(ValueError):
        graph(h, {"max_gap": 200})
