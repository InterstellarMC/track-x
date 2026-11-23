import numpy as np
import torch

from trackx.barrel import directional_graph
from trackx.phase2 import LayerNet
from trackx.triplets import hypotheses


def test_whole_path_and_competing_noise():
    r = np.array([0.03, 0.07, 0.11, 0.17])
    phi = 3.13 + 0.3 * r
    h = np.column_stack(
        (r * np.cos(phi), r * np.sin(phi), 0.04 + 0.8 * r, np.arange(4))
    )
    h = np.vstack((h, [h[2, 0], h[2, 1], h[2, 2] + 0.01, 2]))
    x, e, a = directional_graph(h)
    out = hypotheses(h, e.numpy(), np.ones(e.shape[1]))
    assert out == [[0, 1, 2, 3]]
    torch.manual_seed(1)
    net = LayerNet(2, 32)
    logits = net(x, e, a)
    logits.sum().backward()
    assert torch.isfinite(net.layer_embedding.weight.grad).all()


def test_no_candidate_triplet():
    assert hypotheses(np.empty((0, 4)), np.empty((2, 0), dtype=int)) == []


def test_branched_triplets_resolve_ownership():
    h = np.array(
        [
            [0.03, 0, 0, 0],
            [0.07, 0, 0, 1],
            [0.11, 0, 0, 2],
            [0.11, 0, 0.001, 2],
            [0.17, 0, 0, 3],
        ]
    )
    g = directional_graph(h)
    paths = hypotheses(h, g[1].numpy())
    assert paths == [[0, 1, 2, 4]]
