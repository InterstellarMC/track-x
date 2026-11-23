import functools
import json
import operator
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from trackx.cache import cached
from trackx.core import RADII, event, graph, intersection, labels, wrap
from trackx.experiment import matching, splits
from trackx.tracking import EdgeNet, assemble, fit
from trackx.trackml import FIELDS, load, rows

CONFIG = Path(__file__).resolve().parents[1] / "configs/smoke.yaml"
C = yaml.safe_load(CONFIG.read_text())


@pytest.mark.parametrize("k", [0.0, 1e-11, -1.0, 0.8])
def test_helix(k):
    angle, arc = intersection(RADII, k)
    assert np.isfinite(arc).all()
    p = np.column_stack(
        (RADII * np.cos(angle), RADII * np.sin(angle), 0.02 + 0.7 * arc)
    )
    result, pt, dz = fit(p, 2)
    assert result == pytest.approx(k, abs=1e-9)
    assert np.max(abs(dz)) < 1e-12
    if abs(k) > 0.1:
        assert pt == pytest.approx(0.599584916 / abs(k))


def test_periodic_graph():
    p = np.array([np.pi - 0.01, -np.pi + 0.01, np.pi - 0.02])
    r = RADII[:3]
    h = np.column_stack((r * np.cos(p), r * np.sin(p), np.zeros(3), np.arange(3)))
    _x, e, a = graph(h)
    assert wrap(p[1] - p[0]) == pytest.approx(0.02)
    assert set(map(tuple, e.T.tolist())) == {(0, 1), (0, 2), (1, 2)}
    assert torch.isfinite(a).all()
    with pytest.raises(ValueError):
        graph(h * np.nan)


def test_truth_bounds_and_replay(tmp_path):
    h, y, _ = event(C, 2)
    np.testing.assert_array_equal(h, event(C, 2)[0])
    x, e, a = cached(h, tmp_path, 100000)
    before = labels(y, e)
    y[:] = 0
    assert before.any() and not labels(y, e).any()
    assert all(
        torch.equal(u, v) for u, v in zip((x, e, a), cached(h, tmp_path, 100000))
    )
    assert e.shape[1] <= 16 * len(h)
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= 100000
    split = splits(C)
    assert len(set(np.concatenate(list(split.values())))) == sum(
        map(len, split.values())
    )
    with pytest.raises(ValueError):
        splits(dict(C, max_events=1))


@pytest.mark.parametrize("dev", ["cpu", "mps"])
def test_gradient(dev):
    if dev == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    g = [t.to(dev) for t in graph(event(C, 1)[0])]
    model = EdgeNet(2).to(dev)
    output = model(*g)
    output.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    torch.testing.assert_close(output, model(*g))


def test_matching_and_ownership():
    ids = np.array([1] * 6 + [2] * 3 + [0] * 3)
    truth, q, d = matching([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], ids)
    assert truth == {1, 2} and len(q) == 2 and d == 1
    h, _, _ = event(dict(C, tracks=50, noise=120), 5)
    _, e, _ = graph(h)
    paths = assemble(h, e.numpy(), np.ones(e.shape[1]), True)
    hits = functools.reduce(operator.iadd, paths, [])
    assert paths and len(hits) == len(set(hits))
    assert all(np.all(np.diff(h[p, 3]) > 0) for p in paths)


def test_official_schema_fixture(tmp_path):
    prefix = tmp_path / "event000000001"
    data = {
        "hits": ["7 .1 0 0 8 2 1"],
        "truth": ["7 9007199254740993 0 0 0 0 0 0 1"],
        "cells": ["7 1 2 3"],
        "particles": ["9007199254740993 0 0 0 1 1 1 1 1"],
    }
    for part, lines in data.items():
        Path(f"{prefix}-{part}.csv").write_text(
            "\n".join([FIELDS[part], *lines]).replace(" ", ",") + "\n"
        )
        assert len(list(rows(prefix, part, 1))) == 1
    h, y = load(prefix)
    assert h[0, 0] == pytest.approx(0.0001) and y.tolist() == [9007199254740993]
    Path(f"{prefix}-hits.csv").write_text("wrong\n")
    with pytest.raises(ValueError, match="schema"):
        load(prefix)


def test_clean_smoke(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "trackx", "smoke", "--config", str(CONFIG)],
        cwd=tmp_path,
        check=True,
    )
    metrics = json.loads(next(tmp_path.rglob("metrics.json")).read_text())
    assert {r["method"] for r in metrics["records"]} == {
        "classical",
        "mlp",
        "gnn",
        "hybrid",
    }
    assert any(r["efficiency"] > 0 for r in metrics["records"] if r["method"] == "gnn")
    assert next(tmp_path.rglob("event.html")).stat().st_size > 1000
