import functools
import json
import operator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trackx.barrel import directional_graph
from trackx.official import (
    Event,
    discover,
    freeze,
    load,
    official_score,
    sector_indices,
)
from trackx.phase2 import truth_pairs
from trackx.trackml import FIELDS


def fixture(root):
    (root / "detectors.csv").write_text(
        "volume_id,layer_id,module_id,cx,cy,cz\n8,2,1,32,0,0\n8,4,1,70,0,0\n8,6,1,110,0,0\n"
    )
    pid = 9007199254740993
    for e in range(3):
        prefix = root / f"event{e:09d}"
        data = {
            "hits": [[i, r, 0, i, 8, 2 * i, 1] for i, r in enumerate([32, 70, 110], 1)],
            "cells": [[i, 1, 1, 0.1] for i in [1, 2, 3]],
            "particles": [[pid, 0, 0, 0, 1, 1, 1, 1, 3]],
            "truth": [[i, pid, 0, 0, 0, 0, 0, 0, 0.3] for i in [3, 1, 2]],
        }
        for k, v in data.items():
            pd.DataFrame(v, columns=FIELDS[k].split()).to_csv(
                f"{prefix}-{k}.csv", index=False
            )
    return freeze(root, root / "manifest.json")


def test_roundtrip(tmp_path):
    doc = fixture(tmp_path)
    parts = list(doc["split"].values())
    assert len(set(functools.reduce(operator.iadd, parts, []))) == 3
    ev = load(tmp_path, doc, parts[0][0])
    ix, h = next(sector_indices(ev.hits))
    g = directional_graph(h)
    pid, total = truth_pairs(ev)
    assert pid.tolist() == [9007199254740993] and total.tolist() == [3]
    assert g[1].shape == (2, 3) and ev.cells == 3
    assert official_score(ev, [ix.tolist()]) == pytest.approx(1.0)
    ev.truth["particle_id"] = 0
    h2 = next(sector_indices(ev.hits))[1]
    assert all(np.array_equal(a, b) for a, b in zip(g, directional_graph(h2)))


def test_manifest_rejects_changes(tmp_path):
    doc = fixture(tmp_path)
    eid = doc["split"]["test"][0]
    path = tmp_path / doc["events"][eid]["cells"]["path"]
    path.write_text("hit_id,ch0,ch1,value\n9,0,0,1\n")
    with pytest.raises(ValueError, match="digest"):
        load(tmp_path, doc, eid)
    doc = freeze(tmp_path, tmp_path / "changed.json")
    with pytest.raises(ValueError, match="association"):
        load(tmp_path, doc, eid)
    path.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        discover(tmp_path)


def test_strict_weighted_score():
    h = pd.DataFrame({"hit_id": range(1, 7)})
    truth = h.assign(
        particle_id=[1, 1, 1, 1, 2, 2], weight=[1.0, 1.0, 1.0, 1.0, 4.0, 2.0]
    )
    ev = Event(h, truth, None, 0)
    assert official_score(ev, [[0, 1], [2, 3], [4, 5]]) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        official_score(ev, [[0, 1], [1, 2]])
    with pytest.raises(ValueError):
        directional_graph(np.ones((4097, 4)))


def test_real_smoke():
    root = Path("data/trackml")
    manifest = Path("reports/phase2/manifest.json")
    if not manifest.exists() or not (root / "event000001000-hits.csv").exists():
        pytest.skip("licensed official files unavailable")
    doc = json.loads(manifest.read_text())
    ev = load(root, doc, doc["split"]["train"][0])
    ix, h = next(sector_indices(ev.hits))
    from trackx.experiment import predict
    from trackx.tracking import assemble

    g = directional_graph(h)
    paths = assemble(h, g[1].numpy(), predict("classical", {}, g, "cpu"), True)
    assert paths and 0 <= official_score(ev, [ix[p].tolist() for p in paths]) <= 1


def test_shifted_sectors_cover_each_hit_once(tmp_path):
    doc = fixture(tmp_path)
    ev = load(tmp_path, doc, doc["split"]["train"][0])
    for offset in (0, np.pi / 32):
        sets = [ix for ix, _ in sector_indices(ev.hits, 32, offset)]
        np.testing.assert_array_equal(np.sort(np.concatenate(sets)), np.arange(3))
    with pytest.raises(ValueError, match="barrel volume"):
        list(sector_indices(ev.hits, volume=9))


def test_bad_particle_count(tmp_path):
    doc = fixture(tmp_path)
    event = doc["split"]["train"][0]
    path = tmp_path / doc["events"][event]["particles"]["path"]
    particles = pd.read_csv(path)
    particles["nhits"] = 0
    particles.to_csv(path, index=False)
    doc = freeze(tmp_path, tmp_path / "modified.json")
    with pytest.raises(ValueError, match="nhits"):
        load(tmp_path, doc, event)
