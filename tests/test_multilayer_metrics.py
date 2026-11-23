"""Exercise inference and scoring together on a schema-sized ten-layer event."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trackx.extension import Barrel
from trackx.official import Event


def pipeline(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "phase3_runner", directory / "phase3.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disjoint_two_tracks_and_reference_score(monkeypatch):
    runner = pipeline(monkeypatch)
    r = np.array([32, 72, 116, 172, 261, 361, 501, 661, 821, 1021])
    volume = [8] * 4 + [13] * 4 + [17] * 2
    layers = [2, 4, 6, 8, 2, 4, 6, 8, 2, 4]
    modules = pd.DataFrame(
        {"cx": r, "cy": 0.0, "volume_id": volume, "layer_id": layers}
    )
    hit = pd.concat(
        [
            modules.assign(x=r * np.cos(angle), y=r * np.sin(angle), z=0.4 * r)
            for angle in (0.01, 0.5)
        ],
        ignore_index=True,
    )
    hit["hit_id"] = np.arange(1, 21)
    truth = pd.DataFrame(
        {"hit_id": hit.hit_id, "particle_id": np.repeat([1, 2], 10), "weight": 0.05}
    )
    ev = Event(hit, truth, None, 0)
    barrel = Barrel(hit, modules)
    seeds = runner.seeds(hit, barrel.xyz)[1][0]
    paths, traces, _ = barrel.extend(seeds)
    assert sorted(map(len, paths)) == [10, 10]
    assert len({hit for path in paths for hit in path}) == 20
    result = runner.evaluate(ev, paths, seeds, traces)
    assert result["score"] == pytest.approx(1)
    assert result["candidate_recall"] == result["efficiency"] == 1
    assert result["mixed"] == result["duplicate"] == 0
    assert all(r["matched_hit_efficiency"] == 1 for r in result["regions"].values())
    ev.truth.loc[0, "particle_id"] = 2
    assert barrel.extend(seeds)[0] == paths
    assert runner.evaluate(ev, paths, seeds, traces)["mixed"] == 0.5


def test_conflicting_predictions_never_share_hit():
    modules = pd.DataFrame(
        {
            "cx": [32, 72, 116, 172, 261, 821],
            "cy": 0,
            "volume_id": [8] * 4 + [13, 17],
            "layer_id": [2, 4, 6, 8, 2, 2],
        }
    )
    h = modules.rename(columns={"cx": "x", "cy": "y"}).assign(z=0.0)
    h = pd.concat([h.iloc[:4], h.iloc[:4], h.iloc[4:]], ignore_index=True)
    barrel = Barrel(h, modules)
    paths, _, _ = barrel.extend([list(range(4)), list(range(4, 8))])
    assert len({hit for path in paths for hit in path}) == sum(map(len, paths))
    assert max(map(len, paths)) == 6


def test_phase3_event_partition():
    import json

    first = json.loads(Path("reports/phase2/manifest.json").read_text())["split"]
    second = json.loads(Path("reports/phase2/confirmation-manifest.json").read_text())[
        "split"
    ]
    future = {f"event{n:09}" for n in (1009, 1010, 1011)}
    assert not future.intersection(
        e for split in [*first.values(), *second.values()] for e in split
    )
    release = Path("reports/phase3/manifest.json")
    if release.exists():
        assert set(json.loads(release.read_text())["events"]) == future


def test_official_barrel_extension(monkeypatch):
    import json

    from trackx.official import geometry, load

    root = Path("data/trackml")
    if not (root / "event000001002-hits.csv").is_file():
        pytest.skip("official sample not installed")
    doc = json.loads(Path("reports/phase2/manifest.json").read_text())
    ev = load(root, doc, doc["split"]["validation"][0])
    runner = pipeline(monkeypatch)
    barrel = Barrel(ev.hits, geometry(root / "detectors.csv"))
    ix, h = next(runner.sector_indices(ev.hits))
    g = runner.directional_graph(h)
    seed = [ix[p].tolist() for p in runner.hypotheses(h, g[1].numpy())]
    tracks, offers, _ = barrel.extend(seed)
    row = runner.evaluate(ev, tracks, seed, offers)
    assert row["added"] > 0 and 0 < row["score"] < 1
    assert all(p[: len(s)] == s for p, s in zip(tracks, seed, strict=True))
    assert len({j for p in tracks for j in p}) == sum(map(len, tracks))
