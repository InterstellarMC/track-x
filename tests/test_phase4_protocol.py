"""Regression checks on independent cohorts and the frozen validation decision."""

import json
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parents[1] / "reports/phase4"


def test_historical_and_prospective_samples_disjoint():
    historical = set(range(1000, 1012))
    seen = set()
    for stage, start, n in (("gap-a", 1012, 5), ("gap-b", 1017, 5), ("final", 1022, 5)):
        expected = set(range(start, start + n))
        assert expected.isdisjoint(seen | historical)
        data = json.loads((OUT / (stage + "-manifest.json")).read_text())
        ids = {int(e[5:]) for e in data["events"]}
        assert ids == expected
        assert set(data["split"][stage]) == set(data["events"])
        seen.update(ids)


def test_validation_selection_and_reference_replay():
    data = json.loads((OUT / "event000001002.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    scores = {m: r["score"] for m, r in data["rows"].items()}
    assert scores == selection["validation_scores"]
    assert selection["backbone"] == max(scores, key=scores.get)
    assert data["stage"] == "validation"


def test_mixed_severity_definitions():
    atlas = json.loads((OUT / "validation-atlas.json").read_text())
    n = atlas["counts"]
    assert (
        n["seed already mixed"] + n["new mixing in extension"] + n["no signal mixing"]
        == atlas["total_paths"]
    )
    data = json.loads((OUT / "event000001002.json").read_text())["rows"]["M3"]
    assert n["severe merge"] / atlas["total_paths"] == pytest.approx(data["severe"])
    assert (n["seed already mixed"] + n["new mixing in extension"]) / atlas[
        "total_paths"
    ] == pytest.approx(data["mixed"])
