import json
import sys
from pathlib import Path

import numpy as np
import pytest

from trackx.seed_order import resolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from omega_review import read_stage, verify
from phase2_boundaries import resolve as scalar_resolve


def test_ranker_identical_ties_duplicates_and_conflicts():
    r = np.array([0.04, 0.08, 0.12, 0.16] * 3)
    xyz = np.column_stack((r, r * 0, r * 0.4))
    proposals = [
        [4, 5, 6],
        [0, 1, 2],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 5, 6, 7],
        [0, 1, 2, 3],
    ]
    assert resolve(proposals, xyz) == scalar_resolve(proposals, xyz)
    assert resolve([], xyz) == []


def test_missing_and_modified_replay_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        read_stage("validation", tmp_path)
    [doc] = read_stage("validation")
    doc["rows"]["beam"]["score"] += 0.01
    (tmp_path / "event000001002.json").write_text(json.dumps(doc))
    with pytest.raises(AssertionError):
        verify("validation", tmp_path)


def test_confirmation_split_and_paired_endpoints():
    rows = read_stage("confirmation")
    assert len(rows) == 12
    for d in rows:
        assert int(d["provenance"]["event"][5:]) > 1026
        assert set(d["rows"]) == {"local", "beam"}
        for r in d["rows"].values():
            assert 0 <= r["severe"] <= r["mixed"] <= 1
            assert 0 <= r["retention"] <= 1
