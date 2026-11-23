"""Validated, bounded TrackML tables and a frozen file-level experiment manifest."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from trackml.score import score_event

from .trackml import FIELDS

KEY = ["volume_id", "layer_id", "module_id"]
LIMITS = {"hits": 250000, "truth": 250000, "particles": 50000, "cells": 2000000}
INTS = {
    "hit_id",
    "particle_id",
    "volume_id",
    "layer_id",
    "module_id",
    "ch0",
    "ch1",
    "nhits",
    "q",
}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def discover(root):
    root = Path(root)
    groups = {}
    for path in sorted(root.rglob("event*-*.csv*")):
        m = re.fullmatch(
            r"(event\d{9})-(hits|cells|truth|particles)\.csv(?:\.gz)?", path.name
        )
        if m:
            parts = groups.setdefault(m[1], {})
            if m[2] in parts:
                raise ValueError("duplicate event table")
            parts[m[2]] = path
    if not groups or any(set(parts) != set(FIELDS) for parts in groups.values()):
        raise ValueError("no events or incomplete event")
    return groups


def freeze(root, output):
    groups = discover(root)
    events = sorted(groups)[:6]
    if len(events) < 3:
        raise ValueError("need at least three disjoint events")

    def record(p):
        return {
            "path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": digest(p),
        }

    doc = {
        "version": 1,
        "source": "https://www.kaggle.com/competitions/trackml-particle-identification/data",
        "archive": "train_sample.zip",
        "geometry": record(Path(root) / "detectors.csv"),
        "events": {e: {k: record(p) for k, p in groups[e].items()} for e in events},
        "split": {
            "train": events[: max(1, len(events) // 3)],
            "validation": [events[max(1, len(events) // 3)]],
            "test": events[max(1, len(events) // 3) + 1 :],
        },
    }
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != text:
        raise ValueError("frozen manifest differs; use a new output path")
    output.write_text(text)
    return doc


def chunks(path, part):
    names = FIELDS[part].split()
    size = 0
    for df in pd.read_csv(
        path,
        usecols=names,
        dtype={n: "int64" if n in INTS else "float64" for n in names},
        chunksize=32768,
    ):
        size += len(df)
        if size > LIMITS[part] or not np.isfinite(df.to_numpy()).all():
            raise ValueError(f"invalid or oversized {part}")
        yield df


def geometry(path):
    df = pd.read_csv(path, usecols=KEY + ["cx", "cy", "cz"], nrows=50001)
    if (
        len(df) > 50000
        or df.duplicated(KEY).any()
        or not np.isfinite(df.to_numpy()).all()
    ):
        raise ValueError("invalid module geometry")
    return df


@dataclass
class Event:
    hits: pd.DataFrame
    truth: pd.DataFrame
    particles: pd.DataFrame
    cells: int


def load(root, doc, event_id):
    parts = doc["events"][event_id]

    def path(meta):
        p = Path(root) / meta["path"]
        if p.stat().st_size != meta["bytes"] or digest(p) != meta["sha256"]:
            raise ValueError("input digest mismatch")
        return p

    modules = geometry(path(doc["geometry"]))
    paths = {k: path(v) for k, v in parts.items()}
    hits, truth, particles = [
        pd.concat(chunks(paths[k], k), ignore_index=True)
        for k in ("hits", "truth", "particles")
    ]
    for df, key in [(hits, "hit_id"), (truth, "hit_id"), (particles, "particle_id")]:
        if df.empty or df[key].duplicated().any() or (df[key] <= 0).any():
            raise ValueError(f"invalid {key}")
    if (
        not pd.MultiIndex.from_frame(hits[KEY])
        .isin(pd.MultiIndex.from_frame(modules[KEY]))
        .all()
    ):
        raise ValueError("unknown module")
    if (
        set(hits.hit_id) != set(truth.hit_id)
        or not truth.particle_id.isin([0, *particles.particle_id]).all()
        or (truth.weight < 0).any()
        or truth.weight.sum() <= 0
    ):
        raise ValueError("invalid truth associations")
    truth_counts = truth.particle_id.value_counts().reindex(
        particles.particle_id, fill_value=0
    )
    if not np.array_equal(particles.nhits.to_numpy(), truth_counts.to_numpy()):
        raise ValueError("particle nhits disagrees with truth")
    n = 0
    known = pd.Index(hits.hit_id)
    for cells in chunks(paths["cells"], "cells"):
        if (
            not cells.hit_id.isin(known).all()
            or (cells[["ch0", "ch1"]] < 0).any().any()
        ):
            raise ValueError("invalid cell association")
        n += len(cells)
    aligned = truth.set_index("hit_id").loc[hits.hit_id].reset_index()
    return Event(hits, aligned, particles, n)


def sector_indices(hits, nsectors=32, offset=0.0, volume=8):
    """Partition a cylindrical barrel volume by measured phi, including the seam."""
    if not 1 <= nsectors <= 128:
        raise ValueError("sector count outside budget")
    layers = hits.layer_id.to_numpy()
    if volume not in (8, 13, 17):
        raise ValueError("radial sectoring requires a barrel volume")
    barrel = hits.volume_id.to_numpy() == volume
    if not np.isin(layers[barrel], [2, 4, 6, 8]).all():
        raise ValueError("unmapped pixel barrel layer")
    phi = np.arctan2(hits.y, hits.x).to_numpy()
    bins = (
        np.floor((phi + np.pi - offset) * nsectors / (2 * np.pi)).astype(int) % nsectors
    )
    for s in range(nsectors):
        ix = np.flatnonzero(barrel & (bins == s))
        if len(ix) > 4096:
            raise ValueError("sector too large")
        if len(ix):
            yield (
                ix,
                np.column_stack(
                    (
                        hits.iloc[ix][["x", "y", "z"]].to_numpy() / 1000,
                        layers[ix] // 2 - 1,
                    )
                ),
            )


def official_score(event, paths):
    track = np.zeros(len(event.hits), dtype=np.int64)
    for i, p in enumerate(paths, 1):
        if (
            not len(p)
            or np.min(p) < 0
            or np.max(p) >= len(track)
            or len(set(p)) != len(p)
            or np.any(track[p])
        ):
            raise ValueError("shared hit ownership")
        track[p] = i
    submission = pd.DataFrame({"hit_id": event.hits.hit_id, "track_id": track})
    return float(score_event(event.truth, submission))
