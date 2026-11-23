"""Evaluate a bounded pixel-barrel sector using synthetic-trained weights."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .core import graph, wrap
from .experiment import device, matching, models, predict
from .tracking import assemble

FIELDS = {
    "hits": "hit_id x y z volume_id layer_id module_id",
    "truth": "hit_id particle_id tx ty tz tpx tpy tpz weight",
    "cells": "hit_id ch0 ch1 value",
    "particles": "particle_id vx vy vz px py pz q nhits",
}


def rows(prefix, part, limit):
    choices = [
        p
        for ext in (".csv", ".csv.gz")
        if (p := Path(f"{prefix}-{part}{ext}")).is_file()
    ]
    if len(choices) != 1:
        raise ValueError(f"expected one {part} file")
    p = choices[0]
    with (gzip.open if p.suffix == ".gz" else open)(p, "rt") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS[part].split()) <= set(reader.fieldnames or []):
            raise ValueError(f"invalid {part} schema")
        for i, row in enumerate(reader):
            if i >= limit:
                raise ValueError(f"{part} row limit exceeded")
            yield row


def load(prefix, center=0.0, half_width=0.15, max_hits=4096, row_limit=2000000):
    """Counts are conditional on the selected volume and phi sector."""
    hits = {}
    for row in rows(prefix, "hits", row_limit):
        if int(row["volume_id"]) != 8:
            continue
        xyz = np.array([float(row[k]) for k in "xyz"]) / 1000
        if not np.isfinite(xyz).all():
            raise ValueError("invalid hit")
        if abs(wrap(np.arctan2(xyz[1], xyz[0]) - center)) > half_width:
            continue
        key = int(row["hit_id"])
        layer = int(row["layer_id"])
        if layer not in (2, 4, 6, 8) or key in hits or len(hits) >= max_hits:
            raise ValueError("invalid or oversized barrel selection")
        hits[key] = [*xyz, layer // 2 - 1]
    ids = {}
    for row in rows(prefix, "truth", row_limit):
        key = int(row["hit_id"])
        if key in hits:
            if key in ids:
                raise ValueError("duplicate truth hit")
            ids[key] = int(row["particle_id"])
    if not hits or ids.keys() != hits.keys():
        raise ValueError("empty sector or missing truth")
    return np.array(list(hits.values())), np.array(
        [ids[k] for k in hits], dtype=np.int64
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--phi", type=float, default=0.0)
    args = p.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(1)
    dev = device(c)
    nets = models(c, dev)
    h, y = load(args.prefix, args.phi)
    g = graph(h, c.get("graph"))
    for name in ("classical", "mlp", "gnn", "hybrid"):
        paths = assemble(
            h,
            g[1].numpy(),
            predict(name, nets, g, dev),
            name in ("classical", "hybrid"),
        )
        eligible, q, dups = matching(paths, y)
        print(
            json.dumps(
                {
                    "dataset": "TrackML sector",
                    "method": name,
                    "hits": len(h),
                    "eligible": len(eligible),
                    "matched": len(q),
                    "duplicates": dups,
                    "candidates": len(paths),
                }
            )
        )


if __name__ == "__main__":
    main()
