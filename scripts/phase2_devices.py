"""Isolated graph-size and device measurements on one frozen validation event."""

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from trackx.barrel import directional_graph
from trackx.official import load, sector_indices
from trackx.phase2 import save
from trackx.tracking import EdgeNet


def one(dev, n):
    torch.set_num_threads(1)
    out = Path("reports/phase2")
    doc = json.loads((out / "manifest.json").read_text())
    meta = json.loads((out / "training.json").read_text())
    ev = load(Path("data/trackml"), doc, doc["split"]["validation"][0])
    _, h = next(sector_indices(ev.hits, n))
    g = directional_graph(h)
    net = EdgeNet(2, 32)
    net.load_state_dict(
        torch.load(
            Path("experiments") / ("official-" + meta["config_id"]) / "local_gnn.pt"
        )
    )
    net.eval()
    with torch.inference_mode():
        expected = net(*g).sigmoid()
        net.to(dev)

        def score():
            return net(*(t.to(dev) for t in g)).sigmoid().cpu()

        actual = score()
        torch.testing.assert_close(actual, expected, rtol=0.001, atol=0.0001)
        times = []
        for _ in range(12):
            if dev == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            score()
            if dev == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - start)
    print(
        json.dumps(
            {
                "device": dev,
                "sectors": n,
                "nodes": len(h),
                "edges": g[1].shape[1],
                "median_ms": 1000 * np.median(times),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024),
                "max_score_delta": float((actual - expected).abs().max()),
            }
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--child", choices=["cpu", "mps"])
    p.add_argument("--sectors", type=int, default=32)
    args = p.parse_args()
    if args.child:
        one(args.child, args.sectors)
        return
    rows = []
    for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        for n in (16, 32, 64):
            rows.append(
                json.loads(
                    subprocess.check_output(
                        [sys.executable, __file__, "--child", dev, "--sectors", str(n)]
                    )
                )
            )
    save(Path("reports/phase2/devices.json"), rows)


if __name__ == "__main__":
    main()
