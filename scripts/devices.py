"""Time the same smoke checkpoint and test graphs on CPU and probed MPS."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from trackx.core import event, graph
from trackx.experiment import device, models, predict, splits

c = yaml.safe_load(Path("configs/smoke.yaml").read_text())
torch.set_num_threads(1)
graphs = [graph(event(c, n)[0]) for n in splits(c)["test"]]
result = {}
for dev in dict.fromkeys(["cpu", device(dict(c, device="auto"))]):
    nets = models(c, dev)
    times = []
    for g in graphs:
        predict("gnn", nets, g, dev)
        for _ in range(10):
            if dev == "mps":
                torch.mps.synchronize()
            t = time.perf_counter()
            scores = predict("gnn", nets, g, dev)
            if dev == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t) * 1000)
    result[dev] = {"median_ms": float(np.median(times)), "calls": len(times)}
print(result)
Path("reports/devices.json").write_text(json.dumps(result, indent=2))
