"""Size-limited, content-addressed cache of truth-free CPU tensors."""

import json
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .core import graph

VERSION = b"barrel-graph-v1"


def cached(h, root, budget, settings=None):
    if budget <= 0:
        return graph(h, settings)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = sha256(
        VERSION
        + json.dumps(settings or {}, sort_keys=True).encode()
        + str((h.shape, h.dtype)).encode()
        + h.tobytes()
    ).hexdigest()
    dest = root / (key + ".pt")
    if dest.exists():
        return torch.load(dest, weights_only=True)
    g = graph(h, settings)
    with NamedTemporaryFile(dir=root, delete=False) as f:
        temp = Path(f.name)
    try:
        torch.save(g, temp)
        size = temp.stat().st_size
        files = sorted(root.glob("*.pt"), key=lambda p: p.stat().st_mtime_ns)
        used = sum(p.stat().st_size for p in files)
        for old in files:
            if used + size <= budget:
                break
            used -= old.stat().st_size
            old.unlink()
        if size <= budget:
            temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)
    return g
