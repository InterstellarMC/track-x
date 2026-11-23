import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .core import event
from .experiment import device, evaluate, folder, splits, training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["smoke", "generate", "train", "evaluate", "visualize"]
    )
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    ids = splits(c)
    torch.set_num_threads(1)
    dev = device(c)
    print("device", dev)
    start = time.perf_counter()
    if args.command in ("smoke", "generate"):
        h, y, _ = event(c, ids["test"][0])
        np.savez(folder(c) / "event.npz", hits=h, truth=y)
    if args.command in ("smoke", "train"):
        training(c, dev)
    if args.command in ("smoke", "evaluate"):
        evaluate(c, dev)
    if args.command in ("smoke", "visualize"):
        from .view import render

        render(c, dev)
    print("elapsed seconds", round(time.perf_counter() - start, 3))


if __name__ == "__main__":
    main()
