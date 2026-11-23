"""Acquire only the committed Omega cohort using bounded official archive ranges."""

import base64
import json
import os
import re
import shutil
import zlib
from pathlib import Path
from urllib.error import HTTPError
from zipfile import ZipFile

import numpy as np
from fetch_trackml import RemoteZip

from trackx.official import digest
from trackx.phase2 import save

OUT = Path("reports/omega")
ROOT = Path("data/omega")


def inventory(z):
    groups = {}
    for info in z.infolist():
        m = re.fullmatch(
            r"(event\d{9})-(hits|truth|cells|particles)\.csv", Path(info.filename).name
        )
        if m and int(m[1][5:]) > 1026:
            g = groups.setdefault(m[1], {})
            if m[2] in g:
                raise ValueError("ambiguous archive path")
            g[m[2]] = info
    events = sorted(e for e, v in groups.items() if len(v) == 4)
    if len(events) < 12:
        raise ValueError("fewer than twelve complete unseen events")
    return {
        events[i]: groups[events[i]]
        for i in np.linspace(0, len(events) - 1, 12, dtype=int)
    }


def main():
    if not (OUT / "selection.json").is_file():
        raise ValueError("freeze selection before acquisition")
    ROOT.mkdir(exist_ok=True)
    if shutil.disk_usage(ROOT).free < 3 * 2**30 + 600_000_000:
        raise OSError("insufficient free space")
    cfg = json.loads(
        (
            Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
            / "kaggle.json"
        ).read_text()
    )
    auth = (
        "Basic "
        + base64.b64encode((cfg["username"] + ":" + cfg["key"]).encode()).decode()
    )
    cohort = OUT / "cohort.json"
    old = json.loads(cohort.read_text()) if cohort.exists() else None
    for archive in [old["archive"]] if old else ["train_1.zip", "train_sample.zip"]:
        try:
            reader = RemoteZip(archive, auth)
            with ZipFile(reader) as z:
                groups = inventory(z)
            break
        except (ValueError, HTTPError) as exc:
            print(archive, "inventory unavailable", type(exc).__name__, flush=True)
            if old or archive == "train_sample.zip":
                raise
    details = {
        "archive": archive,
        "events": list(groups),
        "source": "TrackML Particle Tracking Challenge",
    }
    if old and old != details:
        raise ValueError("cohort changed")
    save(cohort, details)
    items = list(groups.items())
    if sum(i.file_size for _, g in items for i in g.values()) > 600_000_000:
        raise ValueError("raw storage cap exceeded")
    doc = json.loads(Path("reports/phase2/manifest.json").read_text())
    doc.update(archive=archive, events={}, split={"confirmation": list(groups)})
    geom = ROOT / "detectors.csv"
    if not geom.exists():
        shutil.copyfile("data/trackml/detectors.csv", geom)
    if digest(geom) != doc["geometry"]["sha256"]:
        raise ValueError("geometry mismatch")
    for start in range(0, 12, 4):
        reader = RemoteZip(archive, auth)
        with ZipFile(reader) as z:
            for e, g in items[start : start + 4]:
                record = {}
                for kind, info in sorted(g.items()):
                    dest = ROOT / Path(info.filename).name
                    if info.file_size > 40_000_000:
                        raise ValueError("oversized member")
                    if not dest.exists():
                        temp = dest.with_suffix(".partial")
                        temp.write_bytes(z.read(info.filename))
                        temp.replace(dest)
                    if (
                        dest.stat().st_size != info.file_size
                        or zlib.crc32(dest.read_bytes()) != info.CRC
                    ):
                        raise ValueError("corrupt member")
                    record[kind] = {
                        "path": dest.name,
                        "bytes": info.file_size,
                        "sha256": digest(dest),
                    }
                doc["events"][e] = record
                print(archive, e, flush=True)
    manifest = OUT / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != doc:
        raise ValueError("frozen manifest differs")
    save(manifest, doc)


if __name__ == "__main__":
    main()
