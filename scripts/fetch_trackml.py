"""Fetch six complete official events via bounded authenticated ZIP range reads."""

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

URL = "https://www.kaggle.com/api/v1/competitions/data/download/trackml-particle-identification/"
DEST = Path("data/trackml")


class RemoteZip(io.RawIOBase):
    """Seekable reader; refuse servers that ignore Range. Credentials stay on Kaggle."""

    def __init__(self, name, auth):
        self.url = URL + name
        self.auth = auth
        self.pos = 0
        self.transferred = 0
        with self.request(0, 0) as response:
            self.size = int(response.headers["Content-Range"].split("/")[-1])
            self.url = response.url
        self.auth = None

    def request(self, start, end):
        req = Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        if self.auth:
            req.add_unredirected_header("Authorization", self.auth)
        response = urlopen(req, timeout=40)
        if response.status != 206 or not response.headers.get(
            "Content-Range", ""
        ).startswith(f"bytes {start}-{end}/"):
            response.close()
            raise ValueError("server did not honor requested range")
        return response

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = ([0, self.pos, self.size][whence]) + offset
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 32_000_000 or self.transferred + n > 80_000_000:
            raise ValueError("transfer budget exceeded")
        with self.request(self.pos, self.pos + n - 1) as response:
            data = response.read(n)
        if len(data) != n:
            raise OSError("truncated range")
        self.pos += n
        self.transferred += n
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="only events 1006–1008, in data/trackml-confirmation",
    )
    parser.add_argument(
        "--phase3", action="store_true", help="only frozen Phase III events 1009–1011"
    )
    args = parser.parse_args()
    dest_root = Path("data/trackml-confirmation") if args.confirmation else DEST
    selection = slice(6, 9) if args.confirmation else slice(0, 6)
    if args.phase3:
        if args.confirmation:
            parser.error("choose one sample")
        dest_root = Path("data/trackml-phase3")
        selection = slice(9, 12)
    dest_root.mkdir(parents=True, exist_ok=True)
    settings = json.loads(
        (
            Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
            / "kaggle.json"
        ).read_text()
    )
    auth = (
        "Basic "
        + base64.b64encode(
            (settings["username"] + ":" + settings["key"]).encode()
        ).decode()
    )
    for name in ("detectors.zip", "train_sample.zip"):
        reader = RemoteZip(name, auth)
        with ZipFile(reader) as z:
            names = sorted(z.namelist())
            if name == "train_sample.zip":
                events = sorted(
                    {
                        m[1]
                        for f in names
                        if (m := re.search(r"(event\d{9})-hits\.csv$", f))
                    }
                )[selection]
                if len(events) != selection.stop - selection.start:
                    raise ValueError("unexpected event archive")
                names = [
                    f
                    for f in names
                    if any(
                        Path(f).name == f"{e}-{p}.csv"
                        for e in events
                        for p in ("hits", "cells", "particles", "truth")
                    )
                ]
                if len(names) != len(events) * 4:
                    raise ValueError("incomplete archive")
            else:
                names = ["detectors.csv"]
            for member in names:
                dest = dest_root / Path(member).name
                info = z.getinfo(member)
                if info.file_size > 40_000_000:
                    raise ValueError("member exceeds output budget")
                import zlib

                if (
                    dest.exists()
                    and dest.stat().st_size == info.file_size
                    and zlib.crc32(dest.read_bytes()) == info.CRC
                ):
                    continue
                if dest.exists():
                    raise ValueError(
                        f"existing file differs from official archive: {dest.name}"
                    )
                temp = dest.with_suffix(".partial")
                temp.write_bytes(z.read(member))
                temp.replace(dest)
                print(dest.name, flush=True)
        print(name, "range bytes", reader.transferred, flush=True)


if __name__ == "__main__":
    main()
