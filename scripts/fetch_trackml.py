"""Fetch six complete official events via bounded authenticated ZIP range reads."""

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
    DEST.mkdir(parents=True, exist_ok=True)
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
                )[:6]
                if len(events) != 6:
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
                if len(names) != 24:
                    raise ValueError("incomplete archive")
            else:
                names = ["detectors.csv"]
            for member in names:
                dest = DEST / Path(member).name
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
                temp = dest.with_suffix(".partial")
                temp.write_bytes(z.read(member))
                temp.replace(dest)
                print(dest.name, flush=True)
        print(name, "range bytes", reader.transferred, flush=True)


if __name__ == "__main__":
    main()
