#!/usr/bin/env python3
"""Download the pinned public USGS Utah DEM, verifying its complete-file hash."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "data/utah/mdrs_1m.tif")
    args = parser.parse_args()
    manifest = json.loads((root / "datasets/utah/mdrs_1m.json").read_text())
    url = manifest["url"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "prd-tnm.s3.amazonaws.com":
        raise ValueError("Expected an HTTPS USGS National Map source")
    if args.output.exists():
        if checksum(args.output) == manifest["sha256"]:
            print(f"Existing DEM verified: {args.output}")
            return
        raise FileExistsError(
            "Existing DEM does not match the pinned source; choose another output path"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    part = args.output.with_suffix(args.output.suffix + ".part")
    if part.exists():
        raise FileExistsError(f"Partial file already exists: {part}")
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--show-error",
                "--retry",
                "2",
                "--max-time",
                "600",
                "--output",
                str(part),
                url,
            ],
            check=True,
        )
        if checksum(part) != manifest["sha256"]:
            raise ValueError(
                "USGS download hash changed; inspect updated source provenance before using it"
            )
        part.rename(args.output)
    finally:
        if part.exists():
            part.unlink()
    print(f'Verified native {manifest["source_resolution_m"]} m DEM: {args.output}')


if __name__ == "__main__":
    main()
