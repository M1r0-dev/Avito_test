#!/usr/bin/env python
"""Download the public Kaggle GPU outputs without a Kaggle account.

The GPU stages (zero-shot dense, SPLADE, fine-tuned dense) ran as public Kaggle
kernels. Re-running them is not bitwise deterministic (LoRA training on two
T4 GPUs), so exact reproduction of ``answer.csv`` relies on the saved kernel
outputs. This script fetches them through Kaggle's anonymous output endpoint
and verifies every file against the SHA-256 of the version used for the
submissions. Only the standard library is required.

Usage::

    python scripts/fetch_public_kaggle_outputs.py            # dense + fine-tuned
    python scripts/fetch_public_kaggle_outputs.py --splade   # also SPLADE (186 MB)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "m1r0tvorxc"
API = "https://www.kaggle.com/api/v1/kernels/output"

# kernel slug -> (local directory, {file name: expected sha256}).
KERNELS: dict[str, tuple[str, dict[str, str]]] = {
    "avito-russian-dense-candidate-retrieval": ("artifacts/dense_kaggle", {
        "dense_rankings.parquet": "b725e38ed198d5de1e6c5a67365dddb9ae0bc3802d64d44824325bdf4a3b2ed1",
        "dense_run.json": "616f54c5568bcd618d5c8bbf766f217d3f712f9c7a302077a94e78afff8a1f94",
        "validation_labels.parquet": "f419c0b43a6a08aa2febe9602ce48feed0d240dcaff6382b3b973c33896755ba",
    }),
    "avito-finetuned-dense-retrieval": ("artifacts/finetuned_dense_kaggle", {
        "finetuned_dense_rankings.parquet": "d08f8fed2cbc4d7fb2f9adb789afe2e7daf789d41f5ef3003b07871541819d3a",
        "finetuned_dense_run.json": "1c4c4b3d95fc7a0ff43eb4200afaa2002e4a24fcca52b935d963d8a268f5518e",
        "validation_labels.parquet": "8eaf3015aa4602aa14e14320aaf98de701987beed98fee0a18d60485a53db6cc",
    }),
    "avito-russian-splade-candidate-retrieval": ("artifacts/splade_kaggle", {
        "splade_rankings.parquet": "535576a255b74baaacbb1456a4b17a1d1d5130d1bace90bea2bcdc776278fc41",
        "splade_run.json": "991695407f6104c6f349f60ef729d7460ca5621879f6194471539980087733a9",
        "validation_labels.parquet": "8eaf3015aa4602aa14e14320aaf98de701987beed98fee0a18d60485a53db6cc",
    }),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_output_urls(slug: str) -> dict[str, str]:
    """Return {file name: signed download URL} for the latest kernel version."""
    urls: dict[str, str] = {}
    token = ""
    while True:
        query = {"userName": OWNER, "kernelSlug": slug}
        if token:
            query["pageToken"] = token
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(query)}", timeout=60) as response:
            payload = json.load(response)
        for item in payload.get("files", []):
            urls[item["fileName"]] = item.get("url") or item.get("urlNullable")
        token = payload.get("nextPageToken") or ""
        if not payload.get("hasNextPageToken") or not token:
            return urls


def fetch(slug: str) -> bool:
    directory, expected = KERNELS[slug]
    output = ROOT / directory
    output.mkdir(parents=True, exist_ok=True)
    urls = list_output_urls(slug)
    ok = True
    for name, checksum in expected.items():
        target = output / name
        # Skip the download when an identical file is already present.
        if not (target.exists() and sha256(target) == checksum):
            if name not in urls:
                print(f"[missing] {slug}/{name} is not in the kernel output")
                ok = False
                continue
            partial = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(urls[name], timeout=600) as response, partial.open("wb") as sink:
                shutil.copyfileobj(response, sink)
            partial.replace(target)
        actual = sha256(target)
        status = "ok" if actual == checksum else "SHA256 MISMATCH"
        ok &= actual == checksum
        print(f"[{status}] {directory}/{name}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splade", action="store_true", help="also fetch SPLADE rankings")
    args = parser.parse_args()
    slugs = ["avito-russian-dense-candidate-retrieval", "avito-finetuned-dense-retrieval"]
    if args.splade:
        slugs.append("avito-russian-splade-candidate-retrieval")
    if not all([fetch(slug) for slug in slugs]):
        sys.exit("Some kernel outputs are missing or differ from the submitted version.")


if __name__ == "__main__":
    main()
