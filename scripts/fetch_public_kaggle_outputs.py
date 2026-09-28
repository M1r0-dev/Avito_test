#!/usr/bin/env python
"""Download the public Kaggle GPU outputs without a Kaggle account.

The GPU stages (zero-shot dense, LoRA v1 / v2 dense, SPLADE) ran as public
Kaggle kernels. Re-running them is not bitwise deterministic (LoRA training on
two T4 GPUs), so exact reproduction of ``answer.csv`` relies on the saved
kernel outputs. This script fetches them through Kaggle's anonymous output
endpoint and verifies every file against the SHA-256 of the version used for
the submissions. Only the standard library is required.

The same ranking files are tracked in Git LFS, so this is an independent way
to obtain them and to check that the LFS copies are the Kaggle outputs.

Usage::

    python scripts/fetch_public_kaggle_outputs.py            # rankings: zero-shot, LoRA v1, LoRA v2
    python scripts/fetch_public_kaggle_outputs.py --splade   # also SPLADE (186 MB)
    python scripts/fetch_public_kaggle_outputs.py --online   # also LoRA v2 vectors + merged model (~1.9 GB)

``--online`` is only needed for the latency benchmark of the online path
(``scripts/export_onnx_encoder.py``, ``scripts/benchmark_final_latency.py``);
``scripts/generate_answer.py`` needs just the rankings.
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
    # Stage 10B: LoRA v2 dense retrieval — the dense channel of attempts 4-7.
    "lora-v2-dense-retrieval": ("artifacts/finetuned_v2_kaggle", {
        "finetuned_v2_dense_rankings.parquet": "e6a2d2f3634ab5a326f3c943dff5e81d9a5a512bb321c5b3bc52507a5c1bf85a",
        "finetuned_v2_dense_run.json": "210f8af39ce47e5dc09f44091cce7413a1323c8654e34870e71b0a80e77074e4",
    }),
    "russian-splade-candidate-retrieval": ("artifacts/splade_kaggle", {
        "splade_rankings.parquet": "535576a255b74baaacbb1456a4b17a1d1d5130d1bace90bea2bcdc776278fc41",
        "splade_run.json": "991695407f6104c6f349f60ef729d7460ca5621879f6194471539980087733a9",
        "validation_labels.parquet": "8eaf3015aa4602aa14e14320aaf98de701987beed98fee0a18d60485a53db6cc",
    }),
}


# Online path only (latency benchmark): LoRA v2 passage/query vectors with their
# row mappings (stage 10B) and the merged LoRA v2 encoder (stage 10A).
ONLINE: dict[str, tuple[str, dict[str, str]]] = {
    "lora-v2-dense-retrieval": ("artifacts/finetuned_v2_kaggle", {
        "passage_embeddings_v2_fp16.npy": "7a5eb3f54c817da8234d38b04773cf0544765df417e91496202bfbaa1c670a07",
        "passage_item_rows.npy": "f63c464e7c6bfc38cd42a7a8bab1c3c40c491d06037e1d4f633b52098a34ce69",
        "item_order.parquet": "2817343dd88cdf6e2e3c790e534c1ec80c59402210ed4d19db6d085aab272bdf",
        "query_embeddings_v2_fp16.npy": "2f0e1b430da4eda1867ebfc44062cb94e578bc5f2ac8df5b51484bcc1a93411f",
        "query_order.parquet": "30685aa55e562ec7ce181f86f4a312f75ebc62757b9275b9fefcc93db1e46fe9",
    }),
    "user-bge-m3-lora-v2": ("artifacts/lora_v2_model", {
        "user_bge_m3_avito_v2/config.json": "72d176925fb37c68eb0faf25656a53f5ce5021297908b3dfec662a95f3a15081",
        "user_bge_m3_avito_v2/model.safetensors": "ac1dc01741785807b368a3fe919af8cef911576da67acd1c92f9738a7d3354c5",
        "user_bge_m3_avito_v2/tokenizer.json": "4a288e568b5ae5079473dd08c4337cb83016f5728c87d50ff9295d0fc523332e",
        "user_bge_m3_avito_v2/tokenizer_config.json": "145cf61acbab00da6287ab49a68835a669367b0199c88ed2ae621ab261a6c2b4",
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


def fetch(slug: str, registry: dict[str, tuple[str, dict[str, str]]] = KERNELS) -> bool:
    directory, expected = registry[slug]
    output = ROOT / directory
    output.mkdir(parents=True, exist_ok=True)
    urls = list_output_urls(slug)
    ok = True
    for name, checksum in expected.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)  # merged model lives in a subdirectory
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
    parser.add_argument("--online", action="store_true", help="also fetch LoRA v2 vectors and merged model")
    args = parser.parse_args()
    slugs = ["avito-russian-dense-candidate-retrieval", "avito-finetuned-dense-retrieval", "lora-v2-dense-retrieval"]
    if args.splade:
        slugs.append("russian-splade-candidate-retrieval")
    results = [fetch(slug) for slug in slugs]
    if args.online:
        results += [fetch(slug, ONLINE) for slug in ONLINE]
    if not all(results):
        sys.exit("Some kernel outputs are missing or differ from the submitted version.")


if __name__ == "__main__":
    main()
