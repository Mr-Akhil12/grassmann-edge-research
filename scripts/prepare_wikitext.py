"""Fetch pinned WikiText-2-raw-v1 Parquet splits with SHA-256 verification.

This *downloads public CC BY-SA dataset files* to ignored data/; it does
NOT add dataset content to the Git repository. Requires optional pyarrow
for conversion; all text processing is intentionally deterministic.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


REVISION = "6231e49f19a707241d6b84d9cff60a3a86b85a85"
DATASET_URL = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
               + REVISION + "/wikitext-2-raw-v1/")
SPLIT_SHA256 = {
    "train": "49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701",
    "validation": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
    "test": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
}
MAX_BYTES = 15 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_download(url: str, target: Path, expected: str):
    if target.exists() and sha256_file(target) == expected:
        return
    temp = target.with_suffix(target.suffix + ".partial")
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "grassmann-edge-research/0.1"}),
                timeout=30) as response, temp.open("wb") as out:
            total = 0
            for chunk in iter(lambda: response.read(65536), b""):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("Dataset file exceeds safety cap")
                out.write(chunk)
        actual = sha256_file(temp)
        if actual != expected:
            raise ValueError(f"Dataset SHA mismatch: actual {actual}")
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/wikitext-2-raw-v1")
    p.add_argument("--splits", nargs="+", choices=tuple(SPLIT_SHA256),
                   default=["train", "validation"])
    a = p.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit('Install optional dependency: pip install -e ".[data]"') from exc
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset": "Salesforce/wikitext",
                "config": "wikitext-2-raw-v1",
                "revision": REVISION, "tokenizer": "UTF-8 byte stream, no token normalization",
                "upstream_license": "CC BY-SA 3.0; refer to source dataset card",
                "splits": {}}
    for split in a.splits:
        filename = f"{split}-00000-of-00001.parquet"
        source = root / filename
        verified_download(DATASET_URL + filename, source, SPLIT_SHA256[split])
        table = pq.read_table(source, columns=["text"])
        # Retain row boundaries; explicitly join rows with a single LF.
        rows = table["text"].to_pylist()
        text = "\n".join(rows)
        data = text.encode("utf-8")
        target = root / f"{split}.bytes"
        target.write_bytes(data)
        metadata["splits"][split] = {
            "upstream_filename": filename, "sha256": SPLIT_SHA256[split],
            "rows": len(rows), "utf8_bytes": len(data),
            "prepared_sha256": hashlib.sha256(data).hexdigest()
        }
        print(split, len(rows), "rows", len(data), "bytes", flush=True)
    (root / "PROVENANCE.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
