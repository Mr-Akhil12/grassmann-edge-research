"""Integrity tests only; no external dataset downloaded in CI."""
import hashlib
from pathlib import Path

from scripts.prepare_wikitext import SPLIT_SHA256, REVISION, sha256_file, verified_download


def test_source_revision_and_split_hashes_are_pinned():
    assert len(REVISION) == 40
    assert set(SPLIT_SHA256) == {"train", "validation", "test"}
    assert all(len(s) == 64 for s in SPLIT_SHA256.values())


def test_existing_correct_file_never_downloads(tmp_path):
    p = tmp_path / "data.parquet"
    p.write_bytes(b"some sample bytes")
    digest = hashlib.sha256(b"some sample bytes").hexdigest()
    verified_download("https://invalid.example/would-not-call", p, digest)
    assert sha256_file(p) == digest


def test_local_source_hash_mismatch_never_replaces_target(tmp_path):
    original = tmp_path / "original.bin"
    original.write_bytes(b"wrong content")
    target = tmp_path / "retrieved.parquet"
    expected = hashlib.sha256(b"other content").hexdigest()
    try:
        verified_download(original.as_uri(), target, expected)
    except ValueError as err:
        assert "SHA mismatch" in str(err)
    else:
        raise AssertionError("Incorrect hash accepted")
    assert not target.exists()
    assert not target.with_suffix(".parquet.partial").exists()
