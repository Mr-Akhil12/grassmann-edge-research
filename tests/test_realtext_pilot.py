"""CPU-only deterministic real-text pilot checks; no external downloads."""
import hashlib
import json

import pytest
import torch

from scripts.controlled_realtext import load_split, batch_from, new_model, loss_on
from scripts.prepare_wikitext import SPLIT_SHA256


def test_pinned_validation_hash_matches_immutable_revision():
    assert SPLIT_SHA256["validation"] == (
        "a10eb5a42c4264c45dd38bfef4da6a74ddef5fc22cb419e1a89d54c4cc6d748f"
    )
    assert SPLIT_SHA256["test"] == (
        "3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925"
    )


def test_realtext_load_and_integrity(tmp_path):
    data = bytes(list(range(64)) * 6)
    (tmp_path / "train.bytes").write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    provenance = {"splits": {"train": {"prepared_sha256": expected}}}
    x = load_split(tmp_path, "train", provenance)
    assert x.dtype == torch.long
    assert x.numel() == len(data)
    assert int(x[255]) == data[255]
    (tmp_path / "train.bytes").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_split(tmp_path, "train", provenance)


def test_deterministic_splits_and_shape():
    stream = torch.arange(1024) % 256
    a = batch_from(stream, 32, 4, torch.Generator().manual_seed(11), "cpu")
    b = batch_from(stream, 32, 4, torch.Generator().manual_seed(11), "cpu")
    assert torch.equal(a, b)
    assert a.shape == (4, 33)


@pytest.mark.parametrize("name", ["grassmann", "transformer", "local"])
def test_realtext_models_compatible(name):
    torch.manual_seed(123)
    model = new_model(name, 32)
    seq = torch.randint(0, 256, (2, 33))
    output = model(seq[:, :-1])
    assert output.shape == (2, 32, 256)
    loss = loss_on(model, seq)
    assert torch.isfinite(loss)
