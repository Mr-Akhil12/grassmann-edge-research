"""CPU-only integrity tests for the read-only P2.3 auditor."""
import hashlib
import json
import math
from pathlib import Path
import pytest
from scripts import p23_analyze as audit


def write_trial(train: Path, seed: int, kind: str, bpb=3.5):
    train.mkdir(parents=True, exist_ok=True)
    name = audit.stem(kind, seed)
    checkpoint = train / f"{name}.ckpt.pt"
    checkpoint.write_bytes(f"fake checkpoint for {seed}/{kind}".encode())
    vals = [bpb * math.log(2)] * 8
    trial = {
        "status": "EXPLORATORY FP32 BYTE-LM P2.2; NOT PAPER REPRODUCTION",
        "kind": kind, "seed": seed,
        "config": {**audit.CONFIG, "kind": kind, "seed": seed,
                   "out_dir": "results/p23_multiseed_v1/train"},
        "steps_completed": 100, "steps_resumed_from": 0, "train_tokens_total": 409600,
        "offsets": audit.OFFSETS, "ff_widths": audit.FF_WIDTHS,
        "parameter_counts": audit.PARAMS, "parameter_mismatch_fraction": .00026225,
        "dataset_revision": audit.DATASET_REVISION,
        "train_sha256": audit.TRAIN_SHA, "validation_sha256": audit.VAL_SHA,
        "test_split_used": False,
        "train_losses_nats": [2.5] * 100,
        "validation_per_batch_nats": vals,
        "mean_validation_bits_per_byte": bpb,
        "block_wall_seconds": 21.,
        "peak_allocated_mib": 550., "peak_reserved_mib": 600.,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    metrics = train / f"{name}.metrics.json"
    metrics.write_text(json.dumps(trial))
    return metrics, checkpoint


def test_missing_and_partial_does_not_rank(tmp_path):
    write_trial(tmp_path / "train", 17, "grassmann")
    partial = audit.audit(tmp_path, pilot=True)
    assert partial["status"] == "incomplete"
    assert partial["valid_trials"] == 1
    assert len(partial["missing"]) == 2
    assert "model_means_bpb" not in partial
    assert audit.main(["--run-dir", str(tmp_path), "--pilot"]) == 2


def test_pilot_complete_and_does_not_infer_superiority(tmp_path):
    for kind, bpb in [("grassmann", 3.55), ("local", 3.50), ("transformer", 3.64)]:
        write_trial(tmp_path / "train", 17, kind, bpb)
    result = audit.audit(tmp_path, pilot=True)
    assert result["status"] == "complete"
    assert len(result["per_seed_complete_pairs"]) == 1
    assert "paired_summaries" not in result
    assert audit.main(["--run-dir", str(tmp_path), "--pilot"]) == 0


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(steps_resumed_from=20),
    lambda m: m.update(train_sha256="0" * 64),
    lambda m: m.update(validation_sha256="0" * 64),
    lambda m: m.update(test_split_used=True),
    lambda m: m["config"].update(batch=16),
    lambda m: m.update(train_losses_nats=[2.5] * 99),
    lambda m: m.update(mean_validation_bits_per_byte=math.nan),
    lambda m: m.update(mean_validation_bits_per_byte=3.99),
])
def test_fails_closed_on_invalid_metrics(tmp_path, mutation):
    path, checkpoint = write_trial(tmp_path / "train", 17, "grassmann")
    data = json.loads(path.read_text())
    mutation(data)
    path.write_text(json.dumps(data))
    result = audit.audit(tmp_path, pilot=True)
    assert result["status"] == "incomplete"
    assert result["valid_trials"] == 0
    assert result["invalid"][0]["trial"] == "grassmann/seed17"


def test_tampered_checkpoint_rejected(tmp_path):
    _, checkpoint = write_trial(tmp_path / "train", 17, "grassmann")
    checkpoint.write_bytes(b"tampered")
    result = audit.audit(tmp_path, pilot=True)
    assert result["invalid"][0]["reason"] == "Checkpoint SHA-256 mismatch"


def test_complete_five_seed_report_is_paired_and_deterministic(tmp_path):
    for seed in audit.SEEDS:
        write_trial(tmp_path / "train", seed, "grassmann", 3.60 + seed / 1000)
        write_trial(tmp_path / "train", seed, "local", 3.50 + seed / 1000)
        write_trial(tmp_path / "train", seed, "transformer", 3.70 + seed / 1000)
    result = audit.audit(tmp_path)
    assert result["status"] == "complete"
    assert result["valid_trials"] == 15
    assert len(result["per_seed_complete_pairs"]) == 5
    assert abs(result["paired_summaries"]["grassmann_minus_local"]["mean"] - .1) < 1e-12
    assert result["paired_summaries"]["grassmann_minus_local"]["exact_two_sided_sign_flip_p_exploratory"] == .0625
    assert result == audit.audit(tmp_path)


def test_five_seed_run_with_one_missing_has_no_aggregate_winner(tmp_path):
    for seed in audit.SEEDS:
        for kind in audit.KINDS:
            if (seed, kind) != (47, "local"):
                write_trial(tmp_path / "train", seed, kind)
    result = audit.audit(tmp_path)
    assert result["status"] == "incomplete"
    assert "model_means_bpb" not in result
    assert len(result["per_seed_complete_pairs"]) == 4


def test_exact_roster_rejects_unplanned_seed():
    with pytest.raises(ValueError):
        audit.stem("local", 22)
