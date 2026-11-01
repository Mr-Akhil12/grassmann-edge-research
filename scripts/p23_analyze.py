"""Read-only, fail-closed P2.3 five-seed checkpoint and metrics audit.

DOES NOT run training, query NVIDIA hardware, change checkpoints or upload data.
Defaults to the fully preregistered five-seed roster. --pilot selects the
sole predeclared seed-17 triplet; never treat that as a confirmatory study.

Usage:
  python -m scripts.p23_analyze --run-dir results/p23_multiseed_v1
  python -m scripts.p23_analyze --run-dir results/p23_pilot_seed17 --pilot
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

KINDS = ("grassmann", "transformer", "local")
SEEDS = (11, 17, 23, 31, 47)
PILOT_SEEDS = (17,)
TRAIN_SHA = "c58d555a94ada0882d1e8c0b3dc7ba68e1c269da5ad9e7a057300245d130c4e8"
VAL_SHA = "714a24a5de7459876f7cfdc64837e551dd7cde2a51f95b763ee083fff273e307"
DATASET_REVISION = "6231e49f19a707241d6b84d9cff60a3a86b85a85"
PARAMS = {"grassmann": 6290604, "transformer": 6289920, "local": 6291570}
FF_WIDTHS = {"grassmann": 1238, "transformer": 1152, "local": 1399}
CONFIG = {"dim": 288, "layers": 6, "rank": 28, "batch": 8,
          "context": 512, "lr": 0.0003, "stop_step": 100,
          "checkpoint_every": 20}
OFFSETS = [1, 2, 4, 8, 16, 32]


class InvalidTrial(ValueError):
    """An expected trial exists but violates a frozen integrity invariant."""


def stem(kind: str, seed: int) -> str:
    if kind not in KINDS or seed not in SEEDS:
        raise ValueError("Architecture/seed is not part of the locked protocol")
    return f"{kind}_seed{seed}_d288_ctx512"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _check(condition: bool, reason: str) -> None:
    if not condition:
        raise InvalidTrial(reason)


def _number(value: object, name: str) -> float:
    _check(isinstance(value, (int, float)) and not isinstance(value, bool),
           name + " is not numeric")
    numeric = float(value)
    _check(math.isfinite(numeric), name + " is not finite")
    return numeric


def validate_trial(train_dir: Path, kind: str, seed: int) -> dict:
    """Require exact run identity, raw data provenance and checkpoint integrity."""
    name = stem(kind, seed)
    metrics = train_dir / (name + ".metrics.json")
    checkpoint = train_dir / (name + ".ckpt.pt")
    if not metrics.is_file():
        raise FileNotFoundError(str(metrics))
    try:
        trial = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InvalidTrial(f"Cannot read metrics JSON: {exc}") from exc
    _check(isinstance(trial, dict), "Metrics must be a JSON object")
    config = trial.get("config")
    _check(isinstance(config, dict), "Missing configuration")
    for key, expected in CONFIG.items():
        _check(config.get(key) == expected, f"Config {key} changed from locked value")
    _check(config.get("kind") == kind and config.get("seed") == seed,
           "Config architecture/seed mismatch")
    _check(trial.get("kind") == kind and trial.get("seed") == seed,
           "Top-level architecture/seed mismatch")
    _check(trial.get("steps_completed") == 100, "Trial is not 100 steps")
    _check(trial.get("steps_resumed_from") == 0, "Trial was resumed, not fresh")
    _check(trial.get("train_tokens_total") == 409600, "Training token budget mismatch")
    _check(trial.get("offsets") == OFFSETS, "Offsets changed")
    _check(trial.get("ff_widths") == FF_WIDTHS, "FFN widths changed")
    _check(trial.get("parameter_counts") == PARAMS, "Parameter counts changed")
    _check(_number(trial.get("parameter_mismatch_fraction"), "parameter mismatch") <= 0.0003,
           "Parameter mismatch exceeds preregistered value")
    _check(trial.get("dataset_revision") == DATASET_REVISION, "Dataset revision mismatch")
    _check(trial.get("train_sha256") == TRAIN_SHA, "Training corpus hash mismatch")
    _check(trial.get("validation_sha256") == VAL_SHA, "Validation corpus hash mismatch")
    _check(trial.get("test_split_used") is False, "Held-out test split was used")
    _check(isinstance(trial.get("status"), str) and
           "EXPLORATORY FP32 BYTE-LM" in trial["status"], "Unexpected training variant")
    losses = trial.get("train_losses_nats")
    _check(isinstance(losses, list) and len(losses) == 100,
           "Expected 100 per-step training losses")
    for loss in losses:
        _number(loss, "training loss")
    val = trial.get("validation_per_batch_nats")
    _check(isinstance(val, list) and len(val) == 8,
           "Expected eight validation batches")
    for loss in val:
        _number(loss, "validation loss")
    bpb = _number(trial.get("mean_validation_bits_per_byte"), "validation BPB")
    _check(math.isclose(bpb, statistics.mean(val) / math.log(2), rel_tol=1e-8,
                        abs_tol=1e-8), "Validation BPB disagrees with raw batches")
    _check(bpb > 0, "BPB must be positive")
    seconds = _number(trial.get("block_wall_seconds"), "training wall time")
    peak_alloc = _number(trial.get("peak_allocated_mib"), "GPU allocated memory")
    peak_res = _number(trial.get("peak_reserved_mib"), "GPU reserved memory")
    _check(seconds > 0 and 0 < peak_alloc <= peak_res, "Invalid wall-time or GPU memory")
    _check(checkpoint.is_file(), "Checkpoint file missing")
    expected_hash = trial.get("checkpoint_sha256")
    _check(isinstance(expected_hash, str) and len(expected_hash) == 64,
           "Checkpoint SHA-256 missing")
    _check(sha256_file(checkpoint) == expected_hash, "Checkpoint SHA-256 mismatch")
    return {"kind": kind, "seed": seed, "validation_bpb": bpb,
            "block_wall_seconds": seconds,
            "peak_allocated_mib": peak_alloc, "peak_reserved_mib": peak_res,
            "checkpoint_sha256": expected_hash, "steps_completed": 100}


def _percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    where = (len(xs) - 1) * q
    low = math.floor(where)
    high = math.ceil(where)
    return xs[low] + (xs[high] - xs[low]) * (where - low)


def paired_summary(differences: list[float]) -> dict:
    """Descriptive paired statistics, not an architecture superiority test."""
    n = len(differences)
    summary = {"paired_seeds": n, "mean": statistics.mean(differences),
               "median": statistics.median(differences),
               "minimum": min(differences), "maximum": max(differences),
               "sample_sd": statistics.stdev(differences) if n > 1 else None,
               "negative_differences": sum(x < 0 for x in differences),
               "positive_differences": sum(x > 0 for x in differences)}
    if n >= 2:
        rng = random.Random(23012026)
        draws = [statistics.mean(rng.choices(differences, k=n)) for _ in range(10000)]
        summary["bootstrap_95pct_descriptive"] = [_percentile(draws, 0.025),
                                                  _percentile(draws, 0.975)]
        observed = abs(statistics.mean(differences))
        more_extreme = 0
        for signs in product((-1, 1), repeat=n):
            flipped = abs(statistics.mean(s * d for s, d in zip(signs, differences)))
            if flipped >= observed - 1e-12:
                more_extreme += 1
        summary["exact_two_sided_sign_flip_p_exploratory"] = more_extreme / (2 ** n)
    else:
        summary["bootstrap_95pct_descriptive"] = None
        summary["exact_two_sided_sign_flip_p_exploratory"] = None
    return summary


def audit(run_dir: Path, *, pilot: bool = False) -> dict:
    """All missing/invalid runs must be visible; incomplete data has no aggregate winner."""
    seeds = PILOT_SEEDS if pilot else SEEDS
    train_dir = run_dir / "train"
    valid = {}
    missing = []
    invalid = []
    for seed in seeds:
        for kind in KINDS:
            key = f"{kind}/seed{seed}"
            try:
                valid[(seed, kind)] = validate_trial(train_dir, kind, seed)
            except FileNotFoundError as exc:
                missing.append({"trial": key, "path": str(exc)})
            except (InvalidTrial, OSError, ValueError) as exc:
                invalid.append({"trial": key, "reason": str(exc)})
    per_seed = []
    paired = defaultdict(list)
    for seed in seeds:
        if all((seed, kind) in valid for kind in KINDS):
            values = {kind: valid[(seed, kind)]["validation_bpb"] for kind in KINDS}
            d_local = values["grassmann"] - values["local"]
            d_attention = values["grassmann"] - values["transformer"]
            paired["grassmann_minus_local"].append(d_local)
            paired["grassmann_minus_transformer"].append(d_attention)
            per_seed.append({"seed": seed, "values": values,
                             "grassmann_minus_local": d_local,
                             "grassmann_minus_transformer": d_attention})
    complete = not missing and not invalid and len(valid) == len(seeds) * 3
    report = {"protocol": "P2.3", "mode": "pilot" if pilot else "five_seed",
              "status": "complete" if complete else "incomplete",
              "intended_seeds": list(seeds), "expected_trials": len(seeds) * 3,
              "valid_trials": len(valid), "missing": missing, "invalid": invalid,
              "per_seed_complete_pairs": per_seed,
              "analysis_note": "Quality at matched 100-step training budget only; not compute-matched. "
                               "Results remain exploratory with small n and unequal receptive fields. "
                               "This script does not attest to live thermal gates; audit supervisor manifests separately."}
    if complete and not pilot:
        report["paired_summaries"] = {key: paired_summary(values)
                                      for key, values in paired.items()}
        report["model_means_bpb"] = {
            kind: statistics.mean(valid[(seed, kind)]["validation_bpb"] for seed in seeds)
            for kind in KINDS}
        report["interpretation"] = (
            "Descriptive five-seed cross-seed replication; no universal superiority claim. "
            "Exact two-sided sign-flip test cannot produce p<0.05 with n=5.")
    elif complete and pilot:
        report["interpretation"] = "Seed-17 systems pilot ONLY; do not rank architectures statistically."
    else:
        report["interpretation"] = "Incomplete matrix: do not report aggregate model rankings."
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--pilot", action="store_true",
                   help="Audit only preregistered seed-17 systems pilot")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="Show partial status without treating incomplete runs as success")
    args = p.parse_args(argv)
    result = audit(args.run_dir, pilot=args.pilot)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" or args.allow_incomplete else 2


if __name__ == "__main__":
    sys.exit(main())
