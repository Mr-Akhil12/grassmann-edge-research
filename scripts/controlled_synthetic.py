"""Predeclared, hardware-bounded synthetic comparison.

This is *not* the paper's WikiText-2 reproduction. It compares causal
Grassmann, causal Transformer and a non-geometric local mixer under exactly
the same task, train examples, optimizer, sequence length and seeds.
Limited parameter matching: report all actual counts and relative mismatch.

Usage:
 python -m scripts.controlled_synthetic --device cpu --steps 40 --seeds 11,17,23
 python -m scripts.controlled_synthetic --device cuda --steps 80 --lag 24
"""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from grassmann_edge import TinyLM, delayed_copy_batch
from grassmann_edge.paper_reference import PaperGrassmannLM


MODELS = ("grassmann", "transformer", "local")


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def build(kind, a):
    kw = dict(vocab=a.vocab, dim=a.dim, layers=a.layers,
              rank=a.rank, context=a.length, offsets=tuple(a.offsets))
    return (PaperGrassmannLM(**kw) if kind == "grassmann"
            else TinyLM(kind=kind, **kw))


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def data_batch(a, seed, step):
    # Independent, deterministic seed for each training step.
    return delayed_copy_batch(a.batch, a.length, a.vocab, a.lag,
                              seed * 100000 + step).to(a.device)


@torch.inference_mode()
def validate(model, a, seed):
    model.eval()
    losses = []
    for idx in range(a.eval_batches):
        seq = delayed_copy_batch(a.batch, a.length, a.vocab, a.lag,
                                 900000000 + seed * 100 + idx).to(a.device)
        pred = model(seq[:, :-1])
        loss = F.cross_entropy(pred.reshape(-1, a.vocab),
                               seq[:, 1:].reshape(-1))
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def run_one(kind, seed, a):
    torch.manual_seed(seed)
    model = build(kind, a).to(a.device)
    params = count(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    train_curve = []
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model.train()
    for i in range(a.steps):
        seq = data_batch(a, seed, i)
        pred = model(seq[:, :-1])
        loss = F.cross_entropy(pred.reshape(-1, a.vocab),
                               seq[:, 1:].reshape(-1))
        if not torch.isfinite(loss):
            raise ArithmeticError(f"Non-finite loss {kind} seed={seed} step={i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_curve.append(float(loss.item()))
    if a.device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    result = {
        "model": kind, "seed": seed, "parameters": params,
        "train_tokens": a.steps * a.batch * a.length,
        "train_losses": train_curve, "first_loss": train_curve[0],
        "last_loss": train_curve[-1],
        "train_seconds": seconds,
        "validation_cross_entropy_nats": validate(model, a, seed),
        "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated()
                                     if a.device == "cuda" else None),
    }
    del model, opt
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seeds", default="11,17,23")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--lag", type=int, default=4)
    p.add_argument("--vocab", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--max-seconds", type=int, default=180)
    p.add_argument("--out", default="results/controlled_synthetic.json")
    a = p.parse_args()
    seeds = [int(i) for i in a.seeds.split(",")]
    if not (1 <= len(seeds) <= 5 and len(set(seeds)) == len(seeds)
            and all(0 <= i <= 100000 for i in seeds)
            and 1 <= a.steps <= 150 and 1 <= a.batch <= 8
            and 16 <= a.length <= 128 and 8 <= a.dim <= 64
            and a.dim % 4 == 0 and 1 <= a.layers <= 4
            and 2 <= a.rank <= 16 and 2 <= a.vocab <= 256
            and 1 <= a.lag <= a.length and 1 <= a.eval_batches <= 8
            and 0 < a.lr <= 0.01 and 5 <= a.max_seconds <= 1200
            and len(a.offsets) <= 8 and min(a.offsets) >= 1):
        p.error("Unsafe or invalid bounded experiment configuration")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    all_counts = {kind: count(build(kind, a)) for kind in MODELS}
    mismatch = ((max(all_counts.values()) - min(all_counts.values())) /
                max(all_counts.values()))
    output = {
        "status": "EXPLORATORY SYNTHETIC ONLY; no paper quality claims",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_rev(), "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": a.device, "config": vars(a),
        "model_counts": all_counts, "parameter_mismatch_fraction": mismatch,
        "training_protocol": "equal steps/tokens/seed/data/batch/optimizer; not equal FLOPs",
        "data": "synthetic delayed-copy autoregressive next-token task",
        "runs": [], "summary": {}, "partial": False
    }
    total_start = time.perf_counter()
    try:
        # Fixed model order is declared, not tuned after results.
        for seed in seeds:
            for kind in MODELS:
                if time.perf_counter() - total_start > a.max_seconds:
                    output["partial"] = True
                    raise TimeoutError("Experiment deadline reached safely")
                result = run_one(kind, seed, a)
                output["runs"].append(result)
                print(f"{kind} seed={seed} loss={result['validation_cross_entropy_nats']:.5f}",
                      flush=True)
    except (TimeoutError, ArithmeticError, RuntimeError) as exc:
        output["partial"] = True
        output["error"] = str(exc)
        print("PARTIAL RUN", str(exc), flush=True)
    for kind in MODELS:
        samples = [x["validation_cross_entropy_nats"]
                   for x in output["runs"] if x["model"] == kind]
        if samples:
            output["summary"][kind] = {
                "n": len(samples), "mean_cross_entropy": statistics.mean(samples),
                "sample_sd": statistics.stdev(samples) if len(samples) > 1 else None}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Saved", out, "partial=", output["partial"], flush=True)
    if output["partial"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
