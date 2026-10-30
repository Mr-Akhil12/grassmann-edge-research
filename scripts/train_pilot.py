"""Bounded synthetic diagnostic ONLY. Not a quality-matched benchmark."""
import argparse
import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch.nn import functional as F
from grassmann_edge import TinyLM, delayed_copy_batch, model_parameters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("all", "grassmann", "transformer", "local"), default="all")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lag", type=int, default=4)
    p.add_argument("--vocab", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    p.add_argument("--out", default="results/pilot.json")
    a = p.parse_args()
    if not (1 <= a.steps <= 100 and 1 <= a.batch <= 8 and 4 <= a.length <= 128
            and 8 <= a.dim <= 128 and 1 <= a.layers <= 4 and 2 <= a.rank <= 24
            and 1 <= a.lag <= a.length and a.dim % 4 == 0):
        p.error("Pilot argument outside conservative safe bounds")
    device = "cuda" if a.device == "auto" and torch.cuda.is_available() else (
        "cpu" if a.device == "auto" else a.device)
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    kinds = ("grassmann", "transformer", "local") if a.model == "all" else (a.model,)
    result = {"status": "synthetic diagnostic - NOT architecture benchmark",
              "date_utc": datetime.now(timezone.utc).isoformat(),
              "device": device, "torch": torch.__version__,
              "python": platform.python_version(), "args": vars(a), "runs": []}
    for kind in kinds:
        random.seed(a.seed)
        torch.manual_seed(a.seed)
        model = TinyLM(kind, vocab=a.vocab, dim=a.dim, layers=a.layers,
                       rank=a.rank, context=a.length).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        first = None
        losses = []
        t0 = time.perf_counter()
        for step in range(a.steps):
            seq = delayed_copy_batch(a.batch, a.length, a.vocab, a.lag,
                                     a.seed * 100000 + step).to(device)
            logits = model(seq[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, a.vocab),
                                   seq[:, 1:].reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {kind} step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.item()))
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        model.eval()
        with torch.inference_mode():
            val = []
            for e in range(3):
                seq = delayed_copy_batch(a.batch, a.length, a.vocab,
                                         a.lag, 900000000 + a.seed * 100 + e).to(device)
                logits = model(seq[:, :-1])
                val.append(float(F.cross_entropy(
                    logits.reshape(-1, a.vocab), seq[:, 1:].reshape(-1)).item()))
        item = {"kind": kind, "parameters": model_parameters(model),
                "first_loss": losses[0], "last_loss": losses[-1],
                "validation_loss": sum(val) / len(val),
                "seconds": elapsed, "losses": losses}
        print(kind, item["parameters"], "params", "val_loss", round(item["validation_loss"], 4))
        result["runs"].append(item)
        del model, optimizer
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
